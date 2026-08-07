"""Strict enterprise change/admission review case domain and persistence.

The model-facing boundary ends at :class:`ModelRecommendation`.  A model can
never turn a case into an approved or rejected state: those transitions require
an explicit, host-bound :class:`HumanActor` whose identity matches the trusted
access context.  Every durable mutation is scope-filtered, revision-CAS guarded,
idempotent, and accompanied by an append-only audit event.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from .case_profiles import ResearchSurveyDepth
from .domain import StrictModel, utc_now

_MAX_JSON_BYTES = 1_000_000
_MAX_JSON_DEPTH = 50
_MAX_JSON_NODES = 100_000
_ADMIN_ROLE = "case_admin"


class ReviewCaseError(RuntimeError):
    """Base error for review case commands."""


class ReviewCaseNotFoundError(ReviewCaseError):
    """The case is absent from the caller's exact ownership scope."""


class CaseAccessDeniedError(ReviewCaseError):
    """The trusted actor is not allowed to use the requested owner scope."""


class CaseIdempotencyConflictError(ReviewCaseError):
    """An idempotency key was reused for a different semantic command."""


class CaseRevisionConflictError(ReviewCaseError):
    """The supplied optimistic revision is stale."""


class CaseInvalidTransitionError(ReviewCaseError):
    """A command is not legal from the case's current state."""


class CaseDecisionRuleError(ReviewCaseError):
    """A recommendation, evidence binding, or human decision rule failed."""


class CaseKind(str, Enum):
    ENTERPRISE_CHANGE = "enterprise_change"
    ENTERPRISE_ADMISSION = "enterprise_admission"
    RESEARCH_SURVEY = "research_survey"


class CaseStatus(str, Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    RUNNING = "running"
    WAITING_HUMAN_REVIEW = "waiting_human_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    FAILED = "failed"


class RecommendationOutcome(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"


class AuditEventType(str, Enum):
    CREATED = "case_created"
    DRAFT_UPDATED = "draft_updated"
    SUBMITTED = "case_submitted"
    STARTED = "review_started"
    RECOMMENDATION_RECORDED = "model_recommendation_recorded"
    APPROVED = "case_approved"
    REJECTED = "case_rejected"
    FAILED = "case_failed"


def _reject_invalid_unicode(value: str) -> str:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError("strings must contain Unicode scalar values")
    return value


def _nonblank(value: Any) -> Any:
    if isinstance(value, str):
        _reject_invalid_unicode(value)
        if not value.strip():
            raise ValueError("value must not be blank")
    return value


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _utc(value: datetime | None) -> datetime:
    return _aware_utc(value or utc_now())


def _validate_json_tree(
    value: Any,
    *,
    depth: int = 0,
    node_count: list[int] | None = None,
) -> None:
    if depth > _MAX_JSON_DEPTH:
        raise ValueError("JSON payload exceeds the maximum nesting depth")
    counter = node_count if node_count is not None else [0]
    counter[0] += 1
    if counter[0] > _MAX_JSON_NODES:
        raise ValueError("JSON payload exceeds the maximum node count")

    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        _reject_invalid_unicode(value)
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            _validate_json_tree(key, depth=depth + 1, node_count=counter)
            _validate_json_tree(item, depth=depth + 1, node_count=counter)
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for item in value:
            _validate_json_tree(item, depth=depth + 1, node_count=counter)
        return
    raise ValueError("value must contain JSON-compatible data")


def _canonical_json(value: Any, *, max_bytes: int = _MAX_JSON_BYTES) -> str:
    _validate_json_tree(value)
    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value must contain finite JSON-compatible data") from exc
    if len(payload.encode("utf-8")) > max_bytes:
        raise ValueError(f"JSON payload exceeds the {max_bytes} byte limit")
    return payload


def _model_json(value: StrictModel) -> str:
    return _canonical_json(value.model_dump(mode="json"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class CaseAccess(StrictModel):
    """Trusted, owner-specific access supplied by the authenticated host.

    An administrator does not gain ambient tenant-wide access merely by having
    a role.  Acting for another owner requires both ``case_admin`` and the
    explicit ``admin_override`` flag, and all queries remain pinned to the
    supplied owner and conversation.
    """

    tenant_id: str = Field(min_length=1, max_length=200)
    owner_user_id: str = Field(min_length=1, max_length=200)
    conversation_id: str = Field(min_length=1, max_length=240)
    actor_user_id: str | None = Field(default=None, min_length=1, max_length=200)
    actor_roles: frozenset[str] = Field(default_factory=frozenset, max_length=32)
    admin_override: bool = False

    @field_validator(
        "tenant_id", "owner_user_id", "conversation_id", "actor_user_id",
        mode="before",
    )
    @classmethod
    def identifiers_are_nonblank(cls, value: Any) -> Any:
        if value is None:
            return value
        return _nonblank(value)

    @field_validator("actor_roles", mode="before")
    @classmethod
    def roles_are_unique_and_safe(cls, value: Any) -> Any:
        if value is None:
            return frozenset()
        if isinstance(value, (str, bytes, bytearray)):
            raise ValueError("actor_roles must be a collection of role names")
        roles = list(value)
        if len(roles) != len(set(roles)):
            raise ValueError("actor roles must be unique")
        for role in roles:
            if not isinstance(role, str) or not role.strip() or len(role) > 120:
                raise ValueError("actor roles must be non-blank bounded strings")
            _reject_invalid_unicode(role)
        return frozenset(roles)

    @model_validator(mode="after")
    def admin_override_is_explicitly_authorized(self) -> CaseAccess:
        if self.admin_override and _ADMIN_ROLE not in self.actor_roles:
            raise ValueError("admin_override requires the case_admin role")
        return self

    @property
    def effective_actor_user_id(self) -> str:
        return self.actor_user_id or self.owner_user_id


class EvidenceRef(StrictModel):
    evidence_id: str = Field(min_length=1, max_length=240)
    source_type: Literal["document", "artifact", "tool_receipt", "url", "case"]
    locator: str = Field(min_length=1, max_length=2_048)
    title: str | None = Field(default=None, max_length=500)
    version: str | None = Field(default=None, max_length=160)
    checksum_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    page_number: int | None = Field(default=None, ge=1, le=1_000_000)
    excerpt: str = Field(min_length=1, max_length=16_000)

    @field_validator(
        "evidence_id", "locator", "title", "version", "excerpt", mode="before"
    )
    @classmethod
    def text_is_safe(cls, value: Any) -> Any:
        if value is None:
            return value
        return _nonblank(value)


class CaseSubmission(StrictModel):
    request_summary: str = Field(min_length=1, max_length=16_000)
    business_justification: str = Field(min_length=1, max_length=16_000)
    attributes: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list, max_length=100)

    @field_validator("request_summary", "business_justification", mode="before")
    @classmethod
    def narrative_is_nonblank(cls, value: Any) -> Any:
        return _nonblank(value)

    @model_validator(mode="after")
    def payload_is_bounded_and_evidence_is_unique(self) -> CaseSubmission:
        identifiers = [item.evidence_id for item in self.evidence_refs]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("submission evidence IDs must be unique")
        # Inspect the Python value before Pydantic's JSON-mode serializer can
        # normalise a non-finite float to null.
        _canonical_json(self.attributes)
        _canonical_json(self.model_dump(mode="json"))
        return self


class ModelRecommendation(StrictModel):
    """Untrusted model conclusion; it carries no final decision authority."""

    recommendation_id: str = Field(min_length=1, max_length=240)
    model_run_id: str = Field(min_length=1, max_length=240)
    model_id: str = Field(min_length=1, max_length=240)
    outcome: RecommendationOutcome
    summary: str = Field(min_length=1, max_length=16_000)
    rationale: str = Field(min_length=1, max_length=32_000)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_refs: list[EvidenceRef] = Field(min_length=1, max_length=100)
    authority: Literal["model_untrusted"] = "model_untrusted"
    produced_at: datetime

    @field_validator(
        "recommendation_id", "model_run_id", "model_id", "summary", "rationale",
        mode="before",
    )
    @classmethod
    def text_is_nonblank(cls, value: Any) -> Any:
        return _nonblank(value)

    @field_validator("produced_at")
    @classmethod
    def produced_at_is_aware(cls, value: datetime) -> datetime:
        return _aware_utc(value)

    @model_validator(mode="after")
    def recommendation_is_bounded(self) -> ModelRecommendation:
        identifiers = [item.evidence_id for item in self.evidence_refs]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("recommendation evidence IDs must be unique")
        _canonical_json(self.model_dump(mode="json"))
        return self


class HumanActor(StrictModel):
    actor_user_id: str = Field(min_length=1, max_length=200)
    display_name: str | None = Field(default=None, max_length=240)
    authority: Literal["human"] = "human"

    @field_validator("actor_user_id", "display_name", mode="before")
    @classmethod
    def text_is_safe(cls, value: Any) -> Any:
        if value is None:
            return value
        return _nonblank(value)


class HostActor(StrictModel):
    """Non-model host identity allowed to start or fail processing."""

    actor_id: str = Field(min_length=1, max_length=240)
    authority: Literal["human", "system", "tool"]

    @field_validator("actor_id", mode="before")
    @classmethod
    def actor_id_is_safe(cls, value: Any) -> Any:
        return _nonblank(value)


class HumanDecision(StrictModel):
    outcome: Literal[CaseStatus.APPROVED, CaseStatus.REJECTED]
    actor: HumanActor
    rationale: str = Field(min_length=1, max_length=16_000)
    evidence_ref_ids: list[str] = Field(default_factory=list, max_length=100)
    decided_at: datetime

    @field_validator("rationale", mode="before")
    @classmethod
    def rationale_is_nonblank(cls, value: Any) -> Any:
        return _nonblank(value)

    @field_validator("evidence_ref_ids", mode="before")
    @classmethod
    def evidence_ids_are_unique(cls, value: Any) -> Any:
        if isinstance(value, (str, bytes, bytearray)):
            raise ValueError("evidence_ref_ids must be a collection of IDs")
        values = list(value or [])
        if len(values) != len(set(values)):
            raise ValueError("decision evidence IDs must be unique")
        for item in values:
            _nonblank(item)
        return values

    @field_validator("decided_at")
    @classmethod
    def decided_at_is_aware(cls, value: datetime) -> datetime:
        return _aware_utc(value)


class CaseFailure(StrictModel):
    reason: str = Field(min_length=1, max_length=16_000)
    actor: HostActor
    failed_at: datetime

    @field_validator("reason", mode="before")
    @classmethod
    def reason_is_nonblank(cls, value: Any) -> Any:
        return _nonblank(value)

    @field_validator("failed_at")
    @classmethod
    def failed_at_is_aware(cls, value: datetime) -> datetime:
        return _aware_utc(value)


class ReviewCase(StrictModel):
    case_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=240)
    tenant_id: str = Field(min_length=1, max_length=200)
    owner_user_id: str = Field(min_length=1, max_length=200)
    conversation_id: str = Field(min_length=1, max_length=240)
    kind: CaseKind
    title: str = Field(min_length=1, max_length=500)
    submission: CaseSubmission
    survey_depth: ResearchSurveyDepth = ResearchSurveyDepth.RIGOROUS
    status: CaseStatus = CaseStatus.DRAFT
    recommendation: ModelRecommendation | None = None
    human_decision: HumanDecision | None = None
    failure: CaseFailure | None = None
    revision: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    submitted_at: datetime | None = None
    started_at: datetime | None = None
    review_requested_at: datetime | None = None
    resolved_at: datetime | None = None

    @field_validator(
        "case_id", "tenant_id", "owner_user_id", "conversation_id", "title",
        mode="before",
    )
    @classmethod
    def text_is_safe(cls, value: Any) -> Any:
        return _nonblank(value)

    @field_validator(
        "created_at", "updated_at", "submitted_at", "started_at",
        "review_requested_at", "resolved_at",
    )
    @classmethod
    def timestamps_are_aware(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _aware_utc(value)

    @model_validator(mode="after")
    def lifecycle_fields_are_consistent(self) -> ReviewCase:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        milestones = [
            value
            for value in (
                self.created_at,
                self.submitted_at,
                self.started_at,
                self.review_requested_at,
                self.resolved_at,
            )
            if value is not None
        ]
        if any(later < earlier for earlier, later in zip(milestones, milestones[1:])):
            raise ValueError("case lifecycle timestamps must be chronological")
        if any(value > self.updated_at for value in milestones):
            raise ValueError("case lifecycle timestamps cannot exceed updated_at")
        if self.status == CaseStatus.DRAFT:
            if any(
                value is not None
                for value in (
                    self.submitted_at,
                    self.started_at,
                    self.review_requested_at,
                    self.resolved_at,
                    self.recommendation,
                    self.human_decision,
                    self.failure,
                )
            ):
                raise ValueError("a draft cannot contain processing or decision state")
        else:
            if self.submitted_at is None:
                raise ValueError("a non-draft case requires submitted_at")

        if self.status == CaseStatus.SUBMITTED:
            if any(
                value is not None
                for value in (
                    self.started_at,
                    self.review_requested_at,
                    self.resolved_at,
                    self.recommendation,
                    self.human_decision,
                    self.failure,
                )
            ):
                raise ValueError("a submitted case cannot contain later state")

        if self.status == CaseStatus.RUNNING:
            if self.started_at is None:
                raise ValueError("a running case requires started_at")
            if any(
                value is not None
                for value in (
                    self.review_requested_at,
                    self.resolved_at,
                    self.recommendation,
                    self.human_decision,
                    self.failure,
                )
            ):
                raise ValueError("a running case cannot contain review outcome state")

        if self.status in {
            CaseStatus.WAITING_HUMAN_REVIEW,
            CaseStatus.APPROVED,
            CaseStatus.REJECTED,
        }:
            if self.started_at is None or self.review_requested_at is None:
                raise ValueError("human review state requires processing timestamps")
            if self.recommendation is None:
                raise ValueError("human review state requires a model recommendation")
            if self.recommendation.produced_at > self.review_requested_at:
                raise ValueError("model recommendation cannot postdate review request")

        if self.status == CaseStatus.WAITING_HUMAN_REVIEW:
            if any(value is not None for value in (self.human_decision, self.failure, self.resolved_at)):
                raise ValueError("waiting review cannot contain a terminal outcome")

        if self.status in {CaseStatus.APPROVED, CaseStatus.REJECTED}:
            if self.human_decision is None or self.resolved_at is None:
                raise ValueError("approved/rejected cases require a human decision")
            if self.human_decision.outcome != self.status:
                raise ValueError("human decision must match the terminal case status")
            if self.human_decision.decided_at != self.resolved_at:
                raise ValueError("human decision time must match resolved_at")
            if self.failure is not None:
                raise ValueError("a decided case cannot also be failed")
        elif self.human_decision is not None:
            raise ValueError("only approved/rejected cases may contain a human decision")

        if self.status == CaseStatus.FAILED:
            if self.failure is None or self.resolved_at is None:
                raise ValueError("a failed case requires failure details")
            if self.failure.failed_at != self.resolved_at:
                raise ValueError("failure time must match resolved_at")
        elif self.failure is not None:
            raise ValueError("only a failed case may contain failure details")

        if self.status not in {
            CaseStatus.APPROVED,
            CaseStatus.REJECTED,
            CaseStatus.FAILED,
        } and self.resolved_at is not None:
            raise ValueError("only terminal cases may contain resolved_at")

        _canonical_json(self.model_dump(mode="json"))
        return self


class CaseAuditEvent(StrictModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    tenant_id: str = Field(min_length=1)
    owner_user_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    event_type: AuditEventType
    revision: int = Field(ge=1)
    from_status: CaseStatus | None = None
    to_status: CaseStatus
    actor_id: str = Field(min_length=1, max_length=240)
    actor_authority: Literal["human", "model_untrusted", "system", "tool"]
    idempotency_key: str = Field(min_length=1, max_length=240)
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("created_at")
    @classmethod
    def created_at_is_aware(cls, value: datetime) -> datetime:
        return _aware_utc(value)

    @model_validator(mode="after")
    def details_are_bounded(self) -> CaseAuditEvent:
        _canonical_json(self.model_dump(mode="json"))
        return self


_ALLOWED_TRANSITIONS: dict[CaseStatus, set[CaseStatus]] = {
    CaseStatus.DRAFT: {CaseStatus.SUBMITTED},
    CaseStatus.SUBMITTED: {CaseStatus.RUNNING, CaseStatus.FAILED},
    CaseStatus.RUNNING: {CaseStatus.WAITING_HUMAN_REVIEW, CaseStatus.FAILED},
    CaseStatus.WAITING_HUMAN_REVIEW: {
        CaseStatus.APPROVED,
        CaseStatus.REJECTED,
        CaseStatus.FAILED,
    },
    CaseStatus.APPROVED: set(),
    CaseStatus.REJECTED: set(),
    CaseStatus.FAILED: set(),
}


class SQLiteReviewCaseStore:
    """SQLite repository for strict, owner-scoped review cases."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialise(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS review_cases (
                    case_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    owner_user_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN (
                        'draft', 'submitted', 'running', 'waiting_human_review',
                        'approved', 'rejected', 'failed'
                    )),
                    revision INTEGER NOT NULL CHECK(revision >= 1),
                    case_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS review_case_scope_idx
                    ON review_cases(
                        tenant_id, owner_user_id, conversation_id,
                        updated_at DESC
                    );

                CREATE TABLE IF NOT EXISTS review_case_commands (
                    tenant_id TEXT NOT NULL,
                    owner_user_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    command_type TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    result_revision INTEGER NOT NULL,
                    result_case_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(
                        tenant_id, owner_user_id, conversation_id,
                        idempotency_key
                    ),
                    FOREIGN KEY(case_id) REFERENCES review_cases(case_id)
                );

                CREATE TABLE IF NOT EXISTS review_case_audit_events (
                    event_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    owner_user_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(case_id, revision),
                    FOREIGN KEY(case_id) REFERENCES review_cases(case_id)
                );
                CREATE INDEX IF NOT EXISTS review_case_audit_scope_idx
                    ON review_case_audit_events(
                        tenant_id, owner_user_id, conversation_id,
                        case_id, revision
                    );
                """
            )
            connection.commit()

    def create_case(
        self,
        access: CaseAccess,
        *,
        kind: CaseKind,
        title: str,
        submission: CaseSubmission | Mapping[str, Any],
        idempotency_key: str,
        survey_depth: ResearchSurveyDepth = ResearchSurveyDepth.RIGOROUS,
        case_id: str | None = None,
        now: datetime | None = None,
    ) -> ReviewCase:
        _require_access(access)
        key = _validate_idempotency_key(idempotency_key)
        validated_submission = (
            submission
            if isinstance(submission, CaseSubmission)
            else CaseSubmission.model_validate(submission)
        )
        kind = CaseKind(kind)
        survey_depth = ResearchSurveyDepth(survey_depth)
        timestamp = _utc(now)
        request_hash = _command_hash(
            access,
            "create_case",
            {
                "kind": kind.value,
                "title": title,
                "submission": validated_submission.model_dump(mode="json"),
                "survey_depth": survey_depth.value,
                "requested_case_id": case_id,
            },
        )

        with self._transaction() as connection:
            replay = self._command_replay(
                connection, access, key, request_hash
            )
            if replay is not None:
                return replay

            review_case = ReviewCase(
                case_id=case_id or str(uuid4()),
                tenant_id=access.tenant_id,
                owner_user_id=access.owner_user_id,
                conversation_id=access.conversation_id,
                kind=kind,
                title=title,
                submission=validated_submission,
                survey_depth=survey_depth,
                created_at=timestamp,
                updated_at=timestamp,
            )
            try:
                connection.execute(
                    """
                    INSERT INTO review_cases(
                        case_id, tenant_id, owner_user_id, conversation_id,
                        status, revision, case_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        review_case.case_id,
                        review_case.tenant_id,
                        review_case.owner_user_id,
                        review_case.conversation_id,
                        review_case.status.value,
                        review_case.revision,
                        _model_json(review_case),
                        timestamp.timestamp(),
                        timestamp.timestamp(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise CaseIdempotencyConflictError(
                    "review case identity already exists"
                ) from exc
            event = self._make_event(
                review_case,
                event_type=AuditEventType.CREATED,
                from_status=None,
                actor_id=access.effective_actor_user_id,
                actor_authority="human",
                idempotency_key=key,
                request_hash=request_hash,
                details={"kind": kind.value},
                created_at=timestamp,
            )
            self._insert_event(connection, event)
            self._insert_command(
                connection, access, key, "create_case", request_hash,
                review_case, timestamp,
            )
            return review_case.model_copy(deep=True)

    def get_case(self, access: CaseAccess, case_id: str) -> ReviewCase:
        _require_access(access)
        case_id = _validate_case_id(case_id)
        with self._connection() as connection:
            return self._case_in_transaction(connection, access, case_id)

    def list_cases(
        self,
        access: CaseAccess,
        *,
        statuses: Sequence[CaseStatus] | None = None,
        limit: int = 100,
    ) -> list[ReviewCase]:
        _require_access(access)
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        validated_statuses = (
            [CaseStatus(status) for status in statuses]
            if statuses is not None
            else None
        )
        if validated_statuses is not None and not validated_statuses:
            return []
        query = (
            "SELECT case_json FROM review_cases "
            "WHERE tenant_id = ? AND owner_user_id = ? AND conversation_id = ?"
        )
        parameters: list[Any] = [
            access.tenant_id,
            access.owner_user_id,
            access.conversation_id,
        ]
        if validated_statuses is not None:
            placeholders = ",".join("?" for _ in validated_statuses)
            query += f" AND status IN ({placeholders})"
            parameters.extend(status.value for status in validated_statuses)
        query += " ORDER BY updated_at DESC, case_id ASC LIMIT ?"
        parameters.append(limit)
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [ReviewCase.model_validate_json(row["case_json"]) for row in rows]

    def list_owned_cases(
        self,
        access: CaseAccess,
        *,
        statuses: Sequence[CaseStatus] | None = None,
        limit: int = 100,
    ) -> list[ReviewCase]:
        """List an owner's cases across their isolated conversations.

        Every case uses a unique conversation so orchestration facts and role
        memory cannot bleed between cases.  The workbench still needs an
        owner-level inbox, hence this separate query.  Point reads and every
        mutation continue to require the exact conversation ID.
        """

        _require_access(access)
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        validated_statuses = (
            [CaseStatus(status) for status in statuses]
            if statuses is not None
            else None
        )
        if validated_statuses is not None and not validated_statuses:
            return []
        query = (
            "SELECT case_json FROM review_cases "
            "WHERE tenant_id = ? AND owner_user_id = ?"
        )
        parameters: list[Any] = [access.tenant_id, access.owner_user_id]
        if validated_statuses is not None:
            placeholders = ",".join("?" for _ in validated_statuses)
            query += f" AND status IN ({placeholders})"
            parameters.extend(status.value for status in validated_statuses)
        query += " ORDER BY updated_at DESC, case_id ASC LIMIT ?"
        parameters.append(limit)
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [ReviewCase.model_validate_json(row["case_json"]) for row in rows]

    def update_draft(
        self,
        access: CaseAccess,
        case_id: str,
        *,
        expected_revision: int,
        idempotency_key: str,
        title: str | None = None,
        submission: CaseSubmission | Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> ReviewCase:
        _require_access(access)
        if title is None and submission is None:
            raise ValueError("a draft update requires title or submission")
        validated_submission = (
            None
            if submission is None
            else submission
            if isinstance(submission, CaseSubmission)
            else CaseSubmission.model_validate(submission)
        )

        def transform(current: ReviewCase, timestamp: datetime) -> ReviewCase:
            if current.status != CaseStatus.DRAFT:
                raise CaseInvalidTransitionError("only a draft may be edited")
            return _replace_case(
                current,
                title=current.title if title is None else title,
                submission=(
                    current.submission
                    if validated_submission is None
                    else validated_submission
                ),
                revision=current.revision + 1,
                updated_at=timestamp,
            )

        return self._mutate(
            access,
            case_id,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            command_type="update_draft",
            request_body={
                "title": title,
                "submission": (
                    None
                    if validated_submission is None
                    else validated_submission.model_dump(mode="json")
                ),
            },
            event_type=AuditEventType.DRAFT_UPDATED,
            actor_id=access.effective_actor_user_id,
            actor_authority="human",
            details={
                "title_changed": title is not None,
                "submission_changed": validated_submission is not None,
            },
            transform=transform,
            now=now,
        )

    def submit_case(
        self,
        access: CaseAccess,
        case_id: str,
        *,
        expected_revision: int,
        idempotency_key: str,
        now: datetime | None = None,
    ) -> ReviewCase:
        _require_access(access)

        def transform(current: ReviewCase, timestamp: datetime) -> ReviewCase:
            _require_transition(current.status, CaseStatus.SUBMITTED)
            if not current.submission.evidence_refs and (
                current.kind != CaseKind.RESEARCH_SURVEY
            ):
                raise CaseDecisionRuleError(
                    "a submitted enterprise review requires at least one evidence reference"
                )
            return _replace_case(
                current,
                status=CaseStatus.SUBMITTED,
                revision=current.revision + 1,
                updated_at=timestamp,
                submitted_at=timestamp,
            )

        return self._mutate(
            access,
            case_id,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            command_type="submit_case",
            request_body={},
            event_type=AuditEventType.SUBMITTED,
            actor_id=access.effective_actor_user_id,
            actor_authority="human",
            details={},
            transform=transform,
            now=now,
        )

    def start_case(
        self,
        access: CaseAccess,
        case_id: str,
        *,
        expected_revision: int,
        idempotency_key: str,
        actor: HostActor,
        now: datetime | None = None,
    ) -> ReviewCase:
        _require_access(access)
        if not isinstance(actor, HostActor):
            raise TypeError("start_case requires a trusted HostActor")
        _require_host_actor_binding(access, actor)

        def transform(current: ReviewCase, timestamp: datetime) -> ReviewCase:
            _require_transition(current.status, CaseStatus.RUNNING)
            return _replace_case(
                current,
                status=CaseStatus.RUNNING,
                revision=current.revision + 1,
                updated_at=timestamp,
                started_at=timestamp,
            )

        return self._mutate(
            access,
            case_id,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            command_type="start_case",
            request_body={"actor": actor.model_dump(mode="json")},
            event_type=AuditEventType.STARTED,
            actor_id=actor.actor_id,
            actor_authority=actor.authority,
            details={},
            transform=transform,
            now=now,
        )

    start_review = start_case

    def submit_model_recommendation(
        self,
        access: CaseAccess,
        case_id: str,
        *,
        expected_revision: int,
        idempotency_key: str,
        recommendation: ModelRecommendation | Mapping[str, Any],
        now: datetime | None = None,
    ) -> ReviewCase:
        _require_access(access)
        validated = (
            recommendation
            if isinstance(recommendation, ModelRecommendation)
            else ModelRecommendation.model_validate(recommendation)
        )

        def transform(current: ReviewCase, timestamp: datetime) -> ReviewCase:
            _require_transition(current.status, CaseStatus.WAITING_HUMAN_REVIEW)
            if current.kind != CaseKind.RESEARCH_SURVEY:
                # A survey's recommendation cites the retrieved corpus, not the
                # submitted evidence list, so the exact-match binding is skipped.
                _require_bound_evidence(
                    current.submission.evidence_refs, validated.evidence_refs
                )
            return _replace_case(
                current,
                status=CaseStatus.WAITING_HUMAN_REVIEW,
                recommendation=validated,
                revision=current.revision + 1,
                updated_at=timestamp,
                review_requested_at=timestamp,
            )

        return self._mutate(
            access,
            case_id,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            command_type="submit_model_recommendation",
            request_body={"recommendation": validated.model_dump(mode="json")},
            event_type=AuditEventType.RECOMMENDATION_RECORDED,
            actor_id=validated.model_run_id,
            actor_authority="model_untrusted",
            details={
                "recommendation_id": validated.recommendation_id,
                "outcome": validated.outcome.value,
                "evidence_count": len(validated.evidence_refs),
            },
            transform=transform,
            now=now,
        )

    record_model_recommendation = submit_model_recommendation

    def decide_case(
        self,
        access: CaseAccess,
        case_id: str,
        *,
        expected_revision: int,
        idempotency_key: str,
        outcome: Literal[CaseStatus.APPROVED, CaseStatus.REJECTED],
        human_actor: HumanActor,
        rationale: str,
        evidence_ref_ids: Sequence[str] = (),
        now: datetime | None = None,
    ) -> ReviewCase:
        _require_access(access)
        if not isinstance(human_actor, HumanActor):
            raise TypeError("approve/reject requires an explicit HumanActor")
        if human_actor.actor_user_id != access.effective_actor_user_id:
            raise CaseAccessDeniedError(
                "human decision actor does not match the trusted access actor"
            )
        outcome = CaseStatus(outcome)
        if outcome not in {CaseStatus.APPROVED, CaseStatus.REJECTED}:
            raise CaseDecisionRuleError("human outcome must be approved or rejected")
        evidence_ids = list(evidence_ref_ids)
        decision_timestamp = _utc(now)
        decision = HumanDecision(
            outcome=outcome,
            actor=human_actor,
            rationale=rationale,
            evidence_ref_ids=evidence_ids,
            decided_at=decision_timestamp,
        )

        def transform(current: ReviewCase, timestamp: datetime) -> ReviewCase:
            _require_transition(current.status, outcome)
            known = {item.evidence_id for item in current.submission.evidence_refs}
            unknown = sorted(set(decision.evidence_ref_ids) - known)
            if unknown:
                raise CaseDecisionRuleError(
                    f"human decision cites unknown evidence IDs: {unknown}"
                )
            return _replace_case(
                current,
                status=outcome,
                human_decision=decision,
                revision=current.revision + 1,
                updated_at=timestamp,
                resolved_at=timestamp,
            )

        return self._mutate(
            access,
            case_id,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            command_type="decide_case",
            request_body={
                "outcome": outcome.value,
                "human_actor": human_actor.model_dump(mode="json"),
                "rationale": rationale,
                "evidence_ref_ids": evidence_ids,
            },
            event_type=(
                AuditEventType.APPROVED
                if outcome == CaseStatus.APPROVED
                else AuditEventType.REJECTED
            ),
            actor_id=human_actor.actor_user_id,
            actor_authority="human",
            details={
                "outcome": outcome.value,
                "evidence_ref_ids": evidence_ids,
            },
            transform=transform,
            now=decision_timestamp,
        )

    def fail_case(
        self,
        access: CaseAccess,
        case_id: str,
        *,
        expected_revision: int,
        idempotency_key: str,
        actor: HostActor,
        reason: str,
        now: datetime | None = None,
    ) -> ReviewCase:
        _require_access(access)
        if not isinstance(actor, HostActor):
            raise TypeError("fail_case requires a trusted non-model HostActor")
        _require_host_actor_binding(access, actor)
        timestamp = _utc(now)
        failure = CaseFailure(reason=reason, actor=actor, failed_at=timestamp)

        def transform(current: ReviewCase, command_time: datetime) -> ReviewCase:
            _require_transition(current.status, CaseStatus.FAILED)
            return _replace_case(
                current,
                status=CaseStatus.FAILED,
                failure=failure,
                revision=current.revision + 1,
                updated_at=command_time,
                resolved_at=command_time,
            )

        return self._mutate(
            access,
            case_id,
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            command_type="fail_case",
            request_body={
                "actor": actor.model_dump(mode="json"),
                "reason": reason,
            },
            event_type=AuditEventType.FAILED,
            actor_id=actor.actor_id,
            actor_authority=actor.authority,
            details={"reason_hash": _sha256(reason)},
            transform=transform,
            now=timestamp,
        )

    def list_audit_events(
        self, access: CaseAccess, case_id: str
    ) -> list[CaseAuditEvent]:
        _require_access(access)
        with self._connection() as connection:
            self._case_in_transaction(connection, access, case_id)
            rows = connection.execute(
                """
                SELECT event_json FROM review_case_audit_events
                 WHERE tenant_id = ? AND owner_user_id = ?
                   AND conversation_id = ? AND case_id = ?
                 ORDER BY revision ASC
                """,
                (
                    access.tenant_id,
                    access.owner_user_id,
                    access.conversation_id,
                    case_id,
                ),
            ).fetchall()
        return [CaseAuditEvent.model_validate_json(row["event_json"]) for row in rows]

    def _mutate(
        self,
        access: CaseAccess,
        case_id: str,
        *,
        expected_revision: int,
        idempotency_key: str,
        command_type: str,
        request_body: Mapping[str, Any],
        event_type: AuditEventType,
        actor_id: str,
        actor_authority: Literal["human", "model_untrusted", "system", "tool"],
        details: Mapping[str, Any],
        transform: Callable[[ReviewCase, datetime], ReviewCase],
        now: datetime | None,
    ) -> ReviewCase:
        _require_access(access)
        case_id = _validate_case_id(case_id)
        if not isinstance(expected_revision, int) or expected_revision < 1:
            raise ValueError("expected_revision must be a positive integer")
        key = _validate_idempotency_key(idempotency_key)
        timestamp = _utc(now)
        request_hash = _command_hash(
            access,
            command_type,
            {
                "case_id": case_id,
                "expected_revision": expected_revision,
                **dict(request_body),
            },
        )
        safe_details = dict(details)
        _canonical_json(safe_details)

        with self._transaction() as connection:
            replay = self._command_replay(
                connection, access, key, request_hash
            )
            if replay is not None:
                return replay
            current = self._case_in_transaction(connection, access, case_id)
            if current.revision != expected_revision:
                raise CaseRevisionConflictError(
                    f"case revision is stale: expected {expected_revision}, "
                    f"current {current.revision}"
                )
            updated = transform(current, timestamp)
            cursor = connection.execute(
                """
                UPDATE review_cases
                   SET status = ?, revision = ?, case_json = ?, updated_at = ?
                 WHERE tenant_id = ? AND owner_user_id = ?
                   AND conversation_id = ? AND case_id = ? AND revision = ?
                """,
                (
                    updated.status.value,
                    updated.revision,
                    _model_json(updated),
                    updated.updated_at.timestamp(),
                    access.tenant_id,
                    access.owner_user_id,
                    access.conversation_id,
                    case_id,
                    expected_revision,
                ),
            )
            if cursor.rowcount != 1:
                raise CaseRevisionConflictError("case revision CAS lost")
            event = self._make_event(
                updated,
                event_type=event_type,
                from_status=current.status,
                actor_id=actor_id,
                actor_authority=actor_authority,
                idempotency_key=key,
                request_hash=request_hash,
                details=safe_details,
                created_at=timestamp,
            )
            self._insert_event(connection, event)
            self._insert_command(
                connection, access, key, command_type, request_hash,
                updated, timestamp,
            )
            return updated.model_copy(deep=True)

    @staticmethod
    def _make_event(
        review_case: ReviewCase,
        *,
        event_type: AuditEventType,
        from_status: CaseStatus | None,
        actor_id: str,
        actor_authority: Literal["human", "model_untrusted", "system", "tool"],
        idempotency_key: str,
        request_hash: str,
        details: Mapping[str, Any],
        created_at: datetime,
    ) -> CaseAuditEvent:
        return CaseAuditEvent(
            tenant_id=review_case.tenant_id,
            owner_user_id=review_case.owner_user_id,
            conversation_id=review_case.conversation_id,
            case_id=review_case.case_id,
            event_type=event_type,
            revision=review_case.revision,
            from_status=from_status,
            to_status=review_case.status,
            actor_id=actor_id,
            actor_authority=actor_authority,
            idempotency_key=idempotency_key,
            request_hash=request_hash,
            details=dict(details),
            created_at=created_at,
        )

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection, event: CaseAuditEvent
    ) -> None:
        connection.execute(
            """
            INSERT INTO review_case_audit_events(
                event_id, tenant_id, owner_user_id, conversation_id, case_id,
                revision, event_type, event_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.tenant_id,
                event.owner_user_id,
                event.conversation_id,
                event.case_id,
                event.revision,
                event.event_type.value,
                _model_json(event),
                event.created_at.timestamp(),
            ),
        )

    @staticmethod
    def _insert_command(
        connection: sqlite3.Connection,
        access: CaseAccess,
        idempotency_key: str,
        command_type: str,
        request_hash: str,
        review_case: ReviewCase,
        created_at: datetime,
    ) -> None:
        try:
            connection.execute(
                """
                INSERT INTO review_case_commands(
                    tenant_id, owner_user_id, conversation_id,
                    idempotency_key, command_type, request_hash, case_id,
                    result_revision, result_case_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    access.tenant_id,
                    access.owner_user_id,
                    access.conversation_id,
                    idempotency_key,
                    command_type,
                    request_hash,
                    review_case.case_id,
                    review_case.revision,
                    _model_json(review_case),
                    created_at.timestamp(),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise CaseIdempotencyConflictError(
                "idempotency receipt insert lost"
            ) from exc

    @staticmethod
    def _command_replay(
        connection: sqlite3.Connection,
        access: CaseAccess,
        idempotency_key: str,
        request_hash: str,
    ) -> ReviewCase | None:
        row = connection.execute(
            """
            SELECT request_hash, result_case_json
              FROM review_case_commands
             WHERE tenant_id = ? AND owner_user_id = ?
               AND conversation_id = ? AND idempotency_key = ?
            """,
            (
                access.tenant_id,
                access.owner_user_id,
                access.conversation_id,
                idempotency_key,
            ),
        ).fetchone()
        if row is None:
            return None
        if row["request_hash"] != request_hash:
            raise CaseIdempotencyConflictError(
                "idempotency key was reused with a different command request"
            )
        return ReviewCase.model_validate_json(row["result_case_json"])

    @staticmethod
    def _case_in_transaction(
        connection: sqlite3.Connection,
        access: CaseAccess,
        case_id: str,
    ) -> ReviewCase:
        row = connection.execute(
            """
            SELECT case_json FROM review_cases
             WHERE tenant_id = ? AND owner_user_id = ?
               AND conversation_id = ? AND case_id = ?
            """,
            (
                access.tenant_id,
                access.owner_user_id,
                access.conversation_id,
                case_id,
            ),
        ).fetchone()
        if row is None:
            raise ReviewCaseNotFoundError(
                "review case was not found in the caller ownership scope"
            )
        return ReviewCase.model_validate_json(row["case_json"])


def _replace_case(review_case: ReviewCase, **updates: Any) -> ReviewCase:
    payload = review_case.model_dump()
    payload.update(updates)
    return ReviewCase.model_validate(payload)


def _validate_idempotency_key(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("idempotency_key must be a string")
    _nonblank(value)
    if len(value) > 240:
        raise ValueError("idempotency_key exceeds 240 characters")
    return value


def _validate_case_id(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("case_id must be a string")
    _nonblank(value)
    if len(value) > 240:
        raise ValueError("case_id exceeds 240 characters")
    return value


def _require_access(access: CaseAccess) -> None:
    if not isinstance(access, CaseAccess):
        raise TypeError("review case operations require a trusted CaseAccess")
    _canonical_json(access.model_dump(mode="json"))
    if access.effective_actor_user_id == access.owner_user_id:
        return
    if access.admin_override and _ADMIN_ROLE in access.actor_roles:
        return
    raise CaseAccessDeniedError(
        "acting for another owner requires explicit case_admin override"
    )


def _require_host_actor_binding(access: CaseAccess, actor: HostActor) -> None:
    if (
        actor.authority == "human"
        and actor.actor_id != access.effective_actor_user_id
    ):
        raise CaseAccessDeniedError(
            "human host actor does not match the trusted access actor"
        )


def _require_transition(current: CaseStatus, target: CaseStatus) -> None:
    if target not in _ALLOWED_TRANSITIONS[current]:
        raise CaseInvalidTransitionError(
            f"invalid review case transition: {current.value} -> {target.value}"
        )


def _command_hash(
    access: CaseAccess, command_type: str, body: Mapping[str, Any]
) -> str:
    return _sha256(
        {
            "scope": {
                "tenant_id": access.tenant_id,
                "owner_user_id": access.owner_user_id,
                "conversation_id": access.conversation_id,
            },
            "actor_user_id": access.effective_actor_user_id,
            "command_type": command_type,
            "body": dict(body),
        }
    )


def _require_bound_evidence(
    submitted: Sequence[EvidenceRef], recommended: Sequence[EvidenceRef]
) -> None:
    submitted_by_id = {item.evidence_id: item for item in submitted}
    for evidence in recommended:
        bound = submitted_by_id.get(evidence.evidence_id)
        if bound is None or _model_json(bound) != _model_json(evidence):
            raise CaseDecisionRuleError(
                "model recommendation evidence must exactly match submitted evidence"
            )


ReviewCaseStore = SQLiteReviewCaseStore


__all__ = [
    "AuditEventType",
    "CaseAccess",
    "CaseAccessDeniedError",
    "CaseAuditEvent",
    "CaseDecisionRuleError",
    "CaseFailure",
    "CaseIdempotencyConflictError",
    "CaseInvalidTransitionError",
    "CaseKind",
    "CaseRevisionConflictError",
    "CaseStatus",
    "CaseSubmission",
    "EvidenceRef",
    "HostActor",
    "HumanActor",
    "HumanDecision",
    "ModelRecommendation",
    "RecommendationOutcome",
    "ReviewCase",
    "ReviewCaseError",
    "ReviewCaseNotFoundError",
    "ReviewCaseStore",
    "SQLiteReviewCaseStore",
]

"""PostgreSQL persistence for TaskForge human-gated review cases.

The review domain remains in :mod:`review_cases`; this module only replaces
the SQLite transport.  Case JSON is stored as JSONB, mutations are protected
by a row lock plus revision CAS, and command receipts/audit events are written
in the same transaction as the case snapshot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from .case_profiles import ResearchSurveyDepth
from .postgres_runtime import PostgresRuntime
from .review_cases import (
    AuditEventType,
    CaseAccess,
    CaseAccessDeniedError,
    CaseAuditEvent,
    CaseDecisionRuleError,
    CaseFailure,
    CaseIdempotencyConflictError,
    CaseInvalidTransitionError,
    CaseKind,
    CaseRevisionConflictError,
    CaseStatus,
    CaseSubmission,
    HostActor,
    HumanActor,
    HumanDecision,
    ModelRecommendation,
    ReviewCase,
    ReviewCaseNotFoundError,
    _canonical_json,
    _command_hash,
    _replace_case,
    _require_access,
    _require_bound_evidence,
    _require_host_actor_binding,
    _require_transition,
    _sha256,
    _utc,
    _validate_case_id,
    _validate_idempotency_key,
)


class PostgresReviewCaseStore:
    """Pooled, tenant-scoped implementation of the review case port."""

    def __init__(
        self,
        dsn: str,
        *,
        tenant_id: str = "local",
        min_size: int = 1,
        max_size: int = 8,
        connect_timeout_seconds: float = 5.0,
        runtime: PostgresRuntime | None = None,
    ) -> None:
        if not tenant_id.strip():
            raise ValueError("tenant_id is required")
        self.tenant_id = tenant_id
        self._owns_runtime = runtime is None
        self.runtime = runtime or PostgresRuntime(
            dsn,
            min_size=min_size,
            max_size=max_size,
            connect_timeout_seconds=connect_timeout_seconds,
        )

    def close(self) -> None:
        if self._owns_runtime:
            self.runtime.close()

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
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            replay = self._command_replay(cursor, access, key, request_hash)
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
                cursor.execute(
                    """
                    INSERT INTO review.review_cases(
                        tenant_id, case_id, owner_user_id, conversation_id,
                        status, revision, case_json, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        access.tenant_id,
                        review_case.case_id,
                        review_case.owner_user_id,
                        review_case.conversation_id,
                        review_case.status.value,
                        review_case.revision,
                        _json(review_case),
                        timestamp,
                        timestamp,
                    ),
                )
            except Exception as exc:
                if _is_unique_violation(exc):
                    raise CaseIdempotencyConflictError(
                        "review case identity already exists"
                    ) from exc
                raise
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
            self._insert_event(cursor, event)
            self._insert_command(cursor, access, key, "create_case", request_hash, review_case, timestamp)
            return review_case.model_copy(deep=True)

    def get_case(self, access: CaseAccess, case_id: str) -> ReviewCase:
        _require_access(access)
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            return self._case(cursor, access, _validate_case_id(case_id), lock=False)

    def list_cases(
        self,
        access: CaseAccess,
        *,
        statuses: Sequence[CaseStatus] | None = None,
        limit: int = 100,
    ) -> list[ReviewCase]:
        return self._list(access, statuses=statuses, limit=limit, owned_only=False)

    def list_owned_cases(
        self,
        access: CaseAccess,
        *,
        statuses: Sequence[CaseStatus] | None = None,
        limit: int = 100,
    ) -> list[ReviewCase]:
        return self._list(access, statuses=statuses, limit=limit, owned_only=True)

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
                submission=current.submission if validated_submission is None else validated_submission,
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
                "submission": None if validated_submission is None else validated_submission.model_dump(mode="json"),
            },
            event_type=AuditEventType.DRAFT_UPDATED,
            actor_id=access.effective_actor_user_id,
            actor_authority="human",
            details={"title_changed": title is not None, "submission_changed": validated_submission is not None},
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
        def transform(current: ReviewCase, timestamp: datetime) -> ReviewCase:
            _require_transition(current.status, CaseStatus.SUBMITTED)
            if not current.submission.evidence_refs and current.kind != CaseKind.RESEARCH_SURVEY:
                raise CaseDecisionRuleError("a submitted enterprise review requires at least one evidence reference")
            return _replace_case(
                current,
                status=CaseStatus.SUBMITTED,
                revision=current.revision + 1,
                updated_at=timestamp,
                submitted_at=timestamp,
            )

        return self._mutate(
            access, case_id, expected_revision=expected_revision, idempotency_key=idempotency_key,
            command_type="submit_case", request_body={}, event_type=AuditEventType.SUBMITTED,
            actor_id=access.effective_actor_user_id, actor_authority="human", details={},
            transform=transform, now=now,
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
            return _replace_case(current, status=CaseStatus.RUNNING, revision=current.revision + 1, updated_at=timestamp, started_at=timestamp)

        return self._mutate(
            access, case_id, expected_revision=expected_revision, idempotency_key=idempotency_key,
            command_type="start_case", request_body={"actor": actor.model_dump(mode="json")},
            event_type=AuditEventType.STARTED, actor_id=actor.actor_id, actor_authority=actor.authority,
            details={}, transform=transform, now=now,
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
        validated = recommendation if isinstance(recommendation, ModelRecommendation) else ModelRecommendation.model_validate(recommendation)

        def transform(current: ReviewCase, timestamp: datetime) -> ReviewCase:
            _require_transition(current.status, CaseStatus.WAITING_HUMAN_REVIEW)
            if current.kind != CaseKind.RESEARCH_SURVEY:
                _require_bound_evidence(current.submission.evidence_refs, validated.evidence_refs)
            return _replace_case(current, status=CaseStatus.WAITING_HUMAN_REVIEW, recommendation=validated, revision=current.revision + 1, updated_at=timestamp, review_requested_at=timestamp)

        return self._mutate(
            access, case_id, expected_revision=expected_revision, idempotency_key=idempotency_key,
            command_type="submit_model_recommendation", request_body={"recommendation": validated.model_dump(mode="json")},
            event_type=AuditEventType.RECOMMENDATION_RECORDED, actor_id=validated.model_run_id,
            actor_authority="model_untrusted", details={"recommendation_id": validated.recommendation_id, "outcome": validated.outcome.value, "evidence_count": len(validated.evidence_refs)},
            transform=transform, now=now,
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
            raise CaseAccessDeniedError("human decision actor does not match the trusted access actor")
        outcome = CaseStatus(outcome)
        if outcome not in {CaseStatus.APPROVED, CaseStatus.REJECTED}:
            raise CaseDecisionRuleError("human outcome must be approved or rejected")
        evidence_ids = list(evidence_ref_ids)
        decision_timestamp = _utc(now)
        decision = HumanDecision(outcome=outcome, actor=human_actor, rationale=rationale, evidence_ref_ids=evidence_ids, decided_at=decision_timestamp)

        def transform(current: ReviewCase, timestamp: datetime) -> ReviewCase:
            _require_transition(current.status, outcome)
            known = {item.evidence_id for item in current.submission.evidence_refs}
            unknown = sorted(set(decision.evidence_ref_ids) - known)
            if unknown:
                raise CaseDecisionRuleError(f"human decision cites unknown evidence IDs: {unknown}")
            return _replace_case(current, status=outcome, human_decision=decision, revision=current.revision + 1, updated_at=timestamp, resolved_at=timestamp)

        return self._mutate(
            access, case_id, expected_revision=expected_revision, idempotency_key=idempotency_key,
            command_type="decide_case", request_body={"outcome": outcome.value, "human_actor": human_actor.model_dump(mode="json"), "rationale": rationale, "evidence_ref_ids": evidence_ids},
            event_type=AuditEventType.APPROVED if outcome == CaseStatus.APPROVED else AuditEventType.REJECTED,
            actor_id=human_actor.actor_user_id, actor_authority="human", details={"outcome": outcome.value, "evidence_ref_ids": evidence_ids},
            transform=transform, now=decision_timestamp,
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
            return _replace_case(current, status=CaseStatus.FAILED, failure=failure, revision=current.revision + 1, updated_at=command_time, resolved_at=command_time)

        return self._mutate(
            access, case_id, expected_revision=expected_revision, idempotency_key=idempotency_key,
            command_type="fail_case", request_body={"actor": actor.model_dump(mode="json"), "reason": reason},
            event_type=AuditEventType.FAILED, actor_id=actor.actor_id, actor_authority=actor.authority,
            details={"reason_hash": _sha256(reason)}, transform=transform, now=timestamp,
        )

    def list_audit_events(self, access: CaseAccess, case_id: str) -> list[CaseAuditEvent]:
        _require_access(access)
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            self._case(cursor, access, _validate_case_id(case_id), lock=False)
            cursor.execute(
                """
                SELECT event_json FROM review.review_case_audit_events
                 WHERE tenant_id = %s AND owner_user_id = %s
                   AND conversation_id = %s AND case_id = %s
                 ORDER BY revision ASC
                """,
                (access.tenant_id, access.owner_user_id, access.conversation_id, case_id),
            )
            rows = cursor.fetchall()
        return [_model_validate(CaseAuditEvent, _row(row, "event_json", 0)) for row in rows]

    def _list(
        self,
        access: CaseAccess,
        *,
        statuses: Sequence[CaseStatus] | None,
        limit: int,
        owned_only: bool,
    ) -> list[ReviewCase]:
        _require_access(access)
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        validated = None if statuses is None else [CaseStatus(status) for status in statuses]
        if validated is not None and not validated:
            return []
        where = ["tenant_id = %s", "owner_user_id = %s"]
        params: list[Any] = [access.tenant_id, access.owner_user_id]
        if not owned_only:
            where.append("conversation_id = %s")
            params.append(access.conversation_id)
        if validated is not None:
            where.append("status = ANY(%s)")
            params.append([status.value for status in validated])
        params.append(limit)
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            cursor.execute(
                "SELECT case_json FROM review.review_cases WHERE "
                + " AND ".join(where)
                + " ORDER BY updated_at DESC, case_id ASC LIMIT %s",
                tuple(params),
            )
            rows = cursor.fetchall()
        return [_model_validate(ReviewCase, _row(row, "case_json", 0)) for row in rows]

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
        transform: Any,
        now: datetime | None,
    ) -> ReviewCase:
        _require_access(access)
        case_id = _validate_case_id(case_id)
        if not isinstance(expected_revision, int) or expected_revision < 1:
            raise ValueError("expected_revision must be a positive integer")
        key = _validate_idempotency_key(idempotency_key)
        timestamp = _utc(now)
        request_hash = _command_hash(access, command_type, {"case_id": case_id, "expected_revision": expected_revision, **dict(request_body)})
        safe_details = dict(details)
        _canonical_json(safe_details)
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            replay = self._command_replay(cursor, access, key, request_hash)
            if replay is not None:
                return replay
            current = self._case(cursor, access, case_id, lock=True)
            if current.revision != expected_revision:
                raise CaseRevisionConflictError(f"case revision is stale: expected {expected_revision}, current {current.revision}")
            updated = transform(current, timestamp)
            cursor.execute(
                """
                UPDATE review.review_cases
                   SET status = %s, revision = %s, case_json = %s, updated_at = %s
                 WHERE tenant_id = %s AND owner_user_id = %s
                   AND conversation_id = %s AND case_id = %s AND revision = %s
                """,
                (updated.status.value, updated.revision, _json(updated), updated.updated_at, access.tenant_id, access.owner_user_id, access.conversation_id, case_id, expected_revision),
            )
            if cursor.rowcount != 1:
                raise CaseRevisionConflictError("case revision CAS lost")
            event = self._make_event(updated, event_type=event_type, from_status=current.status, actor_id=actor_id, actor_authority=actor_authority, idempotency_key=key, request_hash=request_hash, details=safe_details, created_at=timestamp)
            self._insert_event(cursor, event)
            self._insert_command(cursor, access, key, command_type, request_hash, updated, timestamp)
            return updated.model_copy(deep=True)

    @staticmethod
    def _make_event(review_case: ReviewCase, *, event_type: AuditEventType, from_status: CaseStatus | None, actor_id: str, actor_authority: Literal["human", "model_untrusted", "system", "tool"], idempotency_key: str, request_hash: str, details: Mapping[str, Any], created_at: datetime) -> CaseAuditEvent:
        return CaseAuditEvent(tenant_id=review_case.tenant_id, owner_user_id=review_case.owner_user_id, conversation_id=review_case.conversation_id, case_id=review_case.case_id, event_type=event_type, revision=review_case.revision, from_status=from_status, to_status=review_case.status, actor_id=actor_id, actor_authority=actor_authority, idempotency_key=idempotency_key, request_hash=request_hash, details=dict(details), created_at=created_at)

    @staticmethod
    def _insert_event(cursor: Any, event: CaseAuditEvent) -> None:
        cursor.execute(
            """
            INSERT INTO review.review_case_audit_events(
                tenant_id, event_id, owner_user_id, conversation_id, case_id,
                revision, event_type, event_json, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (event.tenant_id, event.event_id, event.owner_user_id, event.conversation_id, event.case_id, event.revision, event.event_type.value, _json(event), event.created_at),
        )

    @staticmethod
    def _insert_command(cursor: Any, access: CaseAccess, idempotency_key: str, command_type: str, request_hash: str, review_case: ReviewCase, created_at: datetime) -> None:
        try:
            cursor.execute(
                """
                INSERT INTO review.review_case_commands(
                    tenant_id, owner_user_id, conversation_id, idempotency_key,
                    command_type, request_hash, case_id, result_revision,
                    result_case_json, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (access.tenant_id, access.owner_user_id, access.conversation_id, idempotency_key, command_type, request_hash, review_case.case_id, review_case.revision, _json(review_case), created_at),
            )
        except Exception as exc:
            if _is_unique_violation(exc):
                raise CaseIdempotencyConflictError("idempotency receipt insert lost") from exc
            raise

    @staticmethod
    def _command_replay(cursor: Any, access: CaseAccess, idempotency_key: str, request_hash: str) -> ReviewCase | None:
        cursor.execute(
            """
            SELECT request_hash, result_case_json
              FROM review.review_case_commands
             WHERE tenant_id = %s AND owner_user_id = %s
               AND conversation_id = %s AND idempotency_key = %s
            """,
            (access.tenant_id, access.owner_user_id, access.conversation_id, idempotency_key),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        if _row(row, "request_hash", 0) != request_hash:
            raise CaseIdempotencyConflictError("idempotency key was reused with a different command request")
        return _model_validate(ReviewCase, _row(row, "result_case_json", 1))

    @staticmethod
    def _case(cursor: Any, access: CaseAccess, case_id: str, *, lock: bool) -> ReviewCase:
        cursor.execute(
            "SELECT case_json FROM review.review_cases "
            "WHERE tenant_id = %s AND owner_user_id = %s "
            "AND conversation_id = %s AND case_id = %s"
            + (" FOR UPDATE" if lock else ""),
            (access.tenant_id, access.owner_user_id, access.conversation_id, case_id),
        )
        row = cursor.fetchone()
        if row is None:
            raise ReviewCaseNotFoundError("review case was not found in the caller ownership scope")
        return _model_validate(ReviewCase, _row(row, "case_json", 0))


def _json(value: object) -> object:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    try:
        from psycopg.types.json import Jsonb
    except ImportError:  # pragma: no cover - exercised by fake-runtime tests
        return payload
    return Jsonb(payload)


def _row(row: Any, key: str, index: int) -> Any:
    return row.get(key) if isinstance(row, Mapping) else row[index]


def _model_validate(model: Any, payload: Any) -> Any:
    if isinstance(payload, str):
        return model.model_validate_json(payload)
    return model.model_validate(payload)


def _is_unique_violation(exc: Exception) -> bool:
    return getattr(exc, "sqlstate", None) == "23505" or exc.__class__.__name__ == "UniqueViolation"


__all__ = ["PostgresReviewCaseStore"]

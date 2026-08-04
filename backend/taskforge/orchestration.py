"""Persistent, controlled multi-role orchestration primitives.

This module deliberately sits *above* ``AgentRuntime``.  A speaker plan is a
fixed host-validated DAG; a model may recommend an allowed ready role, but it
cannot add roles, dependencies, tools, or execution authority.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from .domain import StrictModel, utc_now


class OrchestrationError(RuntimeError):
    pass


class OrchestrationNotFoundError(OrchestrationError):
    pass


class IdempotencyConflictError(OrchestrationError):
    pass


class VersionConflictError(OrchestrationError):
    pass


class InvalidTransitionError(OrchestrationError):
    pass


class RoleNotAllowedError(OrchestrationError):
    pass


class SlotNotReadyError(OrchestrationError):
    pass


class FactRuleError(OrchestrationError):
    pass


class ExecutionClaimUnavailableError(OrchestrationError):
    """Another executor owns the unexpired RoleRun execution lease."""


class OrchestrationAccess(StrictModel):
    """Trusted caller scope for every orchestration operation.

    This object must be constructed by the authenticated host boundary, never
    from model-provided arguments.  ``allowed_role_ids`` is an optional
    capability reduction: ``None`` means the host may act for every role in
    the owned conversation, while a tuple restricts model-facing access to the
    listed roles.
    """

    tenant_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    allowed_role_ids: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def role_allowlist_is_unambiguous(self) -> OrchestrationAccess:
        if self.allowed_role_ids is not None:
            if any(not role_id for role_id in self.allowed_role_ids):
                raise ValueError("allowed orchestration roles must be non-empty")
            if len(self.allowed_role_ids) != len(set(self.allowed_role_ids)):
                raise ValueError("allowed orchestration roles must be unique")
        return self

    def require_role(self, role_id: str) -> None:
        if (
            self.allowed_role_ids is not None
            and role_id not in self.allowed_role_ids
        ):
            raise RoleNotAllowedError("role is outside the caller access scope")


class PlanStatus(str, Enum):
    READY = "ready"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    DEGRADED = "degraded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RoleRunStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FactStatus(str, Enum):
    PROPOSED = "proposed"
    VERIFIED = "verified"


class SpeakerSlot(StrictModel):
    slot_id: str = Field(min_length=1, max_length=120)
    role_id: str = Field(min_length=1, max_length=120)
    agent_profile_id: str = Field(min_length=1, max_length=160)
    instruction: str = Field(min_length=1, max_length=8_000)
    depends_on: list[str] = Field(default_factory=list)
    order: int = Field(default=0, ge=0)
    required: bool = True
    max_attempts: int = Field(default=2, ge=1, le=20)

    @model_validator(mode="after")
    def dependency_list_is_unambiguous(self) -> SpeakerSlot:
        if self.slot_id in self.depends_on:
            raise ValueError("a speaker slot cannot depend on itself")
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("slot dependencies must be unique")
        return self


class SpeakerPlan(StrictModel):
    plan_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    tenant_id: str = Field(min_length=1)
    owner_user_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    objective: str = Field(min_length=1, max_length=16_000)
    strategy: Literal["explicit", "static", "round_robin", "model_router"] = "static"
    allowed_role_ids: list[str] = Field(min_length=1)
    slots: list[SpeakerSlot] = Field(min_length=1)
    client_idempotency_key: str = Field(min_length=1, max_length=240)
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: PlanStatus = PlanStatus.READY
    version: int = Field(default=1, ge=1)
    max_role_runs: int = Field(default=20, ge=1, le=1_000)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def fixed_dag_is_valid(self) -> SpeakerPlan:
        if len(self.allowed_role_ids) != len(set(self.allowed_role_ids)):
            raise ValueError("allowed roles must be unique")
        slot_ids = [slot.slot_id for slot in self.slots]
        if len(slot_ids) != len(set(slot_ids)):
            raise ValueError("speaker slot IDs must be unique")
        known = set(slot_ids)
        allowed = set(self.allowed_role_ids)
        for slot in self.slots:
            if slot.role_id not in allowed:
                raise ValueError(f"slot role is not allowed: {slot.role_id}")
            missing = set(slot.depends_on) - known
            if missing:
                raise ValueError(
                    f"slot {slot.slot_id!r} has unknown dependencies: {sorted(missing)}"
                )
        _assert_acyclic(self.slots)
        required_closure = _required_slot_closure(self.slots)
        if self.max_role_runs < len(required_closure):
            raise ValueError(
                "max_role_runs is smaller than the minimum required DAG executions"
            )
        expected = compute_plan_request_hash(
            tenant_id=self.tenant_id,
            owner_user_id=self.owner_user_id,
            conversation_id=self.conversation_id,
            objective=self.objective,
            strategy=self.strategy,
            allowed_role_ids=self.allowed_role_ids,
            slots=self.slots,
            max_role_runs=self.max_role_runs,
        )
        if self.request_hash != expected:
            raise ValueError("request_hash does not match the fixed plan request")
        return self


class RoleRun(StrictModel):
    role_run_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    run_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    slot_id: str = Field(min_length=1)
    role_id: str = Field(min_length=1)
    agent_profile_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    status: RoleRunStatus = RoleRunStatus.PENDING
    plan_version: int = Field(ge=1)
    version: int = Field(default=1, ge=1)
    output: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def terminal_payload_is_consistent(self) -> RoleRun:
        if self.status == RoleRunStatus.FAILED and not self.error:
            raise ValueError("a failed RoleRun requires an error")
        if self.status != RoleRunStatus.FAILED and self.error is not None:
            raise ValueError("only a failed RoleRun may retain an error")
        return self


class RoleRunExecutionClaim(StrictModel):
    claim_token: str = Field(min_length=16, max_length=240)
    role_run_id: str = Field(min_length=1, max_length=240)
    tenant_id: str = Field(min_length=1)
    owner_user_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    acquired_at: datetime
    expires_at: datetime

    @model_validator(mode="after")
    def expiry_follows_acquisition(self) -> RoleRunExecutionClaim:
        if _utc(self.expires_at) <= _utc(self.acquired_at):
            raise ValueError("execution claim expiry must follow acquisition")
        return self


class Handoff(StrictModel):
    handoff_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    tenant_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    from_role_run_id: str = Field(min_length=1)
    to_slot_id: str = Field(min_length=1)
    summary: str = Field(min_length=1, max_length=12_000)
    summary_authority: Literal["model_untrusted"] = "model_untrusted"
    shared_fact_ids: list[str] = Field(default_factory=list)
    payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime = Field(default_factory=utc_now)


class SharedFact(StrictModel):
    fact_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    tenant_id: str = Field(min_length=1)
    owner_user_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    fact_key: str = Field(min_length=1, max_length=240)
    value: Any
    status: FactStatus
    authority: Literal["model", "tool", "user", "system"]
    version: int = Field(ge=1)
    source_role_run_id: str | None = None
    supersedes_fact_id: str | None = None
    verifier_ref: str | None = Field(default=None, min_length=1, max_length=500)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def authority_matches_status(self) -> SharedFact:
        if self.status == FactStatus.PROPOSED and self.authority != "model":
            raise ValueError("proposed facts must be model-attributed")
        if self.status == FactStatus.VERIFIED and self.authority == "model":
            raise ValueError("a model cannot verify a shared fact")
        if self.status == FactStatus.VERIFIED and not self.verifier_ref:
            raise ValueError("verified facts require verifier_ref")
        return self


class FactVerificationReceipt(StrictModel):
    """Host-issued evidence that may verify exactly one proposed fact."""

    receipt_id: str = Field(min_length=1, max_length=500)
    tenant_id: str = Field(min_length=1)
    owner_user_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    fact_key: str = Field(min_length=1, max_length=240)
    authority: Literal["tool", "user", "system"]
    value_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_ref: str = Field(min_length=1, max_length=500)
    created_at: datetime = Field(default_factory=utc_now)


class PrivateRoleMemory(StrictModel):
    memory_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    tenant_id: str = Field(min_length=1)
    owner_user_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    role_id: str = Field(min_length=1)
    kind: Literal["episode", "preference", "commitment", "relationship"] = "episode"
    content: str = Field(min_length=1, max_length=16_000)
    provenance_role_run_id: str | None = None
    extractor_version: str = Field(default="v1", min_length=1, max_length=80)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    version: int = Field(default=1, ge=1)
    expires_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)


def _assert_acyclic(slots: Sequence[SpeakerSlot]) -> None:
    dependencies = {slot.slot_id: set(slot.depends_on) for slot in slots}
    remaining = set(dependencies)
    while remaining:
        ready = {
            slot_id
            for slot_id in remaining
            if not (dependencies[slot_id] & remaining)
        }
        if not ready:
            raise ValueError("speaker slots must form an acyclic DAG")
        remaining -= ready


def _required_slot_closure(slots: Sequence[SpeakerSlot]) -> set[str]:
    by_id = {slot.slot_id: slot for slot in slots}
    required = {slot.slot_id for slot in slots if slot.required}
    frontier = list(required)
    while frontier:
        slot_id = frontier.pop()
        for dependency in by_id[slot_id].depends_on:
            if dependency not in required:
                required.add(dependency)
                frontier.append(dependency)
    return required


def _validate_json_tree(value: Any, *, depth: int = 0) -> None:
    if depth > 50:
        raise ValueError("JSON payload exceeds the maximum nesting depth")
    if isinstance(value, str):
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("JSON strings must contain Unicode scalar values")
        return
    if value is None or isinstance(value, (bool, int, float)):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_json_tree(key, depth=depth + 1)
            _validate_json_tree(item, depth=depth + 1)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _validate_json_tree(item, depth=depth + 1)


def _canonical_json(value: Any) -> str:
    _validate_json_tree(value)
    try:
        payload = json.dumps(
            value,
            # Escaping non-ASCII also makes isolated surrogate code points
            # deterministic JSON instead of leaking UnicodeEncodeError.
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value must contain finite JSON data") from exc
    if len(payload.encode("utf-8")) > 1_000_000:
        raise ValueError("JSON payload exceeds the 1000000 byte limit")
    return payload


def _model_json(value: StrictModel) -> str:
    return _canonical_json(value.model_dump(mode="json"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def compute_plan_request_hash(
    *,
    tenant_id: str,
    owner_user_id: str,
    conversation_id: str,
    objective: str,
    strategy: str,
    allowed_role_ids: Sequence[str],
    slots: Sequence[SpeakerSlot],
    max_role_runs: int,
) -> str:
    ordered_slots = sorted(slots, key=lambda slot: (slot.order, slot.slot_id))
    return _sha256(
        {
            "tenant_id": tenant_id,
            "owner_user_id": owner_user_id,
            "conversation_id": conversation_id,
            "objective": objective,
            "strategy": strategy,
            "allowed_role_ids": sorted(allowed_role_ids),
            "slots": [
                {
                    **slot.model_dump(mode="json"),
                    "depends_on": sorted(slot.depends_on),
                }
                for slot in ordered_slots
            ],
            "max_role_runs": max_role_runs,
        }
    )


_PLAN_TRANSITIONS: dict[PlanStatus, set[PlanStatus]] = {
    PlanStatus.READY: {PlanStatus.RUNNING, PlanStatus.CANCELLED},
    PlanStatus.RUNNING: {
        PlanStatus.WAITING_APPROVAL,
        PlanStatus.COMPLETED,
        PlanStatus.DEGRADED,
        PlanStatus.FAILED,
        PlanStatus.CANCELLED,
    },
    PlanStatus.WAITING_APPROVAL: {
        PlanStatus.RUNNING,
        PlanStatus.FAILED,
        PlanStatus.CANCELLED,
    },
    PlanStatus.COMPLETED: set(),
    PlanStatus.DEGRADED: set(),
    PlanStatus.FAILED: set(),
    PlanStatus.CANCELLED: set(),
}

_ROLE_RUN_TRANSITIONS: dict[RoleRunStatus, set[RoleRunStatus]] = {
    RoleRunStatus.PENDING: {
        RoleRunStatus.QUEUED,
        RoleRunStatus.RUNNING,
        RoleRunStatus.FAILED,
        RoleRunStatus.CANCELLED,
    },
    RoleRunStatus.QUEUED: {
        RoleRunStatus.RUNNING,
        RoleRunStatus.FAILED,
        RoleRunStatus.CANCELLED,
    },
    RoleRunStatus.RUNNING: {
        RoleRunStatus.WAITING_APPROVAL,
        RoleRunStatus.SUCCEEDED,
        RoleRunStatus.FAILED,
        RoleRunStatus.CANCELLED,
    },
    RoleRunStatus.WAITING_APPROVAL: {
        RoleRunStatus.RUNNING,
        RoleRunStatus.FAILED,
        RoleRunStatus.CANCELLED,
    },
    RoleRunStatus.SUCCEEDED: set(),
    RoleRunStatus.FAILED: set(),
    RoleRunStatus.CANCELLED: set(),
}


class SQLiteOrchestrationStore:
    """SQLite persistence for the fixed multi-role orchestration contract."""

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
        """Apply sqlite transaction semantics and close on every path."""

        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialise(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS speaker_plans (
                    plan_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    owner_user_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    client_idempotency_key TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK (version >= 1),
                    plan_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(tenant_id, owner_user_id, client_idempotency_key)
                );
                CREATE INDEX IF NOT EXISTS speaker_plan_scope_idx
                    ON speaker_plans(
                        tenant_id, owner_user_id, conversation_id, created_at
                    );

                CREATE TABLE IF NOT EXISTS role_runs (
                    role_run_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE,
                    tenant_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    plan_id TEXT NOT NULL,
                    slot_id TEXT NOT NULL,
                    role_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL CHECK (attempt >= 1),
                    status TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK (version >= 1),
                    role_run_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(plan_id, slot_id, attempt),
                    FOREIGN KEY(plan_id) REFERENCES speaker_plans(plan_id)
                );
                CREATE INDEX IF NOT EXISTS role_run_plan_idx
                    ON role_runs(tenant_id, plan_id, slot_id, attempt);

                CREATE TABLE IF NOT EXISTS role_run_execution_claims (
                    role_run_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    owner_user_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    claim_token TEXT NOT NULL UNIQUE,
                    expires_at REAL NOT NULL,
                    claim_json TEXT NOT NULL,
                    FOREIGN KEY(role_run_id) REFERENCES role_runs(role_run_id)
                );
                CREATE INDEX IF NOT EXISTS role_run_execution_claim_scope_idx
                    ON role_run_execution_claims(
                        tenant_id, owner_user_id, conversation_id, expires_at
                    );

                CREATE TABLE IF NOT EXISTS shared_facts (
                    fact_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    owner_user_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    fact_key TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK (version >= 1),
                    status TEXT NOT NULL CHECK (status IN ('proposed', 'verified')),
                    fact_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(
                        tenant_id, owner_user_id, conversation_id,
                        fact_key, version
                    )
                );
                CREATE INDEX IF NOT EXISTS shared_fact_scope_idx
                    ON shared_facts(
                        tenant_id, owner_user_id, conversation_id,
                        fact_key, version DESC
                    );

                CREATE TABLE IF NOT EXISTS fact_verification_receipts (
                    receipt_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    owner_user_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    fact_key TEXT NOT NULL,
                    authority TEXT NOT NULL CHECK (authority IN ('tool', 'user', 'system')),
                    value_hash TEXT NOT NULL,
                    evidence_ref TEXT NOT NULL,
                    receipt_json TEXT NOT NULL,
                    consumed_by_fact_id TEXT,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS fact_receipt_scope_idx
                    ON fact_verification_receipts(
                        tenant_id, owner_user_id, conversation_id,
                        fact_key, created_at
                    );

                CREATE TABLE IF NOT EXISTS handoffs (
                    handoff_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    plan_id TEXT NOT NULL,
                    from_role_run_id TEXT NOT NULL,
                    to_slot_id TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    handoff_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(from_role_run_id, to_slot_id),
                    FOREIGN KEY(plan_id) REFERENCES speaker_plans(plan_id),
                    FOREIGN KEY(from_role_run_id) REFERENCES role_runs(role_run_id)
                );

                CREATE TABLE IF NOT EXISTS private_role_memories (
                    memory_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    owner_user_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    role_id TEXT NOT NULL,
                    provenance_key TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    memory_json TEXT NOT NULL,
                    expires_at REAL,
                    created_at REAL NOT NULL,
                    UNIQUE(
                        tenant_id, owner_user_id, conversation_id, role_id,
                        provenance_key, content_hash
                    )
                );
                CREATE INDEX IF NOT EXISTS private_memory_scope_idx
                    ON private_role_memories(
                        tenant_id, owner_user_id, conversation_id,
                        role_id, created_at
                    );
                """
            )

    def create_plan(
        self,
        access: OrchestrationAccess,
        *,
        objective: str,
        allowed_role_ids: Sequence[str],
        slots: Sequence[SpeakerSlot | Mapping[str, Any]],
        client_idempotency_key: str,
        strategy: Literal["explicit", "static", "round_robin", "model_router"] = "static",
        max_role_runs: int = 20,
        plan_id: str | None = None,
        now: datetime | None = None,
    ) -> SpeakerPlan:
        _require_access(access)
        for role_id in allowed_role_ids:
            access.require_role(role_id)
        validated_slots = [
            slot if isinstance(slot, SpeakerSlot) else SpeakerSlot.model_validate(slot)
            for slot in slots
        ]
        timestamp = _utc(now)
        request_hash = compute_plan_request_hash(
            tenant_id=access.tenant_id,
            owner_user_id=access.user_id,
            conversation_id=access.conversation_id,
            objective=objective,
            strategy=strategy,
            allowed_role_ids=allowed_role_ids,
            slots=validated_slots,
            max_role_runs=max_role_runs,
        )
        plan = SpeakerPlan(
            plan_id=plan_id or str(uuid4()),
            tenant_id=access.tenant_id,
            owner_user_id=access.user_id,
            conversation_id=access.conversation_id,
            objective=objective,
            strategy=strategy,
            allowed_role_ids=list(allowed_role_ids),
            slots=validated_slots,
            client_idempotency_key=client_idempotency_key,
            request_hash=request_hash,
            max_role_runs=max_role_runs,
            created_at=timestamp,
            updated_at=timestamp,
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT plan_json, request_hash FROM speaker_plans
                 WHERE tenant_id = ? AND owner_user_id = ?
                   AND client_idempotency_key = ?
                """,
                (access.tenant_id, access.user_id, client_idempotency_key),
            ).fetchone()
            if existing is not None:
                if existing["request_hash"] != request_hash:
                    raise IdempotencyConflictError(
                        "client idempotency key was reused with a different plan request"
                    )
                replay = SpeakerPlan.model_validate_json(existing["plan_json"])
                connection.commit()
                return replay
            try:
                connection.execute(
                    """
                    INSERT INTO speaker_plans(
                        plan_id, tenant_id, owner_user_id, conversation_id,
                        client_idempotency_key, request_hash, status, version,
                        plan_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        plan.plan_id,
                        plan.tenant_id,
                        plan.owner_user_id,
                        plan.conversation_id,
                        plan.client_idempotency_key,
                        plan.request_hash,
                        plan.status.value,
                        plan.version,
                        _model_json(plan),
                        timestamp.timestamp(),
                        timestamp.timestamp(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise IdempotencyConflictError("plan identity already exists") from exc
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return plan.model_copy(deep=True)

    def get_plan(
        self, access: OrchestrationAccess, plan_id: str
    ) -> SpeakerPlan:
        _require_access(access)
        with self._connection() as connection:
            return self._plan_in_transaction(connection, access, plan_id)

    def transition_plan(
        self,
        access: OrchestrationAccess,
        plan_id: str,
        *,
        expected_version: int,
        status: PlanStatus,
        now: datetime | None = None,
    ) -> SpeakerPlan:
        _require_access(access)
        timestamp = _utc(now)
        status = PlanStatus(status)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            plan = self._plan_in_transaction(connection, access, plan_id)
            if plan.version != expected_version:
                raise VersionConflictError("speaker plan version is stale")
            if status not in _PLAN_TRANSITIONS[plan.status]:
                raise InvalidTransitionError(
                    f"invalid plan transition: {plan.status.value} -> {status.value}"
                )
            if status == PlanStatus.COMPLETED:
                self._assert_plan_completion_in_transaction(connection, plan)
            updated = plan.model_copy(
                update={
                    "status": status,
                    "version": plan.version + 1,
                    "updated_at": timestamp,
                },
                deep=True,
            )
            cursor = connection.execute(
                """
                UPDATE speaker_plans
                   SET status = ?, version = ?, plan_json = ?, updated_at = ?
                 WHERE tenant_id = ? AND owner_user_id = ?
                   AND conversation_id = ? AND plan_id = ? AND version = ?
                """,
                (
                    updated.status.value,
                    updated.version,
                    _model_json(updated),
                    timestamp.timestamp(),
                    access.tenant_id,
                    access.user_id,
                    access.conversation_id,
                    plan_id,
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                raise VersionConflictError("speaker plan CAS lost")
            connection.commit()
            return updated
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    cas_plan_status = transition_plan

    def next_ready_slots(
        self, access: OrchestrationAccess, plan_id: str
    ) -> list[SpeakerSlot]:
        _require_access(access)
        with self._connection() as connection:
            plan = self._plan_in_transaction(connection, access, plan_id)
            ready = self._next_ready_in_transaction(connection, plan)
        if access.allowed_role_ids is None:
            return ready
        allowed = set(access.allowed_role_ids)
        return [slot for slot in ready if slot.role_id in allowed]

    def validate_model_role_proposal(
        self,
        access: OrchestrationAccess,
        plan_id: str,
        role_id: str,
        *,
        expected_plan_version: int | None = None,
    ) -> SpeakerSlot:
        _require_access(access)
        access.require_role(role_id)
        plan = self.get_plan(access, plan_id)
        if expected_plan_version is not None and plan.version != expected_plan_version:
            raise VersionConflictError("speaker plan version is stale")
        if role_id not in plan.allowed_role_ids:
            raise RoleNotAllowedError("model proposed a role outside the plan allowlist")
        for slot in self.next_ready_slots(access, plan_id):
            if slot.role_id == role_id:
                return slot
        raise SlotNotReadyError("allowed role has no ready slot in the fixed DAG")

    resolve_model_role_proposal = validate_model_role_proposal

    def create_role_run(
        self,
        access: OrchestrationAccess,
        plan_id: str,
        slot_id: str,
        *,
        expected_plan_version: int,
        role_run_id: str | None = None,
        now: datetime | None = None,
    ) -> RoleRun:
        _require_access(access)
        timestamp = _utc(now)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            plan = self._plan_in_transaction(connection, access, plan_id)
            if plan.version != expected_plan_version:
                raise VersionConflictError("speaker plan version is stale")
            try:
                slot = next(item for item in plan.slots if item.slot_id == slot_id)
            except StopIteration as exc:
                raise SlotNotReadyError("slot is not present in the fixed DAG") from exc
            access.require_role(slot.role_id)
            if role_run_id is not None:
                replay = connection.execute(
                    """
                    SELECT role_runs.role_run_json
                      FROM role_runs
                      JOIN speaker_plans
                        ON speaker_plans.plan_id = role_runs.plan_id
                     WHERE role_runs.tenant_id = ?
                       AND role_runs.conversation_id = ?
                       AND role_runs.role_run_id = ?
                       AND speaker_plans.tenant_id = ?
                       AND speaker_plans.owner_user_id = ?
                       AND speaker_plans.conversation_id = ?
                    """,
                    (
                        access.tenant_id,
                        access.conversation_id,
                        role_run_id,
                        access.tenant_id,
                        access.user_id,
                        access.conversation_id,
                    ),
                ).fetchone()
                if replay is not None:
                    existing = RoleRun.model_validate_json(replay["role_run_json"])
                    if existing.plan_id != plan_id or existing.slot_id != slot_id:
                        raise IdempotencyConflictError(
                            "RoleRun identity was reused for another plan slot"
                        )
                    connection.commit()
                    return existing
            ready_ids = {
                slot.slot_id for slot in self._next_ready_in_transaction(connection, plan)
            }
            if slot_id not in ready_ids:
                existing = self._latest_role_run(
                    connection, access, plan_id, slot_id
                )
                if existing is not None and existing.status in {
                    RoleRunStatus.PENDING,
                    RoleRunStatus.QUEUED,
                    RoleRunStatus.RUNNING,
                    RoleRunStatus.WAITING_APPROVAL,
                    RoleRunStatus.SUCCEEDED,
                    RoleRunStatus.CANCELLED,
                }:
                    if (
                        role_run_id is not None
                        and existing.role_run_id != role_run_id
                    ):
                        raise IdempotencyConflictError(
                            "explicit RoleRun identity conflicts with the existing slot materialization"
                        )
                    connection.commit()
                    return existing
                raise SlotNotReadyError("slot is not ready in the fixed DAG")
            row = connection.execute(
                "SELECT COALESCE(MAX(attempt), 0) AS attempt FROM role_runs "
                "WHERE tenant_id = ? AND conversation_id = ? "
                "AND plan_id = ? AND slot_id = ?",
                (access.tenant_id, access.conversation_id, plan_id, slot_id),
            ).fetchone()
            attempt = int(row["attempt"]) + 1
            if attempt > slot.max_attempts:
                raise SlotNotReadyError("slot exhausted its attempt budget")
            identifier = role_run_id or str(uuid4())
            run = RoleRun(
                role_run_id=identifier,
                run_id=identifier,
                tenant_id=access.tenant_id,
                conversation_id=plan.conversation_id,
                plan_id=plan_id,
                slot_id=slot.slot_id,
                role_id=slot.role_id,
                agent_profile_id=slot.agent_profile_id,
                attempt=attempt,
                plan_version=plan.version,
                created_at=timestamp,
                updated_at=timestamp,
            )
            connection.execute(
                """
                INSERT INTO role_runs(
                    role_run_id, run_id, tenant_id, conversation_id, plan_id,
                    slot_id, role_id, attempt, status, version,
                    role_run_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.role_run_id,
                    run.run_id,
                    run.tenant_id,
                    run.conversation_id,
                    run.plan_id,
                    run.slot_id,
                    run.role_id,
                    run.attempt,
                    run.status.value,
                    run.version,
                    _model_json(run),
                    timestamp.timestamp(),
                    timestamp.timestamp(),
                ),
            )
            connection.commit()
            return run
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def transition_role_run(
        self,
        access: OrchestrationAccess,
        role_run_id: str,
        *,
        expected_version: int,
        status: RoleRunStatus,
        output: Mapping[str, Any] | None = None,
        error: str | None = None,
        execution_claim_token: str | None = None,
        now: datetime | None = None,
    ) -> RoleRun:
        _require_access(access)
        timestamp = _utc(now)
        status = RoleRunStatus(status)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            run = self._role_run_in_transaction(connection, access, role_run_id)
            plan = self._plan_in_transaction(connection, access, run.plan_id)
            claim = self._execution_claim_in_transaction(
                connection, access, role_run_id
            )
            if claim is not None and claim.expires_at > timestamp:
                if execution_claim_token != claim.claim_token:
                    raise ExecutionClaimUnavailableError(
                        "RoleRun transition requires its active execution claim"
                    )
            elif execution_claim_token is not None:
                raise ExecutionClaimUnavailableError(
                    "RoleRun execution claim is missing or expired"
                )
            elif claim is not None:
                connection.execute(
                    "DELETE FROM role_run_execution_claims WHERE role_run_id = ?",
                    (role_run_id,),
                )
            if run.version != expected_version:
                raise VersionConflictError("RoleRun version is stale")
            if status not in _ROLE_RUN_TRANSITIONS[run.status]:
                raise InvalidTransitionError(
                    f"invalid RoleRun transition: {run.status.value} -> {status.value}"
                )
            if plan.status in {
                PlanStatus.COMPLETED,
                PlanStatus.DEGRADED,
                PlanStatus.FAILED,
                PlanStatus.CANCELLED,
            } and status != RoleRunStatus.CANCELLED:
                raise InvalidTransitionError(
                    "a terminal speaker plan cannot advance a RoleRun"
                )
            output_value = dict(output) if output is not None else run.output
            if output_value is not None:
                _canonical_json(output_value)
            updated = run.model_copy(
                update={
                    "status": status,
                    "version": run.version + 1,
                    "output": output_value,
                    "error": error if status == RoleRunStatus.FAILED else None,
                    "updated_at": timestamp,
                },
                deep=True,
            )
            updated = RoleRun.model_validate(updated.model_dump())
            cursor = connection.execute(
                """
                UPDATE role_runs
                   SET status = ?, version = ?, role_run_json = ?, updated_at = ?
                 WHERE tenant_id = ? AND conversation_id = ?
                   AND role_run_id = ? AND version = ?
                   AND plan_id IN (
                       SELECT plan_id FROM speaker_plans
                        WHERE tenant_id = ? AND owner_user_id = ?
                          AND conversation_id = ?
                   )
                """,
                (
                    updated.status.value,
                    updated.version,
                    _model_json(updated),
                    timestamp.timestamp(),
                    access.tenant_id,
                    access.conversation_id,
                    role_run_id,
                    expected_version,
                    access.tenant_id,
                    access.user_id,
                    access.conversation_id,
                ),
            )
            if cursor.rowcount != 1:
                raise VersionConflictError("RoleRun CAS lost")
            if execution_claim_token is not None and status != RoleRunStatus.RUNNING:
                released = connection.execute(
                    """
                    DELETE FROM role_run_execution_claims
                     WHERE role_run_id = ? AND claim_token = ?
                    """,
                    (role_run_id, execution_claim_token),
                )
                if released.rowcount != 1:
                    raise ExecutionClaimUnavailableError(
                        "RoleRun execution claim release CAS lost"
                    )
            connection.commit()
            return updated
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_role_run(
        self, access: OrchestrationAccess, role_run_id: str
    ) -> RoleRun:
        _require_access(access)
        with self._connection() as connection:
            run = self._role_run_in_transaction(connection, access, role_run_id)
        return run

    def list_role_runs(
        self, access: OrchestrationAccess, plan_id: str
    ) -> list[RoleRun]:
        _require_access(access)
        self.get_plan(access, plan_id)
        role_clause = ""
        parameters: list[Any] = [
            access.tenant_id,
            access.conversation_id,
            plan_id,
        ]
        if access.allowed_role_ids is not None:
            placeholders = ",".join("?" for _ in access.allowed_role_ids)
            role_clause = f" AND role_id IN ({placeholders})"
            parameters.extend(access.allowed_role_ids)
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT role_run_json FROM role_runs
                 WHERE tenant_id = ? AND conversation_id = ? AND plan_id = ?
                   {role_clause}
                 ORDER BY created_at ASC, role_run_id ASC
                """,
                parameters,
            ).fetchall()
        return [RoleRun.model_validate_json(row["role_run_json"]) for row in rows]

    def claim_role_run_execution(
        self,
        access: OrchestrationAccess,
        role_run_id: str,
        *,
        claim_token: str,
        lease_seconds: int = 120,
        now: datetime | None = None,
    ) -> RoleRunExecutionClaim:
        """Atomically claim one active RoleRun before provider or tool work."""

        _require_access(access)
        if not isinstance(claim_token, str) or not 16 <= len(claim_token) <= 240:
            raise ValueError("claim_token must contain 16 to 240 characters")
        if not 15 <= int(lease_seconds) <= 3_600:
            raise ValueError("lease_seconds must be between 15 and 3600")
        timestamp = _utc(now)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            run = self._role_run_in_transaction(connection, access, role_run_id)
            if run.status not in {
                RoleRunStatus.PENDING,
                RoleRunStatus.QUEUED,
                RoleRunStatus.RUNNING,
                RoleRunStatus.WAITING_APPROVAL,
            }:
                raise InvalidTransitionError("only an active RoleRun may be claimed")
            existing = self._execution_claim_in_transaction(
                connection, access, role_run_id
            )
            if existing is not None and existing.claim_token == claim_token:
                connection.commit()
                return existing
            if existing is not None and existing.expires_at > timestamp:
                raise ExecutionClaimUnavailableError(
                    "RoleRun is already claimed by another executor"
                )
            if existing is not None:
                connection.execute(
                    "DELETE FROM role_run_execution_claims WHERE role_run_id = ?",
                    (role_run_id,),
                )
            claim = RoleRunExecutionClaim(
                claim_token=claim_token,
                role_run_id=role_run_id,
                tenant_id=access.tenant_id,
                owner_user_id=access.user_id,
                conversation_id=access.conversation_id,
                acquired_at=timestamp,
                expires_at=timestamp + timedelta(seconds=int(lease_seconds)),
            )
            connection.execute(
                """
                INSERT INTO role_run_execution_claims(
                    role_run_id, tenant_id, owner_user_id, conversation_id,
                    claim_token, expires_at, claim_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    claim.role_run_id,
                    claim.tenant_id,
                    claim.owner_user_id,
                    claim.conversation_id,
                    claim.claim_token,
                    claim.expires_at.timestamp(),
                    _model_json(claim),
                ),
            )
            connection.commit()
            return claim
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def renew_role_run_execution(
        self,
        access: OrchestrationAccess,
        role_run_id: str,
        *,
        claim_token: str,
        lease_seconds: int = 120,
        now: datetime | None = None,
    ) -> RoleRunExecutionClaim:
        _require_access(access)
        if not 15 <= int(lease_seconds) <= 3_600:
            raise ValueError("lease_seconds must be between 15 and 3600")
        timestamp = _utc(now)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._role_run_in_transaction(connection, access, role_run_id)
            existing = self._execution_claim_in_transaction(
                connection, access, role_run_id
            )
            if (
                existing is None
                or existing.claim_token != claim_token
                or existing.expires_at <= timestamp
            ):
                raise ExecutionClaimUnavailableError(
                    "RoleRun execution claim cannot be renewed"
                )
            renewed = RoleRunExecutionClaim.model_validate(
                existing.model_copy(
                    update={
                        "expires_at": timestamp
                        + timedelta(seconds=int(lease_seconds))
                    },
                    deep=True,
                ).model_dump()
            )
            updated = connection.execute(
                """
                UPDATE role_run_execution_claims
                   SET expires_at = ?, claim_json = ?
                 WHERE role_run_id = ? AND claim_token = ? AND expires_at > ?
                """,
                (
                    renewed.expires_at.timestamp(),
                    _model_json(renewed),
                    role_run_id,
                    claim_token,
                    timestamp.timestamp(),
                ),
            )
            if updated.rowcount != 1:
                raise ExecutionClaimUnavailableError(
                    "RoleRun execution claim renewal CAS lost"
                )
            connection.commit()
            return renewed
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def release_role_run_execution(
        self,
        access: OrchestrationAccess,
        role_run_id: str,
        *,
        claim_token: str,
    ) -> bool:
        _require_access(access)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._role_run_in_transaction(connection, access, role_run_id)
            existing = self._execution_claim_in_transaction(
                connection, access, role_run_id
            )
            if existing is None:
                connection.commit()
                return False
            if existing.claim_token != claim_token:
                raise ExecutionClaimUnavailableError(
                    "another executor owns the RoleRun claim"
                )
            deleted = connection.execute(
                """
                DELETE FROM role_run_execution_claims
                 WHERE role_run_id = ? AND claim_token = ?
                """,
                (role_run_id, claim_token),
            )
            if deleted.rowcount != 1:
                raise ExecutionClaimUnavailableError(
                    "RoleRun execution claim release CAS lost"
                )
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def propose_shared_fact(
        self,
        access: OrchestrationAccess,
        fact_key: str,
        value: Any,
        *,
        source_role_run_id: str,
        expected_version: int | None = None,
        now: datetime | None = None,
    ) -> SharedFact:
        _require_access(access)
        _canonical_json(value)
        timestamp = _utc(now)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            source = self._role_run_in_transaction(
                connection, access, source_role_run_id
            )
            if source.status != RoleRunStatus.SUCCEEDED:
                raise FactRuleError("only a succeeded RoleRun may propose shared facts")
            head = self._fact_head(connection, access, fact_key)
            current_version = head.version if head else 0
            if expected_version is None:
                if head is not None:
                    raise VersionConflictError(
                        "expected_version is required when a fact head already exists"
                    )
            elif expected_version != current_version:
                raise VersionConflictError("shared fact version is stale")
            fact = SharedFact(
                tenant_id=access.tenant_id,
                owner_user_id=access.user_id,
                conversation_id=access.conversation_id,
                fact_key=fact_key,
                value=value,
                status=FactStatus.PROPOSED,
                authority="model",
                version=current_version + 1,
                source_role_run_id=source_role_run_id,
                supersedes_fact_id=head.fact_id if head else None,
                created_at=timestamp,
            )
            self._insert_fact(connection, fact)
            connection.commit()
            return fact
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def verify_shared_fact(
        self,
        access: OrchestrationAccess,
        fact_key: str,
        *,
        expected_version: int,
        verifier: Literal["tool", "user", "system"],
        verifier_ref: str,
        now: datetime | None = None,
    ) -> SharedFact:
        _require_access(access)
        if not verifier_ref:
            raise FactRuleError("verifier_ref is required")
        if verifier not in {"tool", "user", "system"}:
            raise FactRuleError("models cannot issue fact verification receipts")
        timestamp = _utc(now)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            head = self._fact_head(connection, access, fact_key)
            if head is None:
                raise OrchestrationNotFoundError("shared fact not found")
            if head.version != expected_version:
                raise VersionConflictError("shared fact version is stale")
            if head.status != FactStatus.PROPOSED:
                raise FactRuleError("only the latest proposed fact can be verified")
            receipt_row = connection.execute(
                """
                SELECT receipt_json, consumed_by_fact_id
                 FROM fact_verification_receipts
                 WHERE receipt_id = ? AND tenant_id = ? AND owner_user_id = ?
                   AND conversation_id = ?
                """,
                (
                    verifier_ref,
                    access.tenant_id,
                    access.user_id,
                    access.conversation_id,
                ),
            ).fetchone()
            if receipt_row is None:
                raise FactRuleError("verification receipt is unavailable in this scope")
            receipt = FactVerificationReceipt.model_validate_json(
                receipt_row["receipt_json"]
            )
            if receipt_row["consumed_by_fact_id"] is not None:
                raise FactRuleError("verification receipt has already been consumed")
            if (
                receipt.fact_key != fact_key
                or receipt.authority != verifier
                or receipt.value_hash != _sha256(head.value)
            ):
                raise FactRuleError(
                    "verification receipt does not bind this fact key, value, and authority"
                )
            verified = SharedFact(
                tenant_id=access.tenant_id,
                owner_user_id=access.user_id,
                conversation_id=access.conversation_id,
                fact_key=head.fact_key,
                value=head.value,
                status=FactStatus.VERIFIED,
                authority=verifier,
                version=head.version + 1,
                source_role_run_id=head.source_role_run_id,
                supersedes_fact_id=head.fact_id,
                verifier_ref=verifier_ref,
                created_at=timestamp,
            )
            self._insert_fact(connection, verified)
            consumed = connection.execute(
                """
                UPDATE fact_verification_receipts
                   SET consumed_by_fact_id = ?
                 WHERE receipt_id = ? AND consumed_by_fact_id IS NULL
                """,
                (verified.fact_id, verifier_ref),
            )
            if consumed.rowcount != 1:
                raise FactRuleError("verification receipt consumption CAS lost")
            connection.commit()
            return verified
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def record_host_verification_receipt(
        self,
        access: OrchestrationAccess,
        fact_key: str,
        value: Any,
        *,
        authority: Literal["tool", "user", "system"],
        receipt_id: str,
        evidence_ref: str,
        now: datetime | None = None,
    ) -> FactVerificationReceipt:
        """Register trusted evidence produced by a gateway or user-approval layer.

        This method is intentionally host-only and must not be mounted as a
        model tool.  The immutable receipt binds scope, key, exact JSON value,
        authority, and the upstream audit/approval reference.
        """

        _require_access(access)
        _canonical_json(value)
        receipt = FactVerificationReceipt(
            receipt_id=receipt_id,
            tenant_id=access.tenant_id,
            owner_user_id=access.user_id,
            conversation_id=access.conversation_id,
            fact_key=fact_key,
            authority=authority,
            value_hash=_sha256(value),
            evidence_ref=evidence_ref,
            created_at=_utc(now),
        )
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT receipt_json FROM fact_verification_receipts
                 WHERE receipt_id = ?
                """,
                (receipt.receipt_id,),
            ).fetchone()
            if existing is not None:
                replay = FactVerificationReceipt.model_validate_json(
                    existing["receipt_json"]
                )
                replay_request = replay.model_dump(
                    mode="json", exclude={"created_at"}
                )
                receipt_request = receipt.model_dump(
                    mode="json", exclude={"created_at"}
                )
                if replay_request != receipt_request:
                    raise IdempotencyConflictError(
                        "verification receipt identity was reused with another payload"
                    )
                return replay
            connection.execute(
                """
                INSERT INTO fact_verification_receipts(
                    receipt_id, tenant_id, owner_user_id, conversation_id, fact_key,
                    authority, value_hash, evidence_ref, receipt_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt.receipt_id,
                    receipt.tenant_id,
                    receipt.owner_user_id,
                    receipt.conversation_id,
                    receipt.fact_key,
                    receipt.authority,
                    receipt.value_hash,
                    receipt.evidence_ref,
                    _model_json(receipt),
                    receipt.created_at.timestamp(),
                ),
            )
        return receipt

    def list_shared_facts(
        self,
        access: OrchestrationAccess,
        *,
        verified_only: bool = False,
        current_only: bool = True,
    ) -> list[SharedFact]:
        _require_access(access)
        clauses = (
            "tenant_id = ? AND owner_user_id = ? AND conversation_id = ?"
        )
        parameters: list[Any] = [
            access.tenant_id,
            access.user_id,
            access.conversation_id,
        ]
        if verified_only:
            clauses += " AND status = 'verified'"
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT fact_json FROM shared_facts
                 WHERE {clauses}
                 ORDER BY fact_key ASC, version DESC
                """,
                parameters,
            ).fetchall()
        facts = [SharedFact.model_validate_json(row["fact_json"]) for row in rows]
        if not current_only:
            return sorted(facts, key=lambda fact: (fact.fact_key, fact.version))
        current: dict[str, SharedFact] = {}
        for fact in facts:
            current.setdefault(fact.fact_key, fact)
        return list(current.values())

    def create_handoff(
        self,
        access: OrchestrationAccess,
        plan_id: str,
        *,
        from_role_run_id: str,
        to_slot_id: str,
        summary: str,
        shared_fact_ids: Sequence[str] = (),
        now: datetime | None = None,
    ) -> Handoff:
        _require_access(access)
        timestamp = _utc(now)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            plan = self._plan_in_transaction(connection, access, plan_id)
            if plan.status not in {PlanStatus.READY, PlanStatus.RUNNING}:
                raise InvalidTransitionError(
                    "handoff cannot be created after the speaker plan is terminal or paused"
                )
            source = self._role_run_in_transaction(
                connection, access, from_role_run_id
            )
            if source.plan_id != plan_id or source.status != RoleRunStatus.SUCCEEDED:
                raise SlotNotReadyError("handoff source must be a succeeded run in the plan")
            try:
                target = next(slot for slot in plan.slots if slot.slot_id == to_slot_id)
            except StopIteration as exc:
                raise SlotNotReadyError("handoff target is not in the fixed DAG") from exc
            access.require_role(target.role_id)
            if source.slot_id not in target.depends_on:
                raise SlotNotReadyError(
                    "handoff target must explicitly depend on the source slot"
                )
            fact_ids = list(shared_fact_ids)
            if not fact_ids:
                raise FactRuleError(
                    "handoff requires at least one current verified shared fact; "
                    "summary is untrusted model prose"
                )
            if len(fact_ids) != len(set(fact_ids)):
                raise FactRuleError("handoff shared fact IDs must be unique")
            for fact_id in fact_ids:
                row = connection.execute(
                    """
                    SELECT fact_json FROM shared_facts
                     WHERE tenant_id = ? AND owner_user_id = ?
                       AND conversation_id = ? AND fact_id = ?
                    """,
                    (
                        access.tenant_id,
                        access.user_id,
                        access.conversation_id,
                        fact_id,
                    ),
                ).fetchone()
                if row is None:
                    raise FactRuleError("handoff references an unavailable shared fact")
                fact = SharedFact.model_validate_json(row["fact_json"])
                if fact.status != FactStatus.VERIFIED:
                    raise FactRuleError("handoff may reference only verified shared facts")
                latest = connection.execute(
                    """
                    SELECT fact_id FROM shared_facts
                     WHERE tenant_id = ? AND owner_user_id = ?
                       AND conversation_id = ?
                       AND fact_key = ? AND status = 'verified'
                     ORDER BY version DESC LIMIT 1
                    """,
                    (
                        access.tenant_id,
                        access.user_id,
                        access.conversation_id,
                        fact.fact_key,
                    ),
                ).fetchone()
                if latest is None or latest["fact_id"] != fact.fact_id:
                    raise FactRuleError(
                        "handoff references a superseded verified fact version"
                    )
            payload_hash = _sha256(
                {
                    "from_role_run_id": from_role_run_id,
                    "to_slot_id": to_slot_id,
                    "summary": summary,
                    "shared_fact_ids": sorted(fact_ids),
                }
            )
            existing = connection.execute(
                """
                SELECT handoff_json, payload_hash FROM handoffs
                 WHERE from_role_run_id = ? AND to_slot_id = ?
                """,
                (from_role_run_id, to_slot_id),
            ).fetchone()
            if existing is not None:
                if existing["payload_hash"] != payload_hash:
                    raise IdempotencyConflictError(
                        "handoff identity was reused with a different payload"
                    )
                connection.commit()
                return Handoff.model_validate_json(existing["handoff_json"])
            handoff = Handoff(
                tenant_id=access.tenant_id,
                conversation_id=plan.conversation_id,
                plan_id=plan_id,
                from_role_run_id=from_role_run_id,
                to_slot_id=to_slot_id,
                summary=summary,
                shared_fact_ids=fact_ids,
                payload_hash=payload_hash,
                created_at=timestamp,
            )
            connection.execute(
                """
                INSERT INTO handoffs(
                    handoff_id, tenant_id, conversation_id, plan_id,
                    from_role_run_id, to_slot_id, payload_hash,
                    handoff_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    handoff.handoff_id,
                    handoff.tenant_id,
                    handoff.conversation_id,
                    handoff.plan_id,
                    handoff.from_role_run_id,
                    handoff.to_slot_id,
                    handoff.payload_hash,
                    _model_json(handoff),
                    timestamp.timestamp(),
                ),
            )
            connection.commit()
            return handoff
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def list_handoffs(
        self, access: OrchestrationAccess, plan_id: str
    ) -> list[Handoff]:
        _require_access(access)
        self.get_plan(access, plan_id)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT handoff_json FROM handoffs
                 WHERE tenant_id = ? AND conversation_id = ? AND plan_id = ?
                 ORDER BY created_at ASC, handoff_id ASC
                """,
                (access.tenant_id, access.conversation_id, plan_id),
            ).fetchall()
        return [Handoff.model_validate_json(row["handoff_json"]) for row in rows]

    def remember_private(
        self,
        access: OrchestrationAccess,
        role_id: str,
        content: str,
        *,
        kind: Literal["episode", "preference", "commitment", "relationship"] = "episode",
        provenance_role_run_id: str | None = None,
        provenance_key: str | None = None,
        extractor_version: str = "v1",
        expires_at: datetime | None = None,
        now: datetime | None = None,
    ) -> PrivateRoleMemory:
        _require_access(access)
        access.require_role(role_id)
        timestamp = _utc(now)
        expiry = _utc(expires_at) if expires_at is not None else None
        if expiry is not None and expiry <= timestamp:
            raise ValueError("private memory expiry must be in the future")
        if provenance_role_run_id is not None:
            source = self.get_role_run(access, provenance_role_run_id)
            if source.role_id != role_id:
                raise RoleNotAllowedError(
                    "private memory provenance must match tenant, conversation, and role"
                )
            if source.status != RoleRunStatus.SUCCEEDED:
                raise InvalidTransitionError(
                    "private long-term memory requires a succeeded RoleRun"
                )
        source_key = provenance_key or provenance_role_run_id or "manual"
        content_hash = _sha256(
            {
                "kind": kind,
                "content": content,
                "extractor_version": extractor_version,
            }
        )
        memory = PrivateRoleMemory(
            tenant_id=access.tenant_id,
            owner_user_id=access.user_id,
            conversation_id=access.conversation_id,
            role_id=role_id,
            kind=kind,
            content=content,
            provenance_role_run_id=provenance_role_run_id,
            extractor_version=extractor_version,
            content_hash=content_hash,
            expires_at=expiry,
            created_at=timestamp,
        )
        with self._connection() as connection:
            existing = connection.execute(
                """
                SELECT memory_json FROM private_role_memories
                 WHERE tenant_id = ? AND owner_user_id = ?
                   AND conversation_id = ? AND role_id = ?
                   AND provenance_key = ? AND content_hash = ?
                """,
                (
                    access.tenant_id,
                    access.user_id,
                    access.conversation_id,
                    role_id,
                    source_key,
                    content_hash,
                ),
            ).fetchone()
            if existing is not None:
                replay = PrivateRoleMemory.model_validate_json(existing["memory_json"])
                replay_expiry = _utc(replay.expires_at) if replay.expires_at else None
                if replay_expiry != expiry:
                    raise IdempotencyConflictError(
                        "private memory replay changed the expiry policy"
                    )
                return replay
            inserted = connection.execute(
                """
                INSERT OR IGNORE INTO private_role_memories(
                    memory_id, tenant_id, owner_user_id, conversation_id, role_id,
                    provenance_key, content_hash, memory_json, expires_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.memory_id,
                    access.tenant_id,
                    access.user_id,
                    access.conversation_id,
                    role_id,
                    source_key,
                    content_hash,
                    _model_json(memory),
                    expiry.timestamp() if expiry else None,
                    timestamp.timestamp(),
                ),
            )
            if inserted.rowcount == 0:
                replay = connection.execute(
                    """
                    SELECT memory_json FROM private_role_memories
                     WHERE tenant_id = ? AND owner_user_id = ?
                       AND conversation_id = ? AND role_id = ?
                       AND provenance_key = ? AND content_hash = ?
                    """,
                    (
                        access.tenant_id,
                        access.user_id,
                        access.conversation_id,
                        role_id,
                        source_key,
                        content_hash,
                    ),
                ).fetchone()
                if replay is None:
                    raise OrchestrationError("private memory idempotent insert lost")
                recovered = PrivateRoleMemory.model_validate_json(replay["memory_json"])
                recovered_expiry = (
                    _utc(recovered.expires_at) if recovered.expires_at else None
                )
                if recovered_expiry != expiry:
                    raise IdempotencyConflictError(
                        "private memory replay changed the expiry policy"
                    )
                return recovered
        return memory

    def list_private_memories(
        self,
        access: OrchestrationAccess,
        role_id: str,
        *,
        now: datetime | None = None,
    ) -> list[PrivateRoleMemory]:
        _require_access(access)
        access.require_role(role_id)
        timestamp = _utc(now).timestamp()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT memory_json FROM private_role_memories
                 WHERE tenant_id = ? AND owner_user_id = ?
                   AND conversation_id = ? AND role_id = ?
                   AND (expires_at IS NULL OR expires_at > ?)
                 ORDER BY created_at ASC, memory_id ASC
                """,
                (
                    access.tenant_id,
                    access.user_id,
                    access.conversation_id,
                    role_id,
                    timestamp,
                ),
            ).fetchall()
        return [
            PrivateRoleMemory.model_validate_json(row["memory_json"])
            for row in rows
        ]

    def get_private_memory(
        self,
        access: OrchestrationAccess,
        role_id: str,
        memory_id: str,
        *,
        now: datetime | None = None,
    ) -> PrivateRoleMemory:
        _require_access(access)
        access.require_role(role_id)
        timestamp = _utc(now).timestamp()
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT memory_json FROM private_role_memories
                 WHERE tenant_id = ? AND owner_user_id = ?
                   AND conversation_id = ? AND role_id = ?
                   AND memory_id = ? AND (expires_at IS NULL OR expires_at > ?)
                """,
                (
                    access.tenant_id,
                    access.user_id,
                    access.conversation_id,
                    role_id,
                    memory_id,
                    timestamp,
                ),
            ).fetchone()
        if row is None:
            raise OrchestrationNotFoundError("private role memory not found")
        return PrivateRoleMemory.model_validate_json(row["memory_json"])

    def _plan_in_transaction(
        self,
        connection: sqlite3.Connection,
        access: OrchestrationAccess,
        plan_id: str,
    ) -> SpeakerPlan:
        row = connection.execute(
            """
            SELECT plan_json FROM speaker_plans
             WHERE tenant_id = ? AND owner_user_id = ?
               AND conversation_id = ? AND plan_id = ?
            """,
            (
                access.tenant_id,
                access.user_id,
                access.conversation_id,
                plan_id,
            ),
        ).fetchone()
        if row is None:
            raise OrchestrationNotFoundError("speaker plan not found")
        return SpeakerPlan.model_validate_json(row["plan_json"])

    def _role_run_in_transaction(
        self,
        connection: sqlite3.Connection,
        access: OrchestrationAccess,
        role_run_id: str,
    ) -> RoleRun:
        row = connection.execute(
            """
            SELECT role_runs.role_run_json
              FROM role_runs
              JOIN speaker_plans
                ON speaker_plans.plan_id = role_runs.plan_id
             WHERE role_runs.tenant_id = ?
               AND role_runs.conversation_id = ?
               AND role_runs.role_run_id = ?
               AND speaker_plans.tenant_id = ?
               AND speaker_plans.owner_user_id = ?
               AND speaker_plans.conversation_id = ?
            """,
            (
                access.tenant_id,
                access.conversation_id,
                role_run_id,
                access.tenant_id,
                access.user_id,
                access.conversation_id,
            ),
        ).fetchone()
        if row is None:
            raise OrchestrationNotFoundError("RoleRun not found")
        run = RoleRun.model_validate_json(row["role_run_json"])
        access.require_role(run.role_id)
        return run

    @staticmethod
    def _execution_claim_in_transaction(
        connection: sqlite3.Connection,
        access: OrchestrationAccess,
        role_run_id: str,
    ) -> RoleRunExecutionClaim | None:
        row = connection.execute(
            """
            SELECT claim_json FROM role_run_execution_claims
             WHERE role_run_id = ? AND tenant_id = ? AND owner_user_id = ?
               AND conversation_id = ?
            """,
            (
                role_run_id,
                access.tenant_id,
                access.user_id,
                access.conversation_id,
            ),
        ).fetchone()
        return (
            RoleRunExecutionClaim.model_validate_json(row["claim_json"])
            if row is not None
            else None
        )

    def _next_ready_in_transaction(
        self,
        connection: sqlite3.Connection,
        plan: SpeakerPlan,
    ) -> list[SpeakerSlot]:
        if plan.status not in {PlanStatus.READY, PlanStatus.RUNNING}:
            return []
        rows = connection.execute(
            """
            SELECT role_run_json FROM role_runs
             WHERE tenant_id = ? AND plan_id = ?
             ORDER BY attempt ASC
            """,
            (plan.tenant_id, plan.plan_id),
        ).fetchall()
        runs = [RoleRun.model_validate_json(row["role_run_json"]) for row in rows]
        if len(runs) >= plan.max_role_runs:
            return []
        latest: dict[str, RoleRun] = {}
        attempts: dict[str, int] = {}
        for run in runs:
            latest[run.slot_id] = run
            attempts[run.slot_id] = max(attempts.get(run.slot_id, 0), run.attempt)
        succeeded = {
            slot_id
            for slot_id, run in latest.items()
            if run.status == RoleRunStatus.SUCCEEDED
        }
        active = {
            RoleRunStatus.PENDING,
            RoleRunStatus.QUEUED,
            RoleRunStatus.RUNNING,
            RoleRunStatus.WAITING_APPROVAL,
        }
        ready = []
        for slot in plan.slots:
            current = latest.get(slot.slot_id)
            if current and current.status in active | {RoleRunStatus.SUCCEEDED}:
                continue
            if current and current.status == RoleRunStatus.CANCELLED:
                continue
            if attempts.get(slot.slot_id, 0) >= slot.max_attempts:
                continue
            if set(slot.depends_on).issubset(succeeded):
                ready.append(slot)
        return sorted(ready, key=lambda slot: (slot.order, slot.slot_id))

    @staticmethod
    def _latest_role_run(
        connection: sqlite3.Connection,
        access: OrchestrationAccess,
        plan_id: str,
        slot_id: str,
    ) -> RoleRun | None:
        row = connection.execute(
            """
            SELECT role_runs.role_run_json
              FROM role_runs
              JOIN speaker_plans
                ON speaker_plans.plan_id = role_runs.plan_id
             WHERE role_runs.tenant_id = ?
               AND role_runs.conversation_id = ?
               AND role_runs.plan_id = ? AND role_runs.slot_id = ?
               AND speaker_plans.owner_user_id = ?
             ORDER BY role_runs.attempt DESC LIMIT 1
            """,
            (
                access.tenant_id,
                access.conversation_id,
                plan_id,
                slot_id,
                access.user_id,
            ),
        ).fetchone()
        return RoleRun.model_validate_json(row["role_run_json"]) if row else None

    @staticmethod
    def _assert_plan_completion_in_transaction(
        connection: sqlite3.Connection,
        plan: SpeakerPlan,
    ) -> None:
        rows = connection.execute(
            """
            SELECT role_run_json FROM role_runs
             WHERE tenant_id = ? AND plan_id = ?
             ORDER BY attempt ASC
            """,
            (plan.tenant_id, plan.plan_id),
        ).fetchall()
        latest: dict[str, RoleRun] = {}
        for row in rows:
            run = RoleRun.model_validate_json(row["role_run_json"])
            latest[run.slot_id] = run
        active = {
            RoleRunStatus.PENDING,
            RoleRunStatus.QUEUED,
            RoleRunStatus.RUNNING,
            RoleRunStatus.WAITING_APPROVAL,
        }
        if any(run.status in active for run in latest.values()):
            raise InvalidTransitionError(
                "speaker plan cannot complete while a RoleRun is active"
            )
        missing = sorted(
            slot_id
            for slot_id in _required_slot_closure(plan.slots)
            if slot_id not in latest
            or latest[slot_id].status != RoleRunStatus.SUCCEEDED
        )
        if missing:
            raise InvalidTransitionError(
                f"speaker plan required slots have not succeeded: {missing}"
            )

    @staticmethod
    def _fact_head(
        connection: sqlite3.Connection,
        access: OrchestrationAccess,
        fact_key: str,
    ) -> SharedFact | None:
        row = connection.execute(
            """
            SELECT fact_json FROM shared_facts
             WHERE tenant_id = ? AND owner_user_id = ?
               AND conversation_id = ? AND fact_key = ?
             ORDER BY version DESC LIMIT 1
            """,
            (
                access.tenant_id,
                access.user_id,
                access.conversation_id,
                fact_key,
            ),
        ).fetchone()
        return SharedFact.model_validate_json(row["fact_json"]) if row else None

    @staticmethod
    def _insert_fact(connection: sqlite3.Connection, fact: SharedFact) -> None:
        connection.execute(
            """
            INSERT INTO shared_facts(
                fact_id, tenant_id, owner_user_id, conversation_id, fact_key,
                version, status, fact_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fact.fact_id,
                fact.tenant_id,
                fact.owner_user_id,
                fact.conversation_id,
                fact.fact_key,
                fact.version,
                fact.status.value,
                _model_json(fact),
                fact.created_at.timestamp(),
            ),
        )


def _utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _require_access(access: OrchestrationAccess) -> None:
    if not isinstance(access, OrchestrationAccess):
        raise TypeError(
            "orchestration operations require a trusted OrchestrationAccess"
        )


OrchestrationStore = SQLiteOrchestrationStore


__all__ = [
    "ExecutionClaimUnavailableError",
    "FactRuleError",
    "FactStatus",
    "FactVerificationReceipt",
    "Handoff",
    "IdempotencyConflictError",
    "InvalidTransitionError",
    "OrchestrationAccess",
    "OrchestrationNotFoundError",
    "OrchestrationStore",
    "PlanStatus",
    "PrivateRoleMemory",
    "RoleNotAllowedError",
    "RoleRun",
    "RoleRunExecutionClaim",
    "RoleRunStatus",
    "SQLiteOrchestrationStore",
    "SharedFact",
    "SlotNotReadyError",
    "SpeakerPlan",
    "SpeakerSlot",
    "VersionConflictError",
    "compute_plan_request_hash",
]

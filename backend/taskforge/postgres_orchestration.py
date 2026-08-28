"""PostgreSQL persistence for the TaskForge orchestration state machine.

The domain validators and transition tables remain in :mod:`orchestration`;
this module supplies the durable transport only.  Every public operation
executes inside a tenant-scoped transaction, locks the identity it is about
to mutate, and uses a version/claim check before writing a new JSONB snapshot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any, Literal

from .orchestration import (
    _PLAN_TRANSITIONS,
    _ROLE_RUN_TRANSITIONS,
    FactRuleError,
    FactStatus,
    FactVerificationReceipt,
    Handoff,
    IdempotencyConflictError,
    InvalidTransitionError,
    OrchestrationAccess,
    OrchestrationNotFoundError,
    PlanStatus,
    PrivateRoleMemory,
    RoleNotAllowedError,
    RoleRun,
    RoleRunExecutionClaim,
    RoleRunStatus,
    SlotNotReadyError,
    SpeakerPlan,
    SpeakerSlot,
    VersionConflictError,
    _canonical_json,
    _require_access,
    _required_slot_closure,
    _sha256,
    _utc,
    compute_plan_request_hash,
)
from .postgres_runtime import PostgresRuntime


def _json(value: object) -> object:
    try:
        from psycopg.types.json import Json
    except ImportError:  # pragma: no cover - optional dependency
        return value
    return Json(value)


def _model_value(value: Any, model: type[Any]) -> Any:
    if isinstance(value, str):
        return model.model_validate_json(value)
    return model.model_validate(value)


def _row_value(row: Any, key: str, index: int = 0) -> Any:
    return row[key] if isinstance(row, dict) else row[index]


def _is_unique_violation(exc: BaseException) -> bool:
    return exc.__class__.__name__ in {"UniqueViolation", "IntegrityError"}


class PostgresOrchestrationStore:
    """Tenant-scoped PostgreSQL implementation of ``OrchestrationStore``."""

    def __init__(
        self,
        dsn: str | None = None,
        *,
        tenant_id: str = "local",
        min_size: int = 1,
        max_size: int = 8,
        connect_timeout_seconds: float = 5.0,
        runtime: PostgresRuntime | None = None,
    ) -> None:
        if not tenant_id.strip():
            raise ValueError("tenant_id is required")
        if runtime is None and not dsn:
            raise ValueError("dsn or runtime is required")
        self.tenant_id = tenant_id
        self._owns_runtime = runtime is None
        self.runtime = runtime or PostgresRuntime(
            dsn or "",
            min_size=min_size,
            max_size=max_size,
            connect_timeout_seconds=connect_timeout_seconds,
        )

    def close(self) -> None:
        if self._owns_runtime:
            self.runtime.close()

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
            item if isinstance(item, SpeakerSlot) else SpeakerSlot.model_validate(item)
            for item in slots
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
            plan_id=plan_id or self._new_id(),
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
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            cursor.execute(
                """
                INSERT INTO orchestration.speaker_plans(
                    tenant_id, plan_id, owner_user_id, conversation_id,
                    client_idempotency_key, request_hash, status, version,
                    plan_json, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (tenant_id, owner_user_id, client_idempotency_key)
                DO NOTHING
                RETURNING plan_json
                """,
                (
                    plan.tenant_id,
                    plan.plan_id,
                    plan.owner_user_id,
                    plan.conversation_id,
                    plan.client_idempotency_key,
                    plan.request_hash,
                    plan.status.value,
                    plan.version,
                    _json(plan.model_dump(mode="json")),
                    timestamp,
                    timestamp,
                ),
            )
            row = cursor.fetchone()
            if row is not None:
                return _model_value(_row_value(row, "plan_json"), SpeakerPlan)
            cursor.execute(
                """
                SELECT plan_json, request_hash
                  FROM orchestration.speaker_plans
                 WHERE tenant_id = %s AND owner_user_id = %s
                   AND conversation_id = %s AND client_idempotency_key = %s
                 FOR UPDATE
                """,
                (
                    access.tenant_id,
                    access.user_id,
                    access.conversation_id,
                    client_idempotency_key,
                ),
            )
            existing = cursor.fetchone()
            if existing is None:
                raise IdempotencyConflictError("plan insert was not durable")
            if _row_value(existing, "request_hash", 1) != request_hash:
                raise IdempotencyConflictError(
                    "client idempotency key was reused with a different plan request"
                )
            return _model_value(_row_value(existing, "plan_json"), SpeakerPlan)

    def get_plan(self, access: OrchestrationAccess, plan_id: str) -> SpeakerPlan:
        _require_access(access)
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            return self._plan(cursor, access, plan_id)

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
        target_status = PlanStatus(status)
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            plan = self._plan(cursor, access, plan_id, for_update=True)
            if plan.version != expected_version:
                raise VersionConflictError("speaker plan version is stale")
            if target_status not in _PLAN_TRANSITIONS[plan.status]:
                raise InvalidTransitionError(
                    f"invalid plan transition: {plan.status.value} -> {target_status.value}"
                )
            if target_status == PlanStatus.COMPLETED:
                self._assert_plan_completion(cursor, plan)
            updated = plan.model_copy(
                update={
                    "status": target_status,
                    "version": plan.version + 1,
                    "updated_at": timestamp,
                },
                deep=True,
            )
            cursor.execute(
                """
                UPDATE orchestration.speaker_plans
                   SET status = %s, version = %s, plan_json = %s, updated_at = %s
                 WHERE tenant_id = %s AND owner_user_id = %s
                   AND conversation_id = %s AND plan_id = %s AND version = %s
                RETURNING plan_json
                """,
                (
                    updated.status.value,
                    updated.version,
                    _json(updated.model_dump(mode="json")),
                    timestamp,
                    access.tenant_id,
                    access.user_id,
                    access.conversation_id,
                    plan_id,
                    expected_version,
                ),
            )
            row = cursor.fetchone()
            if row is None:
                raise VersionConflictError("speaker plan CAS lost")
            return _model_value(_row_value(row, "plan_json"), SpeakerPlan)

    def next_ready_slots(self, access: OrchestrationAccess, plan_id: str) -> list[SpeakerSlot]:
        _require_access(access)
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            plan = self._plan(cursor, access, plan_id)
            ready = self._next_ready(cursor, plan)
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
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            plan = self._plan(cursor, access, plan_id, for_update=True)
            if plan.version != expected_plan_version:
                raise VersionConflictError("speaker plan version is stale")
            slot = next((item for item in plan.slots if item.slot_id == slot_id), None)
            if slot is None:
                raise SlotNotReadyError("slot is not present in the fixed DAG")
            access.require_role(slot.role_id)
            if role_run_id is not None:
                cursor.execute(
                    """
                    SELECT role_run_json FROM orchestration.role_runs
                     WHERE tenant_id = %s AND conversation_id = %s
                       AND role_run_id = %s AND plan_id = %s
                    FOR UPDATE
                    """,
                    (access.tenant_id, access.conversation_id, role_run_id, plan_id),
                )
                replay = cursor.fetchone()
                if replay is not None:
                    existing = _model_value(_row_value(replay, "role_run_json"), RoleRun)
                    if existing.slot_id != slot_id:
                        raise IdempotencyConflictError(
                            "RoleRun identity was reused for another plan slot"
                        )
                    return existing
            ready_ids = {item.slot_id for item in self._next_ready(cursor, plan)}
            if slot_id not in ready_ids:
                existing = self._latest_role_run(cursor, access, plan_id, slot_id)
                if existing is not None and existing.status in {
                    RoleRunStatus.PENDING,
                    RoleRunStatus.QUEUED,
                    RoleRunStatus.RUNNING,
                    RoleRunStatus.WAITING_APPROVAL,
                    RoleRunStatus.SUCCEEDED,
                    RoleRunStatus.CANCELLED,
                }:
                    if role_run_id is not None and existing.role_run_id != role_run_id:
                        raise IdempotencyConflictError(
                            "explicit RoleRun identity conflicts with slot materialization"
                        )
                    return existing
                raise SlotNotReadyError("slot is not ready in the fixed DAG")
            cursor.execute(
                """
                SELECT COALESCE(MAX(attempt), 0) AS attempt
                  FROM orchestration.role_runs
                 WHERE tenant_id = %s AND conversation_id = %s
                   AND plan_id = %s AND slot_id = %s
                """,
                (access.tenant_id, access.conversation_id, plan_id, slot_id),
            )
            attempt = int(_row_value(cursor.fetchone(), "attempt")) + 1
            if attempt > slot.max_attempts:
                raise SlotNotReadyError("slot exhausted its attempt budget")
            identifier = role_run_id or self._new_id()
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
            try:
                cursor.execute(
                    """
                    INSERT INTO orchestration.role_runs(
                        tenant_id, role_run_id, run_id, conversation_id, plan_id,
                        slot_id, role_id, attempt, status, version,
                        role_run_json, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run.tenant_id,
                        run.role_run_id,
                        run.run_id,
                        run.conversation_id,
                        run.plan_id,
                        run.slot_id,
                        run.role_id,
                        run.attempt,
                        run.status.value,
                        run.version,
                        _json(run.model_dump(mode="json")),
                        timestamp,
                        timestamp,
                    ),
                )
            except Exception as exc:
                if _is_unique_violation(exc):
                    raise VersionConflictError("RoleRun slot materialization raced") from exc
                raise
            return run

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
        target_status = RoleRunStatus(status)
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            run = self._role_run(cursor, access, role_run_id, for_update=True)
            plan = self._plan(cursor, access, run.plan_id, for_update=True)
            claim = self._claim(cursor, access, role_run_id, for_update=True)
            if claim is not None and claim.expires_at > timestamp:
                if execution_claim_token != claim.claim_token:
                    from .orchestration import ExecutionClaimUnavailableError

                    raise ExecutionClaimUnavailableError(
                        "RoleRun transition requires its active execution claim"
                    )
            elif execution_claim_token is not None:
                from .orchestration import ExecutionClaimUnavailableError

                raise ExecutionClaimUnavailableError(
                    "RoleRun execution claim is missing or expired"
                )
            elif claim is not None:
                cursor.execute(
                    "DELETE FROM orchestration.role_run_execution_claims "
                    "WHERE tenant_id = %s AND role_run_id = %s",
                    (access.tenant_id, role_run_id),
                )
            if run.version != expected_version:
                raise VersionConflictError("RoleRun version is stale")
            if target_status not in _ROLE_RUN_TRANSITIONS[run.status]:
                raise InvalidTransitionError(
                    f"invalid RoleRun transition: {run.status.value} -> {target_status.value}"
                )
            if (
                plan.status in {
                    PlanStatus.COMPLETED,
                    PlanStatus.DEGRADED,
                    PlanStatus.FAILED,
                    PlanStatus.CANCELLED,
                }
                and target_status != RoleRunStatus.CANCELLED
            ):
                raise InvalidTransitionError(
                    "a terminal speaker plan cannot advance a RoleRun"
                )
            output_value = dict(output) if output is not None else run.output
            if output_value is not None:
                _canonical_json(output_value)
            updated = run.model_copy(
                update={
                    "status": target_status,
                    "version": run.version + 1,
                    "output": output_value,
                    "error": error if target_status == RoleRunStatus.FAILED else None,
                    "updated_at": timestamp,
                },
                deep=True,
            )
            updated = RoleRun.model_validate(updated.model_dump())
            cursor.execute(
                """
                UPDATE orchestration.role_runs
                   SET status = %s, version = %s, role_run_json = %s, updated_at = %s
                 WHERE tenant_id = %s AND conversation_id = %s
                   AND role_run_id = %s AND version = %s
                RETURNING role_run_json
                """,
                (
                    updated.status.value,
                    updated.version,
                    _json(updated.model_dump(mode="json")),
                    timestamp,
                    access.tenant_id,
                    access.conversation_id,
                    role_run_id,
                    expected_version,
                ),
            )
            row = cursor.fetchone()
            if row is None:
                raise VersionConflictError("RoleRun CAS lost")
            return _model_value(_row_value(row, "role_run_json"), RoleRun)

    def get_role_run(self, access: OrchestrationAccess, role_run_id: str) -> RoleRun:
        _require_access(access)
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            return self._role_run(cursor, access, role_run_id)

    def list_role_runs(self, access: OrchestrationAccess, plan_id: str) -> list[RoleRun]:
        _require_access(access)
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            self._plan(cursor, access, plan_id)
            cursor.execute(
                """
                SELECT role_run_json FROM orchestration.role_runs
                 WHERE tenant_id = %s AND conversation_id = %s AND plan_id = %s
                 ORDER BY created_at ASC, role_run_id ASC
                """,
                (access.tenant_id, access.conversation_id, plan_id),
            )
            runs = [_model_value(_row_value(row, "role_run_json"), RoleRun) for row in cursor.fetchall()]
        if access.allowed_role_ids is None:
            return runs
        return [run for run in runs if run.role_id in set(access.allowed_role_ids)]

    def claim_role_run_execution(
        self,
        access: OrchestrationAccess,
        role_run_id: str,
        *,
        claim_token: str,
        lease_seconds: int = 120,
        now: datetime | None = None,
    ) -> RoleRunExecutionClaim:
        from .orchestration import ExecutionClaimUnavailableError

        _require_access(access)
        if not isinstance(claim_token, str) or not 16 <= len(claim_token) <= 240:
            raise ValueError("claim_token must contain 16 to 240 characters")
        if not 15 <= int(lease_seconds) <= 3_600:
            raise ValueError("lease_seconds must be between 15 and 3600")
        timestamp = _utc(now)
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            run = self._role_run(cursor, access, role_run_id, for_update=True)
            if run.status not in {
                RoleRunStatus.PENDING,
                RoleRunStatus.QUEUED,
                RoleRunStatus.RUNNING,
                RoleRunStatus.WAITING_APPROVAL,
            }:
                raise InvalidTransitionError("only an active RoleRun may be claimed")
            existing = self._claim(cursor, access, role_run_id, for_update=True)
            if existing is not None and existing.claim_token == claim_token:
                return existing
            if existing is not None and existing.expires_at > timestamp:
                raise ExecutionClaimUnavailableError(
                    "RoleRun is already claimed by another executor"
                )
            if existing is not None:
                cursor.execute(
                    "DELETE FROM orchestration.role_run_execution_claims "
                    "WHERE tenant_id = %s AND role_run_id = %s",
                    (access.tenant_id, role_run_id),
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
            cursor.execute(
                """
                INSERT INTO orchestration.role_run_execution_claims(
                    tenant_id, role_run_id, owner_user_id, conversation_id,
                    claim_token, expires_at, claim_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    claim.tenant_id,
                    claim.role_run_id,
                    claim.owner_user_id,
                    claim.conversation_id,
                    claim.claim_token,
                    claim.expires_at,
                    _json(claim.model_dump(mode="json")),
                ),
            )
            return claim

    def renew_role_run_execution(
        self,
        access: OrchestrationAccess,
        role_run_id: str,
        *,
        claim_token: str,
        lease_seconds: int = 120,
        now: datetime | None = None,
    ) -> RoleRunExecutionClaim:
        from .orchestration import ExecutionClaimUnavailableError

        _require_access(access)
        if not 15 <= int(lease_seconds) <= 3_600:
            raise ValueError("lease_seconds must be between 15 and 3600")
        timestamp = _utc(now)
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            self._role_run(cursor, access, role_run_id, for_update=True)
            existing = self._claim(cursor, access, role_run_id, for_update=True)
            if (
                existing is None
                or existing.claim_token != claim_token
                or existing.expires_at <= timestamp
            ):
                raise ExecutionClaimUnavailableError(
                    "RoleRun execution claim cannot be renewed"
                )
            renewed = existing.model_copy(
                update={
                    "expires_at": timestamp + timedelta(seconds=int(lease_seconds))
                },
                deep=True,
            )
            cursor.execute(
                """
                UPDATE orchestration.role_run_execution_claims
                   SET expires_at = %s, claim_json = %s
                 WHERE tenant_id = %s AND role_run_id = %s
                   AND claim_token = %s AND expires_at > %s
                """,
                (
                    renewed.expires_at,
                    _json(renewed.model_dump(mode="json")),
                    access.tenant_id,
                    role_run_id,
                    claim_token,
                    timestamp,
                ),
            )
            if cursor.rowcount != 1:
                raise ExecutionClaimUnavailableError("RoleRun claim renewal CAS lost")
            return renewed

    def release_role_run_execution(
        self,
        access: OrchestrationAccess,
        role_run_id: str,
        *,
        claim_token: str,
    ) -> bool:
        from .orchestration import ExecutionClaimUnavailableError

        _require_access(access)
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            self._role_run(cursor, access, role_run_id, for_update=True)
            existing = self._claim(cursor, access, role_run_id, for_update=True)
            if existing is None:
                return False
            if existing.claim_token != claim_token:
                raise ExecutionClaimUnavailableError("another executor owns the RoleRun claim")
            cursor.execute(
                "DELETE FROM orchestration.role_run_execution_claims "
                "WHERE tenant_id = %s AND role_run_id = %s AND claim_token = %s",
                (access.tenant_id, role_run_id, claim_token),
            )
            if cursor.rowcount != 1:
                raise ExecutionClaimUnavailableError("RoleRun claim release CAS lost")
            return True

    def propose_shared_fact(
        self,
        access: OrchestrationAccess,
        fact_key: str,
        value: Any,
        *,
        source_role_run_id: str,
        expected_version: int | None = None,
        now: datetime | None = None,
    ) -> Any:
        _require_access(access)
        _canonical_json(value)
        timestamp = _utc(now)
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            source = self._role_run(cursor, access, source_role_run_id, for_update=True)
            if source.status != RoleRunStatus.SUCCEEDED:
                raise FactRuleError("only a succeeded RoleRun may propose shared facts")
            head = self._fact_head(cursor, access, fact_key, for_update=True)
            current_version = head.version if head else 0
            if expected_version is None:
                if head is not None:
                    raise VersionConflictError(
                        "expected_version is required when a fact head already exists"
                    )
            elif expected_version != current_version:
                raise VersionConflictError("shared fact version is stale")
            from .orchestration import SharedFact

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
            self._insert_fact(cursor, fact)
            return fact

    def verify_shared_fact(
        self,
        access: OrchestrationAccess,
        fact_key: str,
        *,
        expected_version: int,
        verifier: Literal["tool", "user", "system"],
        verifier_ref: str,
        now: datetime | None = None,
    ) -> Any:
        _require_access(access)
        if not verifier_ref or verifier not in {"tool", "user", "system"}:
            raise FactRuleError("verification authority and receipt are invalid")
        timestamp = _utc(now)
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            head = self._fact_head(cursor, access, fact_key, for_update=True)
            if head is None:
                raise OrchestrationNotFoundError("shared fact not found")
            if head.version != expected_version:
                raise VersionConflictError("shared fact version is stale")
            if head.status != FactStatus.PROPOSED:
                raise FactRuleError("only the latest proposed fact can be verified")
            cursor.execute(
                """
                SELECT receipt_json, consumed_by_fact_id
                  FROM orchestration.fact_verification_receipts
                 WHERE tenant_id = %s AND owner_user_id = %s
                   AND conversation_id = %s AND receipt_id = %s
                 FOR UPDATE
                """,
                (
                    access.tenant_id,
                    access.user_id,
                    access.conversation_id,
                    verifier_ref,
                ),
            )
            receipt_row = cursor.fetchone()
            if receipt_row is None:
                raise FactRuleError("verification receipt is unavailable in this scope")
            if _row_value(receipt_row, "consumed_by_fact_id", 1) is not None:
                raise FactRuleError("verification receipt has already been consumed")
            receipt = _model_value(_row_value(receipt_row, "receipt_json"), FactVerificationReceipt)
            if (
                receipt.fact_key != fact_key
                or receipt.authority != verifier
                or receipt.value_hash != _sha256(head.value)
            ):
                raise FactRuleError("verification receipt does not bind this fact")
            from .orchestration import SharedFact

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
            self._insert_fact(cursor, verified)
            cursor.execute(
                """
                UPDATE orchestration.fact_verification_receipts
                   SET consumed_by_fact_id = %s
                 WHERE tenant_id = %s AND receipt_id = %s
                   AND consumed_by_fact_id IS NULL
                """,
                (verified.fact_id, access.tenant_id, verifier_ref),
            )
            if cursor.rowcount != 1:
                raise FactRuleError("verification receipt consumption CAS lost")
            return verified

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
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            cursor.execute(
                """
                SELECT receipt_json FROM orchestration.fact_verification_receipts
                 WHERE tenant_id = %s AND receipt_id = %s
                 FOR UPDATE
                """,
                (access.tenant_id, receipt_id),
            )
            existing = cursor.fetchone()
            if existing is not None:
                replay = _model_value(_row_value(existing, "receipt_json"), FactVerificationReceipt)
                if replay.model_dump(mode="json", exclude={"created_at"}) != receipt.model_dump(
                    mode="json", exclude={"created_at"}
                ):
                    raise IdempotencyConflictError(
                        "verification receipt identity was reused with another payload"
                    )
                return replay
            cursor.execute(
                """
                INSERT INTO orchestration.fact_verification_receipts(
                    tenant_id, receipt_id, owner_user_id, conversation_id,
                    fact_key, authority, value_hash, evidence_ref, receipt_json, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    receipt.tenant_id,
                    receipt.receipt_id,
                    receipt.owner_user_id,
                    receipt.conversation_id,
                    receipt.fact_key,
                    receipt.authority,
                    receipt.value_hash,
                    receipt.evidence_ref,
                    _json(receipt.model_dump(mode="json")),
                    receipt.created_at,
                ),
            )
            return receipt

    def list_shared_facts(
        self,
        access: OrchestrationAccess,
        *,
        verified_only: bool = False,
        current_only: bool = True,
    ) -> list[Any]:
        _require_access(access)
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            status_clause = " AND status = 'verified'" if verified_only else ""
            cursor.execute(
                "SELECT fact_json FROM orchestration.shared_facts "
                "WHERE tenant_id = %s AND owner_user_id = %s AND conversation_id = %s"
                + status_clause
                + " ORDER BY fact_key ASC, version DESC",
                (access.tenant_id, access.user_id, access.conversation_id),
            )
            from .orchestration import SharedFact

            facts = [_model_value(_row_value(row, "fact_json"), SharedFact) for row in cursor.fetchall()]
        if not current_only:
            return sorted(facts, key=lambda fact: (fact.fact_key, fact.version))
        current: dict[str, Any] = {}
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
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            plan = self._plan(cursor, access, plan_id, for_update=True)
            if plan.status not in {PlanStatus.READY, PlanStatus.RUNNING}:
                raise InvalidTransitionError("handoff cannot be created after plan termination")
            source = self._role_run(cursor, access, from_role_run_id, for_update=True)
            if source.plan_id != plan_id or source.status != RoleRunStatus.SUCCEEDED:
                raise SlotNotReadyError("handoff source must be a succeeded run in the plan")
            target = next((slot for slot in plan.slots if slot.slot_id == to_slot_id), None)
            if target is None:
                raise SlotNotReadyError("handoff target is not in the fixed DAG")
            access.require_role(target.role_id)
            if source.slot_id not in target.depends_on:
                raise SlotNotReadyError("handoff target must depend on the source slot")
            fact_ids = list(shared_fact_ids)
            if not fact_ids:
                raise FactRuleError("handoff requires a current verified shared fact")
            if len(fact_ids) != len(set(fact_ids)):
                raise FactRuleError("handoff shared fact IDs must be unique")
            for fact_id in fact_ids:
                cursor.execute(
                    """
                    SELECT fact_json FROM orchestration.shared_facts
                     WHERE tenant_id = %s AND owner_user_id = %s
                       AND conversation_id = %s AND fact_id = %s
                    FOR UPDATE
                    """,
                    (
                        access.tenant_id,
                        access.user_id,
                        access.conversation_id,
                        fact_id,
                    ),
                )
                row = cursor.fetchone()
                if row is None:
                    raise FactRuleError("handoff references an unavailable shared fact")
                from .orchestration import SharedFact

                fact = _model_value(_row_value(row, "fact_json"), SharedFact)
                if fact.status != FactStatus.VERIFIED:
                    raise FactRuleError("handoff may reference only verified facts")
                latest = self._fact_head(cursor, access, fact.fact_key)
                if latest is None or latest.fact_id != fact.fact_id:
                    raise FactRuleError("handoff references a superseded verified fact")
            payload_hash = _sha256(
                {
                    "from_role_run_id": from_role_run_id,
                    "to_slot_id": to_slot_id,
                    "summary": summary,
                    "shared_fact_ids": sorted(fact_ids),
                }
            )
            cursor.execute(
                """
                SELECT handoff_json, payload_hash
                  FROM orchestration.handoffs
                 WHERE tenant_id = %s AND from_role_run_id = %s AND to_slot_id = %s
                 FOR UPDATE
                """,
                (access.tenant_id, from_role_run_id, to_slot_id),
            )
            existing = cursor.fetchone()
            if existing is not None:
                if _row_value(existing, "payload_hash", 1) != payload_hash:
                    raise IdempotencyConflictError("handoff identity was reused with another payload")
                return _model_value(_row_value(existing, "handoff_json"), Handoff)
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
            cursor.execute(
                """
                INSERT INTO orchestration.handoffs(
                    tenant_id, handoff_id, conversation_id, plan_id,
                    from_role_run_id, to_slot_id, payload_hash, handoff_json, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    handoff.tenant_id,
                    handoff.handoff_id,
                    handoff.conversation_id,
                    handoff.plan_id,
                    handoff.from_role_run_id,
                    handoff.to_slot_id,
                    handoff.payload_hash,
                    _json(handoff.model_dump(mode="json")),
                    handoff.created_at,
                ),
            )
            return handoff

    def list_handoffs(self, access: OrchestrationAccess, plan_id: str) -> list[Handoff]:
        _require_access(access)
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            self._plan(cursor, access, plan_id)
            cursor.execute(
                """
                SELECT handoff_json FROM orchestration.handoffs
                 WHERE tenant_id = %s AND conversation_id = %s AND plan_id = %s
                 ORDER BY created_at ASC, handoff_id ASC
                """,
                (access.tenant_id, access.conversation_id, plan_id),
            )
            return [_model_value(_row_value(row, "handoff_json"), Handoff) for row in cursor.fetchall()]

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
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            if provenance_role_run_id is not None:
                source = self._role_run(cursor, access, provenance_role_run_id)
                if source.role_id != role_id:
                    raise RoleNotAllowedError("private memory provenance role mismatch")
                if source.status != RoleRunStatus.SUCCEEDED:
                    raise InvalidTransitionError("private memory requires a succeeded RoleRun")
            source_key = provenance_key or provenance_role_run_id or "manual"
            content_hash = _sha256(
                {"kind": kind, "content": content, "extractor_version": extractor_version}
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
            cursor.execute(
                """
                SELECT memory_json FROM orchestration.private_role_memories
                 WHERE tenant_id = %s AND owner_user_id = %s
                   AND conversation_id = %s AND role_id = %s
                   AND provenance_key = %s AND content_hash = %s
                 FOR UPDATE
                """,
                (
                    access.tenant_id,
                    access.user_id,
                    access.conversation_id,
                    role_id,
                    source_key,
                    content_hash,
                ),
            )
            existing = cursor.fetchone()
            if existing is not None:
                replay = _model_value(_row_value(existing, "memory_json"), PrivateRoleMemory)
                if replay.expires_at != expiry:
                    raise IdempotencyConflictError("private memory replay changed expiry policy")
                return replay
            cursor.execute(
                """
                INSERT INTO orchestration.private_role_memories(
                    tenant_id, memory_id, owner_user_id, conversation_id, role_id,
                    provenance_key, content_hash, memory_json, expires_at, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (
                    memory.tenant_id,
                    memory.memory_id,
                    memory.owner_user_id,
                    memory.conversation_id,
                    memory.role_id,
                    source_key,
                    memory.content_hash,
                    _json(memory.model_dump(mode="json")),
                    memory.expires_at,
                    memory.created_at,
                ),
            )
            if cursor.rowcount == 0:
                cursor.execute(
                    """
                    SELECT memory_json FROM orchestration.private_role_memories
                     WHERE tenant_id = %s AND owner_user_id = %s
                       AND conversation_id = %s AND role_id = %s
                       AND provenance_key = %s AND content_hash = %s
                    """,
                    (
                        access.tenant_id,
                        access.user_id,
                        access.conversation_id,
                        role_id,
                        source_key,
                        content_hash,
                    ),
                )
                replay_row = cursor.fetchone()
                if replay_row is None:
                    raise IdempotencyConflictError("private memory idempotent insert was lost")
                replay = _model_value(_row_value(replay_row, "memory_json"), PrivateRoleMemory)
                if replay.expires_at != expiry:
                    raise IdempotencyConflictError("private memory replay changed expiry policy")
                return replay
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
        timestamp = _utc(now)
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            cursor.execute(
                """
                SELECT memory_json FROM orchestration.private_role_memories
                 WHERE tenant_id = %s AND owner_user_id = %s
                   AND conversation_id = %s AND role_id = %s
                   AND (expires_at IS NULL OR expires_at > %s)
                 ORDER BY created_at ASC, memory_id ASC
                """,
                (
                    access.tenant_id,
                    access.user_id,
                    access.conversation_id,
                    role_id,
                    timestamp,
                ),
            )
            return [_model_value(_row_value(row, "memory_json"), PrivateRoleMemory) for row in cursor.fetchall()]

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
        timestamp = _utc(now)
        with self.runtime.transaction(access.tenant_id) as (_, cursor):
            cursor.execute(
                """
                SELECT memory_json FROM orchestration.private_role_memories
                 WHERE tenant_id = %s AND owner_user_id = %s
                   AND conversation_id = %s AND role_id = %s AND memory_id = %s
                   AND (expires_at IS NULL OR expires_at > %s)
                """,
                (
                    access.tenant_id,
                    access.user_id,
                    access.conversation_id,
                    role_id,
                    memory_id,
                    timestamp,
                ),
            )
            row = cursor.fetchone()
            if row is None:
                raise OrchestrationNotFoundError("private role memory not found")
            return _model_value(_row_value(row, "memory_json"), PrivateRoleMemory)

    def _plan(
        self,
        cursor: Any,
        access: OrchestrationAccess,
        plan_id: str,
        *,
        for_update: bool = False,
    ) -> SpeakerPlan:
        cursor.execute(
            "SELECT plan_json FROM orchestration.speaker_plans "
            "WHERE tenant_id = %s AND owner_user_id = %s "
            "AND conversation_id = %s AND plan_id = %s"
            + (" FOR UPDATE" if for_update else ""),
            (access.tenant_id, access.user_id, access.conversation_id, plan_id),
        )
        row = cursor.fetchone()
        if row is None:
            raise OrchestrationNotFoundError("speaker plan not found")
        return _model_value(_row_value(row, "plan_json"), SpeakerPlan)

    def _role_run(
        self,
        cursor: Any,
        access: OrchestrationAccess,
        role_run_id: str,
        *,
        for_update: bool = False,
    ) -> RoleRun:
        cursor.execute(
            """
            SELECT rr.role_run_json
              FROM orchestration.role_runs AS rr
              JOIN orchestration.speaker_plans AS sp
                ON sp.tenant_id = rr.tenant_id AND sp.plan_id = rr.plan_id
             WHERE rr.tenant_id = %s AND rr.conversation_id = %s
               AND rr.role_run_id = %s AND sp.owner_user_id = %s
               AND sp.conversation_id = %s
            """ + (" FOR UPDATE OF rr, sp" if for_update else ""),
            (
                access.tenant_id,
                access.conversation_id,
                role_run_id,
                access.user_id,
                access.conversation_id,
            ),
        )
        row = cursor.fetchone()
        if row is None:
            raise OrchestrationNotFoundError("RoleRun not found")
        run = _model_value(_row_value(row, "role_run_json"), RoleRun)
        access.require_role(run.role_id)
        return run

    def _claim(
        self,
        cursor: Any,
        access: OrchestrationAccess,
        role_run_id: str,
        *,
        for_update: bool = False,
    ) -> RoleRunExecutionClaim | None:
        cursor.execute(
            "SELECT claim_json FROM orchestration.role_run_execution_claims "
            "WHERE tenant_id = %s AND role_run_id = %s AND owner_user_id = %s "
            "AND conversation_id = %s"
            + (" FOR UPDATE" if for_update else ""),
            (access.tenant_id, role_run_id, access.user_id, access.conversation_id),
        )
        row = cursor.fetchone()
        return _model_value(_row_value(row, "claim_json"), RoleRunExecutionClaim) if row else None

    def _fact_head(
        self,
        cursor: Any,
        access: OrchestrationAccess,
        fact_key: str,
        *,
        for_update: bool = False,
    ) -> Any | None:
        cursor.execute(
            """
            SELECT fact_json FROM orchestration.shared_facts
             WHERE tenant_id = %s AND owner_user_id = %s
               AND conversation_id = %s AND fact_key = %s
             ORDER BY version DESC LIMIT 1
            """ + (" FOR UPDATE" if for_update else ""),
            (access.tenant_id, access.user_id, access.conversation_id, fact_key),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        from .orchestration import SharedFact

        return _model_value(_row_value(row, "fact_json"), SharedFact)

    def _next_ready(self, cursor: Any, plan: SpeakerPlan) -> list[SpeakerSlot]:
        if plan.status not in {PlanStatus.READY, PlanStatus.RUNNING}:
            return []
        cursor.execute(
            """
            SELECT role_run_json FROM orchestration.role_runs
             WHERE tenant_id = %s AND plan_id = %s
             ORDER BY attempt ASC, role_run_id ASC
            """,
            (plan.tenant_id, plan.plan_id),
        )
        runs = [_model_value(_row_value(row, "role_run_json"), RoleRun) for row in cursor.fetchall()]
        if len(runs) >= plan.max_role_runs:
            return []
        latest: dict[str, RoleRun] = {}
        attempts: dict[str, int] = {}
        for run in runs:
            latest[run.slot_id] = run
            attempts[run.slot_id] = max(attempts.get(run.slot_id, 0), run.attempt)
        succeeded = {slot_id for slot_id, run in latest.items() if run.status == RoleRunStatus.SUCCEEDED}
        active = {
            RoleRunStatus.PENDING,
            RoleRunStatus.QUEUED,
            RoleRunStatus.RUNNING,
            RoleRunStatus.WAITING_APPROVAL,
        }
        ready: list[SpeakerSlot] = []
        for slot in plan.slots:
            current = latest.get(slot.slot_id)
            if current and current.status in active | {RoleRunStatus.SUCCEEDED, RoleRunStatus.CANCELLED}:
                continue
            if attempts.get(slot.slot_id, 0) >= slot.max_attempts:
                continue
            if set(slot.depends_on).issubset(succeeded):
                ready.append(slot)
        return sorted(ready, key=lambda item: (item.order, item.slot_id))

    def _latest_role_run(
        self,
        cursor: Any,
        access: OrchestrationAccess,
        plan_id: str,
        slot_id: str,
    ) -> RoleRun | None:
        cursor.execute(
            """
            SELECT role_run_json FROM orchestration.role_runs
             WHERE tenant_id = %s AND conversation_id = %s
               AND plan_id = %s AND slot_id = %s
             ORDER BY attempt DESC LIMIT 1
            """,
            (access.tenant_id, access.conversation_id, plan_id, slot_id),
        )
        row = cursor.fetchone()
        return _model_value(_row_value(row, "role_run_json"), RoleRun) if row else None

    def _assert_plan_completion(self, cursor: Any, plan: SpeakerPlan) -> None:
        cursor.execute(
            "SELECT role_run_json FROM orchestration.role_runs "
            "WHERE tenant_id = %s AND plan_id = %s ORDER BY attempt ASC",
            (plan.tenant_id, plan.plan_id),
        )
        latest: dict[str, RoleRun] = {}
        for row in cursor.fetchall():
            run = _model_value(_row_value(row, "role_run_json"), RoleRun)
            latest[run.slot_id] = run
        active = {
            RoleRunStatus.PENDING,
            RoleRunStatus.QUEUED,
            RoleRunStatus.RUNNING,
            RoleRunStatus.WAITING_APPROVAL,
        }
        if any(run.status in active for run in latest.values()):
            raise InvalidTransitionError("speaker plan cannot complete while a RoleRun is active")
        missing = sorted(
            slot_id
            for slot_id in _required_slot_closure(plan.slots)
            if slot_id not in latest or latest[slot_id].status != RoleRunStatus.SUCCEEDED
        )
        if missing:
            raise InvalidTransitionError(
                f"speaker plan required slots have not succeeded: {missing}"
            )

    @staticmethod
    def _insert_fact(cursor: Any, fact: Any) -> None:
        cursor.execute(
            """
            INSERT INTO orchestration.shared_facts(
                tenant_id, fact_id, owner_user_id, conversation_id, fact_key,
                version, status, fact_json, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                fact.tenant_id,
                fact.fact_id,
                fact.owner_user_id,
                fact.conversation_id,
                fact.fact_key,
                fact.version,
                fact.status.value,
                _json(fact.model_dump(mode="json")),
                fact.created_at,
            ),
        )

    @staticmethod
    def _new_id() -> str:
        from uuid import uuid4

        return str(uuid4())


__all__ = ["PostgresOrchestrationStore"]

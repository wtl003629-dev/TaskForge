"""PostgreSQL queue and audit-log backend for TaskForge."""

from __future__ import annotations

import secrets
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from .operations import (
    AuditEvent,
    AuditUsage,
    DuplicateJobError,
    JobNotFoundError,
    JobStatus,
    LeaseLostError,
    MetricsSnapshot,
    OperationJob,
    OperationsError,
    _assert_secret_free,
    _percentile,
    sanitize_failure,
)
from .postgres_runtime import PostgresRuntime

_JOB_COLUMNS = (
    "run_id, tenant_id, status, priority, attempt, max_attempts, available_at, "
    "owner, lease_token, lease_version, lease_expires_at, result_status, "
    "last_error, created_at, updated_at"
)
_AUDIT_COLUMNS = (
    "event_id, tenant_id, run_id, action, outcome, duration_ms, tool, provider, "
    "input_tokens, output_tokens, total_tokens, cost_usd, safety_violation, "
    "metadata_json, occurred_at"
)


class PostgresOperationsStore:
    """Pooled queue with SKIP LOCKED claims and lease-version CAS updates."""

    def __init__(
        self,
        dsn: str,
        *,
        tenant_id: str = "local",
        base_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 300.0,
        min_size: int = 1,
        max_size: int = 8,
        connect_timeout_seconds: float = 5.0,
        runtime: PostgresRuntime | None = None,
    ) -> None:
        if base_backoff_seconds < 0:
            raise ValueError("base_backoff_seconds must be non-negative")
        if max_backoff_seconds < base_backoff_seconds:
            raise ValueError("max_backoff_seconds must be at least the base backoff")
        if not tenant_id.strip():
            raise ValueError("tenant_id is required")
        self.tenant_id = tenant_id
        self._owns_runtime = runtime is None
        self.base_backoff_seconds = float(base_backoff_seconds)
        self.max_backoff_seconds = float(max_backoff_seconds)
        self.runtime = runtime or PostgresRuntime(
            dsn,
            min_size=min_size,
            max_size=max_size,
            connect_timeout_seconds=connect_timeout_seconds,
        )

    def close(self) -> None:
        if self._owns_runtime:
            self.runtime.close()

    def enqueue(
        self,
        run_id: str,
        tenant_id: str,
        *,
        max_attempts: int = 3,
        priority: int = 0,
        available_at: datetime | None = None,
        now: datetime | None = None,
    ) -> OperationJob:
        if not run_id or not tenant_id:
            raise ValueError("run_id and tenant_id are required")
        if max_attempts < 1 or max_attempts > 100:
            raise ValueError("max_attempts must be between 1 and 100")
        current = _as_utc(now)
        available = _as_utc(available_at) if available_at else current
        with self.runtime.transaction(tenant_id) as (_, cursor):
            try:
                cursor.execute(
                    f"""
                    INSERT INTO operations.operation_jobs(
                        tenant_id, run_id, status, priority, attempt, max_attempts,
                        available_at, lease_version, created_at, updated_at
                    ) VALUES (%s, %s, 'queued', %s, 0, %s, %s, 0, %s, %s)
                    RETURNING {_JOB_COLUMNS}
                    """,
                    (
                        tenant_id,
                        run_id,
                        int(priority),
                        int(max_attempts),
                        available,
                        current,
                        current,
                    ),
                )
            except Exception as exc:
                if _is_unique_violation(exc):
                    raise DuplicateJobError(f"run already enqueued: {run_id}") from exc
                raise
            row = cursor.fetchone()
        if row is None:
            raise OperationsError("queue insert did not return a job")
        return _job(row)

    def claim(
        self,
        owner: str,
        *,
        lease_seconds: float = 30.0,
        tenant_id: str | None = None,
        now: datetime | None = None,
    ) -> OperationJob | None:
        if not owner:
            raise ValueError("owner is required")
        if lease_seconds <= 0 or lease_seconds > 86_400:
            raise ValueError("lease_seconds must be in (0, 86400]")
        current_tenant = self._tenant(tenant_id)
        current = _as_utc(now)
        token = secrets.token_urlsafe(32)
        with self.runtime.transaction(current_tenant) as (_, cursor):
            cursor.execute(
                f"""
                SELECT {_JOB_COLUMNS}
                  FROM operations.operation_jobs
                 WHERE tenant_id = %s
                   AND (
                        (status = 'queued' AND available_at <= %s)
                     OR (status = 'leased' AND lease_expires_at <= %s)
                   )
                 ORDER BY priority DESC, available_at ASC, created_at ASC, run_id ASC
                 FOR UPDATE SKIP LOCKED
                 LIMIT 1
                """,
                (current_tenant, current, current),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            run_id = _value(row, "run_id", 0)
            cursor.execute(
                f"""
                UPDATE operations.operation_jobs
                   SET status = 'leased', owner = %s, lease_token = %s,
                       lease_version = lease_version + 1,
                       lease_expires_at = %s, attempt = attempt + 1,
                       last_error = CASE WHEN status = 'leased'
                                         THEN 'lease_expired' ELSE last_error END,
                       updated_at = %s
                 WHERE tenant_id = %s AND run_id = %s
                   AND (
                        (status = 'queued' AND available_at <= %s)
                     OR (status = 'leased' AND lease_expires_at <= %s)
                   )
                RETURNING {_JOB_COLUMNS}
                """,
                (
                    owner,
                    token,
                    current + timedelta(seconds=float(lease_seconds)),
                    current,
                    current_tenant,
                    run_id,
                    current,
                    current,
                ),
            )
            claimed = cursor.fetchone()
        if claimed is None:
            raise OperationsError("atomic claim lost inside PostgreSQL transaction")
        return _job(claimed)

    def heartbeat(
        self,
        job: OperationJob,
        *,
        lease_seconds: float = 30.0,
        now: datetime | None = None,
    ) -> OperationJob:
        if lease_seconds <= 0 or lease_seconds > 86_400:
            raise ValueError("lease_seconds must be in (0, 86400]")
        current = _as_utc(now)
        return self._lease_cas(
            job,
            """
            SET lease_expires_at = %s, lease_version = lease_version + 1,
                updated_at = %s
            WHERE tenant_id = %s AND run_id = %s AND status = 'leased'
              AND owner = %s AND lease_token = %s AND lease_version = %s
              AND lease_expires_at > %s
            """,
            (
                current + timedelta(seconds=float(lease_seconds)),
                current,
                job.tenant_id,
                job.run_id,
                job.owner,
                job.lease_token,
                job.lease_version,
                current,
            ),
        )

    def complete(
        self,
        job: OperationJob,
        *,
        result_status: str,
        now: datetime | None = None,
    ) -> OperationJob:
        if not result_status:
            raise ValueError("result_status is required")
        current = _as_utc(now)
        return self._lease_cas(
            job,
            """
            SET status = 'completed', result_status = %s, owner = NULL,
                lease_token = NULL, lease_expires_at = NULL,
                lease_version = lease_version + 1, updated_at = %s
            WHERE tenant_id = %s AND run_id = %s AND status = 'leased'
              AND owner = %s AND lease_token = %s AND lease_version = %s
              AND lease_expires_at > %s
            """,
            (
                result_status,
                current,
                job.tenant_id,
                job.run_id,
                job.owner,
                job.lease_token,
                job.lease_version,
                current,
            ),
        )

    def fail(
        self,
        job: OperationJob,
        error: BaseException | str,
        *,
        now: datetime | None = None,
    ) -> OperationJob:
        current = _as_utc(now)
        terminal = job.attempt >= job.max_attempts
        if terminal:
            status = JobStatus.DEAD_LETTER.value
            result_status = "failed"
            available = current
        else:
            status = JobStatus.QUEUED.value
            result_status = "retry_scheduled"
            delay = min(
                self.max_backoff_seconds,
                self.base_backoff_seconds * (2 ** max(0, job.attempt - 1)),
            )
            available = current + timedelta(seconds=delay)
        return self._lease_cas(
            job,
            """
            SET status = %s, result_status = %s, last_error = %s,
                available_at = %s, owner = NULL, lease_token = NULL,
                lease_expires_at = NULL, lease_version = lease_version + 1,
                updated_at = %s
            WHERE tenant_id = %s AND run_id = %s AND status = 'leased'
              AND owner = %s AND lease_token = %s AND lease_version = %s
              AND lease_expires_at > %s
            """,
            (
                status,
                result_status,
                sanitize_failure(error),
                available,
                current,
                job.tenant_id,
                job.run_id,
                job.owner,
                job.lease_token,
                job.lease_version,
                current,
            ),
        )

    def _lease_cas(
        self,
        job: OperationJob,
        assignment_sql: str,
        parameters: Sequence[Any],
    ) -> OperationJob:
        if job.status != JobStatus.LEASED or not job.owner or not job.lease_token:
            raise LeaseLostError("job does not carry an active lease")
        with self.runtime.transaction(job.tenant_id) as (_, cursor):
            cursor.execute(
                f"UPDATE operations.operation_jobs {assignment_sql} RETURNING {_JOB_COLUMNS}",
                tuple(parameters),
            )
            row = cursor.fetchone()
        if row is None:
            raise LeaseLostError("lease CAS rejected: stale, expired, or wrong owner")
        return _job(row)

    def get_job(self, run_id: str, *, tenant_id: str | None = None) -> OperationJob:
        current_tenant = self._tenant(tenant_id)
        with self.runtime.transaction(current_tenant) as (_, cursor):
            cursor.execute(
                f"SELECT {_JOB_COLUMNS} FROM operations.operation_jobs "
                "WHERE tenant_id = %s AND run_id = %s",
                (current_tenant, run_id),
            )
            row = cursor.fetchone()
        if row is None:
            raise JobNotFoundError(f"job not found: {run_id}")
        return _job(row)

    def append_audit(self, event: AuditEvent | Mapping[str, Any]) -> AuditEvent:
        validated = _validate_audit(event)
        with self.runtime.transaction(validated.tenant_id) as (_, cursor):
            cursor.execute(
                f"""
                INSERT INTO operations.audit_events(
                    event_id, tenant_id, run_id, action, outcome, duration_ms,
                    tool, provider, input_tokens, output_tokens, total_tokens,
                    cost_usd, safety_violation, metadata_json, occurred_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING {_AUDIT_COLUMNS}
                """,
                _audit_parameters(validated),
            )
            row = cursor.fetchone()
        if row is None:
            raise OperationsError("audit insert did not return an event")
        return _audit(row)

    def append_audit_once(self, event: AuditEvent | Mapping[str, Any]) -> AuditEvent:
        validated = _validate_audit(event)
        with self.runtime.transaction(validated.tenant_id) as (_, cursor):
            cursor.execute(
                f"""
                INSERT INTO operations.audit_events(
                    event_id, tenant_id, run_id, action, outcome, duration_ms,
                    tool, provider, input_tokens, output_tokens, total_tokens,
                    cost_usd, safety_violation, metadata_json, occurred_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (event_id) DO NOTHING
                RETURNING {_AUDIT_COLUMNS}
                """,
                _audit_parameters(validated),
            )
            row = cursor.fetchone()
            if row is not None:
                return _audit(row)
            cursor.execute(
                f"SELECT {_AUDIT_COLUMNS} FROM operations.audit_events "
                "WHERE tenant_id = %s AND event_id = %s",
                (validated.tenant_id, validated.event_id),
            )
            existing = cursor.fetchone()
            if existing is None:
                raise OperationsError("audit event idempotent insert lost")
            replay = _audit(existing)
            if _audit_identity(replay) != _audit_identity(validated):
                raise OperationsError("audit event_id collision")
            return replay

    append_event = append_audit

    def list_audit(
        self,
        tenant_id: str,
        *,
        run_id: str | None = None,
        limit: int = 200,
        latest: bool = False,
    ) -> list[AuditEvent]:
        if not tenant_id:
            raise ValueError("tenant_id is required")
        if limit < 1 or limit > 1_000:
            raise ValueError("limit must be between 1 and 1000")
        where = "tenant_id = %s"
        params: list[Any] = [tenant_id]
        if run_id is not None:
            where += " AND run_id = %s"
            params.append(run_id)
        if latest:
            query = (
                f"SELECT {_AUDIT_COLUMNS} FROM operations.audit_events "
                f"WHERE {where} ORDER BY sequence DESC LIMIT %s"
            )
        else:
            query = (
                f"SELECT {_AUDIT_COLUMNS} FROM operations.audit_events "
                f"WHERE {where} ORDER BY sequence ASC LIMIT %s"
            )
        params.append(limit)
        with self.runtime.transaction(tenant_id) as (_, cursor):
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
        converted = [_audit(row) for row in rows]
        return list(reversed(converted)) if latest else converted

    list_events = list_audit

    def metrics(self, tenant_id: str, *, run_id: str | None = None) -> MetricsSnapshot:
        rows = self.list_audit(tenant_id, run_id=run_id, limit=1_000, latest=False)
        success = {"ok", "success", "succeeded", "completed"}
        latest_runs: dict[str, AuditEvent] = {}
        tool_rows: list[AuditEvent] = []
        durations: list[float] = []
        for row in rows:
            if row.action == "run" or row.action.startswith("run."):
                latest_runs[row.run_id] = row
            if row.tool is not None:
                tool_rows.append(row)
            if row.duration_ms is not None:
                durations.append(float(row.duration_ms))
        run_successes = sum(
            1 for row in latest_runs.values() if row.outcome.lower() in success
        )
        tool_successes = sum(1 for row in tool_rows if row.outcome.lower() in success)

        def optional_sum(name: str, *, integral: bool) -> int | float | None:
            values = [
                getattr(row.usage, name)
                for row in rows
                if row.usage is not None and getattr(row.usage, name) is not None
            ]
            if not values:
                return None
            total = sum(values)
            return int(total) if integral else float(total)

        return MetricsSnapshot(
            tenant_id=tenant_id,
            run_id=run_id,
            run_count=len(latest_runs),
            run_success_count=run_successes,
            run_success_rate=(run_successes / len(latest_runs) if latest_runs else None),
            tool_count=len(tool_rows),
            tool_success_count=tool_successes,
            tool_success_rate=(tool_successes / len(tool_rows) if tool_rows else None),
            duration_p50_ms=_percentile(durations, 0.50),
            duration_p95_ms=_percentile(durations, 0.95),
            input_tokens=optional_sum("input_tokens", integral=True),
            output_tokens=optional_sum("output_tokens", integral=True),
            total_tokens=optional_sum("total_tokens", integral=True),
            cost_usd=optional_sum("cost_usd", integral=False),
            safety_violation_count=sum(int(row.safety_violation) for row in rows),
        )

    aggregate_metrics = metrics

    def _tenant(self, tenant_id: str | None) -> str:
        current = tenant_id or self.tenant_id
        if not current.strip():
            raise ValueError("tenant_id is required")
        return current


def _validate_audit(event: AuditEvent | Mapping[str, Any]) -> AuditEvent:
    if isinstance(event, AuditEvent):
        return event
    _assert_secret_free(event, path="event")
    return AuditEvent.model_validate(event)


def _json(value: object) -> object:
    try:
        from psycopg.types.json import Json
    except ImportError:  # pragma: no cover - real runtime imports psycopg first
        return value
    return Json(value)


def _audit_parameters(event: AuditEvent) -> tuple[Any, ...]:
    usage = event.usage
    return (
        event.event_id,
        event.tenant_id,
        event.run_id,
        event.action,
        event.outcome,
        event.duration_ms,
        event.tool,
        event.provider,
        usage.input_tokens if usage else None,
        usage.output_tokens if usage else None,
        usage.total_tokens if usage else None,
        usage.cost_usd if usage else None,
        event.safety_violation,
        _json(event.metadata),
        _as_utc(event.occurred_at),
    )


def _job(row: Any) -> OperationJob:
    return OperationJob(
        run_id=_value(row, "run_id", 0),
        tenant_id=_value(row, "tenant_id", 1),
        status=JobStatus(_value(row, "status", 2)),
        priority=int(_value(row, "priority", 3)),
        attempt=int(_value(row, "attempt", 4)),
        max_attempts=int(_value(row, "max_attempts", 5)),
        available_at=_timestamp(_value(row, "available_at", 6)),
        owner=_value(row, "owner", 7),
        lease_token=_value(row, "lease_token", 8),
        lease_version=int(_value(row, "lease_version", 9)),
        lease_expires_at=_optional_timestamp(_value(row, "lease_expires_at", 10)),
        result_status=_value(row, "result_status", 11),
        last_error=_value(row, "last_error", 12),
        created_at=_timestamp(_value(row, "created_at", 13)),
        updated_at=_timestamp(_value(row, "updated_at", 14)),
    )


def _audit(row: Any) -> AuditEvent:
    input_tokens = _value(row, "input_tokens", 8)
    output_tokens = _value(row, "output_tokens", 9)
    total_tokens = _value(row, "total_tokens", 10)
    cost_usd = _value(row, "cost_usd", 11)
    usage = None
    if any(value is not None for value in (input_tokens, output_tokens, total_tokens, cost_usd)):
        usage = AuditUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
        )
    metadata = _value(row, "metadata_json", 13)
    if isinstance(metadata, str):
        import json

        metadata = json.loads(metadata)
    return AuditEvent(
        event_id=_value(row, "event_id", 0),
        tenant_id=_value(row, "tenant_id", 1),
        run_id=_value(row, "run_id", 2),
        action=_value(row, "action", 3),
        outcome=_value(row, "outcome", 4),
        duration_ms=_value(row, "duration_ms", 5),
        tool=_value(row, "tool", 6),
        provider=_value(row, "provider", 7),
        usage=usage,
        safety_violation=bool(_value(row, "safety_violation", 12)),
        metadata=metadata,
        occurred_at=_timestamp(_value(row, "occurred_at", 14)),
    )


def _audit_identity(event: AuditEvent) -> tuple[Any, ...]:
    return (
        event.tenant_id,
        event.run_id,
        event.action,
        event.outcome,
        event.tool,
        event.provider,
        event.usage,
        event.safety_violation,
        event.metadata,
    )


def _value(row: Any, key: str, index: int) -> Any:
    if isinstance(row, dict):
        return row[key]
    return row[index]


def _timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return _as_utc(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return datetime.fromtimestamp(float(value), tz=UTC)
    raise ValueError("PostgreSQL timestamp column returned an invalid value")


def _optional_timestamp(value: Any) -> datetime | None:
    return None if value is None else _timestamp(value)


def _as_utc(value: datetime | None) -> datetime:
    current = datetime.now(UTC) if value is None else value
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return current.astimezone(UTC)


def _is_unique_violation(exc: BaseException) -> bool:
    return getattr(exc, "sqlstate", None) == "23505" or "UniqueViolation" in type(exc).__name__


__all__ = ["PostgresOperationsStore"]

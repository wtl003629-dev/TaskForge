"""Durable worker operations, audit events, and aggregate metrics.

SQLite is used as a local deployment backend, but the contracts mirror a
database queue: workers obtain expiring leases, and every mutation after claim
is a compare-and-swap on owner, opaque token, and monotonically increasing
lease version.
"""

from __future__ import annotations

import json
import math
import re
import secrets
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import Field, model_validator

from .domain import RunState, StrictModel, ToolResult, utc_now


class OperationsError(RuntimeError):
    """Base error for the durable operations store."""


class DuplicateJobError(OperationsError):
    pass


class JobNotFoundError(OperationsError):
    pass


class LeaseLostError(OperationsError):
    """The caller no longer owns the current, unexpired job lease."""


class AuditSecretError(ValueError):
    """Audit payload contains a credential-like key or value."""


class JobStatus(str, Enum):
    QUEUED = "queued"
    LEASED = "leased"
    COMPLETED = "completed"
    DEAD_LETTER = "dead_letter"


class OperationJob(StrictModel):
    run_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    status: JobStatus
    priority: int = 0
    attempt: int = Field(ge=0)
    max_attempts: int = Field(ge=1)
    available_at: datetime
    owner: str | None = None
    lease_token: str | None = None
    lease_version: int = Field(ge=0)
    lease_expires_at: datetime | None = None
    result_status: str | None = None
    last_error: str | None = None
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def lease_fields_match_status(self) -> OperationJob:
        leased = self.status == JobStatus.LEASED
        lease_values = (self.owner, self.lease_token, self.lease_expires_at)
        if leased and not all(value is not None for value in lease_values):
            raise ValueError("lease owner, token, and expiry are required only for leased jobs")
        if not leased and any(value is not None for value in lease_values):
            raise ValueError("non-leased jobs cannot retain lease identity")
        return self


# A descriptive alias for callers that prefer queue terminology.
JobRecord = OperationJob


class AuditUsage(StrictModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)


def audit_usage_from_state(
    state: RunState,
    *,
    previous: RunState | None = None,
) -> AuditUsage | None:
    """Aggregate model usage, optionally returning only a resumed-run delta."""

    fields = ("input_tokens", "output_tokens", "total_tokens", "cost_usd")

    def totals(run: RunState | None) -> tuple[dict[str, float], set[str]]:
        values = {field: 0.0 for field in fields}
        seen: set[str] = set()
        if run is None:
            return values, seen
        for step in run.steps:
            if step.model_turn is None:
                continue
            raw = step.model_turn.metadata.get("usage")
            if not isinstance(raw, Mapping):
                continue
            for field in fields:
                aliases = {
                    "input_tokens": ("input_tokens", "prompt_tokens"),
                    "output_tokens": ("output_tokens", "completion_tokens"),
                    "total_tokens": ("total_tokens",),
                    "cost_usd": ("cost_usd",),
                }[field]
                value = next((raw.get(alias) for alias in aliases if raw.get(alias) is not None), None)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                    values[field] += float(value)
                    seen.add(field)
        return values, seen

    current, current_seen = totals(state)
    prior, _ = totals(previous)
    if not current_seen:
        return None
    delta = {field: max(0.0, current[field] - prior[field]) for field in fields}
    return AuditUsage(
        input_tokens=int(delta["input_tokens"]) if "input_tokens" in current_seen else None,
        output_tokens=int(delta["output_tokens"]) if "output_tokens" in current_seen else None,
        total_tokens=int(delta["total_tokens"]) if "total_tokens" in current_seen else None,
        cost_usd=float(delta["cost_usd"]) if "cost_usd" in current_seen else None,
    )


_SAFETY_TOOL_ERRORS = frozenset(
    {
        "approval_invalidated",
        "capability_denied",
        "idempotency_key_required",
        "policy_denied",
        "tool_not_allowed",
    }
)


def tool_result_is_safety_violation(result: ToolResult) -> bool:
    return not result.ok and result.error in _SAFETY_TOOL_ERRORS


_SECRET_KEY = re.compile(
    r"(?:^|[_-])(?:password|passwd|secret|api[_-]?key|authorization|cookie|"
    r"credential|private[_-]?key|token|access[_-]?token|refresh[_-]?token|session[_-]?token)"
    r"(?:$|[_-])",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"(?:"
    r"\bBearer\s+[A-Za-z0-9._~+/-]{6,}=*|"
    r"\bsk-[A-Za-z0-9_-]{6,}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"\b(?:password|passwd|secret|api[_-]?key|authorization|cookie|"
    r"access[_-]?token|refresh[_-]?token)\s*[:=]\s*[^\s,;]+"
    r")",
    re.IGNORECASE,
)


def _assert_secret_free(value: Any, *, path: str = "metadata") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            if _SECRET_KEY.search(key_text):
                raise AuditSecretError(f"secret-like audit key rejected at {path}.{key_text}")
            _assert_secret_free(nested, path=f"{path}.{key_text}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _assert_secret_free(nested, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and _SECRET_VALUE.search(value):
        raise AuditSecretError(f"secret-like audit value rejected at {path}")


def _metadata_json(value: Mapping[str, Any]) -> str:
    """Encode only real JSON values so ``default=str`` cannot leak secrets."""

    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("audit metadata must contain finite JSON values") from exc


def sanitize_failure(value: BaseException | str, *, max_chars: int = 500) -> str:
    """Return a bounded error summary with credential-like material removed."""

    if isinstance(value, BaseException):
        prefix = f"{type(value).__name__}: "
        text = str(value)
    else:
        prefix = ""
        text = str(value)
    text = _SECRET_VALUE.sub("[REDACTED]", text)
    # Also redact assignments whose value is empty/short and therefore does not
    # meet the stronger secret-value detector above.
    text = re.sub(
        r"(?i)\b(password|passwd|secret|api[_-]?key|authorization|cookie|"
        r"access[_-]?token|refresh[_-]?token)\s*[:=]\s*\S+",
        r"\1=[REDACTED]",
        text,
    )
    return (prefix + text).strip()[: max(1, max_chars)]


class AuditEvent(StrictModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    tenant_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    action: str = Field(min_length=1, max_length=120)
    outcome: str = Field(min_length=1, max_length=80)
    duration_ms: float | None = Field(default=None, ge=0)
    tool: str | None = Field(default=None, min_length=1, max_length=160)
    provider: str | None = Field(default=None, min_length=1, max_length=160)
    usage: AuditUsage | None = None
    safety_violation: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def contains_no_credentials(self) -> AuditEvent:
        _assert_secret_free(
            {
                "action": self.action,
                "outcome": self.outcome,
                "tool": self.tool,
                "provider": self.provider,
                "metadata": self.metadata,
            },
            path="event",
        )
        encoded = _metadata_json(self.metadata)
        if len(encoded) > 32_000:
            raise ValueError("audit metadata exceeds 32000 characters")
        return self


class MetricsSnapshot(StrictModel):
    tenant_id: str
    run_id: str | None = None
    run_count: int = Field(ge=0)
    run_success_count: int = Field(ge=0)
    run_success_rate: float | None = Field(default=None, ge=0, le=1)
    tool_count: int = Field(ge=0)
    tool_success_count: int = Field(ge=0)
    tool_success_rate: float | None = Field(default=None, ge=0, le=1)
    duration_p50_ms: float | None = Field(default=None, ge=0)
    duration_p95_ms: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    safety_violation_count: int = Field(ge=0)


def _as_utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _from_epoch(value: float | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=UTC)


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * probability
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


class OperationsStore:
    """SQLite-backed queue, append-only audit log, and metrics view."""

    def __init__(
        self,
        path: str | Path,
        *,
        base_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 300.0,
    ) -> None:
        if base_backoff_seconds < 0:
            raise ValueError("base_backoff_seconds must be non-negative")
        if max_backoff_seconds < base_backoff_seconds:
            raise ValueError("max_backoff_seconds must be at least the base backoff")
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.base_backoff_seconds = float(base_backoff_seconds)
        self.max_backoff_seconds = float(max_backoff_seconds)
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialise(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS operation_jobs (
                    run_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('queued', 'leased', 'completed', 'dead_letter')
                    ),
                    priority INTEGER NOT NULL DEFAULT 0,
                    attempt INTEGER NOT NULL DEFAULT 0 CHECK (attempt >= 0),
                    max_attempts INTEGER NOT NULL CHECK (max_attempts >= 1),
                    available_at REAL NOT NULL,
                    owner TEXT,
                    lease_token TEXT,
                    lease_version INTEGER NOT NULL DEFAULT 0 CHECK (lease_version >= 0),
                    lease_expires_at REAL,
                    result_status TEXT,
                    last_error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    CHECK (
                        (status = 'leased' AND owner IS NOT NULL AND lease_token IS NOT NULL
                         AND lease_expires_at IS NOT NULL)
                        OR
                        (status <> 'leased' AND owner IS NULL AND lease_token IS NULL
                         AND lease_expires_at IS NULL)
                    )
                );
                CREATE INDEX IF NOT EXISTS operation_jobs_claim_idx
                    ON operation_jobs(status, available_at, lease_expires_at, priority, created_at);

                CREATE TABLE IF NOT EXISTS audit_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    tenant_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    duration_ms REAL,
                    tool TEXT,
                    provider TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    cost_usd REAL,
                    safety_violation INTEGER NOT NULL DEFAULT 0 CHECK (safety_violation IN (0, 1)),
                    metadata_json TEXT NOT NULL,
                    occurred_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS audit_tenant_run_idx
                    ON audit_events(tenant_id, run_id, sequence);
                CREATE INDEX IF NOT EXISTS audit_tenant_time_idx
                    ON audit_events(tenant_id, occurred_at, sequence);

                CREATE TRIGGER IF NOT EXISTS audit_events_no_update
                BEFORE UPDATE ON audit_events
                BEGIN
                    SELECT RAISE(ABORT, 'audit events are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS audit_events_no_delete
                BEFORE DELETE ON audit_events
                BEGIN
                    SELECT RAISE(ABORT, 'audit events are append-only');
                END;
                """
            )

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
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO operation_jobs(
                        run_id, tenant_id, status, priority, attempt, max_attempts,
                        available_at, lease_version, created_at, updated_at
                    ) VALUES (?, ?, 'queued', ?, 0, ?, ?, 0, ?, ?)
                    """,
                    (
                        run_id,
                        tenant_id,
                        int(priority),
                        int(max_attempts),
                        available.timestamp(),
                        current.timestamp(),
                        current.timestamp(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateJobError(f"run already enqueued: {run_id}") from exc
        return self.get_job(run_id, tenant_id=tenant_id)

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
        current = _as_utc(now)
        timestamp = current.timestamp()
        token = secrets.token_urlsafe(32)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            # Even an expired final-attempt lease gets one reconciliation
            # claim. The worker must inspect the checkpoint: it can complete a
            # state that became durable just before the old process crashed,
            # or dead-letter a non-terminal state without calling the provider.
            tenant_clause = " AND tenant_id = ?" if tenant_id is not None else ""
            parameters: list[Any] = [timestamp, timestamp]
            if tenant_id is not None:
                parameters.append(tenant_id)
            row = connection.execute(
                f"""
                SELECT run_id
                  FROM operation_jobs
                 WHERE (
                        (status = 'queued' AND available_at <= ?)
                     OR (status = 'leased' AND lease_expires_at <= ?)
                 )
                 {tenant_clause}
                 ORDER BY priority DESC, available_at ASC, created_at ASC, run_id ASC
                 LIMIT 1
                """,
                parameters,
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            run_id = str(row["run_id"])
            updated = connection.execute(
                """
                UPDATE operation_jobs
                   SET status = 'leased', owner = ?, lease_token = ?,
                       lease_version = lease_version + 1,
                       lease_expires_at = ?, attempt = attempt + 1,
                       last_error = CASE WHEN status = 'leased'
                                         THEN 'lease_expired' ELSE last_error END,
                       updated_at = ?
                 WHERE run_id = ?
                   AND (
                        (status = 'queued' AND available_at <= ?)
                     OR (status = 'leased' AND lease_expires_at <= ?)
                   )
                """,
                (
                    owner,
                    token,
                    timestamp + float(lease_seconds),
                    timestamp,
                    run_id,
                    timestamp,
                    timestamp,
                ),
            )
            if updated.rowcount != 1:
                raise OperationsError("atomic claim lost inside an immediate transaction")
            claimed = connection.execute(
                "SELECT * FROM operation_jobs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            connection.commit()
            assert claimed is not None
            return self._job(claimed)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

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
        timestamp = current.timestamp()
        return self._lease_cas(
            job,
            """
            UPDATE operation_jobs
               SET lease_expires_at = ?, lease_version = lease_version + 1,
                   updated_at = ?
             WHERE run_id = ? AND status = 'leased' AND owner = ?
               AND lease_token = ? AND lease_version = ? AND lease_expires_at > ?
            """,
            (
                timestamp + float(lease_seconds),
                timestamp,
                job.run_id,
                job.owner,
                job.lease_token,
                job.lease_version,
                timestamp,
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
        timestamp = current.timestamp()
        return self._lease_cas(
            job,
            """
            UPDATE operation_jobs
               SET status = 'completed', result_status = ?, owner = NULL,
                   lease_token = NULL, lease_expires_at = NULL,
                   lease_version = lease_version + 1, updated_at = ?
             WHERE run_id = ? AND status = 'leased' AND owner = ?
               AND lease_token = ? AND lease_version = ? AND lease_expires_at > ?
            """,
            (
                result_status,
                timestamp,
                job.run_id,
                job.owner,
                job.lease_token,
                job.lease_version,
                timestamp,
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
        timestamp = current.timestamp()
        safe_error = sanitize_failure(error)
        terminal = job.attempt >= job.max_attempts
        if terminal:
            status = JobStatus.DEAD_LETTER.value
            available_at = timestamp
            result_status = "failed"
        else:
            status = JobStatus.QUEUED.value
            delay = min(
                self.max_backoff_seconds,
                self.base_backoff_seconds * (2 ** max(0, job.attempt - 1)),
            )
            available_at = timestamp + delay
            result_status = "retry_scheduled"
        return self._lease_cas(
            job,
            """
            UPDATE operation_jobs
               SET status = ?, result_status = ?, last_error = ?, available_at = ?,
                   owner = NULL, lease_token = NULL, lease_expires_at = NULL,
                   lease_version = lease_version + 1, updated_at = ?
             WHERE run_id = ? AND status = 'leased' AND owner = ?
               AND lease_token = ? AND lease_version = ? AND lease_expires_at > ?
            """,
            (
                status,
                result_status,
                safe_error,
                available_at,
                timestamp,
                job.run_id,
                job.owner,
                job.lease_token,
                job.lease_version,
                timestamp,
            ),
        )

    def _lease_cas(
        self,
        job: OperationJob,
        query: str,
        parameters: Sequence[Any],
    ) -> OperationJob:
        if job.status != JobStatus.LEASED or not job.owner or not job.lease_token:
            raise LeaseLostError("job does not carry an active lease")
        with self._connect() as connection:
            cursor = connection.execute(query, tuple(parameters))
            if cursor.rowcount != 1:
                raise LeaseLostError("lease CAS rejected: stale, expired, or wrong owner")
            row = connection.execute(
                "SELECT * FROM operation_jobs WHERE run_id = ?",
                (job.run_id,),
            ).fetchone()
        assert row is not None
        return self._job(row)

    def get_job(self, run_id: str, *, tenant_id: str | None = None) -> OperationJob:
        query = "SELECT * FROM operation_jobs WHERE run_id = ?"
        parameters: list[Any] = [run_id]
        if tenant_id is not None:
            query += " AND tenant_id = ?"
            parameters.append(tenant_id)
        with self._connect() as connection:
            row = connection.execute(query, parameters).fetchone()
        if row is None:
            raise JobNotFoundError(f"job not found: {run_id}")
        return self._job(row)

    def append_audit(self, event: AuditEvent | Mapping[str, Any]) -> AuditEvent:
        if isinstance(event, AuditEvent):
            validated = event
        else:
            # Validate before Pydantic so callers receive the specific security
            # exception rather than a generic wrapped model ValidationError.
            _assert_secret_free(event, path="event")
            validated = AuditEvent.model_validate(event)
        usage = validated.usage
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO audit_events(
                    event_id, tenant_id, run_id, action, outcome, duration_ms,
                    tool, provider, input_tokens, output_tokens, total_tokens,
                    cost_usd, safety_violation, metadata_json, occurred_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    validated.event_id,
                    validated.tenant_id,
                    validated.run_id,
                    validated.action,
                    validated.outcome,
                    validated.duration_ms,
                    validated.tool,
                    validated.provider,
                    usage.input_tokens if usage else None,
                    usage.output_tokens if usage else None,
                    usage.total_tokens if usage else None,
                    usage.cost_usd if usage else None,
                    int(validated.safety_violation),
                    _metadata_json(validated.metadata),
                    _as_utc(validated.occurred_at).timestamp(),
                ),
            )
        return validated.model_copy(deep=True)

    def append_audit_once(self, event: AuditEvent | Mapping[str, Any]) -> AuditEvent:
        """Append an idempotent host event, returning an identical prior identity.

        Durable workers may recover after writing audit data but before their
        lease completion CAS.  A deterministic ``event_id`` lets that retry
        converge without either duplicating metrics or weakening append-only
        storage.  A collision with a different event identity fails closed.
        """

        if isinstance(event, AuditEvent):
            validated = event
        else:
            _assert_secret_free(event, path="event")
            validated = AuditEvent.model_validate(event)
        try:
            return self.append_audit(validated)
        except sqlite3.IntegrityError:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM audit_events WHERE event_id = ?",
                    (validated.event_id,),
                ).fetchone()
            if row is None:
                raise
            existing = self._audit(row)
            if (
                existing.tenant_id,
                existing.run_id,
                existing.action,
                existing.outcome,
                existing.tool,
                existing.provider,
                existing.usage,
                existing.safety_violation,
                existing.metadata,
            ) != (
                validated.tenant_id,
                validated.run_id,
                validated.action,
                validated.outcome,
                validated.tool,
                validated.provider,
                validated.usage,
                validated.safety_violation,
                validated.metadata,
            ):
                raise OperationsError("audit event_id collision") from None
            return existing

    # Friendly aliases for callers that treat the audit log as an event stream.
    append_event = append_audit

    def list_audit(
        self,
        tenant_id: str,
        *,
        run_id: str | None = None,
        limit: int = 200,
        latest: bool = False,
    ) -> list[AuditEvent]:
        """List a bounded append-order prefix or newest append-order window."""

        if not tenant_id:
            raise ValueError("tenant_id is required")
        if limit < 1 or limit > 1_000:
            raise ValueError("limit must be between 1 and 1000")
        query = "SELECT * FROM audit_events WHERE tenant_id = ?"
        parameters: list[Any] = [tenant_id]
        if run_id is not None:
            query += " AND run_id = ?"
            parameters.append(run_id)
        if latest:
            # Select the newest bounded window, then return that window in
            # append/sequence order so API consumers can render it directly.
            query = (
                f"SELECT * FROM ({query} ORDER BY sequence DESC LIMIT ?) "
                "ORDER BY sequence ASC"
            )
        else:
            query += " ORDER BY sequence ASC LIMIT ?"
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._audit(row) for row in rows]

    list_events = list_audit

    def metrics(
        self,
        tenant_id: str,
        *,
        run_id: str | None = None,
    ) -> MetricsSnapshot:
        if not tenant_id:
            raise ValueError("tenant_id is required")
        query = "SELECT * FROM audit_events WHERE tenant_id = ?"
        parameters: list[Any] = [tenant_id]
        if run_id is not None:
            query += " AND run_id = ?"
            parameters.append(run_id)
        query += " ORDER BY occurred_at ASC, sequence ASC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()

        success = {"ok", "success", "succeeded", "completed"}
        latest_runs: dict[str, sqlite3.Row] = {}
        tool_rows: list[sqlite3.Row] = []
        durations: list[float] = []
        for row in rows:
            if str(row["action"]) == "run" or str(row["action"]).startswith("run."):
                latest_runs[str(row["run_id"])] = row
            if row["tool"] is not None:
                tool_rows.append(row)
            if row["duration_ms"] is not None:
                durations.append(float(row["duration_ms"]))
        run_successes = sum(
            1 for row in latest_runs.values() if str(row["outcome"]).lower() in success
        )
        tool_successes = sum(
            1 for row in tool_rows if str(row["outcome"]).lower() in success
        )

        def optional_sum(column: str, *, integral: bool) -> int | float | None:
            values = [row[column] for row in rows if row[column] is not None]
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
            safety_violation_count=sum(int(row["safety_violation"]) for row in rows),
        )

    aggregate_metrics = metrics

    @staticmethod
    def _job(row: sqlite3.Row) -> OperationJob:
        return OperationJob(
            run_id=row["run_id"],
            tenant_id=row["tenant_id"],
            status=JobStatus(row["status"]),
            priority=row["priority"],
            attempt=row["attempt"],
            max_attempts=row["max_attempts"],
            available_at=_from_epoch(row["available_at"]),
            owner=row["owner"],
            lease_token=row["lease_token"],
            lease_version=row["lease_version"],
            lease_expires_at=_from_epoch(row["lease_expires_at"]),
            result_status=row["result_status"],
            last_error=row["last_error"],
            created_at=_from_epoch(row["created_at"]),
            updated_at=_from_epoch(row["updated_at"]),
        )

    @staticmethod
    def _audit(row: sqlite3.Row) -> AuditEvent:
        usage = None
        if any(
            row[column] is not None
            for column in ("input_tokens", "output_tokens", "total_tokens", "cost_usd")
        ):
            usage = AuditUsage(
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
                total_tokens=row["total_tokens"],
                cost_usd=row["cost_usd"],
            )
        return AuditEvent(
            event_id=row["event_id"],
            tenant_id=row["tenant_id"],
            run_id=row["run_id"],
            action=row["action"],
            outcome=row["outcome"],
            duration_ms=row["duration_ms"],
            tool=row["tool"],
            provider=row["provider"],
            usage=usage,
            safety_violation=bool(row["safety_violation"]),
            metadata=json.loads(row["metadata_json"]),
            occurred_at=_from_epoch(row["occurred_at"]),
        )


__all__ = [
    "AuditEvent",
    "AuditSecretError",
    "AuditUsage",
    "DuplicateJobError",
    "JobNotFoundError",
    "JobRecord",
    "JobStatus",
    "LeaseLostError",
    "MetricsSnapshot",
    "OperationJob",
    "OperationsStore",
    "audit_usage_from_state",
    "sanitize_failure",
    "tool_result_is_safety_violation",
]

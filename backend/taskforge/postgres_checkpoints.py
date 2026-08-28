"""PostgreSQL checkpoint backend for TaskForge runs.

The legacy checkpoint interface did not carry tenant identity on every call.
This backend therefore accepts an explicit tenant on each operation and uses a
constructor default only for single-tenant deployments.  PostgreSQL RLS is
still the final authority; the tenant value is never interpolated into SQL.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from .checkpoints import CheckpointCorruptError, CheckpointNotFoundError
from .domain import AgentProfile, RunState, Task
from .postgres_runtime import PostgresRuntime


class PostgresCheckpointStore:
    """Pooled PostgreSQL implementation of the durable run checkpoint port."""

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

    def save_task(self, task: Task, *, tenant_id: str | None = None) -> None:
        current_tenant = self._tenant(task.tenant_id, tenant_id)
        with self.runtime.transaction(current_tenant) as (_, cursor):
            cursor.execute(
                """
                INSERT INTO core.tasks(tenant_id, task_id, task_json, created_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (tenant_id, task_id) DO UPDATE SET
                    task_json = EXCLUDED.task_json
                """,
                (
                    current_tenant,
                    task.id,
                    _json(task.model_dump(mode="json")),
                    _as_utc(task.created_at),
                ),
            )

    def save_profile(
        self,
        profile: AgentProfile,
        *,
        tenant_id: str | None = None,
    ) -> None:
        current_tenant = tenant_id or self.tenant_id
        if not current_tenant.strip():
            raise ValueError("tenant_id is required")
        with self.runtime.transaction(current_tenant) as (_, cursor):
            cursor.execute(
                """
                INSERT INTO core.profiles(
                    tenant_id, profile_id, profile_json, updated_at
                ) VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (tenant_id, profile_id) DO UPDATE SET
                    profile_json = EXCLUDED.profile_json,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    current_tenant,
                    profile.id,
                    _json(profile.model_dump(mode="json")),
                ),
            )

    def save(self, state: RunState, *, tenant_id: str | None = None) -> int:
        current_tenant = self._tenant_required(tenant_id)
        with self.runtime.transaction(current_tenant) as (_, cursor):
            cursor.execute(
                """
                INSERT INTO core.runs AS existing(
                    tenant_id, run_id, task_id, profile_id, state_json,
                    version, updated_at
                ) VALUES (%s, %s, %s, %s, %s, 1, %s)
                ON CONFLICT (tenant_id, run_id) DO UPDATE SET
                    state_json = EXCLUDED.state_json,
                    version = existing.version + 1,
                    updated_at = EXCLUDED.updated_at
                WHERE existing.task_id = EXCLUDED.task_id
                  AND existing.profile_id = EXCLUDED.profile_id
                RETURNING version
                """,
                (
                    current_tenant,
                    state.run_id,
                    state.task_id,
                    state.agent_profile_id,
                    _json(state.model_dump(mode="json")),
                    _as_utc(state.updated_at),
                ),
            )
            row = cursor.fetchone()
            if row is None:
                cursor.execute(
                    """
                    SELECT task_id, profile_id, version
                      FROM core.runs
                     WHERE tenant_id = %s AND run_id = %s
                    """,
                    (current_tenant, state.run_id),
                )
                existing = cursor.fetchone()
                if existing is None:
                    raise RuntimeError("checkpoint upsert did not create a row")
                if _value(existing, "task_id", 0) != state.task_id or _value(
                    existing, "profile_id", 1
                ) != state.agent_profile_id:
                    raise ValueError(
                        "run_id already belongs to another task or profile"
                    )
                return int(_value(existing, "version", 2))
            return int(_value(row, "version", 0))

    def load(self, run_id: str, *, tenant_id: str | None = None) -> RunState:
        row = self._one(
            "SELECT state_json FROM core.runs "
            "WHERE tenant_id = %s AND run_id = %s",
            run_id,
            "run",
            tenant_id=tenant_id,
        )
        return _validate(RunState, _value(row, "state_json", 0), f"run {run_id}")

    def load_task(self, task_id: str, *, tenant_id: str | None = None) -> Task:
        row = self._one(
            "SELECT task_json FROM core.tasks "
            "WHERE tenant_id = %s AND task_id = %s",
            task_id,
            "task",
            tenant_id=tenant_id,
        )
        return _validate(Task, _value(row, "task_json", 0), f"task {task_id}")

    def load_profile(
        self,
        profile_id: str,
        *,
        tenant_id: str | None = None,
    ) -> AgentProfile:
        row = self._one(
            "SELECT profile_json FROM core.profiles "
            "WHERE tenant_id = %s AND profile_id = %s",
            profile_id,
            "profile",
            tenant_id=tenant_id,
        )
        return _validate(
            AgentProfile,
            _value(row, "profile_json", 0),
            f"profile {profile_id}",
        )

    def version(self, run_id: str, *, tenant_id: str | None = None) -> int:
        row = self._one(
            "SELECT version FROM core.runs "
            "WHERE tenant_id = %s AND run_id = %s",
            run_id,
            "run",
            tenant_id=tenant_id,
        )
        return int(_value(row, "version", 0))

    def _one(
        self,
        query: str,
        identifier: str,
        kind: str,
        *,
        tenant_id: str | None,
    ) -> Any:
        current_tenant = self._tenant_required(tenant_id)
        with self.runtime.transaction(current_tenant) as (_, cursor):
            cursor.execute(query, (current_tenant, identifier))
            row = cursor.fetchone()
        if row is None:
            raise CheckpointNotFoundError(f"{kind} not found: {identifier}")
        return row

    def _tenant(self, task_tenant: str, tenant_id: str | None) -> str:
        current_tenant = tenant_id or task_tenant or self.tenant_id
        if not current_tenant.strip() or current_tenant != task_tenant:
            raise PermissionError("checkpoint tenant does not match the task")
        return current_tenant

    def _tenant_required(self, tenant_id: str | None) -> str:
        current_tenant = tenant_id or self.tenant_id
        if not current_tenant.strip():
            raise ValueError("tenant_id is required")
        return current_tenant


def _json(value: object) -> object:
    try:
        from psycopg.types.json import Json
    except ImportError:  # pragma: no cover - real runtime imports psycopg first
        return value
    return Json(value)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _value(row: Any, key: str, index: int) -> Any:
    if isinstance(row, dict):
        return row[key]
    return row[index]


def _validate(model: type[RunState | Task | AgentProfile], payload: Any, label: str):
    try:
        if isinstance(payload, str):
            return model.model_validate_json(payload)
        return model.model_validate(payload)
    except (TypeError, ValueError, ValidationError) as exc:
        raise CheckpointCorruptError(f"invalid persisted {label}") from exc


__all__ = ["PostgresCheckpointStore"]

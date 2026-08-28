from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import UTC, datetime, timedelta
from typing import Any

from taskforge.operations import JobStatus
from taskforge.postgres_operations import PostgresOperationsStore

NOW = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)


def _job_row(*, status: str = "queued", lease: bool = False) -> tuple[Any, ...]:
    return (
        "run-1",
        "tenant-a",
        status,
        2,
        1 if lease else 0,
        3,
        NOW,
        "worker-1" if lease else None,
        "opaque-token" if lease else None,
        1 if lease else 0,
        NOW + timedelta(seconds=30) if lease else None,
        None,
        None,
        NOW,
        NOW,
    )


class FakeCursor(AbstractContextManager["FakeCursor"]):
    def __init__(self, connection: "FakeConnection") -> None:
        self.connection = connection
        self.rows: list[tuple[Any, ...]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def execute(self, sql: str, parameters: Any = ()) -> None:
        self.connection.calls.append((sql, parameters))
        normalised = " ".join(sql.casefold().split())
        if "insert into operations.operation_jobs" in normalised:
            self.rows = [_job_row()]
        elif "update operations.operation_jobs" in normalised:
            self.rows = [_job_row(status="leased", lease=True)]
        elif "from operations.operation_jobs" in normalised:
            self.rows = [_job_row()]
        else:
            self.rows = []

    def fetchone(self) -> tuple[Any, ...] | None:
        return self.rows[0] if self.rows else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self.rows)


class FakeConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def transaction(self) -> "FakeTransaction":
        return FakeTransaction()

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)


class FakeTransaction(AbstractContextManager["FakeTransaction"]):
    def __enter__(self) -> "FakeTransaction":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None


class FakeRuntime:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    def transaction(self, tenant_id: str):
        assert tenant_id == "tenant-a"
        self.connection.calls.append(
            ("SELECT set_config('taskforge.tenant_id', %s, true)", (tenant_id,))
        )
        return _Transaction(self.connection)

    def close(self) -> None:
        return None


class _Transaction(AbstractContextManager[tuple[FakeConnection, FakeCursor]]):
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection
        self.cursor = connection.cursor()

    def __enter__(self) -> tuple[FakeConnection, FakeCursor]:
        self.cursor.__enter__()
        return self.connection, self.cursor

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.cursor.__exit__(exc_type, exc, traceback)


def test_postgres_queue_uses_skip_locked_and_lease_cas() -> None:
    connection = FakeConnection()
    store = PostgresOperationsStore(
        "postgresql://unused",
        tenant_id="tenant-a",
        runtime=FakeRuntime(connection),  # type: ignore[arg-type]
    )
    queued = store.enqueue("run-1", "tenant-a", now=NOW)
    assert queued.status is JobStatus.QUEUED
    claimed = store.claim("worker-1", tenant_id="tenant-a", now=NOW)
    assert claimed is not None
    assert claimed.status is JobStatus.LEASED
    heartbeat = store.heartbeat(claimed, now=NOW)
    assert heartbeat.status is JobStatus.LEASED

    sql = "\n".join(call[0] for call in connection.calls)
    assert "FOR UPDATE SKIP LOCKED" in sql
    assert "lease_token = %s" in sql
    assert "lease_version = %s" in sql
    assert "tenant-a" not in sql


def test_postgres_audit_input_is_validated_before_database_io() -> None:
    connection = FakeConnection()
    store = PostgresOperationsStore(
        "postgresql://unused",
        tenant_id="tenant-a",
        runtime=FakeRuntime(connection),  # type: ignore[arg-type]
    )
    try:
        store.append_audit(
            {
                "tenant_id": "tenant-a",
                "run_id": "run-1",
                "action": "tool.execute",
                "outcome": "ok",
                "metadata": {"api_key": "not-stored"},
            }
        )
    except ValueError:
        pass
    assert not any("INSERT INTO operations.audit_events" in call[0] for call in connection.calls)

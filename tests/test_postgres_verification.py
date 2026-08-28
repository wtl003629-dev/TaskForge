from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import UTC, datetime
from typing import Any

import pytest

from taskforge.postgres_verification import PostgresVerificationStore
from taskforge.verification import VerificationRecord, VerificationSignatureError


class FakeTransaction(AbstractContextManager["FakeTransaction"]):
    def __init__(self, connection: "FakeConnection") -> None:
        self.connection = connection

    def __enter__(self) -> "FakeTransaction":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.connection.commits += 1
        else:
            self.connection.rollbacks += 1


class FakeCursor(AbstractContextManager["FakeCursor"]):
    def __init__(self, connection: "FakeConnection") -> None:
        self.connection = connection
        self.rows: list[dict[str, object]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def execute(self, sql: str, parameters: Any) -> None:
        self.connection.calls.append((sql, parameters))
        if sql.lstrip().casefold().startswith("select record_json"):
            self.rows = list(self.connection.rows)

    def fetchall(self) -> list[dict[str, object]]:
        return self.rows


class FakeConnection:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []
        self.calls: list[tuple[str, Any]] = []
        self.commits = 0
        self.rollbacks = 0

    def transaction(self) -> FakeTransaction:
        return FakeTransaction(self)

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)


class FakeRuntime:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    def transaction(self, tenant_id: str):
        assert tenant_id == "tenant-a"
        self.connection.calls.append(
            ("SELECT set_config('taskforge.tenant_id', %s, true)", ("taskforge.tenant_id", tenant_id))
        )
        return _transaction(self.connection)

    def close(self) -> None:
        return None


class _transaction(AbstractContextManager[tuple[FakeConnection, FakeCursor]]):
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection
        self.cursor = connection.cursor()
        self.transaction = connection.transaction()

    def __enter__(self) -> tuple[FakeConnection, FakeCursor]:
        self.transaction.__enter__()
        self.cursor.__enter__()
        return self.connection, self.cursor

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.cursor.__exit__(exc_type, exc, traceback)
        self.transaction.__exit__(exc_type, exc, traceback)


def test_postgres_verification_store_validates_signatures_and_tenant_scope() -> None:
    connection = FakeConnection()
    store = PostgresVerificationStore(
        "postgresql://unused",
        tenant_id="tenant-a",
        runtime=FakeRuntime(connection),  # type: ignore[arg-type]
    )
    record = VerificationRecord.signed(
        kind="live_smoke",
        provider="openai",
        model="model-1",
        produced_at=datetime(2026, 8, 27, tzinfo=UTC),
    )
    with pytest.raises(VerificationSignatureError):
        store.save(record.model_copy(update={"signature": "sig:" + "0" * 64}))

    # The valid path exercises the transaction-local tenant contract and SQL
    # parameterisation without claiming a live PostgreSQL server.
    saved = store.save(record)
    assert saved == record
    assert connection.commits == 1
    assert "set_config" in connection.calls[0][0]
    assert connection.calls[0][1] == ("taskforge.tenant_id", "tenant-a")
    assert "tenant-a" not in connection.calls[1][0]


def test_postgres_verification_store_reads_and_rejects_tampering() -> None:
    connection = FakeConnection()
    store = PostgresVerificationStore(
        "postgresql://unused",
        tenant_id="tenant-a",
        runtime=FakeRuntime(connection),  # type: ignore[arg-type]
    )
    record = VerificationRecord.signed(
        kind="business_e2e",
        provider="openai",
        evidence={"passed": True},
    )
    connection.rows = [{"record_json": record.model_dump(mode="json")}]
    assert store.latest("business_e2e") == record
    connection.rows = [{
        "record_json": record.model_copy(update={"evidence": {"passed": False}}).model_dump(mode="json")
    }]
    with pytest.raises(VerificationSignatureError, match="tampered"):
        store.all()

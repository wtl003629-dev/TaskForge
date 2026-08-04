from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

import taskforge.postgres_context as postgres_context
from taskforge.knowledge import KnowledgeChunk
from taskforge.memory import MemoryItem, MemoryProvenance, MemoryScope
from taskforge.postgres_context import (
    MAX_JSON_BYTES,
    PostgresContextAccess,
    PostgresContextError,
    PostgresContextRepository,
    PostgresDependencyError,
)


NOW = datetime(2026, 8, 4, 10, 0, tzinfo=timezone.utc)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeTransaction(AbstractContextManager["FakeTransaction"]):
    def __init__(self, connection: "FakeConnection") -> None:
        self.connection = connection

    def __enter__(self) -> "FakeTransaction":
        self.connection.transaction_entries += 1
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
        self.connection.cursor_closes += 1

    def execute(self, sql: str, parameters: dict[str, object]) -> None:
        assert isinstance(sql, str)
        assert isinstance(parameters, dict)
        self.connection.calls.append((sql, dict(parameters)))
        normalised = " ".join(sql.casefold().split())
        if self.connection.fail_on and self.connection.fail_on in normalised:
            raise RuntimeError("injected database failure")
        if "from taskforge.knowledge_chunks as kc" in normalised:
            self.rows = list(self.connection.knowledge_rows)
        elif "from taskforge.memory_items as mi" in normalised:
            self.rows = list(self.connection.memory_rows)
        else:
            self.rows = []

    def fetchall(self) -> list[dict[str, object]]:
        return list(self.rows)


class FakeConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.knowledge_rows: list[dict[str, object]] = []
        self.memory_rows: list[dict[str, object]] = []
        self.fail_on: str | None = None
        self.transaction_entries = 0
        self.commits = 0
        self.rollbacks = 0
        self.cursor_closes = 0
        self.close_calls = 0

    def transaction(self) -> FakeTransaction:
        return FakeTransaction(self)

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def close(self) -> None:
        self.close_calls += 1


def access(**overrides: Any) -> PostgresContextAccess:
    values: dict[str, object] = {
        "tenant_id": "tenant-secret",
        "user_id": "alice",
        "conversation_id": "case-7",
        "org_id": "risk",
        "agent_id": "reviewer",
        "acl_principals": frozenset({"role:approver", "group:legal"}),
    }
    values.update(overrides)
    return PostgresContextAccess(**values)  # type: ignore[arg-type]


def knowledge_row(
    chunk_id: str,
    *,
    tenant_id: str = "tenant-secret",
    acl: object = None,
    source_uri: str = "kb://policy",
    metadata: object = None,
    valid_until: datetime | None = None,
) -> dict[str, object]:
    return {
        "tenant_id": tenant_id,
        "chunk_id": chunk_id,
        "text_content": f"evidence {chunk_id}",
        "source_uri": source_uri,
        "document_id": "policy",
        "version": "2",
        "version_order": 2,
        "acl_json": ["tenant"] if acl is None else acl,
        "valid_from": None,
        "valid_until": valid_until,
        "created_at": NOW,
        "metadata_json": {"knowledge_base_id": "support"} if metadata is None else metadata,
    }


def memory_row(
    memory_id: str,
    scope: str,
    scope_id: str,
    *,
    tenant_id: str = "tenant-secret",
    expires_at: datetime | None = None,
) -> dict[str, object]:
    return {
        "tenant_id": tenant_id,
        "memory_id": memory_id,
        "content": f"memory {memory_id}",
        "scope": scope,
        "scope_id": scope_id,
        "provenance_json": {
            "source_type": "test",
            "source_id": None,
            "source_uri": None,
            "actor_id": "host",
            "observed_at": NOW.isoformat(),
            "confidence": 1.0,
        },
        "importance": 0.7,
        "created_at": NOW,
        "updated_at": NOW,
        "expires_at": expires_at,
        "tags_json": ["review"],
        "metadata_json": {},
    }


def test_knowledge_upsert_is_transactional_idempotent_and_parameterised() -> None:
    connection = FakeConnection()
    repository = PostgresContextRepository(connection)
    principal = access()
    chunk = KnowledgeChunk(
        "chunk-1",
        principal.tenant_id,
        "sensitive evidence",
        "kb://policy",
        acl=frozenset({"user:alice"}),
        metadata={"page": 3},
    )

    assert repository.upsert_knowledge([chunk], principal) == 1
    assert repository.upsert_knowledge([chunk], principal) == 1

    assert connection.commits == 2
    assert connection.rollbacks == 0
    assert len(connection.calls) == 4
    for index in (0, 2):
        set_sql, set_parameters = connection.calls[index]
        assert "set_config('taskforge.tenant_id'" in set_sql
        assert set_parameters == {"tenant_id": "tenant-secret"}
        insert_sql, insert_parameters = connection.calls[index + 1]
        assert "on conflict (tenant_id, chunk_id) do update" in insert_sql.casefold()
        assert "excluded.version_order >= existing.version_order" in insert_sql.casefold()
        assert "%(tenant_id)s" in insert_sql
        assert "tenant-secret" not in insert_sql
        assert insert_parameters["tenant_id"] == "tenant-secret"
        assert insert_parameters["metadata_json"] == '{"page":3}'


def test_knowledge_candidate_sql_filters_identity_acl_and_catalog_before_limit() -> None:
    connection = FakeConnection()
    connection.knowledge_rows = [
        knowledge_row("tenant-wide"),
        knowledge_row("user", acl=["user:alice"]),
        knowledge_row("conversation", acl=["conversation:case-7"]),
        knowledge_row("role", acl=["role:approver"]),
        knowledge_row("wrong-acl", acl=["user:bob"]),
        knowledge_row("wrong-tenant", tenant_id="other"),
        knowledge_row("wrong-source", source_uri="kb://other"),
        knowledge_row("wrong-kb", metadata={"knowledge_base_id": "finance"}),
        knowledge_row("expired", valid_until=NOW),
        {"not": "a valid row"},
    ]
    repository = PostgresContextRepository(connection)

    candidates = repository.fetch_knowledge_candidates(
        access(),
        now=NOW,
        candidate_limit=20,
        source_uris=["kb://policy"],
        knowledge_base_ids=["support"],
    )

    # The fake intentionally ignores SQL; the second host check still drops
    # tenant, ACL, and expiry leaks. Catalog assertions below ensure source/KB
    # are not left to a later ranker either.
    assert {chunk.chunk_id for chunk in candidates}.issuperset(
        {"tenant-wide", "user", "conversation", "role"}
    )
    assert not {
        "wrong-acl",
        "wrong-tenant",
        "wrong-source",
        "wrong-kb",
        "expired",
    }.intersection(
        chunk.chunk_id for chunk in candidates
    )
    sql, parameters = connection.calls[-1]
    normalised = " ".join(sql.casefold().split())
    assert "kc.tenant_id = %(tenant_id)s" in normalised
    assert "'user:' || %(user_id)s" in normalised
    assert "'conversation:' || %(conversation_id)s" in normalised
    assert "kc.acl_json ?| %(additional_acl)s::text[]" in normalised
    assert normalised.index("kc.acl_json") < normalised.index("limit %(candidate_limit)s")
    assert parameters["source_uris"] == ["kb://policy"]
    assert parameters["knowledge_base_ids"] == ["support"]
    assert parameters["candidate_limit"] == 20
    assert "tenant-secret" not in sql


def test_memory_candidates_include_shared_and_bound_private_scopes_only() -> None:
    connection = FakeConnection()
    connection.memory_rows = [
        memory_row("shared", "tenant", "tenant-secret"),
        memory_row("user", "user", "alice"),
        memory_row("conversation", "task", "case-7"),
        memory_row("org", "org", "risk"),
        memory_row("agent", "agent", "reviewer"),
        memory_row("wrong-user", "user", "bob"),
        memory_row("wrong-conversation", "task", "case-8"),
        memory_row("wrong-tenant", "tenant", "other", tenant_id="other"),
        memory_row("expired", "user", "alice", expires_at=NOW),
    ]
    repository = PostgresContextRepository(connection)

    candidates = repository.fetch_memory_candidates(access(), now=NOW, candidate_limit=20)

    assert {item.memory_id for item in candidates} == {
        "shared",
        "user",
        "conversation",
        "org",
        "agent",
    }
    sql, parameters = connection.calls[-1]
    normalised = " ".join(sql.casefold().split())
    assert "mi.tenant_id = %(tenant_id)s" in normalised
    assert "mi.scope = 'user' and mi.scope_id = %(user_id)s" in normalised
    assert "mi.scope = 'task' and mi.scope_id = %(conversation_id)s" in normalised
    assert "mi.scope = 'tenant' and mi.scope_id = %(tenant_id)s" in normalised
    assert parameters["user_id"] == "alice"
    assert parameters["conversation_id"] == "case-7"

    connection.memory_rows = [
        memory_row("shared", "tenant", "tenant-secret"),
        memory_row("user", "user", "alice"),
    ]
    only_user = repository.fetch_memory_candidates(
        access(), scopes=[MemoryScope.USER], now=NOW
    )
    assert [item.memory_id for item in only_user] == ["user"]


def test_memory_writes_require_bound_scope_and_explicit_shared_authority() -> None:
    connection = FakeConnection()
    repository = PostgresContextRepository(connection)
    principal = access()
    shared = MemoryItem("shared", principal.tenant_id, "policy")
    wrong_user = MemoryItem(
        "wrong-user", principal.tenant_id, "preference", MemoryScope.USER, "bob"
    )
    private = MemoryItem(
        "private", principal.tenant_id, "preference", MemoryScope.USER, "alice"
    )
    conversation = MemoryItem(
        "conversation", principal.tenant_id, "case note", MemoryScope.TASK, "case-7"
    )

    with pytest.raises(PermissionError, match="shared memory"):
        repository.upsert_memories([shared], principal)
    with pytest.raises(PermissionError, match="scope"):
        repository.upsert_memories([wrong_user], principal)
    assert connection.calls == []

    assert repository.upsert_memories(
        [shared], principal, allow_shared_writes=True
    ) == 1
    assert repository.upsert_memories([private, conversation], principal) == 2
    memory_sql = [
        sql for sql, _ in connection.calls if "insert into taskforge.memory_items" in sql.casefold()
    ]
    assert memory_sql
    assert all("on conflict (tenant_id, memory_id) do update" in sql.casefold() for sql in memory_sql)
    assert all("excluded.updated_at >= existing.updated_at" in sql.casefold() for sql in memory_sql)


def test_batch_is_prevalidated_and_database_failures_roll_back() -> None:
    connection = FakeConnection()
    repository = PostgresContextRepository(connection)
    principal = access()
    good = KnowledgeChunk("good", principal.tenant_id, "good", "kb://good")
    bad = KnowledgeChunk(
        "bad", principal.tenant_id, "bad", "kb://bad", metadata={"nan": float("nan")}
    )

    with pytest.raises(ValueError, match="finite"):
        repository.upsert_knowledge([good, bad], principal)
    assert connection.calls == []
    assert connection.transaction_entries == 0

    connection.fail_on = "insert into taskforge.knowledge_chunks"
    with pytest.raises(RuntimeError, match="injected"):
        repository.upsert_knowledge([good], principal)
    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert connection.cursor_closes == 1


def test_json_depth_size_nan_and_unicode_are_rejected_before_io() -> None:
    connection = FakeConnection()
    repository = PostgresContextRepository(connection)
    principal = access()

    deep: object = "leaf"
    for _ in range(52):
        deep = {"next": deep}
    invalid_metadata = [
        {"deep": deep},
        {"large": "x" * (MAX_JSON_BYTES + 1)},
        {"nan": float("nan")},
        {"surrogate": "\ud800"},
        {"nul": "before\x00after"},
    ]
    for index, metadata in enumerate(invalid_metadata):
        chunk = KnowledgeChunk(
            f"bad-{index}", principal.tenant_id, "text", "kb://bad", metadata=metadata
        )
        with pytest.raises((TypeError, ValueError)):
            repository.upsert_knowledge([chunk], principal)
    assert connection.calls == []


def test_access_and_limits_fail_closed() -> None:
    with pytest.raises(ValueError, match="conversation_id"):
        access(conversation_id=" ")
    with pytest.raises(ValueError, match="Unicode"):
        access(user_id="bad\ud800")

    repository = PostgresContextRepository(FakeConnection())
    with pytest.raises(ValueError, match="candidate_limit"):
        repository.fetch_knowledge_candidates(access(), candidate_limit=10_001)
    with pytest.raises(ValueError, match="candidate_limit"):
        repository.fetch_memory_candidates(access(), candidate_limit=True)
    with pytest.raises(TypeError, match="latest_only"):
        repository.fetch_knowledge_candidates(access(), latest_only="false")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="iterable"):
        repository.fetch_knowledge_candidates(access(), source_uris="kb://one")
    with pytest.raises(TypeError, match="scopes"):
        repository.fetch_memory_candidates(access(), scopes="user")
    shared = MemoryItem("shared", "tenant-secret", "shared")
    with pytest.raises(TypeError, match="allow_shared_writes"):
        repository.upsert_memories(
            [shared], access(), allow_shared_writes="yes"  # type: ignore[arg-type]
        )


def test_external_connection_must_be_idle_to_bound_rls_tenant() -> None:
    class Info:
        transaction_status = 2

    connection = FakeConnection()
    connection.info = Info()  # type: ignore[attr-defined]
    repository = PostgresContextRepository(connection)

    with pytest.raises(PostgresContextError, match="idle"):
        repository.fetch_knowledge_candidates(access())
    assert connection.calls == []


def test_connection_ownership_closed_state_and_missing_dependency(monkeypatch) -> None:
    external = FakeConnection()
    repository = PostgresContextRepository(external)
    repository.close()
    repository.close()
    assert external.close_calls == 0
    with pytest.raises(PostgresContextError, match="closed"):
        repository.fetch_knowledge_candidates(access())

    owned = FakeConnection()
    owned_repository = PostgresContextRepository(owned, owns_connection=True)
    assert owned_repository.owns_connection is True
    owned_repository.close()
    owned_repository.close()
    assert owned.close_calls == 1

    def unavailable(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(postgres_context.importlib, "import_module", unavailable)
    with pytest.raises(PostgresDependencyError, match="postgres.*extra"):
        PostgresContextRepository.connect("postgresql://db.invalid/taskforge")


def test_postgres_0002_migration_forces_default_deny_rls_and_acl_indexes() -> None:
    sql = (PROJECT_ROOT / "migrations/0002_context_postgres.sql").read_text(
        encoding="utf-8"
    )
    normalised = " ".join(sql.casefold().split())
    for table in ("knowledge_chunks", "memory_items"):
        assert f"alter table taskforge.{table} enable row level security" in normalised
        assert f"alter table taskforge.{table} force row level security" in normalised
    assert normalised.count("current_setting('taskforge.tenant_id', true)") >= 4
    assert normalised.count("with check") >= 2
    assert "using gin (acl_json jsonb_ops)" in normalised
    assert "revoke all on schema taskforge from public" in normalised
    assert "bypassrls" in normalised


def test_fake_contract_is_not_a_live_postgres_claim() -> None:
    """Documents the exact verification boundary of this test module."""

    assert PostgresContextRepository.__doc__
    assert "Synchronous psycopg3-style" in PostgresContextRepository.__doc__
    assert "search engine" in postgres_context.__doc__

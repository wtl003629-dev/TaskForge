from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

import pytest

from taskforge.knowledge import AccessContext, KnowledgeChunk
from taskforge.memory import MemoryItem, MemoryProvenance, MemoryScope
from taskforge.persistent_context import SQLiteKnowledgeStore, SQLiteMemoryStore


NOW = datetime(2026, 8, 4, 10, 0, tzinfo=timezone.utc)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_records_survive_close_and_reopen(tmp_path) -> None:
    database = tmp_path / "context.sqlite3"
    principal = AccessContext("acme", user_id="u1")
    with SQLiteKnowledgeStore(database) as knowledge:
        knowledge.upsert(KnowledgeChunk("k1", "acme", "refund approval", "kb://support"))
    with SQLiteMemoryStore(database) as memory:
        memory.remember(
            MemoryItem(
                "m1",
                "acme",
                "refund answers are concise",
                MemoryScope.USER,
                "u1",
                MemoryProvenance(source_type="user_statement", source_id="turn-7"),
            )
        )

    with SQLiteKnowledgeStore(database) as knowledge, SQLiteMemoryStore(database) as memory:
        assert knowledge.get("k1", principal, now=NOW).text == "refund approval"
        restored = memory.get("m1", principal, now=NOW)
        assert restored.content == "refund answers are concise"
        assert restored.provenance.source_id == "turn-7"
        assert [hit.chunk.chunk_id for hit in knowledge.search("refund", principal, now=NOW)] == ["k1"]
        assert [hit.item.memory_id for hit in memory.recall("refund", principal, now=NOW)] == ["m1"]


def test_same_ids_are_tenant_local_and_acl_scope_remain_enforced(tmp_path) -> None:
    database = tmp_path / "context.sqlite3"
    with SQLiteKnowledgeStore(database) as knowledge:
        knowledge.upsert_many(
            [
                KnowledgeChunk("same", "acme", "acme refund", "kb://acme"),
                KnowledgeChunk("same", "other", "other refund", "kb://other"),
                KnowledgeChunk("private", "acme", "private refund", "kb://private", acl=frozenset({"user:u2"})),
            ]
        )
        assert knowledge.get("same", AccessContext("acme"), now=NOW).text == "acme refund"
        assert knowledge.get("same", AccessContext("other"), now=NOW).text == "other refund"
        assert knowledge.get("private", AccessContext("acme", user_id="u1"), now=NOW) is None

    with SQLiteMemoryStore(database) as memory:
        memory.remember_many(
            [
                MemoryItem("same", "acme", "acme preference"),
                MemoryItem("same", "other", "other preference"),
                MemoryItem("private", "acme", "private preference", MemoryScope.USER, "u2"),
            ]
        )
        assert memory.get("same", AccessContext("acme"), now=NOW).content == "acme preference"
        assert memory.get("same", AccessContext("other"), now=NOW).content == "other preference"
        assert memory.get("private", AccessContext("acme", user_id="u1"), now=NOW) is None


def test_search_filters_expiry_and_keeps_all_chunks_of_latest_valid_version(tmp_path) -> None:
    database = tmp_path / "context.sqlite3"
    with SQLiteKnowledgeStore(database) as knowledge:
        knowledge.upsert_many(
            [
                KnowledgeChunk("old", "acme", "refund old", "kb://refund", version="1", version_order=1),
                KnowledgeChunk("new-a", "acme", "refund approval", "kb://refund", version="2", version_order=2),
                KnowledgeChunk("new-b", "acme", "refund evidence", "kb://refund", version="2", version_order=2),
                KnowledgeChunk("expired", "acme", "refund expired", "kb://expired", valid_until=NOW),
                KnowledgeChunk("future", "acme", "refund future", "kb://future", valid_from=NOW + timedelta(seconds=1)),
            ]
        )

        hits = knowledge.search("refund", AccessContext("acme"), now=NOW, top_k=20)

    assert {hit.chunk.chunk_id for hit in hits} == {"new-a", "new-b"}


def test_memory_recall_filters_expired_before_scope_and_scoring(tmp_path) -> None:
    database = tmp_path / "context.sqlite3"
    with SQLiteMemoryStore(database) as memory:
        memory.remember_many(
            [
                MemoryItem("live", "acme", "concise report", MemoryScope.USER, "u1"),
                MemoryItem("expired", "acme", "concise report", MemoryScope.USER, "u1", expires_at=NOW),
                MemoryItem("wrong-user", "acme", "concise report", MemoryScope.USER, "u2"),
                MemoryItem("other", "other", "concise report"),
            ]
        )

        hits = memory.recall("concise report", AccessContext("acme", user_id="u1"), now=NOW, top_k=20)

    assert [hit.item.memory_id for hit in hits] == ["live"]


def test_knowledge_batch_upsert_rolls_back_every_row_on_serialisation_failure(tmp_path) -> None:
    database = tmp_path / "context.sqlite3"
    good = KnowledgeChunk("good", "acme", "refund good", "kb://good")
    bad = KnowledgeChunk("bad", "acme", "refund bad", "kb://bad", metadata={"not_json": {1, 2}})
    with SQLiteKnowledgeStore(database) as knowledge:
        with pytest.raises(TypeError):
            knowledge.upsert_many([good, bad])

        assert knowledge.get("good", AccessContext("acme"), now=NOW) is None
        assert knowledge.get("bad", AccessContext("acme"), now=NOW) is None


def test_memory_batch_upsert_rolls_back_every_row_on_serialisation_failure(tmp_path) -> None:
    database = tmp_path / "context.sqlite3"
    good = MemoryItem("good", "acme", "refund good")
    bad = MemoryItem("bad", "acme", "refund bad", metadata={"not_json": object()})
    with SQLiteMemoryStore(database) as memory:
        with pytest.raises(TypeError):
            memory.remember_many([good, bad])

        assert memory.get("good", AccessContext("acme"), now=NOW) is None
        assert memory.get("bad", AccessContext("acme"), now=NOW) is None


def test_corrupt_rows_fail_closed_without_hiding_healthy_rows(tmp_path) -> None:
    database = tmp_path / "context.sqlite3"
    with SQLiteKnowledgeStore(database) as knowledge:
        knowledge.upsert(KnowledgeChunk("healthy", "acme", "refund healthy", "kb://healthy"))
    with SQLiteMemoryStore(database) as memory:
        memory.remember(MemoryItem("healthy", "acme", "refund healthy"))

    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO knowledge_chunks (
                tenant_id, chunk_id, text_content, source_uri, document_id,
                version, version_order, acl_json, valid_from, valid_until,
                created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "acme", "corrupt", "refund corrupt", "kb://corrupt", None,
                "1", 1, '{"not":"an acl list"}', None, None,
                "not-a-timestamp", "{}",
            ),
        )
        connection.execute(
            """
            INSERT INTO memory_items (
                tenant_id, memory_id, content, scope, scope_id,
                provenance_json, importance, created_at, updated_at,
                expires_at, tags_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "acme", "corrupt", "refund corrupt", "tenant", "acme",
                '{"source_type":"bad","observed_at":"naive-time","confidence":1}',
                0.5, "not-a-timestamp", "not-a-timestamp", None, "[]", "{}",
            ),
        )

    with SQLiteKnowledgeStore(database) as knowledge, SQLiteMemoryStore(database) as memory:
        knowledge_hits = knowledge.search("refund", AccessContext("acme"), now=NOW, top_k=20)
        memory_hits = memory.recall("refund", AccessContext("acme"), now=NOW, top_k=20)
        assert [hit.chunk.chunk_id for hit in knowledge_hits] == ["healthy"]
        assert [hit.item.memory_id for hit in memory_hits] == ["healthy"]
        assert knowledge.get("corrupt", AccessContext("acme"), now=NOW) is None
        assert memory.get("corrupt", AccessContext("acme"), now=NOW) is None


def test_persistent_memory_forget_is_scope_and_tenant_safe(tmp_path) -> None:
    database = tmp_path / "context.sqlite3"
    item = MemoryItem(
        "memory-delete",
        "tenant-a",
        "user-owned preference",
        MemoryScope.USER,
        "user-a",
    )
    with SQLiteMemoryStore(database) as memory:
        shared = MemoryItem(
            "tenant-shared",
            "tenant-a",
            "shared retention policy",
            MemoryScope.TENANT,
        )
        memory.remember_many((item, shared))
        assert memory.forget(
            shared.memory_id,
            AccessContext("tenant-a", user_id="user-a"),
        ) is False
        assert memory.get(
            shared.memory_id,
            AccessContext("tenant-a", user_id="user-a"),
        ) is not None
        assert memory.forget(
            item.memory_id,
            AccessContext("tenant-a", user_id="other-user"),
        ) is False
        assert memory.forget(
            item.memory_id,
            AccessContext("tenant-b", user_id="user-a"),
        ) is False
        assert memory.forget(
            item.memory_id,
            AccessContext("tenant-a", user_id="user-a"),
        ) is True
        assert memory.get(
            item.memory_id,
            AccessContext("tenant-a", user_id="user-a"),
        ) is None


def test_postgres_migration_declares_forced_default_deny_tenant_rls() -> None:
    sql = (PROJECT_ROOT / "migrations/postgres/001_context.sql").read_text(
        encoding="utf-8"
    )
    normalised = " ".join(sql.casefold().split())
    for table in ("knowledge_chunks", "memory_items"):
        assert f"alter table taskforge.{table} enable row level security" in normalised
        assert f"alter table taskforge.{table} force row level security" in normalised
    assert "current_setting('taskforge.tenant_id', true)" in normalised
    assert normalised.count("with check") >= 2
    assert "revoke all on schema taskforge from public" in normalised

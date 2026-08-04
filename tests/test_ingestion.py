from __future__ import annotations

import pytest

from taskforge.ingestion import ingest_workspace_document
from taskforge.knowledge import AccessContext
from taskforge.persistent_context import SQLiteKnowledgeStore
from taskforge.security import ToolInputError


def test_ingestion_chunks_reopens_and_latest_version_wins(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    document = workspace / "guide.md"
    document.write_text(("Agent permissions and evidence.\n" * 80), encoding="utf-8")
    database = tmp_path / "context.sqlite3"
    with SQLiteKnowledgeStore(database) as store:
        first = ingest_workspace_document(
            store,
            workspace_root=workspace,
            relative_path="guide.md",
            tenant_id="tenant-a",
            knowledge_base_id="docs",
            version="1",
            version_order=1,
            chunk_chars=400,
            overlap_chars=40,
        )
        assert first.chunks > 1

        document.write_text("Agent lease recovery is the new guidance.\n", encoding="utf-8")
        second = ingest_workspace_document(
            store,
            workspace_root=workspace,
            relative_path="guide.md",
            tenant_id="tenant-a",
            knowledge_base_id="docs",
            version="2",
            version_order=2,
            chunk_chars=400,
            overlap_chars=40,
        )
        assert second.chunks == 1

    with SQLiteKnowledgeStore(database) as reopened:
        hits = reopened.search(
            "lease recovery",
            AccessContext("tenant-a"),
            knowledge_base_ids=["docs"],
            top_k=20,
        )
        assert len(hits) == 1
        assert hits[0].chunk.version == "2"
        assert hits[0].chunk.metadata["line_start"] == 1


def test_reingesting_same_version_removes_stale_chunks(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    document = workspace / "guide.md"
    document.write_text("evidence\n" * 200, encoding="utf-8")
    database = tmp_path / "context.sqlite3"
    with SQLiteKnowledgeStore(database) as store:
        large = ingest_workspace_document(
            store,
            workspace_root=workspace,
            relative_path="guide.md",
            tenant_id="tenant-a",
            knowledge_base_id="docs",
            version="1",
            version_order=1,
            chunk_chars=300,
            overlap_chars=20,
        )
        assert large.chunks > 1
        document.write_text("evidence replacement", encoding="utf-8")
        replacement = ingest_workspace_document(
            store,
            workspace_root=workspace,
            relative_path="guide.md",
            tenant_id="tenant-a",
            knowledge_base_id="docs",
            version="1",
            version_order=1,
            chunk_chars=300,
            overlap_chars=20,
        )
        assert replacement.chunks == 1
        hits = store.search("evidence", AccessContext("tenant-a"), top_k=100)
        assert len(hits) == 1


def test_ingestion_rejects_escape_sensitive_and_binary_files(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".env").write_text("SECRET=x", encoding="utf-8")
    (workspace / "binary.bin").write_bytes(b"hello\x00world")
    store = SQLiteKnowledgeStore(tmp_path / "context.sqlite3")
    common = dict(
        store=store,
        workspace_root=workspace,
        tenant_id="tenant-a",
        knowledge_base_id="docs",
        version="1",
        version_order=1,
    )
    with pytest.raises(ToolInputError):
        ingest_workspace_document(relative_path="../escape.md", **common)
    with pytest.raises(ToolInputError):
        ingest_workspace_document(relative_path=".env", **common)
    with pytest.raises(ToolInputError):
        ingest_workspace_document(relative_path="binary.bin", **common)

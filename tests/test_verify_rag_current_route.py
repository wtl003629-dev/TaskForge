from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from verify_rag_current_route import verify_current_route  # noqa: E402

from taskforge.config import Settings  # noqa: E402
from taskforge.knowledge import KnowledgeChunk  # noqa: E402
from taskforge.literature.repository import (  # noqa: E402
    LiteratureAccess,
    SQLiteLiteratureRepository,
)
from taskforge.persistent_context import SQLiteKnowledgeStore  # noqa: E402
from taskforge.research_protocol import (  # noqa: E402
    LiteratureRequest,
    PaperCard,
    ResearchScope,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        rag_active_profile="current",
        rag_experiment_profile="current",
        general_text_backend="fastembed",
        semantic_model="BAAI/bge-small-en-v1.5",
        semantic_cache_path=tmp_path / "embeddings.sqlite3",
        context_sqlite_path=tmp_path / "context.sqlite3",
        literature_sqlite_path=tmp_path / "literature.sqlite3",
    )


def test_static_rollback_check_resolves_original_profile(tmp_path: Path) -> None:
    report = verify_current_route(_settings(tmp_path))

    assert report["status"] == "passed"
    assert report["resolved_profile"] == "current-a"
    assert all(report["checks"].values())


def test_scope_smoke_reads_current_index_when_optimized_index_coexists(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    access = LiteratureAccess("tenant-a", "user-a", "conversation-a")
    repository = SQLiteLiteratureRepository(settings.literature_sqlite_path)
    repository.save_request(
        access,
        LiteratureRequest(request_id="request-1", query="rollback evidence"),
    )
    repository.upsert_paper(
        access,
        PaperCard(
            paper_id="paper-1",
            canonical_title="Rollback Evidence",
            abstract="abstract",
            verification_status="provider_verified",
            full_text_status="ingested",
        ),
    )
    repository.create_scope(
        access,
        ResearchScope(
            scope_id="scope-1",
            tenant_id="tenant-a",
            owner_user_id="user-a",
            conversation_id="conversation-a",
            request_id="request-1",
            selected_paper_ids=["paper-1"],
            user_intent="Verify rollback.",
        ),
    )
    knowledge = SQLiteKnowledgeStore(settings.context_sqlite_path)
    common = {
        "tenant_id": "tenant-a",
        "text": "The current route keeps the rollback evidence available.",
        "source_uri": "paper://paper-1",
        "acl": frozenset({"user:user-a"}),
    }
    knowledge.upsert_many(
        [
            KnowledgeChunk(
                chunk_id="chunk-current",
                document_id="research-paper:scope-1:paper-1",
                metadata={
                    "knowledge_base_id": "research-scope:scope-1:v1",
                    "evidence_id": "evidence-current",
                },
                **common,
            ),
            KnowledgeChunk(
                chunk_id="chunk-optimized",
                document_id="research-paper:scope-1:paper-1:rag:optimized-e",
                metadata={
                    "knowledge_base_id": "research-scope:scope-1:v1:rag:optimized-e",
                    "evidence_id": "evidence-optimized",
                    "rag_profile": "optimized",
                    "rag_ablation": "e",
                },
                **common,
            ),
        ]
    )

    report = verify_current_route(
        settings,
        scope_id="scope-1",
        tenant_id="tenant-a",
        user_id="user-a",
        query="rollback evidence",
        expected_evidence_id="evidence-current",
    )

    assert report["status"] == "passed"
    assert report["scope_smoke"]["current_visible_chunk_count"] == 1
    assert report["scope_smoke"]["optimized_visible_chunk_count"] == 1
    assert report["scope_smoke"]["returned_evidence_ids"] == ["evidence-current"]

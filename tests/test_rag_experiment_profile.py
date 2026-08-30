from __future__ import annotations

import pytest

from taskforge.config import Settings
from taskforge.knowledge import AccessContext, InMemoryKnowledgeStore, KnowledgeChunk
from taskforge.rag_experiment_profile import resolve_rag_experiment_profile
from taskforge.research_retrieval import ResearchRetrievalService


def _principal() -> AccessContext:
    return AccessContext(tenant_id="tenant-a", user_id="user-a")


def test_current_is_the_default_and_resolves_to_original_ablation() -> None:
    settings = Settings(_env_file=None, database_backend="sqlite")
    profile = resolve_rag_experiment_profile(
        settings.rag_active_profile,
        settings.rag_optimized_ablation,
    )

    assert settings.rag_active_profile == "current"
    assert settings.rag_experiment_profile == "current"
    assert settings.pdf_chunking_mode == "parent_child"
    assert settings.research_dual_route_enabled is False
    assert profile.label == "current-a"
    assert profile.retrieval_text_enabled is False
    assert profile.parent_aware_rerank_enabled is False
    assert profile.lineage_diversity_enabled is False
    assert profile.structure_aware_chunking_enabled is False


def test_live_optimized_profile_requires_a_passed_promotion_manifest() -> None:
    with pytest.raises(ValueError, match="promotion manifest"):
        Settings(
            _env_file=None,
            database_backend="sqlite",
            rag_active_profile="optimized",
        )


def test_offline_evaluation_can_run_optimized_without_promotion() -> None:
    settings = Settings(
        _env_file=None,
        database_backend="sqlite",
        rag_active_profile="optimized",
        rag_experiment_profile="optimized",
        rag_evaluation_mode=True,
    )

    assert settings.rag_active_profile == "optimized"
    assert settings.rag_evaluation_mode is True


def test_hybrid_chunking_is_explicit_and_keeps_dual_route_opt_in() -> None:
    settings = Settings(
        _env_file=None,
        database_backend="sqlite",
        pdf_chunking_mode="hybrid",
        research_dual_route_enabled=True,
    )

    assert settings.pdf_chunking_mode == "hybrid"
    assert settings.research_dual_route_enabled is True

    with pytest.raises(ValueError, match="requires pdf_chunking_mode=hybrid"):
        Settings(
            _env_file=None,
            database_backend="sqlite",
            research_dual_route_enabled=True,
        )


def test_optimized_ablation_features_are_incremental() -> None:
    expected = {
        "a": (False, False, False, False),
        "b": (True, False, False, False),
        "c": (True, True, False, False),
        "d": (True, True, True, False),
        "e": (True, True, True, True),
    }

    for stage, flags in expected.items():
        profile = resolve_rag_experiment_profile("optimized", stage)
        assert (
            profile.retrieval_text_enabled,
            profile.parent_aware_rerank_enabled,
            profile.lineage_diversity_enabled,
            profile.structure_aware_chunking_enabled,
        ) == flags


def test_optimized_identity_cannot_replace_current_document_version() -> None:
    current = resolve_rag_experiment_profile("current")
    optimized = resolve_rag_experiment_profile("optimized", "e")
    base_document = "research-paper:scope-1:paper-1"
    base_kb = "research-scope:scope-1:v1"
    store = InMemoryKnowledgeStore()

    current_chunk = KnowledgeChunk(
        chunk_id="current-child",
        tenant_id="tenant-a",
        text="current evidence",
        source_uri="paper://paper-1",
        document_id=current.document_id(base_document),
        version="1",
        metadata={"knowledge_base_id": current.knowledge_base_id(base_kb)},
    )
    optimized_chunk = KnowledgeChunk(
        chunk_id="optimized-child",
        tenant_id="tenant-a",
        text="optimized evidence",
        source_uri="paper://paper-1",
        document_id=optimized.document_id(base_document),
        version="1",
        metadata={
            "knowledge_base_id": optimized.knowledge_base_id(base_kb),
            **optimized.metadata(),
        },
    )

    store.replace_document_version([current_chunk])
    store.replace_document_version([optimized_chunk])

    visible = store.visible_chunks(_principal())
    assert {chunk.chunk_id for chunk in visible} == {
        "current-child",
        "optimized-child",
    }
    assert current.document_id(base_document) == base_document
    assert current.knowledge_base_id(base_kb) == base_kb
    assert optimized.document_id(base_document) != base_document
    assert optimized.knowledge_base_id(base_kb) != base_kb


def test_retrieval_and_read_are_profile_isolated() -> None:
    optimized = resolve_rag_experiment_profile("optimized", "e")
    store = InMemoryKnowledgeStore(
        [
            KnowledgeChunk(
                chunk_id="current-child",
                tenant_id="tenant-a",
                text="shared needle current",
                source_uri="paper://paper-1",
                document_id="current-paper",
                metadata={"evidence_id": "current-evidence"},
            ),
            KnowledgeChunk(
                chunk_id="optimized-child",
                tenant_id="tenant-a",
                text="shared needle optimized",
                source_uri="paper://paper-1",
                document_id="optimized-paper",
                metadata={
                    "evidence_id": "optimized-evidence",
                    **optimized.metadata(),
                },
            ),
        ]
    )
    current_service = ResearchRetrievalService(store, graph_enabled=False)
    optimized_service = ResearchRetrievalService(
        store,
        graph_enabled=False,
        experiment_profile=optimized,
    )

    assert [
        item.chunk_id for item in current_service.search("shared needle", _principal()).evidence
    ] == ["current-child"]
    assert [
        item.chunk_id
        for item in optimized_service.search("shared needle", _principal()).evidence
    ] == ["optimized-child"]
    with pytest.raises(KeyError, match="not found"):
        current_service.read_evidence("optimized-evidence", _principal())
    with pytest.raises(KeyError, match="not found"):
        optimized_service.read_evidence("current-evidence", _principal())


def test_retrieval_text_is_used_only_from_stage_b() -> None:
    retrieval_text = "Document: unique-title-token\n\nContent:\nplain body"
    current_chunk = KnowledgeChunk(
        chunk_id="current-child",
        tenant_id="tenant-a",
        text="plain body",
        source_uri="paper://paper-1",
        document_id="current-paper",
        metadata={"retrieval_text": retrieval_text},
    )
    optimized_b = resolve_rag_experiment_profile("optimized", "b")
    optimized_chunk = KnowledgeChunk(
        chunk_id="optimized-child",
        tenant_id="tenant-a",
        text="plain body",
        source_uri="paper://paper-1",
        document_id="optimized-paper",
        metadata={"retrieval_text": retrieval_text, **optimized_b.metadata()},
    )
    store = InMemoryKnowledgeStore([current_chunk, optimized_chunk])

    current = ResearchRetrievalService(store, graph_enabled=False)
    optimized = ResearchRetrievalService(
        store,
        graph_enabled=False,
        experiment_profile=optimized_b,
    )

    assert not current.search("unique-title-token", _principal()).evidence
    assert [
        item.chunk_id
        for item in optimized.search("unique-title-token", _principal()).evidence
    ] == ["optimized-child"]

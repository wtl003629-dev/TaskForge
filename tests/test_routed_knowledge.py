from __future__ import annotations

import pytest

from taskforge.config import Settings
from taskforge.context import ContextAssembler
from taskforge.knowledge import AccessContext, InMemoryKnowledgeStore, KnowledgeChunk
from taskforge.routed_knowledge import RoutedKnowledgeStore


def _chunk(
    chunk_id: str,
    text: str,
    *,
    source: str = "handbook",
    document_id: str | None = None,
    acl: frozenset[str] = frozenset({"tenant"}),
    metadata: dict[str, object] | None = None,
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        tenant_id="tenant-a",
        text=text,
        source_uri=f"file://{source}",
        document_id=document_id or source,
        acl=acl,
        metadata={"knowledge_base_id": "kb", "source": source, **(metadata or {})},
    )


def _profile(query: str, chunks: list[KnowledgeChunk], *, top_k: int = 5) -> tuple[str, str]:
    hits = RoutedKnowledgeStore(InMemoryKnowledgeStore(chunks)).search(
        query,
        AccessContext("tenant-a"),
        top_k=top_k,
    )
    assert hits
    assert hits[0].retrieval_profile is not None
    assert hits[0].retrieval_backend is not None
    return hits[0].retrieval_profile, hits[0].retrieval_backend


def test_online_router_selects_all_four_profiles() -> None:
    assert _profile(
        "approval policy",
        [_chunk("general", "The approval policy requires two reviewers.")],
    ) == ("general_text", "bm25_general_text")

    table_profile, table_backend = _profile(
        "What was the total revenue in 2021?",
        [
            _chunk(
                "table",
                "Table row: 2021 | revenue | 42",
                metadata={
                    "kind": "table",
                    "table_rows": [["year", "revenue"], ["2021", "42"]],
                },
            )
        ],
    )
    assert table_profile == "table_numeric"
    assert table_backend == "bm25_table_numeric_feature_rerank"

    cross_profile, cross_backend = _profile(
        "According to both TechCrunch and The Verge reports, what changed?",
        [
            _chunk("tc", "TechCrunch reports the policy changed.", source="TechCrunch"),
            _chunk("verge", "The Verge reports the policy changed.", source="The Verge"),
        ],
    )
    assert cross_profile == "cross_document"
    assert cross_backend == "bm25_source_coverage_anchor_rrf"

    pdf_profile, pdf_backend = _profile(
        "What appears on the previous page?",
        [
            _chunk(
                "p1",
                "Page one introduces the approval rule.",
                source="policy.pdf",
                metadata={"pages": [1], "previous_chunk_id": None, "next_chunk_id": "p2"},
            ),
            _chunk(
                "p2",
                "Page two contains reviewer evidence.",
                source="policy.pdf",
                metadata={"pages": [2], "previous_chunk_id": "p1", "next_chunk_id": None},
            ),
        ],
        top_k=3,
    )
    assert pdf_profile == "pdf_layout"
    assert pdf_backend == "structure_aware_pdf_bm25_neighbor"


def test_inaccessible_sources_cannot_change_profile_selection() -> None:
    visible = _chunk("visible", "Both reports use the same approval policy.")
    hidden = _chunk(
        "hidden",
        "A second source reports a different policy.",
        source="secret-source",
        acl=frozenset({"role:secret"}),
    )
    profile, backend = _profile(
        "According to both reports, what changed?",
        [visible, hidden],
    )
    assert profile == "general_text"
    assert backend == "bm25_general_text"


def test_context_main_path_exposes_selected_profile_and_backend() -> None:
    routed = RoutedKnowledgeStore(
        InMemoryKnowledgeStore(
            [_chunk("policy", "The approval policy requires reviewer evidence.")]
        )
    )
    context = ContextAssembler(routed).assemble(
        "approval policy",
        principal=AccessContext("tenant-a"),
    )
    assert context.retrieval_profile == "general_text"
    assert context.retrieval_backend == "bm25_general_text"
    assert context.knowledge_hits[0].chunk.chunk_id == "policy"


def test_bailian_route_uses_isolated_provider_configuration(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class FakeBailianEmbedder:
        model_name = "text-embedding-v4"
        index_name = "knowledge-bailian-text-embedding-v4-1024-v1"
        dimension = 3

        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def embed_documents(self, texts):  # type: ignore[no-untyped-def]
            return [[float(len(text)), 1.0, 0.0] for text in texts]

        def embed_query(self, text):  # type: ignore[no-untyped-def]
            return [float(len(text)), 1.0, 0.0]

    monkeypatch.setattr(
        "taskforge.routed_knowledge.BailianDenseEmbedder",
        FakeBailianEmbedder,
    )
    cache = tmp_path / "embeddings-bailian.sqlite3"
    routed = RoutedKnowledgeStore(
        InMemoryKnowledgeStore(
            [_chunk("policy", "The approval policy requires reviewer evidence.")]
        ),
        general_text_backend="bailian",
        bailian_api_key="configured-secret",
        bailian_cache_path=str(cache),
    )
    hits = routed.search(
        "approval policy",
        AccessContext("tenant-a"),
    )
    assert hits[0].retrieval_backend == "bailian_dense:text-embedding-v4"
    assert captured["api_key"] == "configured-secret"
    assert captured["cache_path"] == str(cache)


def test_bailian_settings_require_key_and_fixed_dimension(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires bailian_api_key"):
        Settings(
            _env_file=None,
            general_text_backend="bailian",
            bailian_api_key=None,
        )

    settings = Settings(
        _env_file=None,
        general_text_backend="bailian",
        bailian_api_key="configured-secret",
        bailian_cache_path=tmp_path / "bailian.sqlite3",
    )
    assert settings.bailian_api_key is not None
    assert settings.bailian_api_key.get_secret_value() == "configured-secret"
    assert settings.bailian_embedding_dimension == 1_024

from __future__ import annotations

import pytest

from taskforge.evidence_graph import EvidenceQueryPlan, LocalEvidenceGraph
from taskforge.hybrid_retrieval import HybridChunk, HybridSearchHit
from taskforge.local_graph import LocalDocumentGraph, entity_phrases


def test_entity_phrases_extract_capitalized_phrases() -> None:
    phrases = entity_phrases(
        "As reported by The Verge and TechCrunch, Apple filed the case."
    )
    assert "The Verge" in phrases
    assert "TechCrunch" in phrases
    assert "Apple" in phrases


def test_graph_seeds_and_one_hop_neighbors() -> None:
    chunks = [
        {
            "document_id": "a",
            "text": "Apple plans a new device and The Verge reported it.",
        },
        {
            "document_id": "b",
            "text": "Apple faces a new investigation per TechCrunch.",
        },
        {
            "document_id": "c",
            "text": "Unrelated baking recipe content.",
        },
    ]
    graph = LocalDocumentGraph(chunks)

    assert graph.node_count == 3
    assert graph.edge_count == 1  # only a-b share "Apple"

    hits = graph.search("Apple")
    # both a and b are seed documents (or neighbors of a seed); c is isolated.
    assert {"a", "b"}.issubset(set(hits))
    assert "c" not in hits

    # A query with no matching entity phrase returns nothing.
    assert graph.search("baking") == []


def _evidence_chunk(
    chunk_id: str,
    text: str,
    *,
    section: str = "paper:section:1",
    parent: str = "paper",
    previous: str | None = None,
    next_id: str | None = None,
) -> HybridChunk:
    return HybridChunk(
        chunk_id=chunk_id,
        tenant_id="tenant",
        text=text,
        source_uri=f"taskforge://{chunk_id}",
        document_id="paper",
        knowledge_base_id="kb",
        acl_principals=frozenset({"user"}),
        previous_chunk_id=previous,
        next_chunk_id=next_id,
        metadata={
            "parent_document_id": parent,
            "section_id": section,
        },
    )


def _evidence_hit(chunk: HybridChunk, rank: int, score: float) -> HybridSearchHit:
    return HybridSearchHit(
        chunk=chunk,
        rank=rank,
        score=score,
        base_score=score,
        retrieval_sources=["python_bm25"],
    )


def test_evidence_graph_builds_structural_edges_and_preserves_candidates() -> None:
    chunks = [
        _evidence_chunk(
            "a",
            "BERT improves accuracy.",
            next_id="b",
        ),
        _evidence_chunk(
            "b",
            "BERT improves robustness.",
            previous="a",
        ),
        _evidence_chunk("c", "Unrelated baking recipe.", section="paper:section:2"),
    ]
    graph = LocalEvidenceGraph(chunks)
    result = graph.rerank(
        "BERT results",
        [
            _evidence_hit(chunks[0], 1, 1.0),
            _evidence_hit(chunks[2], 2, 0.5),
            _evidence_hit(chunks[1], 3, 0.4),
        ],
        seed_k=1,
        graph_weight=0.5,
        entity_weight=0.0,
        section_weight=0.0,
        adjacency_weight=0.0,
    )

    assert graph.node_count == 3
    assert graph.edge_count >= 1
    assert {hit.chunk.chunk_id for hit in result.hits} == {"a", "b", "c"}
    assert set(result.features) == {"a", "b", "c"}
    assert result.features["a"].base_rank == 1
    assert result.features["b"].graph_support > 0
    assert all("graph_feature_rerank" in hit.retrieval_sources for hit in result.hits)
    assert result.seed_chunk_ids == ("a",)


def test_evidence_graph_rejects_invalid_feature_weights() -> None:
    chunk = _evidence_chunk("a", "BERT result")
    graph = LocalEvidenceGraph([chunk])
    with pytest.raises(ValueError, match="sum to at most 1"):
        graph.rerank(
            "BERT",
            [_evidence_hit(chunk, 1, 1.0)],
            seed_k=1,
            graph_weight=0.8,
            entity_weight=0.3,
        )


def test_evidence_graph_expansion_is_bounded_and_scoped() -> None:
    chunks = [
        _evidence_chunk("a", "BERT improves accuracy.", next_id="b"),
        _evidence_chunk("b", "BERT improves robustness.", previous="a", parent="paper-a"),
        _evidence_chunk("c", "BERT improves recall.", previous="b", parent="paper-a"),
        _evidence_chunk("d", "BERT improves precision.", previous="c", parent="paper-b"),
    ]
    graph = LocalEvidenceGraph(chunks)
    expanded = graph.expand_candidates(
        [_evidence_hit(chunks[0], 1, 1.0)],
        max_add=2,
        hops=2,
        allowed_parent_document_ids=frozenset({"paper-a"}),
    )
    assert [hit.chunk.chunk_id for hit in expanded] == ["b", "c"]
    assert all(hit.neighbor_distance in {1, 2} for hit in expanded)
    assert all(hit.chunk.metadata["parent_document_id"] == "paper-a" for hit in expanded)


def test_evidence_query_plan_is_structured_and_bounded() -> None:
    plan = EvidenceQueryPlan.from_text("How did BERT improve results in the methods section?", max_hops=2)
    assert "bert" in plan.terms
    assert "methods" in plan.section_hints
    assert plan.max_hops == 2
    with pytest.raises(ValueError, match="must not be blank"):
        EvidenceQueryPlan.from_text(" ")

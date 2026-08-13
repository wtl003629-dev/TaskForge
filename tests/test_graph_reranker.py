from __future__ import annotations

from pathlib import Path

from taskforge.evidence_graph import GraphRerankFeatures, GraphRerankResult
from taskforge.graph_reranker import (
    LearnedGraphReranker,
    train_pairwise_graph_reranker,
)
from taskforge.hybrid_retrieval import HybridChunk, HybridSearchHit


def _chunk(chunk_id: str) -> HybridChunk:
    return HybridChunk(
        chunk_id=chunk_id,
        tenant_id="tenant",
        text=f"evidence {chunk_id}",
        source_uri=f"taskforge://{chunk_id}",
        document_id="paper",
        acl_principals=frozenset({"user"}),
    )


def _hit(chunk: HybridChunk, rank: int) -> HybridSearchHit:
    return HybridSearchHit(
        chunk=chunk,
        rank=rank,
        score=1.0 / rank,
        base_score=1.0 / rank,
        retrieval_sources=["graph_feature_rerank"],
    )


def _row(case_id: str, positive_id: str) -> dict[str, object]:
    ids = ["a", "b", "c"]
    features = {}
    for index, chunk_id in enumerate(ids, start=1):
        features[chunk_id] = {
            "base_rank": index,
            "base_score": 1.0 / index,
            "graph_support": 1.0 if chunk_id == positive_id else 0.0,
            "entity_support": 1.0 if chunk_id == positive_id else 0.0,
            "section_support": 0.0,
            "adjacency_support": 0.0,
            "final_score": 1.0 / index,
        }
    return {
        "case_id": case_id,
        "retrieved_ids": ids,
        "relevant_ids": [positive_id],
        "graph": {"features": features},
    }


def test_pairwise_ranker_trains_and_preserves_candidates(tmp_path: Path) -> None:
    model = train_pairwise_graph_reranker(
        [_row("case-a", "c"), _row("case-b", "b")],
        fit_run_id="fit-1",
        dataset_sha256="a" * 64,
        epochs=5,
    )
    result = GraphRerankResult(
        hits=(_hit(_chunk("a"), 1), _hit(_chunk("b"), 2), _hit(_chunk("c"), 3)),
        features={
            "a": GraphRerankFeatures(1, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            "b": GraphRerankFeatures(2, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5),
            "c": GraphRerankFeatures(3, 1 / 3, 1.0, 1.0, 0.0, 0.0, 1.0),
        },
        seed_chunk_ids=("a",),
        node_count=3,
        edge_count=1,
    )
    reranked = model.rerank(result)
    assert {hit.chunk.chunk_id for hit in reranked.hits} == {"a", "b", "c"}
    assert all("learned_graph_rerank" in hit.retrieval_sources for hit in reranked.hits)

    path = tmp_path / "model.json"
    model.save(path)
    loaded = LearnedGraphReranker.load(path)
    assert loaded.model_dump() == model.model_dump()

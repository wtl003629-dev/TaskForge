from __future__ import annotations

import math

import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient, models

import taskforge.hybrid_retrieval as hybrid_module
from taskforge.hybrid_retrieval import (
    BM25Index,
    DeterministicHashEmbedder,
    EmbeddingContractError,
    HybridChunk,
    HybridSearchRequest,
    LexicalOverlapFallbackReranker,
    QdrantBackendError,
    QdrantHybridIndex,
    QdrantUnavailableError,
    RerankerContractError,
)


def chunk(
    chunk_id: str,
    text: str,
    *,
    tenant: str = "tenant-a",
    acl: frozenset[str] = frozenset({"user:alice"}),
    version: str = "1",
    version_order: int = 1,
    document: str = "policy",
    previous: str | None = None,
    next_: str | None = None,
) -> HybridChunk:
    return HybridChunk(
        chunk_id=chunk_id,
        tenant_id=tenant,
        text=text,
        source_uri=f"docs/{document}.pdf",
        document_id=document,
        knowledge_base_id="governance",
        version=version,
        version_order=version_order,
        acl_principals=acl,
        previous_chunk_id=previous,
        next_chunk_id=next_,
        metadata={"page": 1, "block_id": f"block-{chunk_id}"},
    )


def request(**updates: object) -> HybridSearchRequest:
    values: dict[str, object] = {
        "query": "approval policy",
        "tenant_id": "tenant-a",
        "acl_principals": {"user:alice", "role:reviewer"},
        "versions": {"1"},
        "version_orders": {1},
        "knowledge_base_ids": {"governance"},
        "top_k": 3,
        "candidate_k": 10,
    }
    values.update(updates)
    return HybridSearchRequest(**values)


class RecordingClient:
    def __init__(self) -> None:
        self.delegate = QdrantClient(location=":memory:")
        self.last_query: dict[str, object] | None = None

    def __getattr__(self, name: str) -> object:
        return getattr(self.delegate, name)

    def query_points(self, **kwargs: object) -> object:
        self.last_query = dict(kwargs)
        return self.delegate.query_points(**kwargs)


def test_request_and_chunk_models_are_strict_and_fail_closed() -> None:
    with pytest.raises(ValidationError, match="acl_principals"):
        request(acl_principals=set())
    with pytest.raises(ValidationError, match="candidate_k"):
        request(top_k=4, candidate_k=3)
    with pytest.raises(ValidationError, match="searchable token"):
        request(query="--- !!!")
    with pytest.raises(ValidationError, match="extra_forbidden"):
        HybridSearchRequest(
            query="policy",
            tenant_id="tenant-a",
            acl_principals={"user:alice"},
            unexpected=True,
        )
    bad_payload = chunk("bad", "policy").model_dump()
    bad_payload["metadata"] = {"bad": math.nan}
    with pytest.raises(ValidationError, match="finite JSON"):
        HybridChunk.model_validate(bad_payload)


def test_pure_python_bm25_is_explainable_and_filters_before_statistics() -> None:
    index = BM25Index(
        [
            chunk("best", "approval policy"),
            chunk("second", "approval policy appendix appendix appendix appendix"),
            # These much stronger-looking records must neither take a slot nor
            # change document frequency/IDF because scope precedes scoring.
            chunk("wrong-tenant", "approval policy " * 50, tenant="tenant-b"),
            chunk("wrong-acl", "approval policy " * 50, acl=frozenset({"user:bob"})),
            chunk("wrong-version", "approval policy " * 50, version="2", version_order=2),
        ]
    )
    result = index.search(request(top_k=1, candidate_k=1))

    assert [hit.chunk.chunk_id for hit in result.hits] == ["best"]
    explanation = result.hits[0].bm25_explanation
    assert explanation is not None
    assert explanation.corpus_size_after_filters == 2
    assert {term.term for term in explanation.terms} == {"approval", "policy"}
    assert math.isclose(
        result.hits[0].score,
        sum(term.contribution for term in explanation.terms),
    )
    assert result.filters_applied_before_ranking.versions == ["1"]


def test_bm25_field_weights_boost_metadata_fields() -> None:
    def titled(chunk_id: str, body: str, title: str) -> HybridChunk:
        base = chunk(chunk_id, body)
        return base.model_copy(
            update={"metadata": {**base.metadata, "title": title}}
        )

    weighted = BM25Index(
        [
            titled("title-only", "unrelated content words here", "renewal"),
            titled("body-match", "renewal and more content here", "unrelated"),
        ],
        field_weights={"title": 10.0},
    )
    result = weighted.search(request(query="renewal", top_k=2, candidate_k=2))
    assert result.hits[0].chunk.chunk_id == "title-only"

    plain = BM25Index(
        [
            titled("title-only", "unrelated content words here", "renewal"),
            titled("body-match", "renewal and more content here", "unrelated"),
        ]
    )
    result = plain.search(request(query="renewal", top_k=2, candidate_k=2))
    assert result.hits[0].chunk.chunk_id == "body-match"

    with pytest.raises(ValueError, match="field weights"):
        BM25Index([], field_weights={"title": 0})


def test_qdrant_uses_named_vectors_real_server_rrf_and_prefetch_filters() -> None:
    client = RecordingClient()
    index = QdrantHybridIndex(
        client,
        collection_name="hybrid-contract",
        embedder=DeterministicHashEmbedder(32),
        backend_label="qdrant_local",
    )
    index.upsert(
        [
            chunk("one", "approval policy evidence"),
            chunk("two", "incident response policy"),
        ]
    )

    result = index.search(request(top_k=2, candidate_k=2))
    collection = client.delegate.get_collection("hybrid-contract")

    assert set(collection.config.params.vectors) == {"dense"}
    assert set(collection.config.params.sparse_vectors or {}) == {"sparse"}
    assert result.backend == "qdrant_local"
    assert result.hits
    assert all(hit.retrieval_sources == ["qdrant_server_rrf"] for hit in result.hits)
    assert client.last_query is not None
    assert isinstance(client.last_query["query"], models.FusionQuery)
    assert client.last_query["query"].fusion == models.Fusion.RRF
    prefetch = client.last_query["prefetch"]
    assert [branch.using for branch in prefetch] == ["dense", "sparse"]
    assert all(branch.filter is not None for branch in prefetch)
    assert client.last_query["query_filter"] is not None


def test_qdrant_tenant_acl_and_version_filters_apply_before_candidate_limit() -> None:
    index = QdrantHybridIndex.in_memory(
        collection_name="isolation",
        embedder=DeterministicHashEmbedder(32),
    )
    index.upsert(
        [
            chunk("visible", "approval policy summary"),
            chunk("tenant-secret", "approval policy " * 50, tenant="tenant-b"),
            chunk("acl-secret", "approval policy " * 50, acl=frozenset({"user:bob"})),
            chunk("old-version", "approval policy " * 50, version="0", version_order=0),
            chunk("order-mismatch", "approval policy " * 50, version="1", version_order=2),
        ]
    )

    # If filtering happened after ranking, candidate_k=1 would be consumed by
    # one of the stronger forbidden records and this visible hit would vanish.
    result = index.search(request(top_k=1, candidate_k=1))
    assert [hit.chunk.chunk_id for hit in result.hits] == ["visible"]
    assert result.filters_applied_before_ranking.tenant_id == "tenant-a"
    assert result.filters_applied_before_ranking.version_orders == [1]


def test_lexical_fallback_reranker_reorders_candidates_and_keeps_base_score() -> None:
    reranker = LexicalOverlapFallbackReranker()
    assert reranker.score("alpha beta", ["alpha", "alpha beta"]) == pytest.approx([0.5, 1.0])
    index = QdrantHybridIndex.in_memory(
        collection_name="rerank",
        embedder=DeterministicHashEmbedder(32),
        reranker=reranker,
    )
    index.upsert(
        [
            chunk("partial", "alpha alpha alpha gamma"),
            chunk("complete", "alpha beta"),
        ]
    )

    result = index.search(
        request(
            query="alpha beta",
            versions={"1"},
            top_k=2,
            candidate_k=2,
            rerank=True,
        )
    )
    assert [hit.chunk.chunk_id for hit in result.hits] == ["complete", "partial"]
    assert all(hit.reranker_score is not None for hit in result.hits)
    assert all("fallback_lexical_rerank" in hit.retrieval_sources for hit in result.hits)
    assert all(hit.base_score > 0 for hit in result.hits)


def test_neighbor_expansion_is_acl_filtered_version_bound_and_deduplicated() -> None:
    index = QdrantHybridIndex.in_memory(
        collection_name="neighbors",
        embedder=DeterministicHashEmbedder(32),
    )
    index.upsert(
        [
            chunk("left", "context before", next_="center"),
            chunk(
                "center",
                "unique-needle approval policy",
                previous="left",
                next_="right",
            ),
            chunk("right", "context after", previous="center"),
            # A readable link that crosses a version is never traversable.
            chunk("old-right", "old context", version="0", version_order=0),
        ]
    )

    result = index.search(
        request(
            query="unique-needle",
            top_k=1,
            candidate_k=5,
            neighbor_window=2,
            max_expanded_hits=10,
        )
    )
    ids = [hit.chunk.chunk_id for hit in result.hits]
    assert ids[0] == "center"
    assert set(ids) == {"left", "center", "right"}
    assert len(ids) == len(set(ids))
    assert result.seed_count == 1
    assert result.expanded_neighbor_count == 2
    neighbors = [hit for hit in result.hits if hit.neighbor_of_chunk_id]
    assert all(hit.neighbor_of_chunk_id == "center" for hit in neighbors)
    assert all(hit.retrieval_sources == ["adjacent_chunk"] for hit in neighbors)


def test_embedding_reranker_collection_and_dependency_failures_are_explicit(monkeypatch) -> None:
    class WrongDimensionEmbedder:
        dimension = 3

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0]

    bad_vectors = QdrantHybridIndex.in_memory(
        collection_name="bad-vectors",
        embedder=WrongDimensionEmbedder(),
    )
    with pytest.raises(EmbeddingContractError, match="dimension"):
        bad_vectors.upsert([chunk("one", "approval")])

    no_reranker = QdrantHybridIndex.in_memory(
        collection_name="no-reranker",
        embedder=DeterministicHashEmbedder(16),
    )
    no_reranker.upsert([chunk("one", "approval policy")])
    with pytest.raises(RerankerContractError, match="configured Reranker"):
        no_reranker.search(request(top_k=1, candidate_k=1, rerank=True))

    client = QdrantClient(location=":memory:")
    client.create_collection(
        "wrong-schema",
        vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )
    with pytest.raises(QdrantBackendError, match="dimension"):
        QdrantHybridIndex(
            client,
            collection_name="wrong-schema",
            embedder=DeterministicHashEmbedder(16),
        )

    monkeypatch.setattr(hybrid_module, "qdrant_models", None)
    with pytest.raises(QdrantUnavailableError, match="qdrant-client"):
        QdrantHybridIndex(
            object(),
            collection_name="missing-client",
            embedder=DeterministicHashEmbedder(16),
        )


def test_fastembed_missing_fails_closed_and_import_never_downloads(monkeypatch) -> None:
    """Without fastembed the semantic embedder fails closed; nothing downloads."""

    from taskforge.hybrid_retrieval import FastEmbedEmbedder

    monkeypatch.setattr(hybrid_module, "TextEmbedding", None)
    with pytest.raises(EmbeddingContractError, match="fastembed"):
        FastEmbedEmbedder()

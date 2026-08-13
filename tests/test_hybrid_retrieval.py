from __future__ import annotations

import math
import sqlite3

import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient, models

import taskforge.hybrid_retrieval as hybrid_module
from taskforge.hybrid_retrieval import (
    AppliedRetrievalFilters,
    BM25DenseRRFIndex,
    BM25Index,
    CandidateTailUnionIndex,
    DeterministicHashEmbedder,
    EmbeddingContractError,
    FastEmbedCrossEncoderReranker,
    FastEmbedEmbedder,
    FastEmbedSparseIndex,
    HybridChunk,
    HybridRetrievalError,
    HybridSearchHit,
    HybridSearchRequest,
    HybridSearchResponse,
    InMemoryDenseIndex,
    LexicalOverlapFallbackReranker,
    MultiQueryRRFIndex,
    ParentChildIndex,
    QdrantBackendError,
    QdrantHybridIndex,
    QdrantUnavailableError,
    RepresentationRRFIndex,
    RerankerContractError,
    SearchRepresentationIndex,
    SourceCoverageRRFIndex,
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


def test_search_representation_returns_raw_evidence_and_preserves_scope() -> None:
    raw = chunk("e1", "the raw evidence sentence", document="paper")
    representation = raw.model_copy(
        update={"text": "Paper title: Retrieval\nSection: Results\nraw evidence"}
    )
    backend = BM25Index([representation])
    projected = SearchRepresentationIndex(backend, [raw])

    response = projected.search(request(query="Paper title Results"))

    assert response.hits[0].chunk.chunk_id == "e1"
    assert response.hits[0].chunk.text == raw.text
    assert "raw_evidence_projection" in response.hits[0].retrieval_sources
    assert response.filters_applied_before_ranking == AppliedRetrievalFilters.from_request(
        request(query="Paper title Results")
    )


def test_search_representation_rejects_unknown_backend_chunk() -> None:
    raw = chunk("e1", "raw", document="paper")
    other = chunk("e2", "other", document="paper")
    with pytest.raises(ValueError, match="duplicate raw"):
        SearchRepresentationIndex(BM25Index([raw]), [raw, raw])
    with pytest.raises(HybridRetrievalError, match="unknown chunk"):
        SearchRepresentationIndex(BM25Index([other]), [raw]).search(
            request(query="other")
        )


def test_in_memory_dense_index_applies_trusted_scope() -> None:
    allowed = chunk("allowed", "approval policy evidence")
    denied = chunk("denied", "approval policy evidence", tenant="tenant-b")
    backend = InMemoryDenseIndex(
        [allowed, denied],
        DeterministicHashEmbedder(32),
    )
    response = backend.search(request(query="approval policy"))
    assert [hit.chunk.chunk_id for hit in response.hits] == ["allowed"]
    assert response.backend == "in_memory_dense"


def test_bounded_reranker_scores_only_prefix_and_preserves_candidate_tail() -> None:
    calls: list[list[str]] = []

    class PrefixReranker:
        def score(self, query: str, documents: list[str]) -> list[float]:
            calls.append(documents)
            return [float(len(documents) - index) for index, _ in enumerate(documents)]

    hits = [
        HybridSearchHit(
            chunk=chunk(f"h-{index}", f"document-{index}"),
            rank=index + 1,
            score=float(10 - index),
            base_score=float(10 - index),
            retrieval_sources=["qdrant_dense"],
        )
        for index in range(4)
    ]
    result = hybrid_module._apply_reranker(
        hits,
        HybridSearchRequest(
            query="document",
            tenant_id="tenant-a",
            acl_principals=frozenset({"user:alice"}),
            rerank=True,
            top_k=4,
            candidate_k=4,
        ),
        PrefixReranker(),
        rerank_limit=2,
    )

    assert calls == [["document-0", "document-1"]]
    assert [hit.chunk.chunk_id for hit in result] == ["h-0", "h-1", "h-2", "h-3"]
    assert result[0].reranker_score == 2.0
    assert result[1].reranker_score == 1.0
    assert result[2].reranker_score is None
    assert result[3].reranker_score is None


def test_adaptive_reranker_only_scores_second_batch_when_margin_is_low() -> None:
    hits = [
        HybridSearchHit(
            chunk=chunk(f"adaptive-{index}", f"document-{index}"),
            rank=index + 1,
            score=float(10 - index),
            base_score=float(10 - index),
            retrieval_sources=["qdrant_dense"],
        )
        for index in range(5)
    ]
    search_request = request(
        query="document",
        top_k=5,
        candidate_k=5,
        rerank=True,
    )

    class LowMarginReranker:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def score(self, query: str, documents: list[str]) -> list[float]:
            self.calls.append(documents)
            return [1.0, 0.6] if len(self.calls) == 1 else [2.0, 0.0]

    low_margin = LowMarginReranker()
    escalated, diagnostics = hybrid_module._apply_adaptive_reranker(
        hits,
        search_request,
        low_margin,
        min_k=2,
        max_k=4,
        margin_threshold=0.7,
    )

    assert low_margin.calls == [
        ["document-0", "document-1"],
        ["document-2", "document-3"],
    ]
    assert diagnostics.escalated is True
    assert diagnostics.applied_k == 4
    assert diagnostics.reason == "low_score_margin"
    assert escalated[0].chunk.chunk_id == "adaptive-2"
    assert escalated[4].reranker_score is None

    class HighMarginReranker:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def score(self, query: str, documents: list[str]) -> list[float]:
            self.calls.append(documents)
            return [2.0, 0.0]

    high_margin = HighMarginReranker()
    retained, diagnostics = hybrid_module._apply_adaptive_reranker(
        hits,
        search_request,
        high_margin,
        min_k=2,
        max_k=4,
        margin_threshold=0.7,
    )

    assert high_margin.calls == [["document-0", "document-1"]]
    assert diagnostics.escalated is False
    assert diagnostics.applied_k == 2
    assert diagnostics.reason == "high_confidence"
    assert [hit.chunk.chunk_id for hit in retained[2:]] == [
        "adaptive-2",
        "adaptive-3",
        "adaptive-4",
    ]


def test_fastembed_sparse_index_filters_before_learned_ranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SparseVector:
        def __init__(self, indices: list[int], values: list[float]) -> None:
            self.indices = indices
            self.values = values

    class FakeSparseEmbedding:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def embed(self, texts: list[str], *, batch_size: int) -> list[SparseVector]:
            assert batch_size == 4
            return [
                SparseVector([1], [10.0])
                if "approval" in text
                else SparseVector([2], [1.0])
                for text in texts
            ]

        def query_embed(
            self,
            texts: list[str],
            *,
            batch_size: int,
        ) -> list[SparseVector]:
            assert texts == ["approval policy"]
            assert batch_size == 1
            return [SparseVector([1], [2.0])]

    monkeypatch.setattr(
        hybrid_module,
        "SparseTextEmbedding",
        FakeSparseEmbedding,
    )
    public = chunk("public", "approval policy", document="public")
    unrelated = chunk("other", "travel policy", document="other")
    denied = chunk(
        "denied",
        "approval policy",
        document="denied",
        acl=frozenset({"user:bob"}),
    )
    index = FastEmbedSparseIndex(
        [public, unrelated, denied],
        model_name="fake/splade",
        batch_size=4,
    )
    response = index.search(request(top_k=2, candidate_k=3))

    assert [hit.chunk.chunk_id for hit in response.hits] == ["public"]
    assert response.backend == "fastembed_sparse"
    assert response.raw_candidate_counts["fastembed_learned_sparse"] == 2
    assert response.hits[0].retrieval_sources == ["fastembed_learned_sparse"]


class RecordingClient:
    def __init__(self) -> None:
        self.delegate = QdrantClient(location=":memory:")
        self.last_query: dict[str, object] | None = None

    def __getattr__(self, name: str) -> object:
        return getattr(self.delegate, name)

    def query_points(self, **kwargs: object) -> object:
        self.last_query = dict(kwargs)
        return self.delegate.query_points(**kwargs)


class TwoDimensionalEmbedder:
    dimension = 2

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0] if text.startswith("semantic") else [0.0, 1.0]
            for text in texts
        ]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


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


def test_parent_child_search_routes_then_preserves_child_scope() -> None:
    children = [
        chunk("a1", "alpha approval terms", document="a").model_copy(
            update={"metadata": {"parent_document_id": "parent-a"}}
        ),
        chunk("a2", "alpha appendix", document="a").model_copy(
            update={"metadata": {"parent_document_id": "parent-a"}}
        ),
        chunk("b1", "beta approval terms", document="b").model_copy(
            update={"metadata": {"parent_document_id": "parent-b"}}
        ),
    ]
    parents = [
        chunk("parent::parent-a", "alpha approval terms alpha appendix", document="parent-a"),
        chunk("parent::parent-b", "beta approval terms", document="parent-b"),
    ]
    index = ParentChildIndex(BM25Index(parents), BM25Index(children), children)
    result = index.search(request(query="beta", top_k=2, candidate_k=2))

    assert [hit.chunk.chunk_id for hit in result.hits] == ["b1"]
    assert result.backend == "parent_child"
    assert result.filters_applied_before_ranking.tenant_id == "tenant-a"

    scoped = index.search(
        request(
            query="beta",
            top_k=2,
            candidate_k=2,
            allowed_chunk_ids=frozenset({"a1", "a2"}),
        )
    )
    assert [hit.chunk.chunk_id for hit in scoped.hits] == []


def test_parent_child_uses_explicit_parent_top_k_budget() -> None:
    children = [
        chunk(
            f"child-{index}",
            f"term-{index} approval policy",
            document=f"doc-{index}",
        ).model_copy(update={"metadata": {"parent_document_id": f"parent-{index}"}})
        for index in range(1, 7)
    ]
    parents = [
        chunk(
            f"parent::{index}",
            f"term-{index} approval policy",
            document=f"parent-{index}",
        )
        for index in range(1, 7)
    ]

    class RecordingBM25(BM25Index):
        def __init__(self, records: list[HybridChunk]) -> None:
            super().__init__(records)
            self.requests: list[HybridSearchRequest] = []

        def search(self, search_request: HybridSearchRequest):  # type: ignore[no-untyped-def]
            self.requests.append(search_request)
            return super().search(search_request)

    parent_index = RecordingBM25(parents)
    child_index = BM25Index(children)
    index = ParentChildIndex(
        parent_index,
        child_index,
        children,
        parent_top_k=5,
    )

    result = index.search(request(query="term-6", top_k=10, candidate_k=50))

    assert parent_index.requests[0].top_k == 5
    assert parent_index.requests[0].candidate_k == 50
    assert len(result.hits) <= 10


def test_parent_child_sibling_coverage_adds_context_evidence_after_scored_hits() -> None:
    children = [
        chunk("table", "target metric", document="doc").model_copy(
            update={"metadata": {"parent_document_id": "parent"}}
        ),
        chunk("paragraph-1", "context explanation", document="doc").model_copy(
            update={"metadata": {"parent_document_id": "parent"}}
        ),
        chunk("paragraph-2", "another context", document="doc").model_copy(
            update={"metadata": {"parent_document_id": "parent"}}
        ),
    ]
    parents = [chunk("parent::parent", "target metric", document="parent")]
    index = ParentChildIndex(BM25Index(parents), BM25Index(children), children)

    result = index.search(request(query="target", top_k=3, candidate_k=3))

    assert [hit.chunk.chunk_id for hit in result.hits] == [
        "table",
        "paragraph-1",
        "paragraph-2",
    ]
    assert result.hits[1].retrieval_sources == ["parent_sibling_coverage"]


def test_representation_max_fusion_keeps_a_strong_single_branch_hit() -> None:
    generic = [
        chunk("g-a", "alpha metric", document="a"),
        chunk("g-b", "beta metric", document="b"),
    ]
    table = [
        chunk("t-a-1", "alpha row one", document="a"),
        chunk("t-a-2", "alpha row two", document="a"),
    ]
    index = RepresentationRRFIndex(
        [("generic", BM25Index(generic), generic), ("table", BM25Index(table), table)],
        fusion="max",
    )

    result = index.search(request(query="beta", top_k=2, candidate_k=2))

    assert [hit.chunk.document_id for hit in result.hits] == ["b"]
    assert result.backend == "multi_representation_rrf"


def test_representation_coverage_can_reserve_same_parent_candidates() -> None:
    primary = chunk("table", "target metric", document="doc").model_copy(
        update={"metadata": {"parent_document_id": "context"}}
    )
    sibling = chunk("paragraph", "context explanation", document="doc-para").model_copy(
        update={"metadata": {"parent_document_id": "context"}}
    )
    generic = [primary, sibling]
    table = [primary]
    index = RepresentationRRFIndex(
        [
            ("generic", BM25Index(generic), generic),
            ("table", BM25Index(table), table),
        ],
        candidate_strategy="coverage",
        context_sibling_coverage=True,
    )

    result = index.search(request(query="target", top_k=2, candidate_k=2))

    assert [hit.chunk.chunk_id for hit in result.hits] == ["table", "paragraph"]
    assert result.hits[1].retrieval_sources == ["context_sibling_coverage"]


def test_representation_context_sibling_limit_bounds_expansion() -> None:
    primary = chunk("table", "target metric", document="doc").model_copy(
        update={"metadata": {"parent_document_id": "context"}}
    )
    siblings = [
        chunk(f"paragraph-{index}", "context explanation", document=f"doc-{index}").model_copy(
            update={"metadata": {"parent_document_id": "context"}}
        )
        for index in range(1, 4)
    ]
    records = [primary, *siblings]
    index = RepresentationRRFIndex(
        [
            ("generic", BM25Index(records), records),
            ("table", BM25Index([primary]), [primary]),
        ],
        candidate_strategy="coverage",
        context_sibling_coverage=True,
        context_sibling_limit=1,
    )

    result = index.search(request(query="target", top_k=4, candidate_k=4))

    assert len(result.hits) == 2
    assert result.hits[1].retrieval_sources == ["context_sibling_coverage"]


def test_representation_coverage_branch_reserve_reaches_deep_table_prefix() -> None:
    generic = [chunk(f"a{index:02d}", "generic", document=f"a{index:02d}") for index in range(1, 15)]
    table = [
        chunk("b01", "table", document="b01"),
        chunk("b02", "table", document="b02"),
        chunk("b03", "table", document="b03"),
        chunk("b99", "table target", document="b99"),
    ]

    class FixedIndex:
        def __init__(self, records: list[HybridChunk]) -> None:
            self.records = records

        def search(self, search_request: HybridSearchRequest) -> HybridSearchResponse:
            hits = [
                HybridSearchHit(
                    chunk=record,
                    rank=index,
                    score=float(len(self.records) - index),
                    base_score=float(len(self.records) - index),
                    retrieval_sources=["python_bm25"],
                )
                for index, record in enumerate(self.records, start=1)
            ][: search_request.candidate_k]
            return HybridSearchResponse(
                backend="python_bm25",
                collection_name=None,
                query=search_request.query,
                filters_applied_before_ranking=AppliedRetrievalFilters.from_request(
                    search_request
                ),
                seed_count=len(hits),
                expanded_neighbor_count=0,
                hits=hits,
            )

    class TargetReranker:
        def score(self, query: str, documents: list[str]) -> list[float]:
            return [100.0 if "target" in document else 0.0 for document in documents]

    plain = RepresentationRRFIndex(
        [("generic", FixedIndex(generic), generic), ("table", FixedIndex(table), table)],
        candidate_strategy="coverage",
        reranker=TargetReranker(),
    )
    reserved = RepresentationRRFIndex(
        [("generic", FixedIndex(generic), generic), ("table", FixedIndex(table), table)],
        candidate_strategy="coverage",
        coverage_branch_reserves={"table": 3},
        reranker=TargetReranker(),
    )

    plain_ids = [
        hit.chunk.document_id
        for hit in plain.search(
            request(query="needle", top_k=5, candidate_k=8, rerank=True)
        ).hits
    ]
    reserved_ids = [
        hit.chunk.document_id
        for hit in reserved.search(
            request(query="needle", top_k=5, candidate_k=8, rerank=True)
        ).hits
    ]

    assert "b99" not in plain_ids
    assert "b99" in reserved_ids


def test_candidate_tail_union_preserves_head_and_adds_only_novel_tail() -> None:
    class FixedIndex:
        def __init__(self, records: list[HybridChunk], backend: str) -> None:
            self.records = records
            self.backend = backend

        def search(self, search_request: HybridSearchRequest) -> HybridSearchResponse:
            hits = [
                HybridSearchHit(
                    chunk=record,
                    rank=rank,
                    score=float(len(self.records) - rank + 1),
                    base_score=float(len(self.records) - rank + 1),
                    retrieval_sources=[
                        "qdrant_dense" if self.backend == "qdrant_local" else "python_bm25"
                    ],
                )
                for rank, record in enumerate(self.records, start=1)
            ][: search_request.candidate_k]
            return HybridSearchResponse(
                backend=self.backend,
                collection_name="candidate" if self.backend == "qdrant_local" else None,
                query=search_request.query,
                filters_applied_before_ranking=AppliedRetrievalFilters.from_request(
                    search_request
                ),
                seed_count=len(hits),
                expanded_neighbor_count=0,
                raw_candidate_counts={"fixed": len(hits)},
                hits=hits,
            )

    primary = [
        chunk(f"p{index}", "primary", document=f"p{index}")
        for index in range(1, 7)
    ]
    candidates = [
        primary[1],
        chunk("d1", "dense one", document="d1"),
        chunk("d2", "dense two", document="d2"),
    ]
    index = CandidateTailUnionIndex(
        FixedIndex(primary, "python_bm25"),
        FixedIndex(candidates, "qdrant_local"),
        preserve_head_k=2,
        candidate_slots=2,
    )

    result = index.search(request(top_k=6, candidate_k=6))

    assert result.backend == "candidate_tail_union"
    assert [hit.chunk.document_id for hit in result.hits] == [
        "p1",
        "p2",
        "p3",
        "p4",
        "d1",
        "d2",
    ]
    assert result.raw_candidate_counts["candidate_novel_tail"] == 2
    assert all(
        "semantic_dense_candidate_tail" in hit.retrieval_sources
        for hit in result.hits[-2:]
    )


def test_representation_reranker_is_bounded_to_top_k_prefix() -> None:
    records = [
        chunk("a", "alpha", document="a"),
        chunk("b", "alpha", document="b"),
        chunk("c", "alpha", document="c"),
        chunk("d", "alpha", document="d"),
    ]

    class PrefixReranker:
        def score(self, query: str, documents: list[str]) -> list[float]:
            return [float(index) for index, _ in enumerate(documents)]

    index = RepresentationRRFIndex(
        [("one", BM25Index(records), records), ("two", BM25Index(records), records)],
        reranker=PrefixReranker(),
        rerank_top_k=2,
    )

    result = index.search(request(query="alpha", top_k=4, candidate_k=4, rerank=True))

    assert [hit.chunk.chunk_id for hit in result.hits] == ["b", "a", "c", "d"]


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


def test_bm25_upsert_invalidates_cached_scope_statistics() -> None:
    index = BM25Index([chunk("first", "existing evidence")])
    assert not index.search(request(query="novel clue", top_k=2, candidate_k=2)).hits

    index.upsert(chunk("second", "novel clue"))

    assert [
        hit.chunk.chunk_id
        for hit in index.search(request(query="novel clue", top_k=2, candidate_k=2)).hits
    ] == ["second"]


def test_bm25_scope_cache_never_crosses_acl_principals() -> None:
    index = BM25Index(
        [
            chunk("alice", "approval summary", acl=frozenset({"user:alice"})),
            chunk(
                "bob",
                "approval secret secret",
                acl=frozenset({"user:bob"}),
            ),
        ]
    )

    alice = index.search(request(top_k=1, candidate_k=1))
    bob = index.search(
        request(
            acl_principals={"user:bob"},
            top_k=1,
            candidate_k=1,
        )
    )

    assert [hit.chunk.chunk_id for hit in alice.hits] == ["alice"]
    assert [hit.chunk.chunk_id for hit in bob.hits] == ["bob"]


def test_parent_document_scope_filters_before_bm25_statistics() -> None:
    first = chunk("first", "shared target exact", document="first").model_copy(
        update={"metadata": {"parent_document_id": "report-a"}}
    )
    second = chunk(
        "second", "shared target exact exact exact", document="second"
    ).model_copy(update={"metadata": {"parent_document_id": "report-b"}})
    index = BM25Index([first, second])

    result = index.search(
        request(
            query="shared target",
            top_k=2,
            candidate_k=2,
            parent_document_ids={"report-a"},
        )
    )

    assert [hit.chunk.chunk_id for hit in result.hits] == ["first"]
    assert result.filters_applied_before_ranking.parent_document_ids == ["report-a"]
    assert result.hits[0].bm25_explanation is not None
    assert result.hits[0].bm25_explanation.corpus_size_after_filters == 1


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


def test_qdrant_dense_only_uses_named_dense_vector_without_sparse_prefetch() -> None:
    client = RecordingClient()
    index = QdrantHybridIndex(
        client,
        collection_name="dense-contract",
        embedder=TwoDimensionalEmbedder(),
        backend_label="qdrant_local",
    )
    index.upsert(
        [
            chunk("lexical", "approval policy exact"),
            chunk("semantic", "semantic concept"),
        ]
    )

    result = index.search_dense(request(top_k=2, candidate_k=2))

    assert result.hits[0].chunk.chunk_id == "semantic"
    assert all(hit.retrieval_sources == ["qdrant_dense"] for hit in result.hits)
    assert client.last_query is not None
    assert client.last_query["using"] == "dense"
    assert "prefetch" not in client.last_query
    assert client.last_query["query_filter"] is not None


def test_bm25_dense_rrf_fuses_real_branch_ranks_and_preserves_scope() -> None:
    visible = [
        chunk("lexical", "approval policy exact"),
        chunk("semantic", "semantic concept"),
    ]
    forbidden = chunk(
        "forbidden",
        "semantic approval policy " * 20,
        acl=frozenset({"user:bob"}),
    )
    qdrant = QdrantHybridIndex.in_memory(
        collection_name="bm25-dense-rrf",
        embedder=TwoDimensionalEmbedder(),
    )
    qdrant.upsert([*visible, forbidden])
    index = BM25DenseRRFIndex(BM25Index([*visible, forbidden]), qdrant)

    result = index.search(request(top_k=2, candidate_k=3))

    assert result.backend == "bm25_dense_rrf"
    assert [hit.chunk.chunk_id for hit in result.hits] == ["lexical", "semantic"]
    assert result.hits[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert result.hits[1].score == pytest.approx(1 / 61)
    assert result.hits[0].retrieval_sources == [
        "python_bm25",
        "qdrant_dense",
        "bm25_dense_rrf",
    ]
    assert result.hits[1].retrieval_sources == ["qdrant_dense", "bm25_dense_rrf"]
    assert result.raw_candidate_counts["bm25"] == 1
    assert result.raw_candidate_counts["dense"] == 2
    assert result.raw_candidate_counts["fused"] == 2
    assert "forbidden" not in {hit.chunk.chunk_id for hit in result.hits}
    assert result.filters_applied_before_ranking.acl_principals == [
        "role:reviewer",
        "user:alice",
    ]


def test_document_diversity_prevents_one_document_from_crowding_top_k() -> None:
    index = BM25Index(
        [
            chunk("a-1", "approval policy exact exact", document="a"),
            chunk("a-2", "approval policy exact", document="a"),
            chunk("a-3", "approval policy", document="a"),
            chunk("b-1", "approval policy secondary", document="b"),
        ]
    )

    crowded = index.search(request(top_k=2, candidate_k=2))
    diverse = index.search(
        request(top_k=2, candidate_k=2, max_chunks_per_document=1)
    )

    assert [hit.chunk.document_id for hit in crowded.hits] == ["a", "a"]
    assert [hit.chunk.document_id for hit in diverse.hits] == ["a", "b"]


def test_fastembed_cross_encoder_adapter_is_explicit_and_contract_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCrossEncoder:
        def __init__(self, *, model_name: str) -> None:
            self.model_name = model_name

        def rerank(self, query, documents, *, batch_size):
            assert query == "approval"
            assert batch_size == 8
            return [float(len(document)) for document in documents]

    monkeypatch.setattr(hybrid_module, "TextCrossEncoder", FakeCrossEncoder)
    reranker = FastEmbedCrossEncoderReranker("fake/model", batch_size=8)

    assert reranker.score("approval", ["a", "abcd"]) == [1.0, 4.0]
    assert reranker.model_name == "fake/model"


def test_multi_query_rrf_returns_document_diverse_evidence() -> None:
    lexical = BM25Index(
        [
            chunk("alpha", "Alpha rollback evidence", document="alpha"),
            chunk("beta", "Beta switching evidence", document="beta"),
            chunk("noise", "unrelated report", document="noise"),
        ]
    )
    index = MultiQueryRRFIndex(
        lexical,
        lambda query: [query, "Alpha rollback", "Beta switching"],
    )

    result = index.search(
        request(
            query="Compare Alpha rollback and Beta switching",
            top_k=2,
            candidate_k=3,
        )
    )

    assert {hit.chunk.document_id for hit in result.hits} == {"alpha", "beta"}
    assert all("multi_query_rrf" in hit.retrieval_sources for hit in result.hits)
    assert result.backend == "multi_query_rrf"


def test_source_coverage_rrf_reserves_named_source_branches() -> None:
    verge = chunk("verge", "Alpha device report", document="verge").model_copy(
        update={"metadata": {"source": "The Verge"}}
    )
    techcrunch = chunk(
        "techcrunch", "Beta company report", document="techcrunch"
    ).model_copy(update={"metadata": {"source": "TechCrunch"}})
    noise = chunk("noise", "Alpha Beta report report", document="noise").model_copy(
        update={"metadata": {"source": "Other"}}
    )
    lexical = BM25Index([verge, techcrunch, noise])
    index = SourceCoverageRRFIndex(lexical, [verge, techcrunch, noise])

    result = index.search(
        request(
            query="Compare Alpha in The Verge with Beta in TechCrunch",
            top_k=3,
            candidate_k=3,
        )
    )

    assert index.matched_sources(result.query) == ["TechCrunch", "The Verge"]
    assert {hit.chunk.document_id for hit in result.hits} >= {"verge", "techcrunch"}
    assert all("source_coverage_rrf" in hit.retrieval_sources for hit in result.hits)


def test_source_coverage_anchor_preserves_lexical_head() -> None:
    verge = chunk("verge", "Alpha device report", document="verge").model_copy(
        update={"metadata": {"source": "The Verge"}}
    )
    techcrunch = chunk(
        "techcrunch", "Beta company report", document="techcrunch"
    ).model_copy(update={"metadata": {"source": "TechCrunch"}})
    noise = chunk("noise", "Alpha Beta report report", document="noise").model_copy(
        update={"metadata": {"source": "Other"}}
    )
    lexical = BM25Index([verge, techcrunch, noise])
    request_value = request(
        query="Compare Alpha in The Verge with Beta in TechCrunch",
        top_k=3,
        candidate_k=3,
    )
    lexical_head = lexical.search(request_value).hits[0].chunk.document_id
    anchored = SourceCoverageRRFIndex(
        lexical,
        [verge, techcrunch, noise],
        lexical_anchor_k=1,
    ).search(request_value)

    assert anchored.hits[0].chunk.document_id == lexical_head


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

    monkeypatch.setattr(hybrid_module, "TextEmbedding", None)
    with pytest.raises(EmbeddingContractError, match="fastembed"):
        FastEmbedEmbedder()


def test_fastembed_cache_reuses_document_and_query_vectors(
    monkeypatch, tmp_path
) -> None:
    class FakeTextEmbedding:
        document_batches: list[list[str]] = []
        query_calls: list[str] = []

        def __init__(self, *, model_name: str) -> None:
            assert model_name == "fake-semantic"

        def embed(self, documents, *, batch_size: int = 256):
            values = list(documents)
            if values == ["dimension probe"]:
                return iter([[0.0, 1.0]])
            self.document_batches.append(values)
            return iter([[float(len(value)), 0.5] for value in values])

        def query_embed(self, query: str):
            self.query_calls.append(query)
            return iter([[float(len(query)), 1.0]])

    monkeypatch.setattr(hybrid_module, "TextEmbedding", FakeTextEmbedding)
    cache_path = tmp_path / "vectors.sqlite3"
    first = FastEmbedEmbedder(
        "fake-semantic",
        cache_path=cache_path,
        batch_size=1,
    )
    document_vectors = first.embed_documents(["same", "second", "same"])
    query_vector = first.embed_query("same")

    second = FastEmbedEmbedder(
        "fake-semantic",
        cache_path=cache_path,
        batch_size=1,
    )
    assert second.embed_documents(["second", "same"]) == [
        document_vectors[1],
        document_vectors[0],
    ]
    assert second.embed_query("same") == query_vector
    assert FakeTextEmbedding.document_batches == [["same"], ["second"]]
    assert FakeTextEmbedding.query_calls == ["same"]


def test_fastembed_cache_corruption_fails_closed(monkeypatch, tmp_path) -> None:
    class FakeTextEmbedding:
        def __init__(self, *, model_name: str) -> None:
            pass

        def embed(self, documents, *, batch_size: int = 256):
            return iter([[1.0, 0.0] for _ in documents])

        def query_embed(self, query: str):
            return iter([[1.0, 0.0]])

    monkeypatch.setattr(hybrid_module, "TextEmbedding", FakeTextEmbedding)
    cache_path = tmp_path / "vectors.sqlite3"
    embedder = FastEmbedEmbedder("fake-semantic", cache_path=cache_path)
    embedder.embed_documents(["document"])
    with sqlite3.connect(cache_path) as connection:
        connection.execute(
            "UPDATE embeddings_v1 SET vector = ? WHERE embedding_kind = 'document'",
            (b"broken",),
        )
    with pytest.raises(EmbeddingContractError, match="corrupt vector"):
        embedder.embed_documents(["document"])


def test_qdrant_upsert_batches_embedding_and_storage() -> None:
    base = DeterministicHashEmbedder(16)

    class CountingEmbedder:
        dimension = 16

        def __init__(self) -> None:
            self.document_batch_sizes: list[int] = []

        def embed_documents(self, texts):
            self.document_batch_sizes.append(len(texts))
            return base.embed_documents(texts)

        def embed_query(self, text: str):
            return base.embed_query(text)

    embedder = CountingEmbedder()
    index = QdrantHybridIndex.in_memory(
        collection_name="batched-upsert",
        embedder=embedder,
        upsert_batch_size=2,
    )
    assert index.upsert(
        [chunk("one", "first"), chunk("two", "second"), chunk("three", "third")]
    ) == 3
    assert embedder.document_batch_sizes == [2, 1]

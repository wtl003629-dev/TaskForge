from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from taskforge.context import ContextAssembler
from taskforge.hybrid_knowledge import (
    HybridCatalogMismatchError,
    HybridKnowledgeError,
    HybridKnowledgeStore,
    UnsupportedHybridScoringError,
    knowledge_to_hybrid_chunk,
)
from taskforge.hybrid_retrieval import (
    BM25Index,
    HybridSearchRequest,
    HybridSearchResponse,
)
from taskforge.knowledge import AccessContext, KnowledgeChunk

NOW = datetime(2026, 8, 4, 8, 0, tzinfo=UTC)


def knowledge_chunk(
    chunk_id: str,
    text: str,
    *,
    document: str = "policy",
    version: str = "1",
    version_order: int = 1,
    acl: frozenset[str] = frozenset({"user:alice"}),
    knowledge_base: str = "governance",
    valid_until: datetime | None = None,
    previous: str | None = None,
    next_: str | None = None,
    published_at: str | None = None,
) -> KnowledgeChunk:
    metadata: dict[str, object] = {
        "knowledge_base_id": knowledge_base,
        "previous_chunk_id": previous,
        "next_chunk_id": next_,
        "page": 1,
    }
    if published_at is not None:
        metadata["published_at"] = published_at
    return KnowledgeChunk(
        chunk_id=chunk_id,
        tenant_id="tenant-a",
        text=text,
        source_uri=f"docs/{document}.pdf",
        document_id=document,
        version=version,
        version_order=version_order,
        acl=acl,
        valid_until=valid_until,
        metadata=metadata,
    )


class RecordingBackend:
    def __init__(self, index: BM25Index) -> None:
        self.index = index
        self.request: HybridSearchRequest | None = None
        self.response: HybridSearchResponse | None = None
        self.calls = 0

    def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        self.calls += 1
        self.request = request
        self.response = self.index.search(request)
        return self.response


def make_store(
    chunks: list[KnowledgeChunk], *, neighbor_window: int = 0
) -> tuple[HybridKnowledgeStore, RecordingBackend]:
    index = BM25Index(knowledge_to_hybrid_chunk(chunk) for chunk in chunks)
    backend = RecordingBackend(index)
    return (
        HybridKnowledgeStore(backend, chunks, neighbor_window=neighbor_window),
        backend,
    )


def test_latest_acl_validity_and_knowledge_base_are_filtered_before_bm25() -> None:
    chunks = [
        knowledge_chunk(
            "old",
            "approval policy " * 100,
            document="policy",
            version="1",
            version_order=1,
        ),
        knowledge_chunk(
            "current",
            "approval policy current",
            document="policy",
            version="2",
            version_order=2,
        ),
        knowledge_chunk("other", "approval policy appendix", document="appendix"),
        knowledge_chunk(
            "forbidden",
            "approval policy " * 100,
            document="secret",
            acl=frozenset({"user:bob"}),
        ),
        knowledge_chunk(
            "expired",
            "approval policy " * 100,
            document="expired",
            valid_until=NOW,
        ),
        knowledge_chunk(
            "wrong-kb",
            "approval policy " * 100,
            document="marketing",
            knowledge_base="marketing",
        ),
    ]
    store, backend = make_store(chunks)

    hits = store.search(
        "approval policy",
        AccessContext("tenant-a", user_id="alice"),
        top_k=5,
        now=NOW,
        knowledge_base_ids={"governance"},
    )

    assert {hit.chunk.chunk_id for hit in hits} == {"current", "other"}
    assert backend.request is not None
    assert backend.request.allowed_chunk_ids == frozenset({"current", "other"})
    assert backend.response is not None
    assert all(
        hit.bm25_explanation is not None
        and hit.bm25_explanation.corpus_size_after_filters == 2
        for hit in backend.response.hits
    )
    applied = backend.response.filters_applied_before_ranking
    assert applied.allowed_chunk_count == 2
    assert applied.allowed_chunk_ids_sha256 is not None
    assert len(applied.allowed_chunk_ids_sha256) == 64


def test_empty_or_wrong_user_scope_avoids_backend_and_legacy_scores_fail_explicitly() -> None:
    store, backend = make_store([knowledge_chunk("private", "approval policy")])

    assert (
        store.search(
            "approval",
            AccessContext("tenant-a", user_id="bob"),
            now=NOW,
        )
        == []
    )
    assert backend.calls == 0
    with pytest.raises(UnsupportedHybridScoringError, match="caller-supplied"):
        store.search(
            "approval",
            AccessContext("tenant-a", user_id="alice"),
            semantic_scores={"private": 0.9},
        )


def test_stale_backend_payload_fails_closed() -> None:
    authoritative = knowledge_chunk("same-id", "current approval policy")
    stale = knowledge_chunk("same-id", "stale approval policy")
    backend = RecordingBackend(BM25Index([knowledge_to_hybrid_chunk(stale)]))
    store = HybridKnowledgeStore(backend, [authoritative], neighbor_window=0)

    with pytest.raises(HybridCatalogMismatchError, match="stale or inconsistent"):
        store.search(
            "approval policy",
            AccessContext("tenant-a", user_id="alice"),
            now=NOW,
        )


def test_neighbor_expansion_stays_in_latest_authorized_allow_list() -> None:
    chunks = [
        knowledge_chunk("left", "context before", previous=None, next_="center"),
        knowledge_chunk(
            "center",
            "unique-needle",
            previous="left",
            next_="right",
        ),
        knowledge_chunk("right", "context after", previous="center"),
        knowledge_chunk(
            "forbidden-neighbor",
            "private context",
            document="secret",
            acl=frozenset({"user:bob"}),
        ),
    ]
    store, backend = make_store(chunks, neighbor_window=1)

    hits = store.search(
        "unique-needle",
        AccessContext("tenant-a", user_id="alice"),
        top_k=3,
        now=NOW,
    )

    assert [hit.chunk.chunk_id for hit in hits] == ["center", "left", "right"]
    assert backend.request is not None
    assert "forbidden-neighbor" not in backend.request.allowed_chunk_ids


def test_context_assembler_consumes_hybrid_store_without_contract_changes() -> None:
    chunk = knowledge_chunk("evidence", "approval requires security evidence")
    store, _ = make_store([chunk])
    assembler = ContextAssembler(store, default_char_budget=500, knowledge_limit=3)

    context = assembler.assemble(
        "security evidence",
        principal=AccessContext("tenant-a", user_id="alice"),
        now=NOW,
    )

    assert [citation.item_id for citation in context.citations] == ["evidence"]
    assert "approval requires security evidence" in context.text
    assert context.knowledge_hits[0].chunk is chunk


def test_catalog_conversion_and_time_window_validation_are_explicit() -> None:
    chunk = knowledge_chunk(
        "indexed",
        "approval evidence",
        valid_until=NOW + timedelta(days=1),
    )
    store, _ = make_store([chunk])
    assert [item.chunk_id for item in store.hybrid_chunks()] == ["indexed"]
    assert store.get("indexed", AccessContext("tenant-a", user_id="alice"), now=NOW) is chunk

    invisible = knowledge_chunk("no-acl", "approval", acl=frozenset())
    with pytest.raises(HybridKnowledgeError, match="cannot be indexed safely"):
        knowledge_to_hybrid_chunk(invisible)


def test_publication_time_window_filters_candidates() -> None:
    chunks = [
        knowledge_chunk(
            "older",
            "October 2023 policy evidence",
            published_at="2023-10-01T00:00:00+00:00",
        ),
        knowledge_chunk(
            "newer",
            "December 2023 policy evidence",
            published_at="2023-12-01T00:00:00+00:00",
        ),
        # Missing publication time must survive (fail-open), not vanish.
        knowledge_chunk("undated", "policy without a date"),
    ]
    store, _ = make_store(chunks)
    principal = AccessContext("tenant-a", user_id="alice")

    after_hits = store.search(
        "policy",
        principal,
        published_after="2023-11-01T00:00:00+00:00",
        now=NOW,
    )
    assert {hit.chunk.chunk_id for hit in after_hits} == {"newer", "undated"}

    before_hits = store.search(
        "policy",
        principal,
        published_before="2023-11-01T00:00:00+00:00",
        now=NOW,
    )
    assert {hit.chunk.chunk_id for hit in before_hits} == {"older", "undated"}

    window_hits = store.search(
        "policy",
        principal,
        published_after="2023-10-01T00:00:00+00:00",
        published_before="2023-12-01T00:00:00+00:00",
        now=NOW,
    )
    assert {hit.chunk.chunk_id for hit in window_hits} == {"older", "undated"}

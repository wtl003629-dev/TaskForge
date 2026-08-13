"""Online retrieval-profile routing for the TaskForge context main path.

The router is intentionally an adapter over the authoritative knowledge store:
tenant, ACL, validity, source/base scope, and latest-version filtering happen
before corpus inspection, profile selection, or ranking.  Benchmark names are
never inputs to routing.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

from .hybrid_knowledge import MAX_ALLOWED_CHUNK_IDS, HybridKnowledgeScopeTooLargeError
from .hybrid_retrieval import (
    AppliedRetrievalFilters,
    BM25Index,
    FastEmbedEmbedder,
    HybridChunk,
    HybridSearchRequest,
    InMemoryDenseIndex,
    SourceCoverageRRFIndex,
    TATQAFeatureReranker,
)
from .knowledge import AccessContext, KnowledgeChunk, KnowledgeHit, lexical_match
from .rag_profiles import knowledge_corpus_metadata, select_retrieval_profile

GeneralTextBackend = Literal["bm25", "fastembed"]


@runtime_checkable
class AuthoritativeKnowledgeStore(Protocol):
    def visible_chunks(
        self,
        principal: AccessContext,
        *,
        now: datetime | None = None,
        source_uris: Iterable[str] | None = None,
        knowledge_base_ids: Iterable[str] | None = None,
        latest_only: bool = True,
    ) -> tuple[KnowledgeChunk, ...]: ...

    def get(
        self,
        chunk_id: str,
        principal: AccessContext,
        *,
        now: datetime | None = None,
    ) -> KnowledgeChunk | None: ...


def _flatten_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        output: list[str] = []
        for child in value.values():
            output.extend(_flatten_strings(child))
        return output
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        output = []
        for child in value:
            output.extend(_flatten_strings(child))
        return output
    if value is None:
        return []
    return [str(value)]


def _runtime_hybrid_chunk(chunk: KnowledgeChunk) -> HybridChunk:
    """Create an index payload with generic structure/layout search fields."""

    metadata = dict(chunk.metadata)
    metadata["source"] = str(
        metadata.get("source") or metadata.get("title") or chunk.source_uri
    )
    pages = _flatten_strings(metadata.get("pages", metadata.get("page")))
    heading = _flatten_strings(metadata.get("heading"))
    block_types = _flatten_strings(metadata.get("block_types", metadata.get("kind")))
    table_rows = _flatten_strings(metadata.get("table_rows"))
    metadata["retrieval_layout"] = " ".join(
        [*(f"page {page}" for page in pages), *heading, *block_types]
    )
    metadata["retrieval_structure"] = " ".join([*block_types, *table_rows])
    knowledge_base_id = str(metadata.get("knowledge_base_id") or "default").strip()
    return HybridChunk(
        chunk_id=chunk.chunk_id,
        tenant_id=chunk.tenant_id,
        text=chunk.text,
        source_uri=chunk.source_uri,
        document_id=chunk.logical_document_id,
        knowledge_base_id=knowledge_base_id,
        version=chunk.version,
        version_order=chunk.version_order,
        acl_principals=chunk.acl,
        previous_chunk_id=(
            str(metadata["previous_chunk_id"])
            if metadata.get("previous_chunk_id")
            else None
        ),
        next_chunk_id=(
            str(metadata["next_chunk_id"])
            if metadata.get("next_chunk_id")
            else None
        ),
        metadata=metadata,
    )


def _bounded(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return max(0.0, min(1.0, score))


class RoutedKnowledgeStore:
    """Knowledge-store contract backed by four query/corpus retrieval profiles."""

    def __init__(
        self,
        authoritative_store: AuthoritativeKnowledgeStore,
        *,
        general_text_backend: GeneralTextBackend = "bm25",
        semantic_model: str = "BAAI/bge-small-en-v1.5",
        semantic_cache_path: str | None = None,
        candidate_multiplier: int = 5,
    ) -> None:
        if not isinstance(authoritative_store, AuthoritativeKnowledgeStore):
            raise TypeError("authoritative_store must expose visible_chunks and get")
        if not 1 <= int(candidate_multiplier) <= 50:
            raise ValueError("candidate_multiplier must be between 1 and 50")
        self.authoritative_store = authoritative_store
        self.general_text_backend = general_text_backend
        self.candidate_multiplier = int(candidate_multiplier)
        # This is explicit host configuration. Missing semantic dependencies or
        # model artifacts fail startup instead of silently changing retrieval.
        self._embedder = (
            FastEmbedEmbedder(semantic_model, cache_path=semantic_cache_path)
            if general_text_backend == "fastembed"
            else None
        )

    def get(
        self,
        chunk_id: str,
        principal: AccessContext,
        *,
        now: datetime | None = None,
    ) -> KnowledgeChunk | None:
        return self.authoritative_store.get(chunk_id, principal, now=now)

    def visible_chunks(self, principal: AccessContext, **kwargs: Any) -> tuple[KnowledgeChunk, ...]:
        return self.authoritative_store.visible_chunks(principal, **kwargs)

    def search(
        self,
        query: str,
        principal: AccessContext,
        *,
        top_k: int = 5,
        now: datetime | None = None,
        source_uris: Iterable[str] | None = None,
        knowledge_base_ids: Iterable[str] | None = None,
        latest_only: bool = True,
        semantic_scores: Mapping[str, float] | None = None,
        lexical_weight: float = 0.70,
        semantic_weight: float = 0.30,
    ) -> list[KnowledgeHit]:
        if top_k <= 0:
            return []
        if top_k > 100:
            raise ValueError("routed top_k must not exceed 100")
        if semantic_scores or not math.isclose(lexical_weight, 0.70) or not math.isclose(
            semantic_weight, 0.30
        ):
            raise ValueError(
                "caller scores cannot override a host-selected retrieval profile"
            )
        corpus = self.authoritative_store.visible_chunks(
            principal,
            now=now,
            source_uris=source_uris,
            knowledge_base_ids=knowledge_base_ids,
            latest_only=latest_only,
        )
        if not corpus:
            return []
        if len(corpus) > MAX_ALLOWED_CHUNK_IDS:
            raise HybridKnowledgeScopeTooLargeError(
                f"authorized candidate scope has {len(corpus)} chunks; "
                f"maximum is {MAX_ALLOWED_CHUNK_IDS}"
            )

        profile = select_retrieval_profile(
            query, knowledge_corpus_metadata(corpus)
        )
        indexed = tuple(_runtime_hybrid_chunk(chunk) for chunk in corpus)
        baseline = BM25Index(indexed)
        rerank = False
        neighbor_window = 0
        if profile == "table_numeric":
            backend: Any = BM25Index(
                indexed,
                reranker=TATQAFeatureReranker(blend_weight=0.2),
                field_weights={"retrieval_structure": 2.0},
            )
            backend_name = "bm25_table_numeric_feature_rerank"
            rerank = True
        elif profile == "cross_document":
            backend = SourceCoverageRRFIndex(
                baseline,
                indexed,
                metadata_field="source",
                lexical_anchor_k=3,
            )
            backend_name = "bm25_source_coverage_anchor_rrf"
        elif profile == "pdf_layout":
            backend = BM25Index(
                indexed,
                field_weights={
                    "retrieval_layout": 2.0,
                    "retrieval_structure": 1.5,
                },
            )
            backend_name = "structure_aware_pdf_bm25_neighbor"
            neighbor_window = 1
        elif self._embedder is not None:
            backend = InMemoryDenseIndex(indexed, self._embedder)
            backend_name = f"fastembed_dense:{self._embedder.model_name}"
        else:
            backend = baseline
            backend_name = "bm25_general_text"

        seed_top_k = (
            max(1, top_k // (1 + 2 * neighbor_window))
            if neighbor_window
            else top_k
        )
        candidate_k = min(
            500,
            max(top_k, seed_top_k * self.candidate_multiplier),
        )
        request = HybridSearchRequest(
            query=query,
            tenant_id=principal.tenant_id,
            acl_principals=principal.acl_tokens,
            allowed_chunk_ids=frozenset(chunk.chunk_id for chunk in corpus),
            top_k=seed_top_k,
            candidate_k=candidate_k,
            rerank=rerank,
            neighbor_window=neighbor_window,
            max_expanded_hits=top_k,
        )
        response = backend.search(request)
        if response.filters_applied_before_ranking != AppliedRetrievalFilters.from_request(
            request
        ):
            raise RuntimeError("retrieval backend changed the authorized search scope")
        if not response.hits and backend is not baseline:
            response = baseline.search(
                request.model_copy(
                    update={"rerank": False, "neighbor_window": 0, "top_k": top_k}
                )
            )
            backend_name = f"{backend_name}:empty_to_bm25_fallback"

        by_id = {chunk.chunk_id: chunk for chunk in corpus}
        output: list[KnowledgeHit] = []
        seen: set[str] = set()
        for result in response.hits:
            chunk_id = result.chunk.chunk_id
            chunk = by_id.get(chunk_id)
            if chunk is None or chunk_id in seen:
                raise RuntimeError("retrieval returned duplicate or unauthorized evidence")
            seen.add(chunk_id)
            if result.chunk != _runtime_hybrid_chunk(chunk):
                raise RuntimeError("retrieval returned stale evidence payload")
            match = lexical_match(query, chunk.text)
            semantic = (
                _bounded(result.base_score)
                if profile == "general_text" and self._embedder is not None
                else _bounded(result.reranker_score)
            )
            output.append(
                KnowledgeHit(
                    chunk=chunk,
                    score=result.score,
                    lexical_score=match.score,
                    semantic_score=semantic,
                    matched_terms=match.matched_terms,
                    retrieval_profile=profile,
                    retrieval_backend=backend_name,
                )
            )
        return output


__all__ = [
    "AuthoritativeKnowledgeStore",
    "GeneralTextBackend",
    "RoutedKnowledgeStore",
]

"""Bridge the runtime knowledge contract to pre-filtered hybrid retrieval.

The existing :mod:`taskforge.context` assembler consumes ``KnowledgeHit``
objects.  This adapter preserves that contract while delegating ranking to a
``BM25Index`` or ``QdrantHybridIndex``.  The catalog remains host-authoritative:
ACL, validity windows, source selection, and latest-document resolution are
turned into a bounded chunk allow-list *before* either backend ranks anything.

Indexing is intentionally explicit.  Use :func:`knowledge_to_hybrid_chunk` to
populate the chosen backend and pass the same ``KnowledgeChunk`` objects to
``HybridKnowledgeStore``.  This avoids surprising embedding/network work in a
constructor and makes catalog/index drift fail closed at query time.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Protocol, runtime_checkable

from .hybrid_retrieval import (
    AppliedRetrievalFilters,
    HybridChunk,
    HybridSearchRequest,
    HybridSearchResponse,
)
from .knowledge import (
    AccessContext,
    KnowledgeChunk,
    KnowledgeHit,
    as_utc,
    lexical_match,
)

MAX_ALLOWED_CHUNK_IDS = 20_000


class HybridKnowledgeError(RuntimeError):
    """Base error for catalog/backend integration failures."""


class HybridCatalogMismatchError(HybridKnowledgeError):
    """Raised when a backend result is absent from or stale against the catalog."""


class HybridKnowledgeScopeTooLargeError(HybridKnowledgeError):
    """Raised instead of weakening a pre-ranking allow-list that is too large."""


class UnsupportedHybridScoringError(HybridKnowledgeError):
    """Raised for legacy caller-supplied scores that hybrid backends cannot honor."""


@runtime_checkable
class HybridSearchBackend(Protocol):
    def search(self, request: HybridSearchRequest) -> HybridSearchResponse: ...


def _metadata_string(metadata: Mapping[str, object], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def knowledge_to_hybrid_chunk(chunk: KnowledgeChunk) -> HybridChunk:
    """Convert one authoritative runtime chunk into the hybrid index payload."""

    metadata = dict(chunk.metadata)
    knowledge_base_id = _metadata_string(metadata, "knowledge_base_id") or "default"
    try:
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
            previous_chunk_id=_metadata_string(metadata, "previous_chunk_id"),
            next_chunk_id=_metadata_string(metadata, "next_chunk_id"),
            metadata=metadata,
        )
    except Exception as exc:
        raise HybridKnowledgeError(
            f"knowledge chunk {chunk.chunk_id!r} cannot be indexed safely: {exc}"
        ) from exc


def _clean_filter(values: Iterable[str] | None) -> frozenset[str] | None:
    if values is None:
        return None
    return frozenset(str(value).strip() for value in values if str(value).strip())


def _bounded(value: float | None) -> float:
    if value is None:
        return 0.0
    score = float(value)
    if not math.isfinite(score):
        return 0.0
    return max(0.0, min(1.0, score))


def _as_bound(value: datetime | str | None) -> datetime | None:
    """Normalise a publication-time filter bound to an aware UTC datetime."""

    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                "published filter must be an ISO-8601 timestamp or datetime"
            ) from exc
        return as_utc(parsed)
    return as_utc(value)


def _published_in_window(
    chunk: KnowledgeChunk,
    *,
    before: datetime | None,
    after: datetime | None,
) -> bool:
    """True when a chunk's ``metadata["published_at"]`` satisfies the window.

    Publication time is an optional host-side constraint, not a security
    boundary: a chunk whose ``published_at`` is missing or unparseable is
    kept (fail-open) so a corpus without dates is not silently empty.
    """

    if before is None and after is None:
        return True
    raw = chunk.metadata.get("published_at")
    if not isinstance(raw, str) or not raw.strip():
        return True
    try:
        published = as_utc(datetime.fromisoformat(raw.replace("Z", "+00:00")))
    except ValueError:
        return True
    if after is not None and published < after:
        return False
    # ``before`` is a strict upper bound: an article published exactly on the
    # boundary is excluded, matching the plain reading of "before <date>".
    if before is not None and published >= before:
        return False
    return True


class HybridKnowledgeStore:
    """Knowledge-store compatible adapter over a hybrid retrieval backend."""

    def __init__(
        self,
        backend: HybridSearchBackend,
        chunks: Iterable[KnowledgeChunk] = (),
        *,
        candidate_multiplier: int = 5,
        rerank: bool = False,
        neighbor_window: int = 1,
    ) -> None:
        if not isinstance(backend, HybridSearchBackend):
            raise TypeError("backend must implement search(HybridSearchRequest)")
        if not 1 <= int(candidate_multiplier) <= 100:
            raise ValueError("candidate_multiplier must be between 1 and 100")
        if not 0 <= int(neighbor_window) <= 5:
            raise ValueError("neighbor_window must be between 0 and 5")
        self.backend = backend
        self.candidate_multiplier = int(candidate_multiplier)
        self.rerank = bool(rerank)
        self.neighbor_window = int(neighbor_window)
        self._catalog: dict[tuple[str, str], tuple[KnowledgeChunk, HybridChunk]] = {}
        for chunk in chunks:
            self.upsert_catalog(chunk)

    def upsert_catalog(self, chunk: KnowledgeChunk) -> None:
        """Update only the authoritative catalog; indexing remains explicit."""

        hybrid = knowledge_to_hybrid_chunk(chunk)
        self._catalog[(chunk.tenant_id, chunk.chunk_id)] = (chunk, hybrid)

    add = upsert_catalog

    def hybrid_chunks(self, *, tenant_id: str | None = None) -> tuple[HybridChunk, ...]:
        """Return deterministic payloads that the caller can explicitly index."""

        values = [
            hybrid
            for (chunk_tenant, _), (_, hybrid) in self._catalog.items()
            if tenant_id is None or chunk_tenant == tenant_id
        ]
        return tuple(sorted(values, key=lambda chunk: (chunk.tenant_id, chunk.chunk_id)))

    def get(
        self,
        chunk_id: str,
        principal: AccessContext,
        *,
        now: datetime | None = None,
    ) -> KnowledgeChunk | None:
        entry = self._catalog.get((principal.tenant_id, str(chunk_id)))
        if entry is None or not entry[0].is_visible_to(principal, now):
            return None
        return entry[0]

    def search(
        self,
        query: str,
        principal: AccessContext,
        *,
        top_k: int = 5,
        now: datetime | None = None,
        source_uris: Iterable[str] | None = None,
        knowledge_base_ids: Iterable[str] | None = None,
        published_before: datetime | str | None = None,
        published_after: datetime | str | None = None,
        latest_only: bool = True,
        semantic_scores: Mapping[str, float] | None = None,
        lexical_weight: float = 0.70,
        semantic_weight: float = 0.30,
    ) -> list[KnowledgeHit]:
        if top_k <= 0:
            return []
        if top_k > 100:
            raise ValueError("hybrid top_k must not exceed 100")
        if semantic_scores or not math.isclose(float(lexical_weight), 0.70) or not math.isclose(
            float(semantic_weight), 0.30
        ):
            raise UnsupportedHybridScoringError(
                "caller-supplied semantic scores/weights are unsupported; configure the hybrid backend"
            )

        instant = as_utc(now)
        allowed_sources = _clean_filter(source_uris)
        allowed_bases = _clean_filter(knowledge_base_ids)
        before = _as_bound(published_before)
        after = _as_bound(published_after)
        if before is not None and after is not None and after > before:
            raise ValueError("published_after must not be later than published_before")
        candidates = [
            entry
            for (tenant_id, _), entry in self._catalog.items()
            if tenant_id == principal.tenant_id
            and entry[0].is_visible_to(principal, instant)
            and (
                allowed_sources is None
                or entry[0].source_uri in allowed_sources
                or entry[0].logical_document_id in allowed_sources
            )
            and (
                allowed_bases is None
                or entry[1].knowledge_base_id in allowed_bases
            )
            and _published_in_window(entry[0], before=before, after=after)
        ]

        if latest_only:
            latest: dict[str, tuple[int, tuple[tuple[int, object], ...]]] = {}
            for chunk, _ in candidates:
                current = latest.get(chunk.logical_document_id)
                if current is None or chunk.version_key > current:
                    latest[chunk.logical_document_id] = chunk.version_key
            candidates = [
                entry
                for entry in candidates
                if entry[0].version_key == latest[entry[0].logical_document_id]
            ]

        allowed_chunk_ids = frozenset(chunk.chunk_id for chunk, _ in candidates)
        if not allowed_chunk_ids:
            return []
        if len(allowed_chunk_ids) > MAX_ALLOWED_CHUNK_IDS:
            raise HybridKnowledgeScopeTooLargeError(
                f"authorized candidate scope has {len(allowed_chunk_ids)} chunks; "
                f"maximum is {MAX_ALLOWED_CHUNK_IDS}"
            )

        # Reserve part of the caller's final budget for adjacent chunks while
        # retaining at least one ranked seed.
        seed_top_k = top_k
        if self.neighbor_window:
            seed_top_k = max(1, top_k // (1 + 2 * self.neighbor_window))
        candidate_k = min(
            500,
            max(seed_top_k, top_k, seed_top_k * self.candidate_multiplier),
        )
        request = HybridSearchRequest(
            query=query,
            tenant_id=principal.tenant_id,
            acl_principals=principal.acl_tokens,
            knowledge_base_ids=allowed_bases,
            allowed_chunk_ids=allowed_chunk_ids,
            top_k=seed_top_k,
            candidate_k=candidate_k,
            rerank=self.rerank,
            neighbor_window=self.neighbor_window,
            max_expanded_hits=top_k,
        )
        response = self.backend.search(request)
        expected_filters = AppliedRetrievalFilters.from_request(request)
        if response.filters_applied_before_ranking != expected_filters:
            raise HybridCatalogMismatchError(
                "retrieval backend did not attest the exact pre-ranking filters"
            )

        results: list[KnowledgeHit] = []
        seen: set[str] = set()
        for result in response.hits:
            chunk_id = result.chunk.chunk_id
            if chunk_id in seen:
                raise HybridCatalogMismatchError("retrieval backend returned a duplicate chunk")
            seen.add(chunk_id)
            entry = self._catalog.get((principal.tenant_id, chunk_id))
            if entry is None or chunk_id not in allowed_chunk_ids:
                raise HybridCatalogMismatchError(
                    "retrieval backend returned a chunk outside the authoritative catalog scope"
                )
            chunk, expected_hybrid = entry
            if result.chunk != expected_hybrid:
                raise HybridCatalogMismatchError(
                    f"retrieval backend payload for {chunk_id!r} is stale or inconsistent"
                )
            if not chunk.is_visible_to(principal, instant):
                raise HybridCatalogMismatchError(
                    "retrieval backend returned a chunk that is no longer visible"
                )
            lexical = lexical_match(query, chunk.text)
            results.append(
                KnowledgeHit(
                    chunk=chunk,
                    score=result.score,
                    lexical_score=lexical.score,
                    semantic_score=_bounded(result.reranker_score),
                    matched_terms=lexical.matched_terms,
                )
            )
        return results


__all__ = [
    "MAX_ALLOWED_CHUNK_IDS",
    "HybridCatalogMismatchError",
    "HybridKnowledgeError",
    "HybridKnowledgeScopeTooLargeError",
    "HybridKnowledgeStore",
    "HybridSearchBackend",
    "UnsupportedHybridScoringError",
    "knowledge_to_hybrid_chunk",
]

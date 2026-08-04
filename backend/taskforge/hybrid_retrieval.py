"""Tenant-safe hybrid retrieval with an explainable offline fallback.

This module deliberately keeps retrieval authority in host code.  Tenant,
principal ACL, knowledge-base, and document-version predicates are applied to
the candidate set *before* either BM25 scoring or Qdrant prefetch/RRF fusion.

``DeterministicHashEmbedder`` and ``LexicalOverlapFallbackReranker`` exist for
offline tests and explicit degraded operation only.  They are not semantic
production models and must not be represented as such in product telemetry.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from typing import Any, Iterable, Literal, Protocol, Sequence, runtime_checkable
from uuid import UUID, uuid5

from pydantic import Field, field_validator, model_validator

from .domain import StrictModel
from .knowledge import tokenise

try:  # Keep the lexical fallback importable without the optional dependency.
    from qdrant_client import QdrantClient
    from qdrant_client import models as qdrant_models
except ImportError:  # pragma: no cover - exercised by explicit monkeypatch test.
    QdrantClient = None  # type: ignore[assignment,misc]
    qdrant_models = None  # type: ignore[assignment]


_POINT_NAMESPACE = UUID("31a673ae-807c-4e12-ad73-68e77f67f99e")


class HybridRetrievalError(RuntimeError):
    """Base error for explicit retrieval failures."""


class QdrantUnavailableError(HybridRetrievalError):
    """Raised when the requested Qdrant implementation is not installed."""


class QdrantBackendError(HybridRetrievalError):
    """Raised when a real Qdrant operation fails or has an unsafe shape."""


class EmbeddingContractError(HybridRetrievalError):
    """Raised when an embedder violates its declared vector contract."""


class RerankerContractError(HybridRetrievalError):
    """Raised when a reranker is missing or returns invalid scores."""


def _clean_required(value: object, field_name: str) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


def _clean_string_set(value: object, field_name: str, *, allow_empty: bool) -> frozenset[str]:
    if isinstance(value, str):
        values: Iterable[object] = (value,)
    elif value is None:
        values = ()
    else:
        values = value  # type: ignore[assignment]
    cleaned = frozenset(str(item).strip() for item in values if str(item).strip())
    if not cleaned and not allow_empty:
        raise ValueError(f"{field_name} must contain at least one non-empty value")
    return cleaned


class HybridChunk(StrictModel):
    """A versioned chunk and the host-selected principals allowed to read it."""

    chunk_id: str = Field(min_length=1, max_length=512)
    tenant_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=2_000_000)
    source_uri: str = Field(min_length=1, max_length=2_048)
    document_id: str = Field(min_length=1, max_length=512)
    knowledge_base_id: str = Field(default="default", min_length=1, max_length=256)
    version: str = Field(default="1", min_length=1, max_length=128)
    version_order: int = Field(default=1, ge=0)
    acl_principals: frozenset[str] = Field(min_length=1, max_length=256)
    previous_chunk_id: str | None = Field(default=None, min_length=1, max_length=512)
    next_chunk_id: str | None = Field(default=None, min_length=1, max_length=512)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "chunk_id",
        "tenant_id",
        "text",
        "source_uri",
        "document_id",
        "knowledge_base_id",
        "version",
        "previous_chunk_id",
        "next_chunk_id",
        mode="before",
    )
    @classmethod
    def clean_strings(cls, value: object, info: Any) -> object:
        if value is None and info.field_name in {"previous_chunk_id", "next_chunk_id"}:
            return None
        return _clean_required(value, info.field_name)

    @field_validator("acl_principals", mode="before")
    @classmethod
    def clean_acl(cls, value: object) -> frozenset[str]:
        return _clean_string_set(value, "acl_principals", allow_empty=False)

    @field_validator("metadata")
    @classmethod
    def metadata_is_json_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be finite JSON data") from exc
        return dict(value)

    @model_validator(mode="after")
    def valid_links_and_searchable_text(self) -> "HybridChunk":
        if self.previous_chunk_id == self.chunk_id or self.next_chunk_id == self.chunk_id:
            raise ValueError("a chunk cannot be its own neighbor")
        if not tokenise(self.text):
            raise ValueError("text must contain at least one searchable token")
        return self


class HybridSearchRequest(StrictModel):
    """Trusted search scope.  ``top_k`` counts seeds, before neighbor expansion."""

    query: str = Field(min_length=1, max_length=32_000)
    tenant_id: str = Field(min_length=1, max_length=256)
    acl_principals: frozenset[str] = Field(min_length=1, max_length=256)
    versions: frozenset[str] | None = Field(default=None, max_length=128)
    version_orders: frozenset[int] | None = Field(default=None, max_length=128)
    knowledge_base_ids: frozenset[str] | None = Field(default=None, max_length=128)
    # Host-resolved visibility, validity-window, source, and latest-version
    # filtering can be expressed as a bounded allow-list.  This keeps those
    # predicates ahead of BM25 corpus statistics and Qdrant prefetch/RRF.
    allowed_chunk_ids: frozenset[str] | None = Field(default=None, max_length=20_000)
    top_k: int = Field(default=5, ge=1, le=100)
    candidate_k: int = Field(default=25, ge=1, le=500)
    rerank: bool = False
    neighbor_window: int = Field(default=0, ge=0, le=5)
    neighbor_score_decay: float = Field(default=0.75, gt=0.0, le=1.0)
    max_expanded_hits: int = Field(default=100, ge=1, le=1_000)

    @field_validator("query", "tenant_id", mode="before")
    @classmethod
    def clean_required_strings(cls, value: object, info: Any) -> str:
        return _clean_required(value, info.field_name)

    @field_validator("query")
    @classmethod
    def query_is_searchable(cls, value: str) -> str:
        if not tokenise(value):
            raise ValueError("query must contain at least one searchable token")
        return value

    @field_validator("acl_principals", mode="before")
    @classmethod
    def clean_principals(cls, value: object) -> frozenset[str]:
        return _clean_string_set(value, "acl_principals", allow_empty=False)

    @field_validator("versions", "knowledge_base_ids", "allowed_chunk_ids", mode="before")
    @classmethod
    def clean_optional_sets(cls, value: object, info: Any) -> frozenset[str] | None:
        if value is None:
            return None
        return _clean_string_set(value, info.field_name, allow_empty=False)

    @field_validator("version_orders", mode="before")
    @classmethod
    def clean_version_orders(cls, value: object) -> frozenset[int] | None:
        if value is None:
            return None
        cleaned = frozenset(int(item) for item in value)  # type: ignore[arg-type]
        if not cleaned or any(item < 0 for item in cleaned):
            raise ValueError("version_orders must contain non-negative integers")
        return cleaned

    @model_validator(mode="after")
    def budgets_are_consistent(self) -> "HybridSearchRequest":
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        if self.max_expanded_hits < self.top_k:
            raise ValueError("max_expanded_hits must be greater than or equal to top_k")
        return self


class AppliedRetrievalFilters(StrictModel):
    tenant_id: str
    acl_principals: list[str]
    versions: list[str] | None = None
    version_orders: list[int] | None = None
    knowledge_base_ids: list[str] | None = None
    allowed_chunk_count: int | None = Field(default=None, ge=1, le=20_000)
    allowed_chunk_ids_sha256: str | None = Field(default=None, min_length=64, max_length=64)

    @classmethod
    def from_request(cls, request: HybridSearchRequest) -> "AppliedRetrievalFilters":
        allowed_ids = request.allowed_chunk_ids
        allowed_hash = None
        if allowed_ids is not None:
            allowed_hash = hashlib.sha256(
                "\0".join(sorted(allowed_ids)).encode("utf-8")
            ).hexdigest()
        return cls(
            tenant_id=request.tenant_id,
            acl_principals=sorted(request.acl_principals),
            versions=None if request.versions is None else sorted(request.versions),
            version_orders=None if request.version_orders is None else sorted(request.version_orders),
            knowledge_base_ids=(
                None if request.knowledge_base_ids is None else sorted(request.knowledge_base_ids)
            ),
            allowed_chunk_count=None if allowed_ids is None else len(allowed_ids),
            allowed_chunk_ids_sha256=allowed_hash,
        )


class BM25TermContribution(StrictModel):
    term: str
    term_frequency: int = Field(ge=0)
    document_frequency: int = Field(ge=0)
    inverse_document_frequency: float
    length_normalized_tf: float
    contribution: float

    @field_validator("inverse_document_frequency", "length_normalized_tf", "contribution")
    @classmethod
    def finite_scores(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("BM25 explanation values must be finite")
        return value


class BM25Explanation(StrictModel):
    corpus_size_after_filters: int = Field(ge=1)
    document_length: int = Field(ge=0)
    average_document_length: float = Field(ge=0)
    k1: float = Field(gt=0)
    b: float = Field(ge=0, le=1)
    terms: list[BM25TermContribution]


RetrievalSource = Literal[
    "python_bm25",
    "qdrant_server_rrf",
    "fallback_lexical_rerank",
    "adjacent_chunk",
]


class HybridSearchHit(StrictModel):
    chunk: HybridChunk
    rank: int = Field(ge=1)
    score: float
    base_score: float
    reranker_score: float | None = None
    retrieval_sources: list[RetrievalSource] = Field(min_length=1)
    neighbor_of_chunk_id: str | None = None
    neighbor_distance: int | None = Field(default=None, ge=1, le=5)
    bm25_explanation: BM25Explanation | None = None

    @field_validator("score", "base_score", "reranker_score")
    @classmethod
    def scores_are_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("retrieval scores must be finite")
        return value

    @model_validator(mode="after")
    def neighbor_fields_are_paired(self) -> "HybridSearchHit":
        if (self.neighbor_of_chunk_id is None) != (self.neighbor_distance is None):
            raise ValueError("neighbor origin and distance must be set together")
        return self


class HybridSearchResponse(StrictModel):
    backend: Literal["python_bm25", "qdrant", "qdrant_local"]
    collection_name: str | None = None
    query: str
    filters_applied_before_ranking: AppliedRetrievalFilters
    seed_count: int = Field(ge=0)
    expanded_neighbor_count: int = Field(ge=0)
    hits: list[HybridSearchHit]


@runtime_checkable
class DenseEmbedder(Protocol):
    """Provider-neutral dense embedding contract."""

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

    def embed_query(self, text: str) -> Sequence[float]: ...


@runtime_checkable
class Reranker(Protocol):
    """Provider-neutral cross-document score contract."""

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]: ...


class DeterministicHashEmbedder:
    """Non-semantic feature hashing for tests and explicit degraded mode only."""

    def __init__(self, dimension: int = 64) -> None:
        if dimension < 8 or dimension > 65_536:
            raise ValueError("dimension must be between 8 and 65536")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token, frequency in Counter(tokenise(text)).items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "big") % self._dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(float(frequency)))
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector


class LexicalOverlapFallbackReranker:
    """Non-semantic lexical reranker for offline tests/degraded operation only."""

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        query_terms = frozenset(tokenise(query))
        if not query_terms:
            return [0.0 for _ in documents]
        scores: list[float] = []
        for document in documents:
            document_terms = frozenset(tokenise(document))
            overlap = len(query_terms.intersection(document_terms))
            coverage = overlap / len(query_terms)
            union = len(query_terms.union(document_terms))
            jaccard = overlap / union if union else 0.0
            scores.append(0.8 * coverage + 0.2 * jaccard)
        return scores


def _matches_scope(chunk: HybridChunk, request: HybridSearchRequest) -> bool:
    """The shared pre-ranking authorization/version predicate."""

    return bool(
        chunk.tenant_id == request.tenant_id
        and chunk.acl_principals.intersection(request.acl_principals)
        and (request.versions is None or chunk.version in request.versions)
        and (request.version_orders is None or chunk.version_order in request.version_orders)
        and (
            request.knowledge_base_ids is None
            or chunk.knowledge_base_id in request.knowledge_base_ids
        )
        and (
            request.allowed_chunk_ids is None
            or chunk.chunk_id in request.allowed_chunk_ids
        )
    )


class BM25Index:
    """Pure Python Okapi BM25 whose corpus is first reduced by trusted scope."""

    def __init__(
        self,
        chunks: Iterable[HybridChunk] = (),
        *,
        k1: float = 1.5,
        b: float = 0.75,
        reranker: Reranker | None = None,
    ) -> None:
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be a finite positive number")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be a finite number between 0 and 1")
        self.k1 = float(k1)
        self.b = float(b)
        self.reranker = reranker
        self._chunks: dict[tuple[str, str], HybridChunk] = {}
        for chunk in chunks:
            self.upsert(chunk)

    def upsert(self, chunk: HybridChunk) -> None:
        self._chunks[(chunk.tenant_id, chunk.chunk_id)] = chunk

    def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        # Security and version scope is established before corpus statistics,
        # so inaccessible documents cannot consume candidate slots or affect IDF.
        eligible = [chunk for chunk in self._chunks.values() if _matches_scope(chunk, request)]
        if not eligible:
            return _response("python_bm25", None, request, [], 0)

        token_counts = {chunk.chunk_id: Counter(tokenise(chunk.text)) for chunk in eligible}
        lengths = {chunk_id: sum(counts.values()) for chunk_id, counts in token_counts.items()}
        average_length = sum(lengths.values()) / len(eligible)
        document_frequencies: Counter[str] = Counter()
        for counts in token_counts.values():
            document_frequencies.update(counts.keys())

        query_terms = tuple(dict.fromkeys(tokenise(request.query)))
        scored: list[HybridSearchHit] = []
        corpus_size = len(eligible)
        for chunk in eligible:
            counts = token_counts[chunk.chunk_id]
            document_length = lengths[chunk.chunk_id]
            term_scores: list[BM25TermContribution] = []
            score = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                df = document_frequencies[term]
                idf = math.log(1.0 + (corpus_size - df + 0.5) / (df + 0.5))
                normalizer = self.k1 * (
                    1.0 - self.b
                    + self.b * (document_length / average_length if average_length else 0.0)
                )
                normalized_tf = frequency * (self.k1 + 1.0) / (frequency + normalizer)
                contribution = idf * normalized_tf
                score += contribution
                term_scores.append(
                    BM25TermContribution(
                        term=term,
                        term_frequency=frequency,
                        document_frequency=df,
                        inverse_document_frequency=idf,
                        length_normalized_tf=normalized_tf,
                        contribution=contribution,
                    )
                )
            if score <= 0:
                continue
            scored.append(
                HybridSearchHit(
                    chunk=chunk,
                    rank=1,
                    score=score,
                    base_score=score,
                    retrieval_sources=["python_bm25"],
                    bm25_explanation=BM25Explanation(
                        corpus_size_after_filters=corpus_size,
                        document_length=document_length,
                        average_document_length=average_length,
                        k1=self.k1,
                        b=self.b,
                        terms=term_scores,
                    ),
                )
            )
        scored.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        candidates = scored[: request.candidate_k]
        candidates = _apply_reranker(candidates, request, self.reranker)
        seeds = _rerank_positions(candidates[: request.top_k])
        expanded = _expand_from_catalog(
            seeds,
            {chunk.chunk_id: chunk for chunk in eligible},
            request,
        )
        return _response(
            "python_bm25",
            None,
            request,
            expanded,
            max(0, len(expanded) - len(seeds)),
        )


def _sparse_vector(text: str) -> tuple[list[int], list[float]]:
    buckets: Counter[int] = Counter()
    for token, frequency in Counter(tokenise(text)).items():
        index = int.from_bytes(
            hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big"
        ) % 2_147_483_647
        buckets[index] += frequency
    ordered = sorted(buckets.items())
    return [index for index, _ in ordered], [float(value) for _, value in ordered]


def _validated_vector(values: Sequence[float], dimension: int, label: str) -> list[float]:
    vector = [float(value) for value in values]
    if len(vector) != dimension:
        raise EmbeddingContractError(
            f"{label} returned dimension {len(vector)}; expected {dimension}"
        )
    if not all(math.isfinite(value) for value in vector):
        raise EmbeddingContractError(f"{label} returned a non-finite vector")
    return vector


class QdrantHybridIndex:
    """Real Qdrant named-vector index using server-side dense/sparse RRF."""

    def __init__(
        self,
        client: Any,
        *,
        collection_name: str,
        embedder: DenseEmbedder,
        reranker: Reranker | None = None,
        backend_label: Literal["qdrant", "qdrant_local"] = "qdrant",
        create_if_missing: bool = True,
    ) -> None:
        if qdrant_models is None:
            raise QdrantUnavailableError(
                "qdrant-client is required for QdrantHybridIndex; install the rag dependency"
            )
        if client is None:
            raise QdrantUnavailableError("a connected Qdrant client is required")
        self.client = client
        self.collection_name = _clean_required(collection_name, "collection_name")
        self.embedder = embedder
        self.reranker = reranker
        self.backend_label = backend_label
        dimension = int(embedder.dimension)
        if dimension < 1 or dimension > 65_536:
            raise EmbeddingContractError("embedder dimension must be between 1 and 65536")
        self.dimension = dimension
        self._ensure_collection(create_if_missing=create_if_missing)

    @classmethod
    def in_memory(
        cls,
        *,
        collection_name: str,
        embedder: DenseEmbedder,
        reranker: Reranker | None = None,
    ) -> "QdrantHybridIndex":
        """Create a genuine qdrant-client local in-memory collection."""

        if QdrantClient is None or qdrant_models is None:
            raise QdrantUnavailableError(
                "qdrant-client is required for local/in-memory Qdrant"
            )
        return cls(
            QdrantClient(location=":memory:"),
            collection_name=collection_name,
            embedder=embedder,
            reranker=reranker,
            backend_label="qdrant_local",
        )

    def _ensure_collection(self, *, create_if_missing: bool) -> None:
        try:
            exists = bool(self.client.collection_exists(self.collection_name))
            if not exists:
                if not create_if_missing:
                    raise QdrantBackendError(
                        f"Qdrant collection {self.collection_name!r} does not exist"
                    )
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": qdrant_models.VectorParams(
                            size=self.dimension,
                            distance=qdrant_models.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={"sparse": qdrant_models.SparseVectorParams()},
                )
            info = self.client.get_collection(self.collection_name)
            dense = info.config.params.vectors
            sparse = info.config.params.sparse_vectors
            if not isinstance(dense, dict) or "dense" not in dense:
                raise QdrantBackendError("collection lacks the required named dense vector")
            if int(dense["dense"].size) != self.dimension:
                raise QdrantBackendError(
                    "collection dense dimension does not match the configured embedder"
                )
            if not isinstance(sparse, dict) or "sparse" not in sparse:
                raise QdrantBackendError("collection lacks the required named sparse vector")
        except QdrantBackendError:
            raise
        except Exception as exc:
            raise QdrantBackendError(
                f"failed to initialize Qdrant collection {self.collection_name!r}: {exc}"
            ) from exc

    def upsert(self, chunks: Sequence[HybridChunk]) -> int:
        if not chunks:
            return 0
        keys = [(chunk.tenant_id, chunk.chunk_id) for chunk in chunks]
        if len(keys) != len(set(keys)):
            raise ValueError("a Qdrant upsert batch cannot contain duplicate tenant/chunk IDs")
        try:
            raw_vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
        except Exception as exc:
            raise EmbeddingContractError(f"document embedding failed: {exc}") from exc
        if len(raw_vectors) != len(chunks):
            raise EmbeddingContractError(
                "embed_documents must return exactly one vector per chunk"
            )

        points = []
        for chunk, raw_vector in zip(chunks, raw_vectors, strict=True):
            dense = _validated_vector(raw_vector, self.dimension, "embed_documents")
            indices, values = _sparse_vector(chunk.text)
            points.append(
                qdrant_models.PointStruct(
                    id=str(uuid5(_POINT_NAMESPACE, f"{chunk.tenant_id}\0{chunk.chunk_id}")),
                    vector={
                        "dense": dense,
                        "sparse": qdrant_models.SparseVector(indices=indices, values=values),
                    },
                    payload=chunk.model_dump(mode="json"),
                )
            )
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
        except Exception as exc:
            raise QdrantBackendError(f"Qdrant upsert failed: {exc}") from exc
        return len(points)

    def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        try:
            dense_query = _validated_vector(
                self.embedder.embed_query(request.query), self.dimension, "embed_query"
            )
        except EmbeddingContractError:
            raise
        except Exception as exc:
            raise EmbeddingContractError(f"query embedding failed: {exc}") from exc
        sparse_indices, sparse_values = _sparse_vector(request.query)
        if not sparse_indices:
            raise ValueError("query must contain at least one searchable token")
        access_filter = self._filter(request)
        try:
            result = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    qdrant_models.Prefetch(
                        query=dense_query,
                        using="dense",
                        filter=access_filter,
                        limit=request.candidate_k,
                    ),
                    qdrant_models.Prefetch(
                        query=qdrant_models.SparseVector(
                            indices=sparse_indices,
                            values=sparse_values,
                        ),
                        using="sparse",
                        filter=access_filter,
                        limit=request.candidate_k,
                    ),
                ],
                query=qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.RRF),
                # Defense in depth: the exact same trusted filter is also on
                # the fusion stage, never applied as post-ranking Python logic.
                query_filter=access_filter,
                limit=request.candidate_k,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise QdrantBackendError(f"Qdrant hybrid RRF query failed: {exc}") from exc

        candidates: list[HybridSearchHit] = []
        for point in result.points:
            chunk = self._payload_chunk(point.payload)
            # Fail closed if a backend/proxy ever violates its own filter.
            if not _matches_scope(chunk, request):
                raise QdrantBackendError("Qdrant returned a point outside the trusted scope")
            score = float(point.score)
            if not math.isfinite(score):
                raise QdrantBackendError("Qdrant returned a non-finite fusion score")
            candidates.append(
                HybridSearchHit(
                    chunk=chunk,
                    rank=1,
                    score=score,
                    base_score=score,
                    retrieval_sources=["qdrant_server_rrf"],
                )
            )
        candidates = _apply_reranker(candidates, request, self.reranker)
        seeds = _rerank_positions(candidates[: request.top_k])
        expanded = self._expand_neighbors(seeds, request)
        return _response(
            self.backend_label,
            self.collection_name,
            request,
            expanded,
            max(0, len(expanded) - len(seeds)),
        )

    def _filter(
        self,
        request: HybridSearchRequest,
        *,
        chunk_ids: frozenset[str] | None = None,
    ) -> Any:
        must: list[Any] = [
            qdrant_models.FieldCondition(
                key="tenant_id", match=qdrant_models.MatchValue(value=request.tenant_id)
            ),
            qdrant_models.FieldCondition(
                key="acl_principals",
                match=qdrant_models.MatchAny(any=sorted(request.acl_principals)),
            ),
        ]
        if request.versions is not None:
            must.append(
                qdrant_models.FieldCondition(
                    key="version", match=qdrant_models.MatchAny(any=sorted(request.versions))
                )
            )
        if request.version_orders is not None:
            must.append(
                qdrant_models.FieldCondition(
                    key="version_order",
                    match=qdrant_models.MatchAny(any=sorted(request.version_orders)),
                )
            )
        if request.knowledge_base_ids is not None:
            must.append(
                qdrant_models.FieldCondition(
                    key="knowledge_base_id",
                    match=qdrant_models.MatchAny(any=sorted(request.knowledge_base_ids)),
                )
            )
        effective_chunk_ids = request.allowed_chunk_ids
        if chunk_ids is not None:
            effective_chunk_ids = (
                chunk_ids
                if effective_chunk_ids is None
                else effective_chunk_ids.intersection(chunk_ids)
            )
        if effective_chunk_ids is not None:
            if not effective_chunk_ids:
                raise ValueError("chunk ID filters do not overlap")
            must.append(
                qdrant_models.FieldCondition(
                    key="chunk_id",
                    match=qdrant_models.MatchAny(any=sorted(effective_chunk_ids)),
                )
            )
        return qdrant_models.Filter(must=must)

    @staticmethod
    def _payload_chunk(payload: dict[str, Any] | None) -> HybridChunk:
        if not isinstance(payload, dict):
            raise QdrantBackendError("Qdrant point is missing its chunk payload")
        try:
            return HybridChunk.model_validate(payload)
        except Exception as exc:
            raise QdrantBackendError(f"Qdrant returned an invalid chunk payload: {exc}") from exc

    def _fetch_chunks(
        self, chunk_ids: frozenset[str], request: HybridSearchRequest
    ) -> dict[str, HybridChunk]:
        if request.allowed_chunk_ids is not None:
            chunk_ids = chunk_ids.intersection(request.allowed_chunk_ids)
        if not chunk_ids:
            return {}
        records: list[Any] = []
        offset: Any = None
        try:
            while True:
                batch, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=self._filter(request, chunk_ids=chunk_ids),
                    limit=min(max(len(chunk_ids), 1), 1_000),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                records.extend(batch)
                if offset is None or len(records) >= len(chunk_ids):
                    break
        except Exception as exc:
            raise QdrantBackendError(f"Qdrant neighbor lookup failed: {exc}") from exc
        chunks: dict[str, HybridChunk] = {}
        for record in records:
            chunk = self._payload_chunk(record.payload)
            if not _matches_scope(chunk, request):
                raise QdrantBackendError("Qdrant returned an out-of-scope neighbor")
            chunks[chunk.chunk_id] = chunk
        return chunks

    def _expand_neighbors(
        self, seeds: list[HybridSearchHit], request: HybridSearchRequest
    ) -> list[HybridSearchHit]:
        if request.neighbor_window == 0 or not seeds:
            return seeds
        output = list(seeds)
        seen = {hit.chunk.chunk_id for hit in seeds}
        # (root seed, current chunk, direction, next target, distance)
        frontier: list[tuple[HybridSearchHit, HybridChunk, str, str, int]] = []
        for seed in seeds:
            if seed.chunk.previous_chunk_id:
                frontier.append((seed, seed.chunk, "previous", seed.chunk.previous_chunk_id, 1))
            if seed.chunk.next_chunk_id:
                frontier.append((seed, seed.chunk, "next", seed.chunk.next_chunk_id, 1))
        traversed: set[tuple[str, str, str]] = set()
        while frontier and len(output) < request.max_expanded_hits:
            target_ids = frozenset(item[3] for item in frontier)
            fetched = self._fetch_chunks(target_ids, request)
            next_frontier: list[tuple[HybridSearchHit, HybridChunk, str, str, int]] = []
            for root, _current, direction, target_id, distance in frontier:
                state = (root.chunk.chunk_id, direction, target_id)
                if state in traversed:
                    continue
                traversed.add(state)
                neighbor = fetched.get(target_id)
                if neighbor is None:
                    continue
                # Links cannot cross document or version even if a malformed
                # payload points at another otherwise-readable chunk.
                if (
                    neighbor.document_id != root.chunk.document_id
                    or neighbor.version != root.chunk.version
                    or neighbor.version_order != root.chunk.version_order
                ):
                    continue
                if neighbor.chunk_id not in seen and len(output) < request.max_expanded_hits:
                    seen.add(neighbor.chunk_id)
                    neighbor_score = root.score * request.neighbor_score_decay**distance
                    output.append(
                        HybridSearchHit(
                            chunk=neighbor,
                            rank=1,
                            score=neighbor_score,
                            base_score=neighbor_score,
                            retrieval_sources=["adjacent_chunk"],
                            neighbor_of_chunk_id=root.chunk.chunk_id,
                            neighbor_distance=distance,
                        )
                    )
                if distance < request.neighbor_window:
                    next_id = (
                        neighbor.previous_chunk_id
                        if direction == "previous"
                        else neighbor.next_chunk_id
                    )
                    if next_id:
                        next_frontier.append((root, neighbor, direction, next_id, distance + 1))
            frontier = next_frontier
        return _rerank_positions(output)


def _apply_reranker(
    hits: list[HybridSearchHit],
    request: HybridSearchRequest,
    reranker: Reranker | None,
) -> list[HybridSearchHit]:
    if not request.rerank:
        return _rerank_positions(hits)
    if reranker is None:
        raise RerankerContractError("rerank=True requires a configured Reranker")
    try:
        raw_scores = reranker.score(request.query, [hit.chunk.text for hit in hits])
    except Exception as exc:
        raise RerankerContractError(f"reranker failed: {exc}") from exc
    if len(raw_scores) != len(hits):
        raise RerankerContractError("reranker must return exactly one score per candidate")
    rescored: list[HybridSearchHit] = []
    for hit, raw_score in zip(hits, raw_scores, strict=True):
        score = float(raw_score)
        if not math.isfinite(score):
            raise RerankerContractError("reranker returned a non-finite score")
        sources = list(hit.retrieval_sources)
        if isinstance(reranker, LexicalOverlapFallbackReranker):
            sources.append("fallback_lexical_rerank")
        rescored.append(
            hit.model_copy(
                update={
                    "score": score,
                    "reranker_score": score,
                    "retrieval_sources": sources,
                }
            )
        )
    rescored.sort(key=lambda hit: (-hit.score, -hit.base_score, hit.chunk.chunk_id))
    return _rerank_positions(rescored)


def _expand_from_catalog(
    seeds: list[HybridSearchHit],
    catalog: dict[str, HybridChunk],
    request: HybridSearchRequest,
) -> list[HybridSearchHit]:
    if request.neighbor_window == 0 or not seeds:
        return seeds
    output = list(seeds)
    seen = {hit.chunk.chunk_id for hit in seeds}
    for seed in seeds:
        for direction in ("previous", "next"):
            target_id = (
                seed.chunk.previous_chunk_id if direction == "previous" else seed.chunk.next_chunk_id
            )
            distance = 1
            path_seen: set[str] = set()
            while (
                target_id
                and distance <= request.neighbor_window
                and len(output) < request.max_expanded_hits
                and target_id not in path_seen
            ):
                path_seen.add(target_id)
                neighbor = catalog.get(target_id)
                if (
                    neighbor is None
                    or neighbor.document_id != seed.chunk.document_id
                    or neighbor.version != seed.chunk.version
                    or neighbor.version_order != seed.chunk.version_order
                ):
                    break
                if neighbor.chunk_id not in seen:
                    seen.add(neighbor.chunk_id)
                    score = seed.score * request.neighbor_score_decay**distance
                    output.append(
                        HybridSearchHit(
                            chunk=neighbor,
                            rank=1,
                            score=score,
                            base_score=score,
                            retrieval_sources=["adjacent_chunk"],
                            neighbor_of_chunk_id=seed.chunk.chunk_id,
                            neighbor_distance=distance,
                        )
                    )
                target_id = (
                    neighbor.previous_chunk_id
                    if direction == "previous"
                    else neighbor.next_chunk_id
                )
                distance += 1
    return _rerank_positions(output)


def _rerank_positions(hits: Sequence[HybridSearchHit]) -> list[HybridSearchHit]:
    return [hit.model_copy(update={"rank": index}) for index, hit in enumerate(hits, start=1)]


def _response(
    backend: Literal["python_bm25", "qdrant", "qdrant_local"],
    collection_name: str | None,
    request: HybridSearchRequest,
    hits: list[HybridSearchHit],
    expanded_neighbors: int,
) -> HybridSearchResponse:
    seed_count = sum(hit.neighbor_of_chunk_id is None for hit in hits)
    return HybridSearchResponse(
        backend=backend,
        collection_name=collection_name,
        query=request.query,
        filters_applied_before_ranking=AppliedRetrievalFilters.from_request(request),
        seed_count=seed_count,
        expanded_neighbor_count=expanded_neighbors,
        hits=hits,
    )


__all__ = [
    "AppliedRetrievalFilters",
    "BM25Explanation",
    "BM25Index",
    "BM25TermContribution",
    "DenseEmbedder",
    "DeterministicHashEmbedder",
    "EmbeddingContractError",
    "HybridChunk",
    "HybridRetrievalError",
    "HybridSearchHit",
    "HybridSearchRequest",
    "HybridSearchResponse",
    "LexicalOverlapFallbackReranker",
    "QdrantBackendError",
    "QdrantHybridIndex",
    "QdrantUnavailableError",
    "Reranker",
    "RerankerContractError",
]

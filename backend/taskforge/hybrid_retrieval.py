"""Tenant-safe hybrid retrieval with an explainable offline fallback.

This module deliberately keeps retrieval authority in host code.  Tenant,
principal ACL, knowledge-base, and document-version predicates are applied to
the candidate set *before* either BM25 scoring or Qdrant prefetch/RRF fusion.

``DeterministicHashEmbedder`` and ``LexicalOverlapFallbackReranker`` exist for
offline tests and explicit degraded operation only.  They are not semantic
production models and must not be represented as such in product telemetry.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
import sqlite3
import struct
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import UUID, uuid5

from pydantic import Field, field_validator, model_validator

from .domain import StrictModel
from .knowledge import tokenise
from .tatqa_reranker import TATQADomainReranker

try:  # Keep the lexical fallback importable without the optional dependency.
    from qdrant_client import QdrantClient
    from qdrant_client import models as qdrant_models
except ImportError:  # pragma: no cover - exercised by explicit monkeypatch test.
    QdrantClient = None  # type: ignore[assignment,misc]
    qdrant_models = None  # type: ignore[assignment]

try:  # Keep everything importable without the optional semantic dependency.
    from fastembed import SparseTextEmbedding, TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder
except ImportError:  # pragma: no cover - exercised by explicit monkeypatch test.
    SparseTextEmbedding = None  # type: ignore[assignment,misc]
    TextEmbedding = None  # type: ignore[assignment,misc]
    TextCrossEncoder = None  # type: ignore[assignment,misc]

try:  # NumPy is a transitive dependency of the semantic/Qdrant extra.
    import numpy as np
except ImportError:  # pragma: no cover - semantic extra is absent.
    np = None  # type: ignore[assignment]


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
    def valid_links_and_searchable_text(self) -> HybridChunk:
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
    # Host-resolved document/report scope.  This is distinct from relevance:
    # callers may constrain retrieval to documents explicitly selected in the
    # UI or supplied by a benchmark's input context, but must never derive it
    # from answer/evidence labels.
    parent_document_ids: frozenset[str] | None = Field(default=None, max_length=128)
    # Host-resolved visibility, validity-window, source, and latest-version
    # filtering can be expressed as a bounded allow-list.  This keeps those
    # predicates ahead of BM25 corpus statistics and Qdrant prefetch/RRF.
    allowed_chunk_ids: frozenset[str] | None = Field(default=None, max_length=20_000)
    top_k: int = Field(default=5, ge=1, le=100)
    candidate_k: int = Field(default=25, ge=1, le=500)
    max_chunks_per_document: int | None = Field(default=None, ge=1, le=100)
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

    @field_validator(
        "versions",
        "knowledge_base_ids",
        "parent_document_ids",
        "allowed_chunk_ids",
        mode="before",
    )
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
    def budgets_are_consistent(self) -> HybridSearchRequest:
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
    parent_document_ids: list[str] | None = None
    allowed_chunk_count: int | None = Field(default=None, ge=1, le=20_000)
    allowed_chunk_ids_sha256: str | None = Field(default=None, min_length=64, max_length=64)

    @classmethod
    def from_request(cls, request: HybridSearchRequest) -> AppliedRetrievalFilters:
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
            parent_document_ids=(
                None
                if request.parent_document_ids is None
                else sorted(request.parent_document_ids)
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
    "qdrant_dense",
    "bm25_dense_rrf",
    "multi_query_rrf",
    "source_coverage_rrf",
    "qdrant_server_rrf",
    "learned_cross_encoder_rerank",
    "fallback_lexical_rerank",
    "tatqa_feature_rerank",
    "tatqa_numeric_scan",
    "tatqa_structured_count",
    "tatqa_structured_arithmetic",
    "tatqa_structured_multi_span",
    "structured_lineage_candidate",
    "structured_lineage_pair_rerank",
    "fastembed_learned_sparse",
    "pgvector_exact",
    "pgvector_hnsw",
    "adjacent_chunk",
    "parent_child_retrieval",
    "parent_sibling_coverage",
    "context_sibling_coverage",
    "same_parent_evidence_closure",
    "multi_representation_rrf",
    "semantic_dense_candidate_tail",
    "graph_feature_rerank",
    "learned_graph_rerank",
]

RetrievalBackend = Literal[
    "python_bm25",
    "qdrant",
    "qdrant_local",
    "bm25_dense_rrf",
    "multi_query_rrf",
    "source_coverage_rrf",
    "profile_routed",
    "parent_child",
    "multi_representation_rrf",
    "fastembed_sparse",
    "candidate_tail_union",
    "in_memory_dense",
    "postgres_pgvector_exact",
    "postgres_pgvector_hnsw",
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
    def neighbor_fields_are_paired(self) -> HybridSearchHit:
        if (self.neighbor_of_chunk_id is None) != (self.neighbor_distance is None):
            raise ValueError("neighbor origin and distance must be set together")
        return self


class AdaptiveRerankDiagnostics(StrictModel):
    """Per-query audit record for a two-step reranker budget."""

    min_k: int = Field(ge=1, le=100)
    max_k: int = Field(ge=1, le=100)
    applied_k: int = Field(ge=0, le=100)
    escalated: bool
    top_score_margin: float | None = None
    margin_threshold: float = Field(ge=0.0)
    reason: Literal["high_confidence", "low_score_margin", "insufficient_candidates"]

    @field_validator("top_score_margin", "margin_threshold")
    @classmethod
    def margins_are_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("adaptive rerank margins must be finite")
        return value


class HybridSearchResponse(StrictModel):
    backend: RetrievalBackend
    collection_name: str | None = None
    query: str
    filters_applied_before_ranking: AppliedRetrievalFilters
    seed_count: int = Field(ge=0)
    expanded_neighbor_count: int = Field(ge=0)
    # Number of candidates emitted by each branch before the outer fusion
    # head is selected.  This is audit metadata, not a ranking input.
    raw_candidate_counts: dict[str, int] = Field(default_factory=dict)
    adaptive_rerank: AdaptiveRerankDiagnostics | None = None
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


class FastEmbedEmbedder:
    """Semantic dense embedder backed by a local ONNX fastembed model.

    Unlike :class:`DeterministicHashEmbedder`, this is a real semantic model;
    the ONNX artifact is downloaded once and cached under the fastembed cache.
    It is deliberately excluded from the offline M1 gate, so a caller must opt
    in explicitly and record the model in run provenance.
    """

    _CACHE_TABLE = "embeddings_v1"

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        cache_path: str | Path | None = None,
        cache_store: Any | None = None,
        model_cache_dir: str | Path | None = None,
        batch_size: int = 64,
    ) -> None:
        if TextEmbedding is None:
            raise EmbeddingContractError(
                "fastembed is required for the semantic embedder; install the semantic extra"
            )
        self._model_name = _clean_required(model_name, "model_name")
        if batch_size < 1 or batch_size > 1_024:
            raise ValueError("batch_size must be between 1 and 1024")
        self._batch_size = int(batch_size)
        self._cache_path = Path(cache_path).resolve() if cache_path is not None else None
        if self._cache_path is not None and cache_store is not None:
            raise EmbeddingContractError(
                "cache_path and cache_store are mutually exclusive"
            )
        self._cache_store = cache_store
        model_kwargs = {"model_name": model_name}
        if model_cache_dir is not None:
            model_kwargs["cache_dir"] = str(Path(model_cache_dir).resolve())
        self._model = TextEmbedding(**model_kwargs)
        # Query embeddings are content-addressed on disk, but keeping the
        # values used by a locked evaluation in memory avoids reopening the
        # ONNX runtime for every case.  ``warm_queries`` is an explicit
        # warm-up step; measured retrieval latency remains the search path.
        self._query_memory: dict[str, list[float]] = {}
        # fastembed exposes no stable .dimension; probe one embedding instead.
        self._dimension = len(next(iter(self._model.embed(["dimension probe"]))))
        if self._dimension < 1 or self._dimension > 65_536:
            raise EmbeddingContractError(
                "fastembed returned an invalid embedding dimension"
            )
        if self._cache_path is not None:
            self._initialize_cache()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def index_name(self) -> str:
        """Stable model-qualified name for an in-memory dense index.

        The name is telemetry and cache identity only; it deliberately does
        not alter the legacy FastEmbed ranking behavior.
        """

        if self._model_name.casefold() == "baai/bge-small-en-v1.5":
            return "knowledge-fastembed-bge-small-v1"
        return "knowledge-fastembed-dense-v1"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        normalized = [str(text) for text in texts]
        if self._cache_path is None and self._cache_store is None:
            return self._encode_documents(normalized)
        return self._cached_documents(normalized)

    def embed_query(self, text: str) -> list[float]:
        normalized = str(text)
        cached_memory = self._query_memory.get(normalized)
        if cached_memory is not None:
            return list(cached_memory)
        if self._cache_path is None and self._cache_store is None:
            vector = self._encode_query(normalized)
        else:
            vector = self._cached_query(normalized)
        self._query_memory[normalized] = list(vector)
        return vector

    def warm_queries(self, texts: Sequence[str]) -> int:
        """Load/encode a bounded query set before timed retrieval begins."""

        warmed = 0
        for text in dict.fromkeys(str(value) for value in texts):
            if text in self._query_memory:
                continue
            self.embed_query(text)
            warmed += 1
        return warmed

    @property
    def cache_path(self) -> Path | None:
        return self._cache_path

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _initialize_cache(self) -> None:
        if self._cache_path is None:  # pragma: no cover - guarded by caller.
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with closing(self._connect_cache()) as connection, connection:
                connection.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._CACHE_TABLE} (
                        cache_key TEXT PRIMARY KEY,
                        model_name TEXT NOT NULL,
                        embedding_kind TEXT NOT NULL,
                        text_sha256 TEXT NOT NULL,
                        dimension INTEGER NOT NULL,
                        vector BLOB NOT NULL
                    )
                    """
                )
        except (OSError, sqlite3.Error) as exc:
            raise EmbeddingContractError(
                f"failed to initialize embedding cache: {exc}"
            ) from exc

    def _connect_cache(self) -> sqlite3.Connection:
        if self._cache_path is None:  # pragma: no cover - internal contract.
            raise EmbeddingContractError("embedding cache is not configured")
        connection = sqlite3.connect(self._cache_path, timeout=60.0)
        connection.execute("PRAGMA busy_timeout = 60000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _cached_documents(self, texts: Sequence[str]) -> list[list[float]]:
        keys = [self._cache_identity("document", text) for text in texts]
        cached = self._load_cached(keys, embedding_kind="document")
        missing_by_key: dict[str, tuple[str, str]] = {}
        for (cache_key, text_sha256), text in zip(keys, texts, strict=True):
            if cache_key not in cached:
                missing_by_key.setdefault(cache_key, (text_sha256, text))
        missing = list(missing_by_key.items())
        for offset in range(0, len(missing), self._batch_size):
            batch = missing[offset : offset + self._batch_size]
            vectors = self._encode_documents([entry[1][1] for entry in batch])
            rows = []
            for (cache_key, (text_sha256, _)), vector in zip(
                batch, vectors, strict=True
            ):
                rows.append(
                    self._cache_row(
                        cache_key,
                        text_sha256,
                        "document",
                        vector,
                    )
                )
                cached[cache_key] = vector
            self._store_cached(rows)
        return [cached[cache_key] for cache_key, _ in keys]

    def _cached_query(self, text: str) -> list[float]:
        cached_memory = self._query_memory.get(text)
        if cached_memory is not None:
            return list(cached_memory)
        identity = self._cache_identity("query", text)
        cached = self._load_cached([identity], embedding_kind="query")
        if identity[0] in cached:
            return cached[identity[0]]
        vector = self._encode_query(text)
        self._store_cached(
            [self._cache_row(identity[0], identity[1], "query", vector)]
        )
        return vector

    def _load_cached(
        self,
        identities: Sequence[tuple[str, str]],
        *,
        embedding_kind: Literal["document", "query"],
    ) -> dict[str, list[float]]:
        if not identities:
            return {}
        expected = dict(identities)
        loaded: dict[str, list[float]] = {}
        if self._cache_store is not None:
            return self._cache_store.load(
                model_name=self._model_name,
                identities=identities,
                embedding_kind=embedding_kind,
                dimension=self._dimension,
            )
        try:
            with closing(self._connect_cache()) as connection, connection:
                keys = list(expected)
                for offset in range(0, len(keys), 400):
                    batch = keys[offset : offset + 400]
                    placeholders = ",".join("?" for _ in batch)
                    rows = connection.execute(
                        f"""
                        SELECT cache_key, model_name, embedding_kind,
                               text_sha256, dimension, vector
                        FROM {self._CACHE_TABLE}
                        WHERE cache_key IN ({placeholders})
                        """,
                        batch,
                    ).fetchall()
                    for row in rows:
                        cache_key, model_name, kind, text_sha256, dimension, blob = row
                        if (
                            model_name != self._model_name
                            or kind != embedding_kind
                            or text_sha256 != expected[cache_key]
                            or int(dimension) != self._dimension
                        ):
                            raise EmbeddingContractError(
                                "embedding cache metadata does not match the request"
                            )
                        loaded[cache_key] = self._decode_vector(blob)
        except EmbeddingContractError:
            raise
        except (OSError, sqlite3.Error, KeyError) as exc:
            raise EmbeddingContractError(f"failed to read embedding cache: {exc}") from exc
        return loaded

    def _store_cached(self, rows: Sequence[tuple[object, ...]]) -> None:
        if not rows:
            return
        if self._cache_store is not None:
            self._cache_store.store(rows)
            return
        try:
            with closing(self._connect_cache()) as connection, connection:
                connection.executemany(
                    f"""
                    INSERT INTO {self._CACHE_TABLE} (
                        cache_key, model_name, embedding_kind,
                        text_sha256, dimension, vector
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO NOTHING
                    """,
                    rows,
                )
        except (OSError, sqlite3.Error) as exc:
            raise EmbeddingContractError(f"failed to write embedding cache: {exc}") from exc

    def _cache_row(
        self,
        cache_key: str,
        text_sha256: str,
        embedding_kind: Literal["document", "query"],
        vector: Sequence[float],
    ) -> tuple[object, ...]:
        validated = _validated_vector(vector, self._dimension, "fastembed")
        blob = struct.pack(f"<{self._dimension}f", *validated)
        return (
            cache_key,
            self._model_name,
            embedding_kind,
            text_sha256,
            self._dimension,
            blob,
        )

    def _decode_vector(self, value: object) -> list[float]:
        if not isinstance(value, bytes) or len(value) != self._dimension * 4:
            raise EmbeddingContractError("embedding cache contains a corrupt vector")
        vector = list(struct.unpack(f"<{self._dimension}f", value))
        return _validated_vector(vector, self._dimension, "embedding cache")

    def _cache_identity(
        self,
        embedding_kind: Literal["document", "query"],
        text: str,
    ) -> tuple[str, str]:
        text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cache_key = hashlib.sha256(
            "\0".join((self._model_name, embedding_kind, text_sha256)).encode("utf-8")
        ).hexdigest()
        return cache_key, text_sha256

    def _encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = [
            self._float_list(vector)
            for vector in self._model.embed(texts, batch_size=self._batch_size)
        ]
        if len(vectors) != len(texts):
            raise EmbeddingContractError(
                "fastembed must return exactly one vector per document"
            )
        return [
            _validated_vector(vector, self._dimension, "fastembed document")
            for vector in vectors
        ]

    def _encode_query(self, text: str) -> list[float]:
        try:
            vector = self._float_list(next(iter(self._model.query_embed(text))))
        except StopIteration as exc:
            raise EmbeddingContractError("fastembed returned no query vector") from exc
        return _validated_vector(vector, self._dimension, "fastembed query")

    @staticmethod
    def _float_list(vector: Any) -> list[float]:
        return [float(value) for value in vector]


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


class TATQAFeatureReranker:
    """Deterministic TAT-QA ranking features for an offline ablation.

    This is intentionally not a learned model.  It combines lexical coverage
    with numeric/year overlap and table-row markers so the experiment can
    isolate feature fusion before paying for a domain-trained reranker.
    """

    _NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")

    def __init__(self, *, blend_weight: float = 0.2) -> None:
        if not math.isfinite(blend_weight) or not 0.0 < blend_weight <= 1.0:
            raise ValueError("blend_weight must be in (0, 1]")
        self.blend_weight = float(blend_weight)

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        query_terms = set(tokenise(query))
        query_numbers = {
            value.replace(",", "")
            for value in self._NUMBER.findall(query)
        }
        query_years = {
            value for value in query_numbers if len(value) == 4 and value[:2] in {"19", "20"}
        }
        count_like = bool(
            re.search(r"\b(how many|number of|count|which years)\b", query, re.I)
        )
        arithmetic_like = bool(
            re.search(r"\b(average|percentage|increase|decrease|change|difference)\b", query, re.I)
        )
        if not query_terms:
            return [0.0 for _ in documents]
        scores: list[float] = []
        for document in documents:
            document_terms = set(tokenise(document))
            overlap = len(query_terms.intersection(document_terms))
            coverage = overlap / len(query_terms)
            union = len(query_terms.union(document_terms))
            jaccard = overlap / union if union else 0.0
            document_numbers = {
                value.replace(",", "")
                for value in self._NUMBER.findall(document)
            }
            number_coverage = (
                len(query_numbers.intersection(document_numbers)) / len(query_numbers)
                if query_numbers
                else 0.0
            )
            year_coverage = (
                len(query_years.intersection(document_numbers)) / len(query_years)
                if query_years
                else 0.0
            )
            table_bonus = 0.08 if "Table row:" in document or "Table cell:" in document else 0.0
            if count_like and table_bonus:
                table_bonus += 0.08
            if arithmetic_like and number_coverage:
                table_bonus += 0.04
            scores.append(
                0.45 * coverage
                + 0.15 * jaccard
                + 0.25 * number_coverage
                + 0.15 * year_coverage
                + table_bonus
            )
        return scores


class FastEmbedCrossEncoderReranker:
    """Learned ONNX cross-encoder reranker loaded only on explicit opt-in."""

    def __init__(
        self,
        model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        *,
        cache_dir: str | Path | None = None,
        batch_size: int = 32,
    ) -> None:
        if TextCrossEncoder is None:
            raise RerankerContractError(
                "fastembed cross-encoder support is required for learned reranking"
            )
        if not 1 <= int(batch_size) <= 512:
            raise ValueError("reranker batch_size must be between 1 and 512")
        self.model_name = _clean_required(model_name, "reranker_model")
        self.batch_size = int(batch_size)
        try:
            model_kwargs = {"model_name": self.model_name}
            if cache_dir is not None:
                model_kwargs["cache_dir"] = str(Path(cache_dir).resolve())
            self._model = TextCrossEncoder(**model_kwargs)
        except Exception as exc:
            raise RerankerContractError(
                f"failed to initialize cross-encoder {self.model_name!r}: {exc}"
            ) from exc
        model_impl = getattr(self._model, "model", None)
        model_dir = getattr(model_impl, "_model_dir", None)
        self.revision = model_dir.name if model_dir is not None else None
        tokenizer = getattr(model_impl, "tokenizer", None)
        truncation = getattr(tokenizer, "truncation", None)
        if isinstance(truncation, Mapping):
            max_length = truncation.get("max_length")
        else:
            max_length = getattr(truncation, "max_length", None)
        self.input_max_tokens = int(max_length or 512)
        self._scored_pairs = 0
        self._estimated_truncated_pairs = 0

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self._scored_pairs += len(documents)
        if self.input_max_tokens is not None:
            for document in documents:
                # This is a conservative, tokenizer-independent diagnostic;
                # the model's own tokenizer remains authoritative for scoring.
                estimated_tokens = len(tokenise(query)) + len(tokenise(document)) + 3
                if estimated_tokens > self.input_max_tokens:
                    self._estimated_truncated_pairs += 1
        try:
            return [
                float(value)
                for value in self._model.rerank(
                    query,
                    documents,
                    batch_size=self.batch_size,
                )
            ]
        except Exception as exc:
            raise RerankerContractError(f"cross-encoder inference failed: {exc}") from exc

    def telemetry(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "batch_size": self.batch_size,
            "input_max_tokens": self.input_max_tokens,
            "estimated_truncation_ratio": (
                self._estimated_truncated_pairs / self._scored_pairs
                if self._scored_pairs
                else 0.0
            ),
            "scored_pairs": self._scored_pairs,
        }


def _matches_scope(chunk: HybridChunk, request: HybridSearchRequest) -> bool:
    """The shared pre-ranking authorization/version predicate."""

    raw_parent = chunk.metadata.get("parent_document_id")
    parent_id = (
        raw_parent.strip()
        if isinstance(raw_parent, str) and raw_parent.strip()
        else chunk.document_id
    )
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
            request.parent_document_ids is None
            or parent_id in request.parent_document_ids
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
        field_weights: Mapping[str, float] | None = None,
    ) -> None:
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be a finite positive number")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be a finite number between 0 and 1")
        weights: dict[str, float] = {}
        for field, weight in (field_weights or {}).items():
            if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
                raise ValueError("field weights must be finite positive numbers")
            weights[str(field)] = float(weight)
        self.k1 = float(k1)
        self.b = float(b)
        self.reranker = reranker
        self._field_weights = weights
        self._chunks: dict[tuple[str, str], HybridChunk] = {}
        self._token_counts: dict[tuple[str, str], Counter[str]] = {}
        self._document_lengths: dict[tuple[str, str], float] = {}
        self._postings: dict[str, set[tuple[str, str]]] = defaultdict(set)
        # Scope-keyed corpus statistics are safe to reuse because the key
        # contains every authorization/version/tenant predicate. Requests with
        # an explicit allow-list are intentionally not cached.
        self._scope_cache: dict[
            tuple[Any, ...],
            tuple[
                dict[tuple[str, str], HybridChunk],
                dict[tuple[str, str], Counter[str]],
                dict[tuple[str, str], float],
                float,
                Counter[str],
            ],
        ] = {}
        for chunk in chunks:
            self.upsert(chunk)

    def upsert(self, chunk: HybridChunk) -> None:
        key = (chunk.tenant_id, chunk.chunk_id)
        previous = self._token_counts.get(key)
        if previous is not None:
            for term in previous:
                self._postings[term].discard(key)
        self._chunks[key] = chunk
        counts = Counter(tokenise(chunk.text))
        for field, weight in self._field_weights.items():
            value = chunk.metadata.get(field)
            if isinstance(value, str) and value:
                for term in tokenise(value):
                    counts[term] += weight
        self._token_counts[key] = counts
        self._document_lengths[key] = sum(counts.values())
        for term in counts:
            self._postings[term].add(key)
        self._scope_cache.clear()

    @staticmethod
    def _scope_cache_key(request: HybridSearchRequest) -> tuple[Any, ...] | None:
        if request.allowed_chunk_ids is not None:
            return None
        return (
            request.tenant_id,
            tuple(sorted(request.acl_principals)),
            tuple(sorted(request.versions)) if request.versions is not None else None,
            (
                tuple(sorted(request.version_orders))
                if request.version_orders is not None
                else None
            ),
            (
                tuple(sorted(request.knowledge_base_ids))
                if request.knowledge_base_ids is not None
                else None
            ),
            (
                tuple(sorted(request.parent_document_ids))
                if request.parent_document_ids is not None
                else None
            ),
        )

    def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        # Security and version scope is established before corpus statistics,
        # so inaccessible documents cannot consume candidate slots or affect IDF.
        cache_key = self._scope_cache_key(request)
        cached = self._scope_cache.get(cache_key) if cache_key is not None else None
        if cached is None:
            eligible_by_key = {
                key: chunk
                for key, chunk in self._chunks.items()
                if _matches_scope(chunk, request)
            }
            token_counts = {
                key: self._token_counts[key] for key in eligible_by_key
            }
            lengths = {
                key: self._document_lengths[key] for key in eligible_by_key
            }
            average_length = (
                sum(lengths.values()) / len(eligible_by_key)
                if eligible_by_key
                else 0.0
            )
            document_frequencies: Counter[str] = Counter()
            for counts in token_counts.values():
                document_frequencies.update(counts.keys())
            if cache_key is not None:
                cached = (
                    eligible_by_key,
                    token_counts,
                    lengths,
                    average_length,
                    document_frequencies,
                )
                self._scope_cache[cache_key] = cached
        if cached is not None:
            (
                eligible_by_key,
                token_counts,
                lengths,
                average_length,
                document_frequencies,
            ) = cached
        if not eligible_by_key:
            return _response("python_bm25", None, request, [], 0)
        query_terms = tuple(dict.fromkeys(tokenise(request.query)))
        corpus_size = len(eligible_by_key)
        scores: dict[tuple[str, str], float] = {}
        normalizers: dict[tuple[str, str], float] = {}
        for term in query_terms:
            df = document_frequencies[term]
            if not df:
                continue
            idf = math.log(1.0 + (corpus_size - df + 0.5) / (df + 0.5))
            for key in self._postings.get(term, ()):
                if key not in eligible_by_key:
                    continue
                counts = token_counts[key]
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                normalizer = normalizers.get(key)
                if normalizer is None:
                    normalizer = self.k1 * (
                        1.0
                        - self.b
                        + self.b
                        * (
                            lengths[key] / average_length
                            if average_length
                            else 0.0
                        )
                    )
                    normalizers[key] = normalizer
                normalized_tf = frequency * (self.k1 + 1.0) / (frequency + normalizer)
                scores[key] = scores.get(key, 0.0) + idf * normalized_tf
        candidate_limit = _raw_candidate_limit(request)
        ranked_keys = heapq.nsmallest(
            candidate_limit,
            scores,
            key=lambda key: (-scores[key], eligible_by_key[key].chunk_id),
        )
        candidates: list[HybridSearchHit] = []
        for key in ranked_keys:
            chunk = eligible_by_key[key]
            counts = token_counts[key]
            document_length = lengths[key]
            normalizer = normalizers[key]
            term_scores: list[BM25TermContribution] = []
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                df = document_frequencies[term]
                idf = math.log(1.0 + (corpus_size - df + 0.5) / (df + 0.5))
                normalized_tf = frequency * (self.k1 + 1.0) / (frequency + normalizer)
                term_scores.append(
                    BM25TermContribution(
                        term=term,
                        term_frequency=int(frequency),
                        document_frequency=df,
                        inverse_document_frequency=idf,
                        length_normalized_tf=normalized_tf,
                        contribution=idf * normalized_tf,
                    )
                )
            candidates.append(
                HybridSearchHit(
                    chunk=chunk,
                    rank=1,
                    score=scores[key],
                    base_score=scores[key],
                    retrieval_sources=["python_bm25"],
                    bm25_explanation=BM25Explanation(
                        corpus_size_after_filters=corpus_size,
                        document_length=int(document_length),
                        average_document_length=average_length,
                        k1=self.k1,
                        b=self.b,
                        terms=term_scores,
                    ),
                )
            )
        candidates = _apply_reranker(candidates, request, self.reranker)
        candidates = _limit_chunks_per_document(candidates, request)
        candidates = candidates[: request.candidate_k]
        seeds = _rerank_positions(candidates[: request.top_k])
        expanded = _expand_from_catalog(
            seeds,
            {chunk.chunk_id: chunk for chunk in eligible_by_key.values()},
            request,
        )
        return _response(
            "python_bm25",
            None,
            request,
            expanded,
            max(0, len(expanded) - len(seeds)),
        )


class FastEmbedSparseIndex:
    """Explicit opt-in learned sparse retrieval with pre-ranking ACL filters."""

    def __init__(
        self,
        chunks: Sequence[HybridChunk],
        *,
        model_name: str = "prithivida/Splade_PP_en_v1",
        batch_size: int = 32,
    ) -> None:
        if SparseTextEmbedding is None:
            raise EmbeddingContractError(
                "fastembed sparse support is required for learned sparse retrieval"
            )
        if not 1 <= int(batch_size) <= 512:
            raise ValueError("sparse batch_size must be between 1 and 512")
        self.model_name = _clean_required(model_name, "sparse_model")
        self.batch_size = int(batch_size)
        self._chunks = tuple(chunks)
        try:
            self._model = SparseTextEmbedding(model_name=self.model_name)
            raw_vectors = list(
                self._model.embed(
                    [chunk.text for chunk in self._chunks],
                    batch_size=self.batch_size,
                )
            )
        except Exception as exc:
            raise EmbeddingContractError(
                f"failed to initialize sparse model {self.model_name!r}: {exc}"
            ) from exc
        if len(raw_vectors) != len(self._chunks):
            raise EmbeddingContractError(
                "learned sparse model must return one vector per indexed chunk"
            )
        self._vectors = tuple(
            self._validated_sparse_vector(vector, "document")
            for vector in raw_vectors
        )

    @staticmethod
    def _validated_sparse_vector(
        vector: Any,
        label: str,
    ) -> dict[int, float]:
        try:
            indices = [int(value) for value in vector.indices]
            values = [float(value) for value in vector.values]
        except (AttributeError, TypeError, ValueError) as exc:
            raise EmbeddingContractError(
                f"{label} sparse vector must expose numeric indices and values"
            ) from exc
        if len(indices) != len(values):
            raise EmbeddingContractError(
                f"{label} sparse vector indices and values must have equal length"
            )
        if any(index < 0 for index in indices):
            raise EmbeddingContractError(
                f"{label} sparse vector contains a negative index"
            )
        if not all(math.isfinite(value) for value in values):
            raise EmbeddingContractError(
                f"{label} sparse vector contains a non-finite value"
            )
        return {
            index: value
            for index, value in zip(indices, values, strict=True)
            if value != 0.0
        }

    def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        try:
            raw_query_vectors = list(
                self._model.query_embed(
                    [request.query],
                    batch_size=1,
                )
            )
        except Exception as exc:
            raise EmbeddingContractError(
                f"learned sparse query embedding failed: {exc}"
            ) from exc
        if len(raw_query_vectors) != 1:
            raise EmbeddingContractError(
                "learned sparse model must return exactly one query vector"
            )
        query_vector = self._validated_sparse_vector(
            raw_query_vectors[0],
            "query",
        )
        scored: list[HybridSearchHit] = []
        eligible_count = 0
        for chunk, vector in zip(self._chunks, self._vectors, strict=True):
            if not _matches_scope(chunk, request):
                continue
            eligible_count += 1
            score = sum(
                query_weight * vector.get(index, 0.0)
                for index, query_weight in query_vector.items()
            )
            if score <= 0.0:
                continue
            scored.append(
                HybridSearchHit(
                    chunk=chunk,
                    rank=1,
                    score=score,
                    base_score=score,
                    retrieval_sources=["fastembed_learned_sparse"],
                )
            )
        scored.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        candidates = _limit_chunks_per_document(
            scored[: _raw_candidate_limit(request)],
            request,
        )[: request.candidate_k]
        hits = _rerank_positions(candidates[: request.top_k])
        return _response(
            "fastembed_sparse",
            None,
            request,
            hits,
            0,
            {"fastembed_learned_sparse": eligible_count},
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
        embedding_metadata_fields: Iterable[str] = (),
        backend_label: Literal["qdrant", "qdrant_local"] = "qdrant",
        create_if_missing: bool = True,
        upsert_batch_size: int = 128,
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
        self.embedding_metadata_fields = tuple(
            dict.fromkeys(
                _clean_required(field, "embedding_metadata_field")
                for field in embedding_metadata_fields
            )
        )
        self.backend_label = backend_label
        if upsert_batch_size < 1 or upsert_batch_size > 1_000:
            raise ValueError("upsert_batch_size must be between 1 and 1000")
        self.upsert_batch_size = int(upsert_batch_size)
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
        embedding_metadata_fields: Iterable[str] = (),
        upsert_batch_size: int = 128,
    ) -> QdrantHybridIndex:
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
            embedding_metadata_fields=embedding_metadata_fields,
            backend_label="qdrant_local",
            upsert_batch_size=upsert_batch_size,
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
        inserted = 0
        for offset in range(0, len(chunks), self.upsert_batch_size):
            batch = chunks[offset : offset + self.upsert_batch_size]
            try:
                raw_vectors = self.embedder.embed_documents(
                    [self._dense_document_text(chunk) for chunk in batch]
                )
            except Exception as exc:
                raise EmbeddingContractError(f"document embedding failed: {exc}") from exc
            if len(raw_vectors) != len(batch):
                raise EmbeddingContractError(
                    "embed_documents must return exactly one vector per chunk"
                )

            points = []
            for chunk, raw_vector in zip(batch, raw_vectors, strict=True):
                dense = _validated_vector(raw_vector, self.dimension, "embed_documents")
                indices, values = _sparse_vector(chunk.text)
                points.append(
                    qdrant_models.PointStruct(
                        id=str(
                            uuid5(
                                _POINT_NAMESPACE,
                                f"{chunk.tenant_id}\0{chunk.chunk_id}",
                            )
                        ),
                        vector={
                            "dense": dense,
                            "sparse": qdrant_models.SparseVector(
                                indices=indices,
                                values=values,
                            ),
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
            inserted += len(points)
        return inserted

    def _dense_document_text(self, chunk: HybridChunk) -> str:
        parts = [chunk.text]
        for field in self.embedding_metadata_fields:
            value = chunk.metadata.get(field)
            if isinstance(value, str) and value.strip():
                parts.append(f"{field}: {value.strip()}")
        return "\n".join(parts)

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
        raw_limit = _raw_candidate_limit(request)
        try:
            result = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    qdrant_models.Prefetch(
                        query=dense_query,
                        using="dense",
                        filter=access_filter,
                        limit=raw_limit,
                    ),
                    qdrant_models.Prefetch(
                        query=qdrant_models.SparseVector(
                            indices=sparse_indices,
                            values=sparse_values,
                        ),
                        using="sparse",
                        filter=access_filter,
                        limit=raw_limit,
                    ),
                ],
                query=qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.RRF),
                # Defense in depth: the exact same trusted filter is also on
                # the fusion stage, never applied as post-ranking Python logic.
                query_filter=access_filter,
                limit=raw_limit,
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
        candidates = _limit_chunks_per_document(candidates, request)
        candidates = candidates[: request.candidate_k]
        seeds = _rerank_positions(candidates[: request.top_k])
        expanded = self._expand_neighbors(seeds, request)
        return _response(
            self.backend_label,
            self.collection_name,
            request,
            expanded,
            max(0, len(expanded) - len(seeds)),
        )

    def search_dense(self, request: HybridSearchRequest) -> HybridSearchResponse:
        """Search only the named dense vector using the same trusted scope."""

        try:
            dense_query = _validated_vector(
                self.embedder.embed_query(request.query), self.dimension, "embed_query"
            )
        except EmbeddingContractError:
            raise
        except Exception as exc:
            raise EmbeddingContractError(f"query embedding failed: {exc}") from exc
        access_filter = self._filter(request)
        raw_limit = _raw_candidate_limit(request)
        try:
            result = self.client.query_points(
                collection_name=self.collection_name,
                query=dense_query,
                using="dense",
                query_filter=access_filter,
                limit=raw_limit,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise QdrantBackendError(f"Qdrant dense query failed: {exc}") from exc

        candidates: list[HybridSearchHit] = []
        for point in result.points:
            chunk = self._payload_chunk(point.payload)
            if not _matches_scope(chunk, request):
                raise QdrantBackendError("Qdrant returned a point outside the trusted scope")
            score = float(point.score)
            if not math.isfinite(score):
                raise QdrantBackendError("Qdrant returned a non-finite dense score")
            candidates.append(
                HybridSearchHit(
                    chunk=chunk,
                    rank=1,
                    score=score,
                    base_score=score,
                    retrieval_sources=["qdrant_dense"],
                )
            )
        candidates = _apply_reranker(candidates, request, self.reranker)
        candidates = _limit_chunks_per_document(candidates, request)
        candidates = candidates[: request.candidate_k]
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
        if request.parent_document_ids is not None:
            must.append(
                qdrant_models.FieldCondition(
                    key="metadata.parent_document_id",
                    match=qdrant_models.MatchAny(
                        any=sorted(request.parent_document_ids)
                    ),
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


class QdrantDenseIndex:
    """Search-backend adapter exposing only Qdrant's named dense vector."""

    def __init__(self, backend: QdrantHybridIndex) -> None:
        if not isinstance(backend, QdrantHybridIndex):
            raise TypeError("backend must be a QdrantHybridIndex")
        self.backend = backend

    def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        return self.backend.search_dense(request)


class InMemoryDenseIndex:
    """Exact cosine dense search for bounded offline/benchmark corpora.

    This keeps the real embedder while avoiding the per-request local-Qdrant
    client overhead.  It is intentionally an explicit benchmark backend, not
    a replacement for the tenant-filtered Qdrant production path.
    """

    def __init__(
        self,
        chunks: Sequence[HybridChunk],
        embedder: DenseEmbedder,
        *,
        collection_name: str = "taskforge-rag-in-memory-dense",
        reranker: Reranker | None = None,
        rerank_top_k: int | None = None,
        adaptive_rerank_min_k: int | None = None,
        adaptive_rerank_margin_threshold: float = 0.7,
    ) -> None:
        if np is None:
            raise EmbeddingContractError("numpy is required for the in-memory dense index")
        if not chunks:
            raise ValueError("in-memory dense index requires at least one chunk")
        self.chunks = tuple(chunks)
        self.embedder = embedder
        self.collection_name = collection_name
        self.reranker = reranker
        if rerank_top_k is not None and (
            isinstance(rerank_top_k, bool) or not 1 <= int(rerank_top_k) <= 100
        ):
            raise ValueError("rerank_top_k must be between 1 and 100")
        self.rerank_top_k = int(rerank_top_k) if rerank_top_k is not None else None
        if adaptive_rerank_min_k is not None:
            if self.rerank_top_k is None:
                raise ValueError("adaptive reranking requires rerank_top_k")
            if (
                isinstance(adaptive_rerank_min_k, bool)
                or not 1 <= int(adaptive_rerank_min_k) < self.rerank_top_k
            ):
                raise ValueError(
                    "adaptive_rerank_min_k must be smaller than rerank_top_k"
                )
        if (
            not math.isfinite(float(adaptive_rerank_margin_threshold))
            or float(adaptive_rerank_margin_threshold) < 0.0
        ):
            raise ValueError(
                "adaptive_rerank_margin_threshold must be finite and non-negative"
            )
        self.adaptive_rerank_min_k = (
            int(adaptive_rerank_min_k)
            if adaptive_rerank_min_k is not None
            else None
        )
        self.adaptive_rerank_margin_threshold = float(
            adaptive_rerank_margin_threshold
        )
        vectors = np.asarray(embedder.embed_documents([chunk.text for chunk in self.chunks]), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(self.chunks):
            raise EmbeddingContractError("dense embedder returned an invalid matrix")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 0) or not np.isfinite(vectors).all():
            raise EmbeddingContractError("dense embedder returned a zero or non-finite vector")
        self._vectors = vectors / norms
        self._catalog = {chunk.chunk_id: chunk for chunk in self.chunks}

    def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        if np is None:  # pragma: no cover - guarded by __init__.
            raise EmbeddingContractError("numpy is required for the in-memory dense index")
        query = np.asarray(self.embedder.embed_query(request.query), dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm <= 0 or not np.isfinite(query).all():
            raise EmbeddingContractError("dense embedder returned an invalid query vector")
        query = query / norm
        eligible = [
            index
            for index, chunk in enumerate(self.chunks)
            if _matches_scope(chunk, request)
        ]
        scores = self._vectors[eligible] @ query if eligible else np.asarray([], dtype=np.float32)
        ranked = sorted(
            zip(eligible, scores.tolist(), strict=True),
            key=lambda item: (-float(item[1]), self.chunks[item[0]].chunk_id),
        )
        candidates = [
            HybridSearchHit(
                chunk=self.chunks[index],
                rank=1,
                score=float(score),
                base_score=float(score),
                retrieval_sources=["qdrant_dense"],
            )
            for index, score in ranked[: _raw_candidate_limit(request)]
        ]
        adaptive_diagnostics: AdaptiveRerankDiagnostics | None = None
        if request.rerank and self.adaptive_rerank_min_k is not None:
            candidates, adaptive_diagnostics = _apply_adaptive_reranker(
                candidates,
                request,
                self.reranker,
                min_k=self.adaptive_rerank_min_k,
                max_k=self.rerank_top_k,
                margin_threshold=self.adaptive_rerank_margin_threshold,
            )
        else:
            candidates = _apply_reranker(
                candidates,
                request,
                self.reranker,
                rerank_limit=self.rerank_top_k,
            )
        candidates = _limit_chunks_per_document(candidates, request)
        candidates = candidates[: request.candidate_k]
        seeds = _rerank_positions(candidates[: request.top_k])
        expanded = _expand_from_catalog(seeds, self._catalog, request)
        return _response(
            "in_memory_dense",
            self.collection_name,
            request,
            expanded,
            max(0, len(expanded) - len(seeds)),
            {"in_memory_dense": len(candidates)},
            adaptive_rerank=adaptive_diagnostics,
        )


class BM25DenseRRFIndex:
    """Rank-fuse genuine BM25 and dense retrieval over an identical scope."""

    def __init__(
        self,
        bm25: BM25Index,
        dense: QdrantHybridIndex,
        *,
        reranker: Reranker | None = None,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> None:
        if not isinstance(bm25, BM25Index):
            raise TypeError("bm25 must be a BM25Index")
        if not isinstance(dense, QdrantHybridIndex):
            raise TypeError("dense must be a QdrantHybridIndex")
        if not 1 <= int(rrf_k) <= 10_000:
            raise ValueError("rrf_k must be between 1 and 10000")
        weights = (float(bm25_weight), float(dense_weight))
        if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
            raise ValueError("RRF branch weights must be finite positive numbers")
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.rrf_k = int(rrf_k)
        self.bm25_weight, self.dense_weight = weights

    def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        branch_request = request.model_copy(
            update={
                "top_k": request.candidate_k,
                "candidate_k": request.candidate_k,
                "rerank": False,
                "neighbor_window": 0,
                "max_expanded_hits": request.candidate_k,
            }
        )
        bm25_response = self.bm25.search(branch_request)
        dense_response = self.dense.search_dense(branch_request)
        if (
            bm25_response.filters_applied_before_ranking
            != dense_response.filters_applied_before_ranking
        ):
            raise HybridRetrievalError("BM25 and dense branches applied different filters")

        fused: dict[str, HybridSearchHit] = {}
        branch_ranks: dict[str, dict[str, int]] = {"bm25": {}, "dense": {}}
        for branch, response in (
            ("bm25", bm25_response),
            ("dense", dense_response),
        ):
            for rank, hit in enumerate(response.hits, start=1):
                chunk_id = hit.chunk.chunk_id
                existing = fused.get(chunk_id)
                if existing is not None and existing.chunk != hit.chunk:
                    raise HybridRetrievalError(
                        "BM25 and dense branches returned conflicting chunk payloads"
                    )
                fused.setdefault(chunk_id, hit)
                branch_ranks[branch][chunk_id] = rank

        candidates: list[HybridSearchHit] = []
        for chunk_id, hit in fused.items():
            bm25_rank = branch_ranks["bm25"].get(chunk_id)
            dense_rank = branch_ranks["dense"].get(chunk_id)
            score = 0.0
            sources: list[RetrievalSource] = []
            if bm25_rank is not None:
                score += self.bm25_weight / (self.rrf_k + bm25_rank)
                sources.append("python_bm25")
            if dense_rank is not None:
                score += self.dense_weight / (self.rrf_k + dense_rank)
                sources.append("qdrant_dense")
            sources.append("bm25_dense_rrf")
            candidates.append(
                hit.model_copy(
                    update={
                        "rank": 1,
                        "score": score,
                        "base_score": score,
                        "reranker_score": None,
                        "retrieval_sources": sources,
                    }
                )
            )
        candidates.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        candidates = _apply_reranker(candidates, request, self.reranker)
        candidates = _limit_chunks_per_document(candidates, request)
        candidates = candidates[: request.candidate_k]
        seeds = _rerank_positions(candidates[: request.top_k])
        expanded = self.dense._expand_neighbors(seeds, request)
        return _response(
            "bm25_dense_rrf",
            self.dense.collection_name,
            request,
            expanded,
            max(0, len(expanded) - len(seeds)),
            {"bm25": len(bm25_response.hits), "dense": len(dense_response.hits), "fused": len(candidates)},
        )


class MultiQueryRRFIndex:
    """Fuse document-diverse rankings from deterministic query decompositions."""

    def __init__(
        self,
        backend: Any,
        query_expander: Callable[[str], Sequence[str]],
        *,
        rrf_k: int = 60,
    ) -> None:
        if not callable(getattr(backend, "search", None)):
            raise TypeError("backend must implement search(HybridSearchRequest)")
        if not callable(query_expander):
            raise TypeError("query_expander must be callable")
        if not 1 <= int(rrf_k) <= 10_000:
            raise ValueError("rrf_k must be between 1 and 10000")
        self.backend = backend
        self.query_expander = query_expander
        self.rrf_k = int(rrf_k)

    def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        queries = list(dict.fromkeys(self.query_expander(request.query)))
        if not queries or queries[0] != request.query:
            queries.insert(0, request.query)
        queries = queries[:8]
        scores: dict[str, float] = {}
        hits: dict[str, HybridSearchHit] = {}
        expected_filters = AppliedRetrievalFilters.from_request(request)
        collection_name: str | None = None
        raw_candidate_counts: dict[str, int] = {}
        for query_index, query in enumerate(queries):
            branch_request = request.model_copy(
                update={
                    "query": query,
                    "top_k": request.candidate_k,
                    "rerank": False,
                    "neighbor_window": 0,
                    "max_expanded_hits": request.candidate_k,
                }
            )
            response = self.backend.search(branch_request)
            raw_candidate_counts[f"query_{query_index}"] = len(response.hits)
            if response.filters_applied_before_ranking != expected_filters:
                raise HybridRetrievalError("multi-query branch changed trusted filters")
            collection_name = response.collection_name or collection_name
            seen_documents: set[str] = set()
            document_rank = 0
            for hit in response.hits:
                document_id = hit.chunk.document_id
                if document_id in seen_documents:
                    continue
                seen_documents.add(document_id)
                document_rank += 1
                scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (
                    self.rrf_k + document_rank
                )
                hits.setdefault(document_id, hit)
        candidates = [
            hit.model_copy(
                update={
                    "rank": 1,
                    "score": scores[document_id],
                    "base_score": scores[document_id],
                    "reranker_score": None,
                    "retrieval_sources": [
                        *hit.retrieval_sources,
                        "multi_query_rrf",
                    ],
                }
            )
            for document_id, hit in hits.items()
        ]
        candidates.sort(key=lambda hit: (-hit.score, hit.chunk.document_id))
        seeds = _rerank_positions(candidates[: request.top_k])
        return _response(
            "multi_query_rrf",
            collection_name,
            request,
            seeds,
            0,
            {**raw_candidate_counts, "fused": len(candidates)},
        )


class SourceCoverageRRFIndex:
    """Reserve retrieval branches for corpus sources explicitly named in a query."""

    def __init__(
        self,
        backend: Any,
        chunks: Sequence[HybridChunk],
        *,
        metadata_field: str = "source",
        rrf_k: int = 60,
        max_sources: int = 6,
        lexical_anchor_k: int = 0,
    ) -> None:
        if not callable(getattr(backend, "search", None)):
            raise TypeError("backend must implement search(HybridSearchRequest)")
        if not 1 <= int(rrf_k) <= 10_000:
            raise ValueError("rrf_k must be between 1 and 10000")
        if not 1 <= int(max_sources) <= 20:
            raise ValueError("max_sources must be between 1 and 20")
        if not 0 <= int(lexical_anchor_k) <= 50:
            raise ValueError("lexical_anchor_k must be between 0 and 50")
        self.backend = backend
        self.rrf_k = int(rrf_k)
        self.max_sources = int(max_sources)
        self.lexical_anchor_k = int(lexical_anchor_k)
        source_ids: dict[str, set[str]] = {}
        source_labels: dict[str, str] = {}
        for chunk in chunks:
            value = chunk.metadata.get(metadata_field)
            if not isinstance(value, str) or len(value.strip()) < 3:
                continue
            label = value.strip()
            key = label.casefold()
            source_labels[key] = label
            source_ids.setdefault(key, set()).add(chunk.chunk_id)
        self._source_labels = source_labels
        self._source_ids = {
            key: frozenset(values) for key, values in source_ids.items()
        }

    def matched_sources(self, query: str) -> list[str]:
        lowered = query.casefold()
        matches = [key for key in self._source_ids if key in lowered]
        matches.sort(key=lambda key: (-len(key), key))
        return [self._source_labels[key] for key in matches[: self.max_sources]]

    def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        branch_requests = [request]
        for source in self.matched_sources(request.query):
            source_ids = self._source_ids[source.casefold()]
            allowed = (
                source_ids
                if request.allowed_chunk_ids is None
                else source_ids.intersection(request.allowed_chunk_ids)
            )
            if allowed:
                branch_requests.append(
                    request.model_copy(update={"allowed_chunk_ids": frozenset(allowed)})
                )
        scores: dict[str, float] = {}
        hits: dict[str, HybridSearchHit] = {}
        raw_candidate_counts: dict[str, int] = {}
        lexical_anchor_ids: list[str] = []
        for branch_index, branch_request in enumerate(branch_requests):
            response = self.backend.search(
                branch_request.model_copy(
                    update={
                        "top_k": request.candidate_k,
                        "rerank": False,
                        "neighbor_window": 0,
                        "max_expanded_hits": request.candidate_k,
                    }
                )
            )
            raw_candidate_counts[f"source_{branch_index}"] = len(response.hits)
            if branch_index == 0 and self.lexical_anchor_k:
                lexical_anchor_ids = [
                    hit.chunk.document_id
                    for hit in response.hits[: self.lexical_anchor_k]
                ]
            if response.filters_applied_before_ranking != AppliedRetrievalFilters.from_request(
                branch_request
            ):
                raise HybridRetrievalError("source coverage branch changed trusted filters")
            seen_documents: set[str] = set()
            rank = 0
            for hit in response.hits:
                document_id = hit.chunk.document_id
                if document_id in seen_documents:
                    continue
                seen_documents.add(document_id)
                rank += 1
                scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (
                    self.rrf_k + rank
                )
                hits.setdefault(document_id, hit)
        candidates = [
            hit.model_copy(
                update={
                    "rank": 1,
                    "score": scores[document_id],
                    "base_score": scores[document_id],
                    "reranker_score": None,
                    "retrieval_sources": [
                        *hit.retrieval_sources,
                        "source_coverage_rrf",
                    ],
                }
            )
            for document_id, hit in hits.items()
        ]
        candidates.sort(key=lambda hit: (-hit.score, hit.chunk.document_id))
        if lexical_anchor_ids:
            by_id = {hit.chunk.document_id: hit for hit in candidates}
            anchors = [
                by_id[document_id]
                for document_id in lexical_anchor_ids
                if document_id in by_id
            ]
            anchor_set = {hit.chunk.document_id for hit in anchors}
            tail = [
                hit for hit in candidates if hit.chunk.document_id not in anchor_set
            ]
            candidates = anchors + tail
        return _response(
            "source_coverage_rrf",
            None,
            request,
            _rerank_positions(candidates[: request.top_k]),
            0,
            {**raw_candidate_counts, "fused": len(candidates)},
        )


class ParentChildIndex:
    """Retrieve authorized parent contexts before searching their children.

    Parent records are a routing index only.  The returned evidence always
    comes from the child backend, and a caller-supplied child allow-list is
    translated into the corresponding parent allow-list before the parent
    search.  This keeps parent routing from widening tenant, ACL, version, or
    host-resolved evidence scope.
    """

    def __init__(
        self,
        parent_backend: Any,
        child_backend: Any,
        children: Sequence[HybridChunk],
        *,
        parent_field: str = "parent_document_id",
        parent_top_k: int = 5,
        include_parent_siblings: bool = True,
    ) -> None:
        if not callable(getattr(parent_backend, "search", None)):
            raise TypeError("parent_backend must implement search(HybridSearchRequest)")
        if not callable(getattr(child_backend, "search", None)):
            raise TypeError("child_backend must implement search(HybridSearchRequest)")
        if not parent_field.strip():
            raise ValueError("parent_field must not be blank")
        if isinstance(parent_top_k, bool) or not isinstance(parent_top_k, int):
            raise TypeError("parent_top_k must be an integer")
        if not 1 <= parent_top_k <= 100:
            raise ValueError("parent_top_k must be between 1 and 100")
        if not isinstance(include_parent_siblings, bool):
            raise TypeError("include_parent_siblings must be a boolean")
        self.parent_backend = parent_backend
        self.child_backend = child_backend
        self.parent_top_k = parent_top_k
        self.include_parent_siblings = include_parent_siblings
        self._parent_by_child: dict[str, str] = {}
        self._children_by_parent: dict[str, set[str]] = {}
        self._child_by_id: dict[str, HybridChunk] = {}
        for child in children:
            raw_parent = child.metadata.get(parent_field)
            parent_id = (
                raw_parent.strip()
                if isinstance(raw_parent, str) and raw_parent.strip()
                else child.document_id
            )
            if child.chunk_id in self._parent_by_child:
                raise ValueError(f"duplicate child chunk id: {child.chunk_id}")
            self._parent_by_child[child.chunk_id] = parent_id
            self._children_by_parent.setdefault(parent_id, set()).add(child.chunk_id)
            self._child_by_id[child.chunk_id] = child

    def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        allowed_child_ids = request.allowed_chunk_ids
        if allowed_child_ids is None:
            allowed_parent_ids = None
        else:
            allowed_parent_ids = frozenset(
                self._parent_by_child[chunk_id]
                for chunk_id in allowed_child_ids
                if chunk_id in self._parent_by_child
            )
            if not allowed_parent_ids:
                return _response("parent_child", None, request, [], 0)

        parent_request = request.model_copy(
            update={
                "allowed_chunk_ids": allowed_parent_ids,
                # Parent routing is intentionally bounded independently from
                # the child candidate budget.  A large child candidate_k must
                # not silently turn this into an unbounded parent fan-out.
                "top_k": min(self.parent_top_k, request.candidate_k),
                "candidate_k": request.candidate_k,
                "rerank": False,
                "neighbor_window": 0,
                "max_expanded_hits": request.candidate_k,
            }
        )
        parent_response = self.parent_backend.search(parent_request)
        if parent_response.filters_applied_before_ranking != AppliedRetrievalFilters.from_request(
            parent_request
        ):
            raise HybridRetrievalError("parent branch changed trusted filters")
        selected_parents: list[str] = []
        seen_parents: set[str] = set()
        for hit in parent_response.hits:
            parent_id = hit.chunk.document_id
            if parent_id in self._children_by_parent and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                selected_parents.append(parent_id)
        if not selected_parents:
            return _response("parent_child", parent_response.collection_name, request, [], 0)

        child_ids = frozenset(
            child_id
            for parent_id in selected_parents
            for child_id in self._children_by_parent[parent_id]
            if allowed_child_ids is None or child_id in allowed_child_ids
        )
        if not child_ids:
            return _response("parent_child", parent_response.collection_name, request, [], 0)
        child_request = request.model_copy(
            update={
                "allowed_chunk_ids": child_ids,
                "top_k": request.candidate_k,
                "candidate_k": request.candidate_k,
                "rerank": request.rerank,
                "neighbor_window": 0,
                "max_expanded_hits": request.candidate_k,
            }
        )
        child_response = self.child_backend.search(child_request)
        if child_response.filters_applied_before_ranking != AppliedRetrievalFilters.from_request(
            child_request
        ):
            raise HybridRetrievalError("child branch changed trusted filters")
        filtered_hits = [
            hit
            for hit in child_response.hits
            if hit.chunk.chunk_id in child_ids
            and self._parent_by_child.get(hit.chunk.chunk_id) in seen_parents
        ]
        if self.include_parent_siblings:
            seen_children = {hit.chunk.chunk_id for hit in filtered_hits}
            # A selected parent is an authorized context route.  Add its
            # sibling evidence units after scored child hits so a table hit
            # can expose the accompanying paragraph(s) required by TAT-QA.
            # Scope is checked again here because this supplement bypasses the
            # child backend's ranking path.
            for parent_id in selected_parents:
                for child_id in sorted(self._children_by_parent[parent_id]):
                    if child_id in seen_children or child_id not in child_ids:
                        continue
                    child = self._child_by_id[child_id]
                    if not _matches_scope(child, request):
                        continue
                    filtered_hits.append(
                        HybridSearchHit(
                            chunk=child,
                            rank=1,
                            score=0.0,
                            base_score=0.0,
                            retrieval_sources=["parent_sibling_coverage"],
                        )
                    )
                    seen_children.add(child_id)
        return _response(
            "parent_child",
            child_response.collection_name,
            request,
            _rerank_positions(filtered_hits[: request.top_k]),
            0,
        )


class SearchRepresentationIndex:
    """Search an alternate representation while returning raw evidence chunks."""

    def __init__(
        self,
        backend: Any,
        raw_chunks: Sequence[HybridChunk],
        *,
        backend_label: str | None = None,
    ) -> None:
        if not callable(getattr(backend, "search", None)):
            raise TypeError("representation backend must implement search(request)")
        mapping: dict[str, HybridChunk] = {}
        for chunk in raw_chunks:
            if chunk.chunk_id in mapping:
                raise ValueError(f"duplicate raw representation chunk: {chunk.chunk_id}")
            mapping[chunk.chunk_id] = chunk
        if not mapping:
            raise ValueError("raw representation chunks must not be empty")
        self.backend = backend
        self._raw_by_id = mapping
        self.backend_label = backend_label

    def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        response = self.backend.search(request)
        mapped_hits: list[HybridSearchHit] = []
        for hit in response.hits:
            raw = self._raw_by_id.get(hit.chunk.chunk_id)
            if raw is None:
                raise HybridRetrievalError(
                    "representation backend returned an unknown chunk"
                )
            if not _matches_scope(raw, request):
                raise HybridRetrievalError(
                    "representation backend returned an out-of-scope chunk"
                )
            mapped_hits.append(
                hit.model_copy(
                    update={
                        "chunk": raw,
                        "retrieval_sources": [
                            *hit.retrieval_sources,
                            "raw_evidence_projection",
                        ],
                    }
                )
            )
        return response.model_copy(
            update={
                "backend": self.backend_label or response.backend,
                "hits": mapped_hits,
            }
        )


class RepresentationRRFIndex:
    """Fuse equivalent evidence represented at different granularities."""

    def __init__(
        self,
        branches: Sequence[tuple[str, Any, Sequence[HybridChunk]]],
        *,
        rrf_k: int = 60,
        fusion: Literal["rrf", "max"] = "rrf",
        candidate_strategy: Literal["score", "coverage"] = "score",
        branch_weights: Mapping[str, float] | None = None,
        coverage_branch_reserves: Mapping[str, int] | None = None,
        reranker: Reranker | None = None,
        context_sibling_coverage: bool = False,
        context_seed_k: int = 10,
        context_sibling_limit: int | None = None,
        rerank_top_k: int = 20,
    ) -> None:
        if len(branches) < 2:
            raise ValueError("representation RRF requires at least two branches")
        if not 1 <= int(rrf_k) <= 10_000:
            raise ValueError("rrf_k must be between 1 and 10000")
        if fusion not in {"rrf", "max"}:
            raise ValueError("fusion must be 'rrf' or 'max'")
        if candidate_strategy not in {"score", "coverage"}:
            raise ValueError("candidate_strategy must be 'score' or 'coverage'")
        if not isinstance(context_sibling_coverage, bool):
            raise TypeError("context_sibling_coverage must be a boolean")
        if isinstance(context_seed_k, bool) or not isinstance(context_seed_k, int):
            raise TypeError("context_seed_k must be an integer")
        if not 1 <= context_seed_k <= 100:
            raise ValueError("context_seed_k must be between 1 and 100")
        if context_sibling_limit is not None and (
            isinstance(context_sibling_limit, bool)
            or not isinstance(context_sibling_limit, int)
            or not 1 <= context_sibling_limit <= 100
        ):
            raise ValueError("context_sibling_limit must be between 1 and 100")
        if isinstance(rerank_top_k, bool) or not isinstance(rerank_top_k, int):
            raise TypeError("rerank_top_k must be an integer")
        if not 1 <= rerank_top_k <= 100:
            raise ValueError("rerank_top_k must be between 1 and 100")
        self.branches = tuple(branches)
        self.rrf_k = int(rrf_k)
        self.fusion = fusion
        self.candidate_strategy = candidate_strategy
        self.reranker = reranker
        self.context_sibling_coverage = context_sibling_coverage
        self.context_seed_k = context_seed_k
        self.context_sibling_limit = context_sibling_limit
        self.rerank_top_k = rerank_top_k
        raw_reserves = coverage_branch_reserves or {}
        if not isinstance(raw_reserves, Mapping):
            raise TypeError("coverage_branch_reserves must be a mapping")
        branch_names = {name for name, _, _ in self.branches}
        unknown_reserves = set(raw_reserves).difference(branch_names)
        if unknown_reserves:
            raise ValueError(
                "coverage_branch_reserves contains unknown branches: "
                + ", ".join(sorted(unknown_reserves))
            )
        self.coverage_branch_reserves = tuple(
            (name, int(limit)) for name, limit in raw_reserves.items()
        )
        if any(
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
            for _, limit in self.coverage_branch_reserves
        ):
            raise ValueError("coverage branch reserves must be positive integers")
        raw_weights = branch_weights or {}
        self.branch_weights = tuple(
            float(raw_weights.get(name, 1.0)) for name, _, _ in self.branches
        )
        if any(
            not math.isfinite(weight) or weight <= 0
            for weight in self.branch_weights
        ):
            raise ValueError("representation branch weights must be finite positive numbers")
        branch_ids: list[dict[str, frozenset[str]]] = []
        for _, _, branch_chunks in self.branches:
            grouped_ids: dict[str, set[str]] = defaultdict(set)
            for chunk in branch_chunks:
                grouped_ids[chunk.document_id].add(chunk.chunk_id)
            branch_ids.append(
                {
                    document_id: frozenset(chunk_ids)
                    for document_id, chunk_ids in grouped_ids.items()
                }
            )
        self._branch_ids = tuple(branch_ids)
        if any(not mapping for mapping in self._branch_ids):
            raise ValueError("representation branches must contain searchable chunks")
        self._context_chunks: dict[str, HybridChunk] = {}
        self._context_children: dict[str, set[str]] = {}
        for _, _, branch_chunks in self.branches:
            for chunk in branch_chunks:
                self._context_chunks.setdefault(chunk.document_id, chunk)
                raw_parent = chunk.metadata.get("parent_document_id")
                parent_id = (
                    raw_parent.strip()
                    if isinstance(raw_parent, str) and raw_parent.strip()
                    else chunk.document_id
                )
                self._context_children.setdefault(parent_id, set()).add(
                    chunk.document_id
                )

    def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        scores: dict[str, float] = {}
        hits: dict[str, HybridSearchHit] = {}
        branch_rankings: list[list[str]] = [[] for _ in self.branches]
        raw_candidate_counts: dict[str, int] = {}
        for branch_index, (name, backend, _) in enumerate(self.branches):
            branch_allowed: frozenset[str] | None
            if request.allowed_chunk_ids is None:
                branch_allowed = None
            else:
                branch_allowed = frozenset(
                    chunk_id
                    for document_id, chunk_ids in self._branch_ids[branch_index].items()
                    if document_id in request.allowed_chunk_ids
                    for chunk_id in chunk_ids
                )
                if not branch_allowed:
                    continue
            branch_request = request.model_copy(
                update={
                    "allowed_chunk_ids": branch_allowed,
                    "top_k": request.candidate_k,
                    "candidate_k": request.candidate_k,
                    "rerank": False,
                    "neighbor_window": 0,
                    "max_expanded_hits": request.candidate_k,
                }
            )
            response = backend.search(branch_request)
            raw_candidate_counts[name] = len(response.hits)
            if response.filters_applied_before_ranking != AppliedRetrievalFilters.from_request(
                branch_request
            ):
                raise HybridRetrievalError(
                    f"representation branch {name!r} changed trusted filters"
                )
            seen_documents: set[str] = set()
            for rank, hit in enumerate(response.hits, start=1):
                document_id = hit.chunk.document_id
                if document_id in seen_documents:
                    continue
                seen_documents.add(document_id)
                branch_rankings[branch_index].append(document_id)
                contribution = self.branch_weights[branch_index] / (
                    self.rrf_k + rank
                )
                if self.fusion == "max":
                    scores[document_id] = max(scores.get(document_id, 0.0), contribution)
                else:
                    scores[document_id] = scores.get(document_id, 0.0) + contribution
                hits.setdefault(document_id, hit)
        candidates = [
            hit.model_copy(
                update={
                    "rank": 1,
                    "score": scores[document_id],
                    "base_score": scores[document_id],
                    "reranker_score": None,
                    "retrieval_sources": [
                        *hit.retrieval_sources,
                        "multi_representation_rrf",
                    ],
                }
            )
            for document_id, hit in hits.items()
        ]
        candidates.sort(key=lambda hit: (-hit.score, hit.chunk.document_id))
        if self.candidate_strategy == "coverage":
            head_limit = min(10, request.top_k, request.candidate_k)
            head = candidates[:head_limit]
            selected_ids = {hit.chunk.document_id for hit in head}
            by_id = {hit.chunk.document_id: hit for hit in candidates}
            tail: list[HybridSearchHit] = []
            branch_positions = [0 for _ in branch_rankings]
            if self.context_sibling_coverage:
                # Preserve the fused head, then reserve candidate slots for
                # evidence units in the same authorized TAT-QA context.  This
                # raises Candidate Recall without changing the initial Top-10
                # ranking; a later reranker may still promote the siblings.
                covered_parents: set[str] = set()
                context_seeds = candidates[: min(self.context_seed_k, request.candidate_k)]
                for hit in context_seeds:
                    raw_parent = hit.chunk.metadata.get("parent_document_id")
                    parent_id = (
                        raw_parent.strip()
                        if isinstance(raw_parent, str) and raw_parent.strip()
                        else hit.chunk.document_id
                    )
                    if parent_id in covered_parents:
                        continue
                    covered_parents.add(parent_id)
                    def sibling_sort_key(sibling_id: str) -> tuple[int, int, str]:
                        sibling = self._context_chunks.get(sibling_id)
                        raw_order = None if sibling is None else sibling.metadata.get("order")
                        try:
                            order = int(raw_order)
                        except (TypeError, ValueError):
                            order = 1_000_000
                        return (0 if order < 1_000_000 else 1, order, sibling_id)

                    sibling_ids = sorted(
                        self._context_children.get(parent_id, ()),
                        key=sibling_sort_key,
                    )
                    siblings_added = 0
                    for sibling_id in sibling_ids:
                        if (
                            self.context_sibling_limit is not None
                            and siblings_added >= self.context_sibling_limit
                        ):
                            break
                        if sibling_id in selected_ids:
                            continue
                        sibling = self._context_chunks.get(sibling_id)
                        if sibling is None or not _matches_scope(sibling, request):
                            continue
                        if request.allowed_chunk_ids is not None and sibling.chunk_id not in request.allowed_chunk_ids:
                            continue
                        selected_ids.add(sibling_id)
                        siblings_added += 1
                        if sibling_id in by_id:
                            tail.append(by_id[sibling_id])
                        else:
                            tail.append(
                                HybridSearchHit(
                                    chunk=sibling,
                                    rank=1,
                                    score=0.0,
                                    base_score=0.0,
                                    retrieval_sources=["context_sibling_coverage"],
                                )
                            )
                        if len(head) + len(tail) >= request.candidate_k:
                            break
                    if len(head) + len(tail) >= request.candidate_k:
                        break
            # A plain round-robin over many representations only exposes a
            # shallow prefix of each branch.  Explicit reserves are used by
            # table candidate generation to inspect deeper schema/row/cell
            # prefixes before the generic fair-fill pass.  This changes only
            # the candidate pool; the fused/reranked head remains untouched.
            if len(head) + len(tail) < request.candidate_k:
                branch_by_name = {
                    name: index for index, (name, _, _) in enumerate(self.branches)
                }
                for name, reserve in self.coverage_branch_reserves:
                    branch_index = branch_by_name[name]
                    ranking = branch_rankings[branch_index]
                    while (
                        branch_positions[branch_index] < min(reserve, len(ranking))
                        and len(head) + len(tail) < request.candidate_k
                    ):
                        document_id = ranking[branch_positions[branch_index]]
                        branch_positions[branch_index] += 1
                        if document_id in selected_ids:
                            continue
                        selected_ids.add(document_id)
                        tail.append(by_id[document_id])
            while len(head) + len(tail) < request.candidate_k:
                progressed = False
                for branch_index, ranking in enumerate(branch_rankings):
                    if len(head) + len(tail) >= request.candidate_k:
                        break
                    while branch_positions[branch_index] < len(ranking):
                        document_id = ranking[branch_positions[branch_index]]
                        branch_positions[branch_index] += 1
                        if document_id in selected_ids:
                            continue
                        selected_ids.add(document_id)
                        tail.append(by_id[document_id])
                        progressed = True
                        break
                if not progressed:
                    break
            candidates = head + tail
        if self.reranker is not None:
            rerank_limit = min(self.rerank_top_k, len(candidates))
            reranked_head = _apply_reranker(
                candidates[:rerank_limit],
                request,
                self.reranker,
            )
            candidates = reranked_head + candidates[rerank_limit:]
        return _response(
            "multi_representation_rrf",
            None,
            request,
            _rerank_positions(candidates[: request.top_k]),
            0,
            {**raw_candidate_counts, "fused": len(candidates)},
        )


class CandidateTailUnionIndex:
    """Reserve candidate-tail slots without changing a promoted ranking head.

    The primary backend remains authoritative for the stable head.  An extra
    retrieval branch may replace only the lowest-ranked tail candidates, so a
    later learned reranker can inspect novel evidence without claiming that
    the candidate generator itself improved final ranking quality.
    """

    def __init__(
        self,
        primary_backend: Any,
        candidate_backend: Any,
        *,
        preserve_head_k: int = 10,
        candidate_slots: int = 10,
    ) -> None:
        if not callable(getattr(primary_backend, "search", None)):
            raise TypeError("primary_backend must implement search(request)")
        if not callable(getattr(candidate_backend, "search", None)):
            raise TypeError("candidate_backend must implement search(request)")
        if not 1 <= preserve_head_k <= 100:
            raise ValueError("preserve_head_k must be between 1 and 100")
        if not 1 <= candidate_slots <= 100:
            raise ValueError("candidate_slots must be between 1 and 100")
        self.primary_backend = primary_backend
        self.candidate_backend = candidate_backend
        self.preserve_head_k = int(preserve_head_k)
        self.candidate_slots = int(candidate_slots)

    def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        primary = self.primary_backend.search(request)
        expected_filters = AppliedRetrievalFilters.from_request(request)
        if primary.filters_applied_before_ranking != expected_filters:
            raise HybridRetrievalError("candidate union primary branch changed trusted filters")
        candidate_request = request.model_copy(
            update={
                "top_k": request.candidate_k,
                "candidate_k": request.candidate_k,
                "rerank": False,
                "neighbor_window": 0,
                "max_expanded_hits": request.candidate_k,
            }
        )
        candidate = self.candidate_backend.search(candidate_request)
        if candidate.filters_applied_before_ranking != expected_filters:
            raise HybridRetrievalError(
                "candidate union additional branch changed trusted filters"
            )

        primary_hits: list[HybridSearchHit] = []
        primary_ids: set[str] = set()
        for hit in primary.hits:
            document_id = hit.chunk.document_id
            if document_id in primary_ids:
                continue
            primary_ids.add(document_id)
            primary_hits.append(hit)
        effective_slots = min(
            self.candidate_slots,
            max(0, request.candidate_k - min(self.preserve_head_k, len(primary_hits))),
        )
        primary_prefix_limit = max(
            min(self.preserve_head_k, request.candidate_k),
            request.candidate_k - effective_slots,
        )
        selected = list(primary_hits[:primary_prefix_limit])
        selected_ids = {hit.chunk.document_id for hit in selected}
        novel_added = 0
        for hit in candidate.hits:
            document_id = hit.chunk.document_id
            if document_id in selected_ids or document_id in primary_ids:
                continue
            selected_ids.add(document_id)
            novel_added += 1
            selected.append(
                hit.model_copy(
                    update={
                        "retrieval_sources": [
                            *hit.retrieval_sources,
                            "semantic_dense_candidate_tail",
                        ]
                    }
                )
            )
            if novel_added >= effective_slots:
                break
        if len(selected) < request.candidate_k:
            for hit in primary_hits[primary_prefix_limit:]:
                if hit.chunk.document_id in selected_ids:
                    continue
                selected_ids.add(hit.chunk.document_id)
                selected.append(hit)
                if len(selected) >= request.candidate_k:
                    break
        if len(selected) < request.candidate_k:
            for hit in candidate.hits:
                if hit.chunk.document_id in selected_ids:
                    continue
                selected_ids.add(hit.chunk.document_id)
                selected.append(
                    hit.model_copy(
                        update={
                            "retrieval_sources": [
                                *hit.retrieval_sources,
                                "semantic_dense_candidate_tail",
                            ]
                        }
                    )
                )
                if len(selected) >= request.candidate_k:
                    break
        output = _rerank_positions(selected[: request.candidate_k])
        return _response(
            "candidate_tail_union",
            candidate.collection_name or primary.collection_name,
            request,
            output,
            primary.expanded_neighbor_count,
            {
                **{
                    f"primary_{name}": count
                    for name, count in primary.raw_candidate_counts.items()
                },
                **{
                    f"candidate_{name}": count
                    for name, count in candidate.raw_candidate_counts.items()
                },
                "primary": len(primary_hits),
                "candidate": len(candidate.hits),
                "candidate_novel_tail": novel_added,
                "fused": len(output),
            },
        )


def _apply_reranker(
    hits: list[HybridSearchHit],
    request: HybridSearchRequest,
    reranker: Reranker | None,
    *,
    rerank_limit: int | None = None,
) -> list[HybridSearchHit]:
    if not request.rerank:
        return _rerank_positions(hits)
    if reranker is None:
        raise RerankerContractError("rerank=True requires a configured Reranker")
    if rerank_limit is not None:
        if isinstance(rerank_limit, bool) or not 1 <= int(rerank_limit) <= 100:
            raise ValueError("rerank_limit must be between 1 and 100")
        if rerank_limit < len(hits):
            reranked_head = _apply_reranker(
                hits[:rerank_limit],
                request,
                reranker,
            )
            # The tail remains in the dense/base ranking.  Reassign positions
            # only after concatenation so callers still receive a valid stable
            # ranking and the candidate pool remains unchanged.
            return _rerank_positions([*reranked_head, *hits[rerank_limit:]])
    try:
        raw_scores = reranker.score(request.query, [hit.chunk.text for hit in hits])
    except Exception as exc:
        raise RerankerContractError(f"reranker failed: {exc}") from exc
    if len(raw_scores) != len(hits):
        raise RerankerContractError("reranker must return exactly one score per candidate")
    feature_scores = [float(value) for value in raw_scores]
    if isinstance(reranker, TATQAFeatureReranker):
        base_values = [float(hit.base_score) for hit in hits]

        def _normalise(values: Sequence[float]) -> list[float]:
            lower = min(values, default=0.0)
            upper = max(values, default=0.0)
            if upper <= lower:
                return [0.0 for _ in values]
            return [(value - lower) / (upper - lower) for value in values]

        base_scores = _normalise(base_values)
        feature_scores = _normalise(feature_scores)
        weight = reranker.blend_weight
        feature_scores = [
            (1.0 - weight) * base_score + weight * feature_score
            for base_score, feature_score in zip(
                base_scores, feature_scores, strict=True
            )
        ]
    rescored: list[HybridSearchHit] = []
    for hit, score in zip(hits, feature_scores, strict=True):
        if not math.isfinite(score):
            raise RerankerContractError("reranker returned a non-finite score")
        sources = list(hit.retrieval_sources)
        if isinstance(reranker, LexicalOverlapFallbackReranker):
            sources.append("fallback_lexical_rerank")
        elif isinstance(reranker, FastEmbedCrossEncoderReranker):
            sources.append("learned_cross_encoder_rerank")
        elif isinstance(reranker, TATQAFeatureReranker):
            sources.append("tatqa_feature_rerank")
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


def _apply_adaptive_reranker(
    hits: list[HybridSearchHit],
    request: HybridSearchRequest,
    reranker: Reranker | None,
    *,
    min_k: int,
    max_k: int | None,
    margin_threshold: float,
) -> tuple[list[HybridSearchHit], AdaptiveRerankDiagnostics]:
    """Score ``min_k`` first and score the remaining prefix only when uncertain.

    Cross-encoder scores are pointwise and therefore comparable across the two
    batches.  The unscored candidate tail remains in its original dense order.
    This makes the saved computation genuine instead of scoring ``max_k`` and
    merely hiding part of the result after the fact.
    """

    if reranker is None:
        raise RerankerContractError("adaptive reranking requires a configured Reranker")
    if max_k is None or not 1 <= min_k < max_k <= 100:
        raise ValueError("adaptive rerank requires 1 <= min_k < max_k <= 100")
    if not math.isfinite(margin_threshold) or margin_threshold < 0.0:
        raise ValueError("adaptive rerank margin must be finite and non-negative")
    if not request.rerank:
        raise ValueError("adaptive rerank requires request.rerank=True")

    first_limit = min(min_k, len(hits))
    first = _apply_reranker(hits[:first_limit], request, reranker)
    first_scores = sorted(
        (
            float(hit.reranker_score)
            for hit in first
            if hit.reranker_score is not None
        ),
        reverse=True,
    )
    margin = (
        first_scores[0] - first_scores[1]
        if len(first_scores) >= 2
        else None
    )
    has_more_candidates = len(hits) > first_limit
    escalated = bool(
        has_more_candidates
        and margin is not None
        and margin < margin_threshold
    )
    applied_k = first_limit
    if escalated:
        second_limit = min(max_k, len(hits))
        second = _apply_reranker(
            hits[first_limit:second_limit],
            request,
            reranker,
        )
        scored = [*first, *second]
        scored.sort(
            key=lambda hit: (-hit.score, -hit.base_score, hit.chunk.chunk_id)
        )
        applied_k = second_limit
        output = _rerank_positions([*scored, *hits[second_limit:]])
        reason: Literal[
            "high_confidence", "low_score_margin", "insufficient_candidates"
        ] = "low_score_margin"
    else:
        output = _rerank_positions([*first, *hits[first_limit:]])
        reason = "high_confidence" if has_more_candidates else "insufficient_candidates"
    return output, AdaptiveRerankDiagnostics(
        min_k=min_k,
        max_k=max_k,
        applied_k=applied_k,
        escalated=escalated,
        top_score_margin=margin,
        margin_threshold=margin_threshold,
        reason=reason,
    )


def _raw_candidate_limit(request: HybridSearchRequest) -> int:
    if request.max_chunks_per_document is None:
        return request.candidate_k
    return min(2_000, request.candidate_k * max(3, request.max_chunks_per_document))


def _limit_chunks_per_document(
    hits: Sequence[HybridSearchHit],
    request: HybridSearchRequest,
) -> list[HybridSearchHit]:
    limit = request.max_chunks_per_document
    if limit is None:
        return list(hits)
    counts: Counter[str] = Counter()
    output: list[HybridSearchHit] = []
    for hit in hits:
        document_id = hit.chunk.document_id
        if counts[document_id] >= limit:
            continue
        counts[document_id] += 1
        output.append(hit)
    return _rerank_positions(output)


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
    backend: RetrievalBackend,
    collection_name: str | None,
    request: HybridSearchRequest,
    hits: list[HybridSearchHit],
    expanded_neighbors: int,
    raw_candidate_counts: Mapping[str, int] | None = None,
    *,
    adaptive_rerank: AdaptiveRerankDiagnostics | None = None,
) -> HybridSearchResponse:
    seed_count = sum(hit.neighbor_of_chunk_id is None for hit in hits)
    return HybridSearchResponse(
        backend=backend,
        collection_name=collection_name,
        query=request.query,
        filters_applied_before_ranking=AppliedRetrievalFilters.from_request(request),
        seed_count=seed_count,
        expanded_neighbor_count=expanded_neighbors,
        raw_candidate_counts=dict(raw_candidate_counts or {}),
        adaptive_rerank=adaptive_rerank,
        hits=hits,
    )


__all__ = [
    "AdaptiveRerankDiagnostics",
    "AppliedRetrievalFilters",
    "BM25Explanation",
    "BM25DenseRRFIndex",
    "BM25Index",
    "BM25TermContribution",
    "CandidateTailUnionIndex",
    "DenseEmbedder",
    "DeterministicHashEmbedder",
    "EmbeddingContractError",
    "FastEmbedCrossEncoderReranker",
    "FastEmbedSparseIndex",
    "InMemoryDenseIndex",
    "TATQAFeatureReranker",
    "TATQADomainReranker",
    "HybridChunk",
    "HybridRetrievalError",
    "HybridSearchHit",
    "HybridSearchRequest",
    "HybridSearchResponse",
    "LexicalOverlapFallbackReranker",
    "MultiQueryRRFIndex",
    "ParentChildIndex",
    "RepresentationRRFIndex",
    "SearchRepresentationIndex",
    "QdrantBackendError",
    "QdrantDenseIndex",
    "QdrantHybridIndex",
    "QdrantUnavailableError",
    "Reranker",
    "RerankerContractError",
    "SourceCoverageRRFIndex",
]

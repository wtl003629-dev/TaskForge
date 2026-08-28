"""Production-capable dense embedding and cross-encoder adapters.

No adapter is enabled implicitly.  The OpenAI adapter requires an explicit API
key and the FastEmbed adapters may be configured for cache-only operation so a
production process never downloads a model unexpectedly.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import sqlite3
import struct
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx

from .hybrid_retrieval import (
    DenseEmbedder,
    EmbeddingContractError,
    Reranker,
    RerankerContractError,
)


class SemanticProviderConfigurationError(ValueError):
    """A semantic provider is unavailable or not explicitly configured."""


class SemanticProviderHTTPError(RuntimeError):
    """Sanitised remote embedding failure that never includes response bodies."""


def _required(value: str, name: str) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise SemanticProviderConfigurationError(f"{name} is required")
    return cleaned


def _validate_texts(
    texts: Sequence[str],
    *,
    max_items: int,
    max_chars: int,
) -> list[str]:
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise EmbeddingContractError("embedding input must be a sequence of strings")
    if not texts:
        return []
    if len(texts) > max_items:
        raise EmbeddingContractError(f"embedding batch exceeds {max_items} items")
    result: list[str] = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingContractError("embedding text must be non-empty")
        if len(text) > max_chars:
            raise EmbeddingContractError(
                f"embedding text exceeds the {max_chars} character limit"
            )
        result.append(text)
    return result


def _vector(value: Any, *, dimension: int) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EmbeddingContractError("embedding vector must be an array")
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError, OverflowError) as exc:
        raise EmbeddingContractError("embedding vector must contain numbers") from exc
    if len(result) != dimension:
        raise EmbeddingContractError(
            f"embedding vector dimension {len(result)} does not match {dimension}"
        )
    if any(not math.isfinite(item) for item in result):
        raise EmbeddingContractError("embedding vector contains non-finite values")
    return result


class OpenAIEmbeddingsDenseEmbedder(DenseEmbedder):
    """Synchronous OpenAI-compatible ``/embeddings`` adapter for Qdrant setup."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimension: int,
        enabled: bool = False,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 60.0,
        batch_size: int = 64,
        max_text_chars: int = 100_000,
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = _required(api_key, "api_key")
        self._model = _required(model, "model")
        if not enabled:
            raise SemanticProviderConfigurationError(
                "OpenAI embeddings are disabled; set enabled=True explicitly"
            )
        if not 1 <= dimension <= 65_536:
            raise SemanticProviderConfigurationError(
                "dimension must be between 1 and 65536"
            )
        if not 1 <= batch_size <= 2_048:
            raise SemanticProviderConfigurationError(
                "batch_size must be between 1 and 2048"
            )
        if not 1 <= max_text_chars <= 2_000_000:
            raise SemanticProviderConfigurationError("max_text_chars is invalid")
        if timeout_seconds <= 0:
            raise SemanticProviderConfigurationError("timeout_seconds must be positive")
        self._dimension = int(dimension)
        self._base_url = _required(base_url, "base_url").rstrip("/")
        self._timeout = httpx.Timeout(timeout_seconds)
        self._batch_size = int(batch_size)
        self._max_text_chars = int(max_text_chars)
        self._owns_client = client is None
        self._client = client or httpx.Client(trust_env=False)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        safe = _validate_texts(
            texts,
            max_items=100_000,
            max_chars=self._max_text_chars,
        )
        result: list[list[float]] = []
        for start in range(0, len(safe), self._batch_size):
            result.extend(self._request(safe[start : start + self._batch_size]))
        return result

    def embed_query(self, text: str) -> list[float]:
        safe = _validate_texts(
            [text],
            max_items=1,
            max_chars=self._max_text_chars,
        )
        return self._request(safe)[0]

    def _request(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            response = self._client.post(
                f"{self._base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self._model, "input": list(texts)},
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise SemanticProviderHTTPError("embedding request timed out") from exc
        except httpx.RequestError as exc:
            raise SemanticProviderHTTPError("embedding request failed") from exc
        if not 200 <= response.status_code < 300:
            raise SemanticProviderHTTPError(
                f"embedding API returned HTTP {response.status_code}"
            )
        try:
            payload = response.json()
        except (ValueError, UnicodeDecodeError) as exc:
            raise EmbeddingContractError("embedding API returned invalid JSON") from exc
        if not isinstance(payload, Mapping):
            raise EmbeddingContractError("embedding response must be an object")
        raw_data = payload.get("data")
        if isinstance(raw_data, (str, bytes)) or not isinstance(raw_data, Sequence):
            raise EmbeddingContractError("embedding response data must be an array")
        indexed: dict[int, list[float]] = {}
        for item in raw_data:
            if not isinstance(item, Mapping):
                raise EmbeddingContractError("embedding data item must be an object")
            index = item.get("index")
            if type(index) is not int or not 0 <= index < len(texts):
                raise EmbeddingContractError("embedding item index is invalid")
            if index in indexed:
                raise EmbeddingContractError("embedding item indexes must be unique")
            indexed[index] = _vector(item.get("embedding"), dimension=self.dimension)
        if set(indexed) != set(range(len(texts))):
            raise EmbeddingContractError("embedding response is missing input items")
        return [indexed[index] for index in range(len(texts))]

    def close(self) -> None:
        if self._owns_client and not self._client.is_closed:
            self._client.close()

    def __enter__(self) -> OpenAIEmbeddingsDenseEmbedder:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class FastEmbedDenseEmbedder(DenseEmbedder):
    """Lazy FastEmbed dense adapter with optional cache-only enforcement."""

    def __init__(
        self,
        *,
        model_name: str,
        dimension: int,
        cache_dir: str | Path | None = None,
        local_files_only: bool = True,
        query_prefix: str = "query: ",
        document_prefix: str = "passage: ",
        model: Any | None = None,
    ) -> None:
        self.model_name = _required(model_name, "model_name")
        if not 1 <= dimension <= 65_536:
            raise SemanticProviderConfigurationError("dimension is invalid")
        self._dimension = dimension
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        if model is None:
            try:
                module = importlib.import_module("fastembed")
                model_type = module.TextEmbedding
            except (ImportError, AttributeError) as exc:
                raise SemanticProviderConfigurationError(
                    "FastEmbed is unavailable; install the semantic extra"
                ) from exc
            kwargs: dict[str, Any] = {
                "model_name": self.model_name,
                "local_files_only": local_files_only,
            }
            if cache_dir is not None:
                kwargs["cache_dir"] = str(Path(cache_dir))
            try:
                model = model_type(**kwargs)
            except Exception as exc:
                mode = "local cache" if local_files_only else "configured source"
                raise SemanticProviderConfigurationError(
                    f"FastEmbed model could not be loaded from {mode}"
                ) from exc
        self._model = model

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        safe = _validate_texts(texts, max_items=100_000, max_chars=100_000)
        return self._embed([self.document_prefix + value for value in safe])

    def embed_query(self, text: str) -> list[float]:
        safe = _validate_texts([text], max_items=1, max_chars=100_000)
        return self._embed([self.query_prefix + safe[0]])[0]

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            raw = list(self._model.embed(list(texts)))
        except Exception as exc:
            raise EmbeddingContractError("FastEmbed dense inference failed") from exc
        if len(raw) != len(texts):
            raise EmbeddingContractError(
                "FastEmbed must return one vector per input"
            )
        return [_vector(value, dimension=self.dimension) for value in raw]


class BailianDenseEmbedder(DenseEmbedder):
    """Alibaba Cloud Model Studio dense embedding adapter.

    The first production profile is deliberately narrow: ``text-embedding-v4``
    with 1024-dimensional float vectors. Document and query vectors share the
    same provider/model identity, while their cache rows remain distinct. The
    cache prevents a transient in-memory index rebuild from issuing paid API
    calls for unchanged text.
    """

    _CACHE_TABLE = "embeddings_bailian_v4_1024_v1"
    _MODEL_NAME = "text-embedding-v4"
    _DIMENSION = 1_024
    _MAX_BATCH_SIZE = 10
    _RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model_name: str = _MODEL_NAME,
        dimension: int = _DIMENSION,
        batch_size: int = _MAX_BATCH_SIZE,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        cache_path: str | Path | None = None,
        cache_store: Any | None = None,
        index_name: str = "knowledge-bailian-text-embedding-v4-1024-v1",
        client: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._api_key = _required(api_key, "bailian_api_key")
        self.model_name = _required(model_name, "bailian_model")
        if self.model_name.casefold() != self._MODEL_NAME:
            raise SemanticProviderConfigurationError(
                "BailianDenseEmbedder only supports text-embedding-v4"
            )
        if dimension != self._DIMENSION:
            raise SemanticProviderConfigurationError(
                "Bailian text-embedding-v4 must use dimension 1024"
            )
        if not 1 <= int(batch_size) <= self._MAX_BATCH_SIZE:
            raise SemanticProviderConfigurationError(
                "Bailian text-embedding-v4 batch_size must be between 1 and 10"
            )
        if timeout_seconds <= 0:
            raise SemanticProviderConfigurationError(
                "Bailian timeout_seconds must be positive"
            )
        if not 0 <= int(max_retries) <= 10:
            raise SemanticProviderConfigurationError(
                "Bailian max_retries must be between 0 and 10"
            )
        cleaned_base_url = _required(base_url, "bailian_base_url").rstrip("/")
        if not cleaned_base_url.startswith("https://"):
            raise SemanticProviderConfigurationError(
                "Bailian base URL must use HTTPS"
            )
        self._base_url = cleaned_base_url
        self._dimension = int(dimension)
        self._batch_size = int(batch_size)
        self._timeout = httpx.Timeout(float(timeout_seconds))
        self._max_retries = int(max_retries)
        self._index_name = _required(index_name, "bailian_index_name")
        self._cache_path = (
            Path(cache_path).resolve() if cache_path is not None else None
        )
        if self._cache_path is not None and cache_store is not None:
            raise SemanticProviderConfigurationError(
                "cache_path and cache_store are mutually exclusive"
            )
        self._cache_store = cache_store
        self._cache_model_id = (
            f"aliyun-bailian|{self.model_name}|dense-v1|{self.dimension}"
        )
        self._query_memory: dict[str, list[float]] = {}
        self._owns_client = client is None
        self._client = client or httpx.Client(trust_env=False)
        self._sleeper = sleeper
        if self._cache_path is not None:
            self._initialize_cache()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def index_name(self) -> str:
        return self._index_name

    @property
    def cache_path(self) -> Path | None:
        return self._cache_path

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        safe = _validate_texts(texts, max_items=100_000, max_chars=200_000)
        if self._cache_path is None and self._cache_store is None:
            return self._request_batches(safe)
        identities = [self._cache_identity("document", text) for text in safe]
        cached = self._load_cached(identities, embedding_kind="document")
        missing_by_key: dict[str, tuple[tuple[str, str], str]] = {}
        for identity, text in zip(identities, safe, strict=True):
            if identity[0] not in cached:
                missing_by_key.setdefault(identity[0], (identity, text))
        missing = list(missing_by_key.values())
        for offset in range(0, len(missing), self._batch_size):
            batch = missing[offset : offset + self._batch_size]
            vectors = self._request([text for _, text in batch])
            rows = [
                self._cache_row(identity, "document", vector)
                for (identity, _), vector in zip(batch, vectors, strict=True)
            ]
            self._store_cached(rows)
            cached.update(
                {
                    entry[0][0]: vector
                    for entry, vector in zip(batch, vectors, strict=True)
                }
            )
        return [list(cached[identity[0]]) for identity in identities]

    def embed_query(self, text: str) -> list[float]:
        safe = _validate_texts([text], max_items=1, max_chars=200_000)[0]
        memory = self._query_memory.get(safe)
        if memory is not None:
            return list(memory)
        if self._cache_path is None and self._cache_store is None:
            vector = self._request([safe])[0]
        else:
            identity = self._cache_identity("query", safe)
            cached = self._load_cached([identity], embedding_kind="query")
            vector = cached.get(identity[0])
            if vector is None:
                vector = self._request([safe])[0]
                self._store_cached([self._cache_row(identity, "query", vector)])
        self._query_memory[safe] = list(vector)
        return list(vector)

    def _request_batches(self, texts: Sequence[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for offset in range(0, len(texts), self._batch_size):
            result.extend(self._request(texts[offset : offset + self._batch_size]))
        return result

    def _request(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {
            "model": self.model_name,
            "input": list(texts),
            "dimensions": self.dimension,
            "encoding_format": "float",
        }
        response: httpx.Response | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(
                    f"{self._base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self._timeout,
                )
            except httpx.TimeoutException as exc:
                if attempt < self._max_retries:
                    self._sleeper(self._retry_delay(None, attempt))
                    continue
                raise SemanticProviderHTTPError(
                    "Bailian embedding request timed out"
                ) from exc
            except httpx.RequestError as exc:
                if attempt < self._max_retries:
                    self._sleeper(self._retry_delay(None, attempt))
                    continue
                raise SemanticProviderHTTPError(
                    "Bailian embedding request failed"
                ) from exc
            if 200 <= response.status_code < 300:
                return self._parse_response(response, expected_count=len(texts))
            if (
                response.status_code in self._RETRYABLE_STATUS_CODES
                and attempt < self._max_retries
            ):
                self._sleeper(self._retry_delay(response, attempt))
                continue
            raise SemanticProviderHTTPError(
                f"Bailian embedding API returned HTTP {response.status_code}"
            )
        raise SemanticProviderHTTPError("Bailian embedding request failed")

    @staticmethod
    def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
        if response is not None:
            raw = response.headers.get("Retry-After")
            if raw is not None:
                try:
                    return max(0.0, min(60.0, float(raw)))
                except (TypeError, ValueError, OverflowError):
                    pass
        return min(4.0, 0.25 * (2**attempt))

    def _parse_response(
        self,
        response: httpx.Response,
        *,
        expected_count: int,
    ) -> list[list[float]]:
        try:
            payload = response.json()
        except (ValueError, UnicodeDecodeError) as exc:
            raise EmbeddingContractError(
                "Bailian embedding API returned invalid JSON"
            ) from exc
        if not isinstance(payload, Mapping):
            raise EmbeddingContractError(
                "Bailian embedding response must be an object"
            )
        raw_data = payload.get("data")
        if isinstance(raw_data, (str, bytes)) or not isinstance(raw_data, Sequence):
            raise EmbeddingContractError(
                "Bailian embedding response data must be an array"
            )
        indexed: dict[int, list[float]] = {}
        for item in raw_data:
            if not isinstance(item, Mapping):
                raise EmbeddingContractError(
                    "Bailian embedding data item must be an object"
                )
            index = item.get("index")
            if type(index) is not int or not 0 <= index < expected_count:
                raise EmbeddingContractError(
                    "Bailian embedding item index is invalid"
                )
            if index in indexed:
                raise EmbeddingContractError(
                    "Bailian embedding item indexes must be unique"
                )
            indexed[index] = _vector(
                item.get("embedding"),
                dimension=self.dimension,
            )
        if set(indexed) != set(range(expected_count)):
            raise EmbeddingContractError(
                "Bailian embedding response is missing input items"
            )
        return [indexed[index] for index in range(expected_count)]

    def _initialize_cache(self) -> None:
        if self._cache_path is None:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self._cache_path, timeout=30.0) as connection:
                connection.execute("PRAGMA busy_timeout = 30000")
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
                f"failed to initialize Bailian embedding cache: {exc}"
            ) from exc

    def _cache_identity(self, embedding_kind: str, text: str) -> tuple[str, str]:
        text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cache_key = hashlib.sha256(
            "\0".join(
                (self._cache_model_id, embedding_kind, text_sha256)
            ).encode("utf-8")
        ).hexdigest()
        return cache_key, text_sha256

    def _load_cached(
        self,
        identities: Sequence[tuple[str, str]],
        *,
        embedding_kind: str,
    ) -> dict[str, list[float]]:
        if not identities or (self._cache_path is None and self._cache_store is None):
            return {}
        expected = dict(identities)
        loaded: dict[str, list[float]] = {}
        if self._cache_store is not None:
            return self._cache_store.load(
                model_name=self._cache_model_id,
                identities=identities,
                embedding_kind=embedding_kind,
                dimension=self.dimension,
            )
        keys = list(expected)
        try:
            with sqlite3.connect(self._cache_path, timeout=30.0) as connection:
                connection.execute("PRAGMA busy_timeout = 30000")
                for offset in range(0, len(keys), 500):
                    batch = keys[offset : offset + 500]
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
                    for cache_key, model_name, kind, text_sha256, dimension, blob in rows:
                        if (
                            model_name != self._cache_model_id
                            or kind != embedding_kind
                            or text_sha256 != expected[cache_key]
                            or int(dimension) != self.dimension
                        ):
                            raise EmbeddingContractError(
                                "Bailian embedding cache metadata does not match the request"
                            )
                        if not isinstance(blob, bytes) or len(blob) != self.dimension * 4:
                            raise EmbeddingContractError(
                                "Bailian embedding cache contains a corrupt vector"
                            )
                        loaded[cache_key] = _vector(
                            struct.unpack(f"<{self.dimension}f", blob),
                            dimension=self.dimension,
                        )
        except EmbeddingContractError:
            raise
        except (OSError, sqlite3.Error, KeyError) as exc:
            raise EmbeddingContractError(
                f"failed to read Bailian embedding cache: {exc}"
            ) from exc
        return loaded

    def _store_cached(self, rows: Sequence[tuple[object, ...]]) -> None:
        if not rows or (self._cache_path is None and self._cache_store is None):
            return
        if self._cache_store is not None:
            self._cache_store.store(rows)
            return
        try:
            with sqlite3.connect(self._cache_path, timeout=30.0) as connection:
                connection.execute("PRAGMA busy_timeout = 30000")
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
            raise EmbeddingContractError(
                f"failed to write Bailian embedding cache: {exc}"
            ) from exc

    def _cache_row(
        self,
        identity: tuple[str, str],
        embedding_kind: str,
        vector: Sequence[float],
    ) -> tuple[object, ...]:
        validated = _vector(vector, dimension=self.dimension)
        return (
            identity[0],
            self._cache_model_id,
            embedding_kind,
            identity[1],
            self.dimension,
            struct.pack(f"<{self.dimension}f", *validated),
        )

    def close(self) -> None:
        if self._owns_client and not self._client.is_closed:
            self._client.close()

    def __enter__(self) -> BailianDenseEmbedder:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class BgeM3DenseEmbedder(DenseEmbedder):
    """BGE-M3 dense adapter with an isolated SQLite vector cache.

    BGE-M3 is intentionally kept separate from the FastEmbed adapter: the
    model emits a 1024-dimensional dense vector through FlagEmbedding and its
    sparse/ColBERT outputs are not enabled in this first controlled rollout.
    """

    _CACHE_TABLE = "embeddings_bge_m3_v1"
    _CACHE_MODEL_ID = "BAAI/bge-m3|dense-v1|1024"

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-m3",
        model_path: str | Path | None = None,
        cache_path: str | Path | None = None,
        cache_store: Any | None = None,
        cache_dir: str | Path | None = None,
        dimension: int = 1_024,
        batch_size: int = 8,
        max_length: int = 8_192,
        device: str = "auto",
        use_fp16: bool | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_name = _required(model_name, "model_name")
        if self.model_name.casefold() != "baai/bge-m3":
            raise SemanticProviderConfigurationError(
                "BgeM3DenseEmbedder only supports model BAAI/bge-m3"
            )
        if dimension != 1_024:
            raise SemanticProviderConfigurationError(
                "BGE-M3 dense output must use dimension 1024"
            )
        if not 1 <= batch_size <= 256:
            raise SemanticProviderConfigurationError(
                "batch_size must be between 1 and 256"
            )
        if not 128 <= max_length <= 8_192:
            raise SemanticProviderConfigurationError(
                "max_length must be between 128 and 8192"
            )
        if device not in {"auto", "cpu", "cuda"}:
            raise SemanticProviderConfigurationError(
                "device must be auto, cpu, or cuda"
            )
        self._dimension = int(dimension)
        self._batch_size = int(batch_size)
        self._max_length = int(max_length)
        self._cache_path = Path(cache_path).resolve() if cache_path is not None else None
        if self._cache_path is not None and cache_store is not None:
            raise SemanticProviderConfigurationError(
                "cache_path and cache_store are mutually exclusive"
            )
        self._cache_store = cache_store
        self._query_memory: dict[str, list[float]] = {}
        if model is None:
            if model_path is None:
                raise SemanticProviderConfigurationError(
                    "BGE-M3 requires an explicit local model_path"
                )
            resolved_model_path = Path(model_path).resolve()
            if not resolved_model_path.is_dir():
                raise SemanticProviderConfigurationError(
                    "BGE-M3 model_path does not exist; download the model to D: first"
                )
            try:
                from FlagEmbedding import BGEM3FlagModel
            except (ImportError, AttributeError) as exc:
                raise SemanticProviderConfigurationError(
                    "FlagEmbedding is required for BGE-M3"
                ) from exc
            kwargs: dict[str, Any] = {
                "normalize_embeddings": True,
                "use_fp16": (
                    device == "cuda" if use_fp16 is None else bool(use_fp16)
                ),
                "batch_size": self._batch_size,
                "query_max_length": self._max_length,
                "passage_max_length": self._max_length,
                "return_dense": True,
                "return_sparse": False,
                "return_colbert_vecs": False,
            }
            if device != "auto":
                kwargs["devices"] = device
            if cache_dir is not None:
                kwargs["cache_dir"] = str(Path(cache_dir).resolve())
            try:
                model = BGEM3FlagModel(str(resolved_model_path), **kwargs)
            except Exception as exc:
                raise SemanticProviderConfigurationError(
                    "BGE-M3 model could not be loaded from local cache"
                ) from exc
        self._model = model
        if self._cache_path is not None:
            self._initialize_cache()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def index_name(self) -> str:
        """Stable model-qualified name for the BGE-M3 dense index."""

        return "knowledge-bge-m3-v1"

    @property
    def cache_path(self) -> Path | None:
        return self._cache_path

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        safe = _validate_texts(texts, max_items=100_000, max_chars=200_000)
        if self._cache_path is None and self._cache_store is None:
            return self._encode(safe)
        identities = [self._cache_identity("document", text) for text in safe]
        cached = self._load_cached(identities, embedding_kind="document")
        missing = [
            (identity, text)
            for identity, text in zip(identities, safe, strict=True)
            if identity[0] not in cached
        ]
        for offset in range(0, len(missing), self._batch_size):
            batch = missing[offset : offset + self._batch_size]
            vectors = self._encode([text for _, text in batch])
            rows = [
                self._cache_row(identity, "document", vector)
                for (identity, _), vector in zip(batch, vectors, strict=True)
            ]
            self._store_cached(rows)
            cached.update(
                {
                    entry[0][0]: vector
                    for entry, vector in zip(batch, vectors, strict=True)
                }
            )
        return [cached[identity[0]] for identity in identities]

    def embed_query(self, text: str) -> list[float]:
        safe = _validate_texts([text], max_items=1, max_chars=200_000)[0]
        memory = self._query_memory.get(safe)
        if memory is not None:
            return list(memory)
        if self._cache_path is None and self._cache_store is None:
            vector = self._encode([safe])[0]
        else:
            identity = self._cache_identity("query", safe)
            cached = self._load_cached([identity], embedding_kind="query")
            vector = cached.get(identity[0])
            if vector is None:
                vector = self._encode([safe])[0]
                self._store_cached([self._cache_row(identity, "query", vector)])
        self._query_memory[safe] = list(vector)
        return vector

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        # FlagEmbedding otherwise pads every short batch to the configured
        # model limit.  Child chunks are normally far shorter than 8k tokens,
        # so derive a batch-local upper bound without truncating any input;
        # the configured limit still applies to genuinely long text.
        max_length = self._effective_max_length(texts)
        try:
            payload = self._model.encode(
                list(texts),
                batch_size=self._batch_size,
                max_length=max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        except Exception as exc:
            raise EmbeddingContractError("BGE-M3 dense inference failed") from exc
        if not isinstance(payload, Mapping):
            raise EmbeddingContractError("BGE-M3 response must be an object")
        dense = payload.get("dense_vecs")
        if hasattr(dense, "tolist"):
            dense = dense.tolist()
        if isinstance(dense, (str, bytes)) or not isinstance(dense, Sequence):
            raise EmbeddingContractError("BGE-M3 dense vectors must be an array")
        if len(dense) != len(texts):
            raise EmbeddingContractError(
                "BGE-M3 must return one dense vector per input"
            )
        return [_vector(value, dimension=self.dimension) for value in dense]

    def _effective_max_length(self, texts: Sequence[str]) -> int:
        tokenizer = getattr(self._model, "tokenizer", None)
        if tokenizer is None:
            return self._max_length
        try:
            encoded = tokenizer(
                list(texts),
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_length=True,
            )
            lengths = encoded.get("length") if isinstance(encoded, Mapping) else None
            if isinstance(lengths, int):
                lengths = [lengths]
            if isinstance(lengths, Sequence) and lengths:
                observed = max(int(length) for length in lengths)
                return min(self._max_length, max(128, observed))
        except (TypeError, ValueError, OverflowError):
            pass
        return self._max_length

    def _initialize_cache(self) -> None:
        if self._cache_path is None:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self._cache_path) as connection:
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
                f"failed to initialize BGE-M3 embedding cache: {exc}"
            ) from exc

    def _cache_identity(
        self,
        embedding_kind: str,
        text: str,
    ) -> tuple[str, str]:
        import hashlib
        text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cache_key = hashlib.sha256(
            "\0".join((self._CACHE_MODEL_ID, embedding_kind, text_sha256)).encode(
                "utf-8"
            )
        ).hexdigest()
        return cache_key, text_sha256

    def _load_cached(
        self,
        identities: Sequence[tuple[str, str]],
        *,
        embedding_kind: str,
    ) -> dict[str, list[float]]:
        if not identities or (self._cache_path is None and self._cache_store is None):
            return {}
        expected = dict(identities)
        loaded: dict[str, list[float]] = {}
        if self._cache_store is not None:
            return self._cache_store.load(
                model_name=self._CACHE_MODEL_ID,
                identities=identities,
                embedding_kind=embedding_kind,
                dimension=self.dimension,
            )
        try:
            with sqlite3.connect(self._cache_path) as connection:
                placeholders = ",".join("?" for _ in expected)
                rows = connection.execute(
                    f"""
                    SELECT cache_key, model_name, embedding_kind,
                           text_sha256, dimension, vector
                    FROM {self._CACHE_TABLE}
                    WHERE cache_key IN ({placeholders})
                    """,
                    list(expected),
                ).fetchall()
            for cache_key, model_name, kind, text_sha256, dimension, blob in rows:
                if (
                    model_name != self._CACHE_MODEL_ID
                    or kind != embedding_kind
                    or text_sha256 != expected[cache_key]
                    or int(dimension) != self.dimension
                ):
                    raise EmbeddingContractError(
                        "BGE-M3 embedding cache metadata does not match the request"
                    )
                if not isinstance(blob, bytes) or len(blob) != self.dimension * 4:
                    raise EmbeddingContractError(
                        "BGE-M3 embedding cache contains a corrupt vector"
                    )
                loaded[cache_key] = _vector(
                    struct.unpack(f"<{self.dimension}f", blob),
                    dimension=self.dimension,
                )
        except EmbeddingContractError:
            raise
        except (OSError, sqlite3.Error, KeyError) as exc:
            raise EmbeddingContractError(
                f"failed to read BGE-M3 embedding cache: {exc}"
            ) from exc
        return loaded

    def _store_cached(self, rows: Sequence[tuple[object, ...]]) -> None:
        if not rows or (self._cache_path is None and self._cache_store is None):
            return
        if self._cache_store is not None:
            self._cache_store.store(rows)
            return
        try:
            with sqlite3.connect(self._cache_path) as connection:
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
            raise EmbeddingContractError(
                f"failed to write BGE-M3 embedding cache: {exc}"
            ) from exc

    def _cache_row(
        self,
        identity: tuple[str, str],
        embedding_kind: str,
        vector: Sequence[float],
    ) -> tuple[object, ...]:
        validated = _vector(vector, dimension=self.dimension)
        return (
            identity[0],
            self._CACHE_MODEL_ID,
            embedding_kind,
            identity[1],
            self.dimension,
            struct.pack(f"<{self.dimension}f", *validated),
        )


class FastEmbedCrossEncoderReranker(Reranker):
    """Lazy FastEmbed cross-encoder reranker; scores are not probabilities."""

    def __init__(
        self,
        *,
        model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        cache_dir: str | Path | None = None,
        local_files_only: bool = True,
        model: Any | None = None,
    ) -> None:
        self.model_name = _required(model_name, "model_name")
        if model is None:
            try:
                module = importlib.import_module("fastembed.rerank.cross_encoder")
                model_type = module.TextCrossEncoder
            except (ImportError, AttributeError) as exc:
                raise SemanticProviderConfigurationError(
                    "FastEmbed cross encoder is unavailable; install the semantic extra"
                ) from exc
            kwargs: dict[str, Any] = {
                "model_name": self.model_name,
                "local_files_only": local_files_only,
            }
            if cache_dir is not None:
                kwargs["cache_dir"] = str(Path(cache_dir))
            try:
                model = model_type(**kwargs)
            except Exception as exc:
                mode = "local cache" if local_files_only else "configured source"
                raise SemanticProviderConfigurationError(
                    f"FastEmbed reranker could not be loaded from {mode}"
                ) from exc
        self._model = model

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not isinstance(query, str) or not query.strip():
            raise RerankerContractError("reranker query must be non-empty")
        safe = _validate_texts(documents, max_items=1_000, max_chars=100_000)
        try:
            raw = list(self._model.rerank(query, safe))
            result = [float(value) for value in raw]
        except Exception as exc:
            raise RerankerContractError("FastEmbed reranker inference failed") from exc
        if len(result) != len(safe):
            raise RerankerContractError(
                "FastEmbed reranker must return one score per document"
            )
        if any(not math.isfinite(value) for value in result):
            raise RerankerContractError("FastEmbed reranker returned non-finite scores")
        return result


__all__ = [
    "BailianDenseEmbedder",
    "BgeM3DenseEmbedder",
    "FastEmbedCrossEncoderReranker",
    "FastEmbedDenseEmbedder",
    "OpenAIEmbeddingsDenseEmbedder",
    "SemanticProviderConfigurationError",
    "SemanticProviderHTTPError",
]

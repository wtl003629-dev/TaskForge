"""Production-capable dense embedding and cross-encoder adapters.

No adapter is enabled implicitly.  The OpenAI adapter requires an explicit API
key and the FastEmbed adapters may be configured for cache-only operation so a
production process never downloads a model unexpectedly.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
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
    "FastEmbedCrossEncoderReranker",
    "FastEmbedDenseEmbedder",
    "OpenAIEmbeddingsDenseEmbedder",
    "SemanticProviderConfigurationError",
    "SemanticProviderHTTPError",
]

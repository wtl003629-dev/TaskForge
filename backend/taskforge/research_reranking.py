"""Provider-neutral research reranker adapters.

The product contract is intentionally tiny: a reranker receives one query and
an ordered candidate list and returns one finite score per candidate. The
FlagEmbedding adapter is optional and is only constructed when explicitly
selected, so an environment without PyTorch cannot silently change the
default retrieval path.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import httpx

from .hybrid_retrieval import (
    FastEmbedCrossEncoderReranker,
    Reranker,
    RerankerContractError,
)

try:  # Optional: installed only for the explicit FlagEmbedding backend.
    from FlagEmbedding import FlagReranker
except ImportError:  # pragma: no cover - depends on deployment extras.
    FlagReranker = None  # type: ignore[assignment,misc]

try:  # Optional PyTorch inference backend for domain-tuned checkpoints.
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:  # pragma: no cover - optional training/inference extra.
    torch = None  # type: ignore[assignment]
    AutoModelForSequenceClassification = None  # type: ignore[assignment,misc]
    AutoTokenizer = None  # type: ignore[assignment,misc]


class BGEV2M3Reranker:
    """BGE cross-encoder adapter backed by ``FlagEmbedding.FlagReranker``."""

    backend = "flagembedding"

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        *,
        device: Literal["auto", "cpu", "cuda"] = "auto",
        batch_size: int = 16,
        use_fp16: bool | None = None,
    ) -> None:
        if FlagReranker is None:
            raise RerankerContractError(
                "FlagEmbedding is required for the flagembedding research reranker; "
                "install the optional reranker dependencies explicitly"
            )
        cleaned = str(model_name).strip()
        if not cleaned:
            raise ValueError("reranker_model must not be empty")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("reranker_device must be auto, cpu, or cuda")
        if not 1 <= int(batch_size) <= 512:
            raise ValueError("reranker batch_size must be between 1 and 512")
        self.model_name = cleaned
        self.device = device
        self.batch_size = int(batch_size)
        # FlagEmbedding's reranker API exposes fp16 rather than a generic
        # device parameter. CUDA is selected by its normal torch environment.
        resolved_fp16 = device == "cuda" if use_fp16 is None else bool(use_fp16)
        try:
            model_kwargs: dict[str, Any] = {"use_fp16": resolved_fp16}
            if device == "cuda":
                model_kwargs["devices"] = "cuda"
            elif device == "cpu":
                model_kwargs["devices"] = "cpu"
            self._model = FlagReranker(cleaned, **model_kwargs)
        except Exception as exc:
            raise RerankerContractError(
                f"failed to initialize FlagEmbedding reranker {cleaned!r}: {exc}"
            ) from exc
        self.use_fp16 = resolved_fp16
        self._scored_pairs = 0

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        pairs = [[str(query), str(document)] for document in documents]
        if not pairs:
            return []
        self._scored_pairs += len(pairs)
        try:
            values = self._model.compute_score(
                pairs,
                batch_size=self.batch_size,
                normalize=True,
            )
        except TypeError:
            # Older FlagEmbedding versions do not expose batch_size or
            # normalize as keyword arguments. Keep the adapter compatible but
            # retain normalization/validation below.
            try:
                values = self._model.compute_score(pairs)
            except Exception as exc:
                raise RerankerContractError(
                    f"BGE reranker inference failed: {exc}"
                ) from exc
        except Exception as exc:
            raise RerankerContractError(f"BGE reranker inference failed: {exc}") from exc
        if isinstance(values, (int, float)):
            values = [values]
        try:
            scores = [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise RerankerContractError("BGE reranker returned non-numeric scores") from exc
        if len(scores) != len(documents) or not all(math.isfinite(value) for value in scores):
            raise RerankerContractError(
                "BGE reranker must return one finite score per candidate"
            )
        return scores

    def telemetry(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model_name,
            "device": self.device,
            "use_fp16": self.use_fp16,
            "batch_size": self.batch_size,
            "scored_pairs": self._scored_pairs,
        }


class FastEmbedEnsembleReranker:
    """Average rank-normalized scores from multiple local cross-encoders.

    The models are deliberately normalized per query because Jina and MiniLM
    use different score scales.  This is an explicit high-recall experiment;
    it is not silently selected when a single model is configured.
    """

    backend = "fastembed_ensemble"

    def __init__(
        self,
        model_names: Sequence[str],
        *,
        cache_dir: str | Path | None = None,
        batch_size: int = 32,
    ) -> None:
        raw = tuple(str(name).strip() for name in model_names if str(name).strip())
        parsed: list[tuple[float, str]] = []
        for name in raw:
            pieces = name.split("::", 1)
            try:
                weight = float(pieces[0]) if len(pieces) == 2 else 1.0
                model_name = pieces[1] if len(pieces) == 2 else name
            except ValueError:
                weight, model_name = 1.0, name
            if weight <= 0 or not model_name:
                raise ValueError("ensemble weights and model names must be positive/non-empty")
            parsed.append((weight, model_name))
        # Keep explicit weighted duplicates; they are useful for giving a
        # domain-tuned checkpoint more influence than a zero-shot model.
        cleaned = tuple(model_name for _, model_name in parsed)
        if len(cleaned) < 2:
            raise ValueError("fastembed ensemble requires at least two model names")
        self.model_names = cleaned
        total_weight = sum(weight for weight, _ in parsed)
        self._weights = tuple(weight / total_weight for weight, _ in parsed)
        self._models = tuple(
            TransformerCrossEncoderReranker(name.removeprefix("transformers::"), batch_size=batch_size)
            if name.startswith("transformers::")
            else FastEmbedCrossEncoderReranker(
                name,
                cache_dir=cache_dir,
                batch_size=batch_size,
            )
            for name in cleaned
        )

    @staticmethod
    def _normalize(values: Sequence[float]) -> list[float]:
        if not values:
            return []
        low, high = min(values), max(values)
        if high <= low:
            return [0.0] * len(values)
        return [(float(value) - low) / (high - low) for value in values]

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        outputs = [self._normalize(model.score(query, documents)) for model in self._models]
        return [
            sum(weight * values[index] for weight, values in zip(self._weights, outputs, strict=True))
            for index in range(len(documents))
        ]

    def telemetry(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "models": list(self.model_names),
            "weights": list(self._weights),
            "model_telemetry": [model.telemetry() for model in self._models],
        }


class TransformerCrossEncoderReranker:
    """Batch PyTorch cross-encoder for locally fine-tuned checkpoints."""

    backend = "transformers"

    def __init__(
        self,
        model_name: str,
        *,
        device: Literal["auto", "cpu", "cuda"] = "auto",
        batch_size: int = 16,
        max_length: int = 512,
        windowed: bool = False,
    ) -> None:
        if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise RerankerContractError("transformers and torch are required for the transformers reranker")
        if device == "cuda" and not torch.cuda.is_available():
            raise RerankerContractError("CUDA was requested but torch.cuda.is_available() is false")
        resolved = "cuda" if device == "cuda" or (device == "auto" and torch.cuda.is_available()) else "cpu"
        self.model_name = str(model_name).strip()
        self.device = resolved
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.windowed = bool(windowed)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        self._scored_pairs = 0

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        # Long PDF/page chunks frequently place the answer after the first
        # 512 tokens. Score overlapping passage windows and keep the best
        # window per candidate. This spends more CPU for a materially higher
        # recall ceiling, while preserving the candidate's identity.
        windows: list[tuple[int, str]] = []
        window_size = max(32, self.max_length - 32)
        stride = max(16, int(window_size * 0.67))
        for index, document in enumerate(documents):
            text = str(document)
            if not self.windowed:
                windows.append((index, text))
                continue
            token_ids = self._tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
            if len(token_ids) <= window_size:
                windows.append((index, text))
                continue
            for start in range(0, len(token_ids), stride):
                end = min(len(token_ids), start + window_size)
                window_text = self._tokenizer.decode(token_ids[start:end], skip_special_tokens=True)
                if window_text.strip():
                    windows.append((index, window_text))
                if end >= len(token_ids):
                    break
        scores = [float("-inf")] * len(documents)
        with torch.inference_mode():
            for start in range(0, len(windows), self.batch_size):
                batch = windows[start : start + self.batch_size]
                encoded = self._tokenizer(
                    [str(query)] * len(batch),
                    [value for _, value in batch],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self._model(**encoded).logits
                values = logits[:, 0] if logits.ndim == 2 else logits
                for (document_index, _), value in zip(batch, values.detach().cpu().tolist(), strict=True):
                    scores[document_index] = max(scores[document_index], float(value))
        self._scored_pairs += len(documents)
        return [0.0 if value == float("-inf") else value for value in scores]

    def telemetry(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "windowed": self.windowed,
            "scored_pairs": self._scored_pairs,
        }


class BailianReranker:
    """Alibaba Cloud Model Studio ``qwen3-rerank`` adapter.

    The adapter follows the provider-neutral ``score(query, documents)``
    contract used by the research retriever.  The API returns ranked indexes;
    this class restores scores to the caller's original candidate order so
    downstream code never loses chunk identity.
    """

    backend = "bailian"
    _MAX_DOCUMENTS = 500
    _RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

    def __init__(
        self,
        model_name: str = "qwen3-rerank",
        *,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-api/v1",
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        client: httpx.Client | None = None,
        sleeper: Any = time.sleep,
    ) -> None:
        cleaned_key = str(api_key or "").strip()
        if not cleaned_key:
            raise RerankerContractError("bailian reranker requires an API key")
        self.model_name = str(model_name or "").strip()
        if self.model_name != "qwen3-rerank":
            raise RerankerContractError(
                "BailianReranker currently supports qwen3-rerank only"
            )
        cleaned_base = str(base_url or "").strip().rstrip("/")
        if cleaned_base.endswith("/compatible-mode/v1"):
            # The existing embedding base is OpenAI-compatible; qwen3-rerank
            # uses Bailian's compatible-api route.
            cleaned_base = cleaned_base[: -len("/compatible-mode/v1")] + "/compatible-api/v1"
        if not cleaned_base.startswith("https://"):
            raise RerankerContractError("Bailian reranker base URL must use HTTPS")
        if timeout_seconds <= 0:
            raise RerankerContractError("Bailian reranker timeout must be positive")
        if not 0 <= int(max_retries) <= 10:
            raise RerankerContractError("Bailian reranker max_retries must be 0..10")
        self._api_key = cleaned_key
        self._base_url = cleaned_base
        self._timeout = httpx.Timeout(float(timeout_seconds))
        self._max_retries = int(max_retries)
        self._owns_client = client is None
        self._client = client or httpx.Client(trust_env=False)
        self._sleeper = sleeper
        self._scored_pairs = 0
        self._requests = 0

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        values = [str(document) for document in documents]
        if not values:
            return []
        if len(values) > self._MAX_DOCUMENTS:
            raise RerankerContractError(
                f"Bailian qwen3-rerank accepts at most {self._MAX_DOCUMENTS} documents"
            )
        payload = {
            "model": self.model_name,
            "query": str(query),
            "documents": values,
            "top_n": len(values),
            "instruct": "Given a web search query, retrieve relevant passages that answer the query.",
        }
        response: httpx.Response | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(
                    f"{self._base_url}/reranks",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self._timeout,
                )
            except httpx.TimeoutException as exc:
                if attempt < self._max_retries:
                    self._sleeper(min(4.0, 0.25 * (2**attempt)))
                    continue
                raise RerankerContractError("Bailian reranker request timed out") from exc
            except httpx.RequestError as exc:
                if attempt < self._max_retries:
                    self._sleeper(min(4.0, 0.25 * (2**attempt)))
                    continue
                raise RerankerContractError("Bailian reranker request failed") from exc
            self._requests += 1
            if 200 <= response.status_code < 300:
                return self._parse_response(response, len(values))
            if response.status_code in self._RETRYABLE_STATUS_CODES and attempt < self._max_retries:
                self._sleeper(min(4.0, 0.25 * (2**attempt)))
                continue
            detail = response.text[:300].replace("\n", " ")
            raise RerankerContractError(
                f"Bailian reranker API returned HTTP {response.status_code}: {detail}"
            )
        raise RerankerContractError("Bailian reranker request failed")

    def _parse_response(self, response: httpx.Response, expected_count: int) -> list[float]:
        try:
            payload = response.json()
        except (ValueError, UnicodeDecodeError) as exc:
            raise RerankerContractError("Bailian reranker returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RerankerContractError("Bailian reranker response must be an object")
        raw_results = payload.get("results")
        if raw_results is None and isinstance(payload.get("output"), dict):
            raw_results = payload["output"].get("results")
        if isinstance(raw_results, (str, bytes)) or not isinstance(raw_results, Sequence):
            raise RerankerContractError("Bailian reranker response results must be an array")
        scores = [float("-inf")] * expected_count
        seen: set[int] = set()
        for item in raw_results:
            if not isinstance(item, dict):
                raise RerankerContractError("Bailian reranker result must be an object")
            raw_index = item.get("index")
            if type(raw_index) is not int or not 0 <= raw_index < expected_count:
                raise RerankerContractError("Bailian reranker result index is invalid")
            if raw_index in seen:
                raise RerankerContractError("Bailian reranker result indexes must be unique")
            raw_score = item.get("relevance_score", item.get("score"))
            try:
                score = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise RerankerContractError("Bailian reranker score is not numeric") from exc
            if not math.isfinite(score):
                raise RerankerContractError("Bailian reranker score is not finite")
            scores[raw_index] = score
            seen.add(raw_index)
        if seen != set(range(expected_count)):
            raise RerankerContractError("Bailian reranker response is missing candidate scores")
        self._scored_pairs += expected_count
        return scores

    def telemetry(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model_name,
            "base_url": self._base_url,
            "scored_pairs": self._scored_pairs,
            "requests": self._requests,
        }

    def close(self) -> None:
        if self._owns_client and not self._client.is_closed:
            self._client.close()


def build_research_reranker(
    backend: Literal["fastembed", "flagembedding", "fastembed_ensemble", "transformers", "bailian"],
    model_name: str,
    *,
    device: Literal["auto", "cpu", "cuda"] = "auto",
    batch_size: int = 32,
    fastembed_cache_dir: str | Path | None = None,
    bailian_api_key: str | None = None,
    bailian_base_url: str = "https://dashscope.aliyuncs.com/compatible-api/v1",
    bailian_timeout_seconds: float = 30.0,
    bailian_max_retries: int = 2,
) -> Reranker:
    """Construct the configured adapter and fail loudly on unavailable extras."""

    if backend == "fastembed":
        return FastEmbedCrossEncoderReranker(
            model_name,
            cache_dir=fastembed_cache_dir,
            batch_size=batch_size,
        )
    if backend == "flagembedding":
        return BGEV2M3Reranker(
            model_name,
            device=device,
            batch_size=batch_size,
        )
    if backend == "fastembed_ensemble":
        return FastEmbedEnsembleReranker(
            str(model_name).split(","),
            cache_dir=fastembed_cache_dir,
            batch_size=batch_size,
        )
    if backend == "transformers":
        return TransformerCrossEncoderReranker(model_name, device=device, batch_size=batch_size)
    if backend == "bailian":
        return BailianReranker(
            model_name,
            api_key=bailian_api_key or "",
            base_url=bailian_base_url,
            timeout_seconds=bailian_timeout_seconds,
            max_retries=bailian_max_retries,
        )
    raise ValueError(f"unsupported research reranker backend: {backend}")


__all__ = [
    "BGEV2M3Reranker",
    "BailianReranker",
    "FastEmbedEnsembleReranker",
    "TransformerCrossEncoderReranker",
    "build_research_reranker",
]

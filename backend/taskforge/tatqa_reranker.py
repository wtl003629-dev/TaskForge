"""Small, deterministic, domain-trainable rerank model for TAT-QA.

The model is deliberately lightweight: it learns a logistic ranking score
over query/document overlap features and has no external runtime dependency.
It is an evaluation provider, not a claim that a remote LLM or BGE-M3 is
available.  A stronger cross-encoder can implement the same ``score``
contract later without changing retrieval orchestration.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .knowledge import tokenise

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")
_FEATURE_NAMES = (
    "token_coverage",
    "token_jaccard",
    "numeric_coverage",
    "year_coverage",
    "query_number_presence",
    "table_marker",
    "count_query_table_marker",
    "arithmetic_query_numeric_presence",
    "short_document_bonus",
)


def _normalised_numbers(text: str) -> set[str]:
    return {value.replace(",", "") for value in _NUMBER.findall(text)}


def _features(query: str, document: str) -> list[float]:
    query_terms = set(tokenise(query))
    document_terms = set(tokenise(document))
    overlap = query_terms.intersection(document_terms)
    token_coverage = len(overlap) / len(query_terms) if query_terms else 0.0
    union = query_terms.union(document_terms)
    token_jaccard = len(overlap) / len(union) if union else 0.0
    query_numbers = _normalised_numbers(query)
    document_numbers = _normalised_numbers(document)
    numeric_coverage = (
        len(query_numbers.intersection(document_numbers)) / len(query_numbers)
        if query_numbers
        else 0.0
    )
    query_years = {
        value
        for value in query_numbers
        if len(value) == 4 and value[:2] in {"19", "20"}
    }
    year_coverage = (
        len(query_years.intersection(document_numbers)) / len(query_years)
        if query_years
        else 0.0
    )
    count_like = bool(
        re.search(r"\b(how many|number of|count|which years)\b", query, re.I)
    )
    arithmetic_like = bool(
        re.search(
            r"\b(average|percentage|increase|decrease|change|difference|ratio)\b",
            query,
            re.I,
        )
    )
    table_marker = float(
        "Table row:" in document
        or "Table cell:" in document
        or "Table schema:" in document
    )
    return [
        token_coverage,
        token_jaccard,
        numeric_coverage,
        year_coverage,
        float(bool(query_numbers and document_numbers.intersection(query_numbers))),
        table_marker,
        float(count_like and bool(table_marker)),
        float(arithmetic_like and numeric_coverage > 0.0),
        float(len(tokenise(document)) <= 160),
    ]


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponent = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(min(value, 60.0))
    return exponent / (1.0 + exponent)


class TATQADomainReranker:
    """A trainable TAT-QA reranker implementing the common score contract."""

    feature_names = _FEATURE_NAMES

    def __init__(
        self,
        weights: Sequence[float],
        bias: float,
        *,
        feature_mean: Sequence[float] | None = None,
        feature_scale: Sequence[float] | None = None,
        model_id: str = "taskforge-tatqa-linear-v1",
    ) -> None:
        if len(weights) != len(_FEATURE_NAMES):
            raise ValueError("weights must match the TAT-QA feature schema")
        mean = list(feature_mean or [0.0] * len(_FEATURE_NAMES))
        scale = list(feature_scale or [1.0] * len(_FEATURE_NAMES))
        if len(mean) != len(_FEATURE_NAMES) or len(scale) != len(_FEATURE_NAMES):
            raise ValueError("feature normalization must match the feature schema")
        if any(not math.isfinite(float(value)) for value in [*weights, bias, *mean, *scale]):
            raise ValueError("reranker parameters must be finite")
        if any(float(value) <= 0.0 for value in scale):
            raise ValueError("feature scales must be positive")
        self.weights = tuple(float(value) for value in weights)
        self.bias = float(bias)
        self.feature_mean = tuple(float(value) for value in mean)
        self.feature_scale = tuple(float(value) for value in scale)
        self.model_id = str(model_id).strip() or "taskforge-tatqa-linear-v1"

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        scores: list[float] = []
        for document in documents:
            values = _features(query, document)
            normalised = [
                (value - mean) / scale
                for value, mean, scale in zip(
                    values,
                    self.feature_mean,
                    self.feature_scale,
                    strict=True,
                )
            ]
            scores.append(
                self.bias
                + sum(weight * value for weight, value in zip(self.weights, normalised, strict=True))
            )
        return scores

    @classmethod
    def fit(
        cls,
        examples: Sequence[tuple[str, str, int]],
        *,
        epochs: int = 160,
        learning_rate: float = 0.08,
        l2: float = 0.01,
        model_id: str = "taskforge-tatqa-linear-v1",
    ) -> TATQADomainReranker:
        if not examples:
            raise ValueError("training examples must not be empty")
        if epochs < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("invalid training hyperparameters")
        if not math.isfinite(l2) or l2 < 0:
            raise ValueError("l2 must be finite and non-negative")
        matrix = [_features(query, document) for query, document, _ in examples]
        labels = [1.0 if int(label) else 0.0 for _, _, label in examples]
        dimension = len(_FEATURE_NAMES)
        mean = [sum(row[index] for row in matrix) / len(matrix) for index in range(dimension)]
        scale = []
        for index in range(dimension):
            variance = sum((row[index] - mean[index]) ** 2 for row in matrix) / len(matrix)
            scale.append(math.sqrt(variance) if variance > 1e-12 else 1.0)
        normalised = [
            [(value - mean[index]) / scale[index] for index, value in enumerate(row)]
            for row in matrix
        ]
        weights = [0.0] * dimension
        bias = 0.0
        count = float(len(normalised))
        for _ in range(epochs):
            gradient = [0.0] * dimension
            bias_gradient = 0.0
            for row, label in zip(normalised, labels, strict=True):
                probability = _sigmoid(
                    bias + sum(weight * value for weight, value in zip(weights, row, strict=True))
                )
                error = probability - label
                bias_gradient += error
                for index, value in enumerate(row):
                    gradient[index] += error * value
            bias -= learning_rate * bias_gradient / count
            for index in range(dimension):
                gradient[index] = gradient[index] / count + l2 * weights[index]
                weights[index] -= learning_rate * gradient[index]
        return cls(
            weights,
            bias,
            feature_mean=mean,
            feature_scale=scale,
            model_id=model_id,
        )

    def model_dump(self) -> Mapping[str, Any]:
        return {
            "schema_version": "1.0",
            "model_id": self.model_id,
            "feature_names": list(_FEATURE_NAMES),
            "weights": list(self.weights),
            "bias": self.bias,
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.model_dump(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> TATQADomainReranker:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("reranker artifact must be a JSON object")
        if payload.get("feature_names") != list(_FEATURE_NAMES):
            raise ValueError("reranker artifact feature schema is incompatible")
        return cls(
            payload["weights"],
            float(payload["bias"]),
            feature_mean=payload["feature_mean"],
            feature_scale=payload["feature_scale"],
            model_id=str(payload.get("model_id", "taskforge-tatqa-linear-v1")),
        )


__all__ = ["TATQADomainReranker"]

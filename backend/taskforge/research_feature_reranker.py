"""Lightweight domain feature fusion for paper evidence ranking."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .knowledge import tokenise

FEATURE_NAMES: tuple[str, ...] = (
    "reranker_score",
    "base_score",
    "reciprocal_rank",
    "lexical_coverage",
    "lexical_frequency",
    "numeric_coverage",
    "section_match",
    "method_section",
    "experiment_section",
    "discussion_section",
    "abstract",
    "length_log",
)

_NUMBER = re.compile(r"(?:19|20)\d{2}|[-+]?\d+(?:\.\d+)?%?")
_METHOD = frozenset({"method", "approach", "architecture", "model", "component", "mechanism"})
_EXPERIMENT = frozenset({"experiment", "evaluation", "result", "dataset", "benchmark", "baseline", "performance", "metric"})
_DISCUSSION = frozenset({"discussion", "limitation", "error", "future", "conclusion"})


def _section_terms(metadata: Mapping[str, Any]) -> set[str]:
    values: list[str] = []
    for key in ("section", "section_title", "subsection_title", "heading"):
        value = metadata.get(key)
        if value is not None:
            values.append(str(value))
    return set(tokenise(" ".join(values)))


def feature_vector(
    query: str,
    text: str,
    metadata: Mapping[str, Any],
    *,
    base_rank: int,
    base_score: float,
    reranker_score: float | None,
) -> list[float]:
    query_tokens = tokenise(query)
    query_set = set(query_tokens)
    document_tokens = tokenise(text)
    document_set = set(document_tokens)
    overlap = query_set.intersection(document_set)
    numbers = set(_NUMBER.findall(query))
    document_numbers = set(_NUMBER.findall(text))
    section = _section_terms(metadata)
    section_overlap = query_set.intersection(section)
    return [
        float(reranker_score if reranker_score is not None else base_score),
        float(base_score),
        1.0 / max(1, int(base_rank)),
        len(overlap) / len(query_set) if query_set else 0.0,
        sum(min(document_tokens.count(term), 3) for term in overlap) / (3 * len(query_set))
        if query_set
        else 0.0,
        len(numbers.intersection(document_numbers)) / len(numbers) if numbers else 0.0,
        len(section_overlap) / len(query_set) if query_set else 0.0,
        float(bool(section & _METHOD and query_set & _METHOD)),
        float(bool(section & _EXPERIMENT and query_set & _EXPERIMENT)),
        float(bool(section & _DISCUSSION and query_set & _DISCUSSION)),
        float(metadata.get("node_type") == "abstract" or metadata.get("section") == "abstract"),
        math.log1p(max(1, len(text))),
    ]


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


@dataclass(frozen=True, slots=True)
class ResearchFeatureReranker:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    training_cases: int
    positive_pairs: int

    def score(self, values: Sequence[float]) -> float:
        normalized = [
            (float(value) - mean) / scale
            for value, mean, scale in zip(values, self.means, self.scales, strict=True)
        ]
        return self.bias + sum(weight * value for weight, value in zip(self.weights, normalized, strict=True))

    def rerank(self, values: Sequence[Sequence[float]]) -> list[int]:
        ordered = sorted(
            range(len(values)),
            key=lambda index: (-self.score(values[index]), index),
        )
        return ordered

    def model_dump(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "algorithm": "pairwise_logistic_linear",
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "weights": list(self.weights),
            "bias": self.bias,
            "training_cases": self.training_cases,
            "positive_pairs": self.positive_pairs,
        }


def train_pairwise_feature_reranker(
    rows: Sequence[Mapping[str, Any]],
    documents: Mapping[str, Mapping[str, Any]],
    *,
    epochs: int = 30,
    learning_rate: float = 0.03,
    l2: float = 0.002,
    max_negatives: int = 20,
    max_rank: int | None = 30,
) -> ResearchFeatureReranker:
    if not rows or epochs < 1 or learning_rate <= 0 or l2 < 0:
        raise ValueError("invalid feature-reranker training arguments")
    all_values: list[list[float]] = []
    for row in rows:
        query = str(row.get("query") or "")
        ids = [str(value) for value in row.get("retrieved_ids", [])]
        relevant = {str(value) for value in row.get("relevant_ids", [])}
        base_scores = list(row.get("base_scores", []))
        reranker_scores = list(row.get("reranker_scores", []))
        positives: list[list[float]] = []
        negatives: list[list[float]] = []
        candidate_indices = range(len(ids))
        if max_rank is not None:
            candidate_indices = range(min(len(ids), max_rank))
        for index in candidate_indices:
            chunk_id = ids[index]
            document = documents.get(chunk_id)
            if document is None:
                continue
            values = feature_vector(
                query,
                str(document.get("text", "")),
                document.get("metadata", {}),
                base_rank=index + 1,
                base_score=float(base_scores[index]) if index < len(base_scores) else 0.0,
                reranker_score=(
                    None
                    if index >= len(reranker_scores) or reranker_scores[index] is None
                    else float(reranker_scores[index])
                ),
            )
            (positives if chunk_id in relevant else negatives).append(values)
        if positives and negatives:
            bounded = negatives[:max_negatives]
            all_values.extend(positives)
            all_values.extend(bounded)
    if not all_values:
        raise ValueError("no positive/negative training pairs")
    dimensions = len(FEATURE_NAMES)
    means = [sum(values[index] for values in all_values) / len(all_values) for index in range(dimensions)]
    scales = [
        math.sqrt(sum((values[index] - means[index]) ** 2 for values in all_values) / len(all_values)) or 1.0
        for index in range(dimensions)
    ]
    pairs: list[list[float]] = []
    pair_count = 0
    case_count = 0
    for row in rows:
        query = str(row.get("query") or "")
        ids = [str(value) for value in row.get("retrieved_ids", [])]
        relevant = {str(value) for value in row.get("relevant_ids", [])}
        base_scores = list(row.get("base_scores", []))
        reranker_scores = list(row.get("reranker_scores", []))
        positives: list[list[float]] = []
        negatives: list[list[float]] = []
        candidate_indices = range(len(ids))
        if max_rank is not None:
            candidate_indices = range(min(len(ids), max_rank))
        for index in candidate_indices:
            chunk_id = ids[index]
            document = documents.get(chunk_id)
            if document is None:
                continue
            raw = feature_vector(
                query,
                str(document.get("text", "")),
                document.get("metadata", {}),
                base_rank=index + 1,
                base_score=float(base_scores[index]) if index < len(base_scores) else 0.0,
                reranker_score=(
                    None
                    if index >= len(reranker_scores) or reranker_scores[index] is None
                    else float(reranker_scores[index])
                ),
            )
            (positives if chunk_id in relevant else negatives).append(raw)
        if not positives or not negatives:
            continue
        case_count += 1
        for positive in positives:
            normalized_positive = [(value - mean) / scale for value, mean, scale in zip(positive, means, scales, strict=True)]
            for negative in negatives[:max_negatives]:
                normalized_negative = [(value - mean) / scale for value, mean, scale in zip(negative, means, scales, strict=True)]
                pairs.append([left - right for left, right in zip(normalized_positive, normalized_negative, strict=True)])
        pair_count += len(positives) * min(len(negatives), max_negatives)
    weights = [0.0] * dimensions
    for _ in range(epochs):
        for difference in pairs:
            probability = _sigmoid(sum(weight * value for weight, value in zip(weights, difference, strict=True)))
            error = 1.0 - probability
            for index, value in enumerate(difference):
                weights[index] += learning_rate * (error * value - l2 * weights[index])
    return ResearchFeatureReranker(
        feature_names=FEATURE_NAMES,
        means=tuple(means),
        scales=tuple(scales),
        weights=tuple(weights),
        bias=0.0,
        training_cases=case_count,
        positive_pairs=pair_count,
    )


__all__ = ["FEATURE_NAMES", "ResearchFeatureReranker", "feature_vector", "train_pairwise_feature_reranker"]

"""Supervised, auditable ranking features for long-form paper evidence.

This module deliberately keeps the model small: a pointwise logistic scorer is
trained on paper-disjoint QASPER candidates, then applied only to the existing
candidate set.  It is a ranking model, not a second retriever, so Candidate@K
and ACL/Scope boundaries are preserved.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .knowledge import tokenise

RANK_FEATURE_NAMES: tuple[str, ...] = (
    "reranker_norm",
    "base_norm",
    "reciprocal_rank",
    "rank_fraction",
    "query_coverage",
    "query_frequency",
    "overlap_count",
    "numeric_coverage",
    "abstract",
    "length_log",
    "section_query_overlap",
    "intent_section_match",
    "section_number",
    "section_results",
    "section_methods",
    "section_discussion",
    "section_experiments",
    "section_dataset",
    "query_numeric",
    "query_comparison",
    "query_method",
    "query_dataset",
    "query_baseline",
    "query_metric",
    "query_limitation",
    "secondary_reranker_norm",
    "reranker_agreement",
    "rank_agreement",
    "ensemble_norm",
)

_NUMBER = re.compile(r"(?:19|20)\d{2}|[-+]?\d+(?:\.\d+)?%?")
_STOPWORDS = frozenset(
    "the a an is are was were of to and in on for with what which how does do their this that from be paper model they it its as by or we our can may have has had been than other used use using".split()
)
_METHOD = frozenset({"method", "methods", "approach", "architecture", "model", "component", "mechanism"})
_DATASET = frozenset({"dataset", "datasets", "corpus", "benchmark", "data", "collection"})
_BASELINE = frozenset({"baseline", "baselines", "compare", "comparison", "compared", "prior", "state-of-the-art"})
_METRIC = frozenset({"metric", "metrics", "score", "accuracy", "f1", "precision", "recall", "performance", "result", "results", "evaluation", "experiment"})
_LIMITATION = frozenset({"limitation", "limitations", "error", "failure", "future", "weakness", "discussion", "conclusion"})
_RESULT_SECTIONS = frozenset({"result", "results", "evaluation", "experiment", "experiments", "dataset", "datasets", "baseline", "baselines"})
_METHOD_SECTIONS = frozenset({"method", "methods", "approach", "architecture", "model"})
_DISCUSSION_SECTIONS = frozenset({"discussion", "limitation", "limitations", "conclusion", "future"})


def _section_terms(metadata: Mapping[str, Any]) -> set[str]:
    values = [str(metadata.get(key, "")) for key in ("section", "section_title", "subsection_title", "heading")]
    return set(tokenise(" ".join(values)))


def _normalise_scores(values: Sequence[float | None], size: int) -> list[float]:
    cleaned = [float(value) if value is not None else -1e9 for value in values[:size]]
    if len(cleaned) < size:
        cleaned.extend([0.0] * (size - len(cleaned)))
    if not cleaned:
        return []
    low, high = min(cleaned), max(cleaned)
    if high <= low:
        return [0.0] * len(cleaned)
    return [(value - low) / (high - low) for value in cleaned]


def _query_flags(query_tokens: set[str], query: str) -> tuple[float, ...]:
    numbers = bool(_NUMBER.search(query))
    return (
        float(numbers),
        float(bool(query_tokens & {"compare", "comparison", "compared", "difference", "differences", "versus", "vs"})),
        float(bool(query_tokens & _METHOD)),
        float(bool(query_tokens & _DATASET)),
        float(bool(query_tokens & _BASELINE)),
        float(bool(query_tokens & _METRIC)),
        float(bool(query_tokens & _LIMITATION)),
    )


def expanded_feature_vector(
    query: str,
    text: str,
    metadata: Mapping[str, Any],
    *,
    base_rank: int,
    candidate_count: int,
    base_norm: float,
    reranker_norm: float,
    secondary_reranker_norm: float = 0.0,
    reranker_agreement: float = 0.0,
    rank_agreement: float = 0.0,
    ensemble_norm: float = 0.0,
) -> list[float]:
    """Build a row of normalized, query-aware structural features."""

    query_tokens = {token for token in tokenise(query) if token not in _STOPWORDS}
    document_tokens = tokenise(text)
    document_set = set(document_tokens)
    overlap = query_tokens.intersection(document_set)
    numbers = set(_NUMBER.findall(query))
    document_numbers = set(_NUMBER.findall(text))
    section = _section_terms(metadata)
    section_overlap = query_tokens.intersection(section)
    query_flags = _query_flags(query_tokens, query)
    intent_section_match = float(
        bool(
            (query_tokens & _METHOD and section & _METHOD_SECTIONS)
            or (query_tokens & (_DATASET | _BASELINE | _METRIC) and section & _RESULT_SECTIONS)
            or (query_tokens & _LIMITATION and section & _DISCUSSION_SECTIONS)
        )
    )
    section_number = 0.0
    for key in ("section", "section_id", "section_index"):
        matched = re.search(r"\d+", str(metadata.get(key, "")))
        if matched:
            section_number = float(matched.group())
            break
    if candidate_count > 1:
        section_number /= float(candidate_count)
    return [
        float(reranker_norm),
        float(base_norm),
        1.0 / max(1, int(base_rank)),
        float(base_rank) / max(1, candidate_count),
        len(overlap) / len(query_tokens) if query_tokens else 0.0,
        sum(min(document_tokens.count(term), 3) for term in overlap) / (3 * len(query_tokens)) if query_tokens else 0.0,
        float(len(overlap)),
        len(numbers.intersection(document_numbers)) / len(numbers) if numbers else 0.0,
        float(metadata.get("node_type") == "abstract" or metadata.get("section") == "abstract"),
        math.log1p(max(1, len(text))),
        len(section_overlap) / len(query_tokens) if query_tokens else 0.0,
        intent_section_match,
        section_number,
        float(bool(section & _RESULT_SECTIONS)),
        float(bool(section & _METHOD_SECTIONS)),
        float(bool(section & _DISCUSSION_SECTIONS)),
        float(bool(section & {"experiment", "experiments", "evaluation"})),
        float(bool(section & _DATASET)),
        *query_flags,
        float(secondary_reranker_norm),
        float(reranker_agreement),
        float(rank_agreement),
        float(ensemble_norm),
    ]


def row_features(row: Mapping[str, Any], documents: Mapping[str, Mapping[str, Any]]) -> list[list[float]]:
    ids = [str(value) for value in row.get("retrieved_ids", [])]
    base_scores = list(row.get("base_scores", []))
    reranker_scores = list(row.get("reranker_scores", []))
    secondary_scores = list(row.get("secondary_reranker_scores", []))
    base_norm = _normalise_scores(base_scores, len(ids))
    reranker_norm = _normalise_scores(reranker_scores, len(ids))
    secondary_norm = _normalise_scores(secondary_scores, len(ids))
    ensemble_norm = _normalise_scores(
        [
            (reranker_norm[index] + secondary_norm[index]) / 2.0
            for index in range(len(ids))
        ],
        len(ids),
    )
    primary_order = {chunk_id: index for index, chunk_id in enumerate(sorted(ids, key=lambda value: (-reranker_norm[ids.index(value)], value)))}
    secondary_order = {chunk_id: index for index, chunk_id in enumerate(sorted(ids, key=lambda value: (-secondary_norm[ids.index(value)], value)))}
    values: list[list[float]] = []
    for index, chunk_id in enumerate(ids):
        document = documents.get(chunk_id)
        if document is None:
            values.append([0.0] * len(RANK_FEATURE_NAMES))
            continue
        values.append(
            expanded_feature_vector(
                str(row.get("query", "")),
                str(document.get("text", "")),
                document.get("metadata", {}),
                base_rank=index + 1,
                candidate_count=len(ids),
                base_norm=base_norm[index],
                reranker_norm=reranker_norm[index],
                secondary_reranker_norm=secondary_norm[index],
                reranker_agreement=1.0 - abs(reranker_norm[index] - secondary_norm[index]),
                rank_agreement=1.0 - abs(primary_order[chunk_id] - secondary_order[chunk_id]) / max(1, len(ids) - 1),
                ensemble_norm=ensemble_norm[index],
            )
        )
    return values


@dataclass(frozen=True, slots=True)
class SupervisedResearchRanker:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    training_cases: int
    positive_examples: int
    algorithm: str = "pointwise_logistic"

    @classmethod
    def from_model_dump(cls, payload: Mapping[str, Any]) -> "SupervisedResearchRanker":
        if str(payload.get("schema_version", "1.0")) != "1.0":
            raise ValueError("unsupported supervised ranker schema")
        names = tuple(str(value) for value in payload.get("feature_names", ()))
        if names != RANK_FEATURE_NAMES:
            raise ValueError("supervised ranker feature contract does not match runtime")
        means = tuple(float(value) for value in payload["means"])
        scales = tuple(float(value) for value in payload["scales"])
        weights = tuple(float(value) for value in payload["weights"])
        if not len(means) == len(scales) == len(weights) == len(names):
            raise ValueError("supervised ranker coefficient dimensions do not match")
        return cls(
            feature_names=names,
            means=means,
            scales=scales,
            weights=weights,
            bias=float(payload.get("bias", 0.0)),
            training_cases=int(payload.get("training_cases", 0)),
            positive_examples=int(payload.get("positive_examples", 0)),
            algorithm=str(payload.get("algorithm", "pointwise_logistic")),
        )

    def score(self, values: Sequence[float]) -> float:
        normalized = [
            (float(value) - mean) / scale
            for value, mean, scale in zip(values, self.means, self.scales, strict=True)
        ]
        return self.bias + sum(weight * value for weight, value in zip(self.weights, normalized, strict=True))

    def rerank(self, values: Sequence[Sequence[float]]) -> list[int]:
        return sorted(range(len(values)), key=lambda index: (-self.score(values[index]), index))

    def model_dump(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "algorithm": self.algorithm,
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "weights": list(self.weights),
            "bias": self.bias,
            "training_cases": self.training_cases,
            "positive_examples": self.positive_examples,
        }


def train_supervised_ranker(
    rows: Sequence[Mapping[str, Any]],
    documents: Mapping[str, Mapping[str, Any]],
    *,
    epochs: int = 250,
    learning_rate: float = 0.08,
    l2: float = 0.3,
    max_rank: int | None = 50,
) -> SupervisedResearchRanker:
    if not rows or epochs < 1 or learning_rate <= 0 or l2 < 0:
        raise ValueError("invalid supervised ranker training arguments")
    all_values: list[list[float]] = []
    labels: list[float] = []
    case_count = 0
    positive_examples = 0
    for row in rows:
        ids = [str(value) for value in row.get("retrieved_ids", [])]
        relevant = {str(value) for value in row.get("relevant_ids", [])}
        values = row_features(row, documents)
        limit = len(ids) if max_rank is None else min(len(ids), max_rank)
        if not relevant or not any(item in relevant for item in ids[:limit]):
            continue
        case_count += 1
        for index, value in enumerate(values[:limit]):
            all_values.append(value)
            label = float(ids[index] in relevant)
            labels.append(label)
            positive_examples += int(label)
    if not all_values or not positive_examples:
        raise ValueError("no supervised ranking examples")
    x = np.asarray(all_values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales < 1e-8] = 1.0
    x = (x - means) / scales
    try:
        # LBFGS converges much more reliably than hand-tuned gradient descent
        # for the highly imbalanced candidate labels.  scikit-learn is only a
        # training-time dependency; the serialized runtime model is numpy-only.
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            # ``l2`` is retained as the public tuning knob for compatibility;
            # its value is interpreted as the inverse regularization strength
            # by the fallback optimizer and as C by LBFGS.
            C=max(1e-6, l2),
            class_weight="balanced",
            max_iter=max(100, epochs),
            solver="lbfgs",
            random_state=20260813,
        )
        model.fit(x, y)
        weights = np.asarray(model.coef_[0], dtype=np.float64)
        bias = float(model.intercept_[0])
    except ImportError:
        positive_weight = len(y) / max(1.0, 2.0 * positive_examples)
        negative_weight = len(y) / max(1.0, 2.0 * (len(y) - positive_examples))
        sample_weights = np.where(y > 0.5, positive_weight, negative_weight)
        weights = np.zeros(x.shape[1], dtype=np.float64)
        bias = 0.0
        for _ in range(epochs):
            logits = np.clip(x @ weights + bias, -30.0, 30.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            residual = (probabilities - y) * sample_weights
            gradient = (x.T @ residual) / len(y) + l2 * weights
            bias_gradient = float(residual.mean())
            weights -= learning_rate * gradient
            bias -= learning_rate * bias_gradient
    return SupervisedResearchRanker(
        feature_names=RANK_FEATURE_NAMES,
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        weights=tuple(float(value) for value in weights),
        bias=float(bias),
        training_cases=case_count,
        positive_examples=positive_examples,
    )


__all__ = [
    "RANK_FEATURE_NAMES",
    "SupervisedResearchRanker",
    "expanded_feature_vector",
    "row_features",
    "train_supervised_ranker",
]

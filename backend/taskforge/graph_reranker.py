"""Small, auditable pairwise ranker for evidence-graph features.

The model is deliberately modest: it learns a linear ordering over the graph
features already emitted by ``LocalEvidenceGraph``.  It does not create
evidence, change ACL scope, or access validation labels at inference time.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evidence_graph import GraphRerankFeatures, GraphRerankResult
from .hybrid_retrieval import HybridSearchHit

FEATURE_NAMES: tuple[str, ...] = (
    "base_score",
    "graph_support",
    "entity_support",
    "section_support",
    "adjacency_support",
    "graph_entity_interaction",
    "graph_section_interaction",
    "raw_score",
    "raw_base_score",
    "reranker_score",
)
SCHEMA_VERSION = "1.0"


def _feature_vector(features: GraphRerankFeatures) -> list[float]:
    graph = float(features.graph_support)
    entity = float(features.entity_support)
    section = float(features.section_support)
    return [
        float(features.base_score),
        graph,
        entity,
        section,
        float(features.adjacency_support),
        graph * entity,
        graph * section,
        float(features.raw_score if features.raw_score is not None else features.base_score),
        float(
            features.raw_base_score
            if features.raw_base_score is not None
            else features.base_score
        ),
        float(
            features.reranker_score
            if features.reranker_score is not None
            else features.raw_score
            if features.raw_score is not None
            else features.base_score
        ),
    ]


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


@dataclass(frozen=True)
class LearnedGraphReranker:
    """A normalized linear pairwise ranker with immutable provenance."""

    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    training_cases: int
    positive_pairs: int
    fit_run_id: str
    dataset_sha256: str

    def _score(self, features: GraphRerankFeatures) -> float:
        values = _feature_vector(features)
        normalized = [
            (value - mean) / scale
            for value, mean, scale in zip(values, self.means, self.scales)
        ]
        return self.bias + sum(weight * value for weight, value in zip(self.weights, normalized))

    def rerank(self, result: GraphRerankResult) -> GraphRerankResult:
        """Reorder the exact graph candidate set using learned graph features."""

        scored: list[tuple[float, int, str, HybridSearchHit]] = []
        for index, hit in enumerate(result.hits):
            feature = result.features.get(hit.chunk.chunk_id)
            if feature is None:
                raise ValueError("learned graph reranker received an unknown candidate")
            scored.append(
                (self._score(feature), index, hit.chunk.chunk_id, hit)
            )
        ordered = sorted(scored, key=lambda item: (-item[0], item[1], item[2]))
        hits: list[HybridSearchHit] = []
        for rank, (score, _, _, hit) in enumerate(ordered, start=1):
            sources = list(hit.retrieval_sources)
            if "learned_graph_rerank" not in sources:
                sources.append("learned_graph_rerank")
            hits.append(
                hit.model_copy(
                    update={
                        "rank": rank,
                        "score": float(score),
                        "retrieval_sources": sources,
                    }
                )
            )
        if {hit.chunk.chunk_id for hit in hits} != {
            hit.chunk.chunk_id for hit in result.hits
        }:
            raise RuntimeError("learned graph reranker changed the candidate set")
        return GraphRerankResult(
            tuple(hits),
            result.features,
            result.seed_chunk_ids,
            result.node_count,
            result.edge_count,
        )

    def model_dump(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "weights": list(self.weights),
            "bias": self.bias,
            "training_cases": self.training_cases,
            "positive_pairs": self.positive_pairs,
            "fit_run_id": self.fit_run_id,
            "dataset_sha256": self.dataset_sha256,
            "algorithm": "pairwise_logistic_linear",
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        payload = json.dumps(
            self.model_dump(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload + b"\n")

    @classmethod
    def load(cls, path: str | Path) -> LearnedGraphReranker:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported learned graph reranker schema")
        names = tuple(str(value) for value in payload.get("feature_names", []))
        if names != FEATURE_NAMES:
            raise ValueError("learned graph feature names do not match the runtime")
        length = len(FEATURE_NAMES)
        fields = ("means", "scales", "weights")
        parsed: dict[str, tuple[float, ...]] = {}
        for field in fields:
            values = tuple(float(value) for value in payload.get(field, []))
            if len(values) != length or not all(math.isfinite(value) for value in values):
                raise ValueError(f"invalid learned graph {field}")
            parsed[field] = values
        if any(value <= 0 for value in parsed["scales"]):
            raise ValueError("learned graph scales must be positive")
        return cls(
            feature_names=names,
            means=parsed["means"],
            scales=parsed["scales"],
            weights=parsed["weights"],
            bias=float(payload.get("bias", 0.0)),
            training_cases=int(payload.get("training_cases", 0)),
            positive_pairs=int(payload.get("positive_pairs", 0)),
            fit_run_id=str(payload.get("fit_run_id", "")),
            dataset_sha256=str(payload.get("dataset_sha256", "")),
        )


def train_pairwise_graph_reranker(
    rows: list[dict[str, Any]],
    *,
    fit_run_id: str,
    dataset_sha256: str,
    epochs: int = 35,
    learning_rate: float = 0.04,
    l2: float = 0.002,
) -> LearnedGraphReranker:
    """Fit a pairwise logistic model from graph-stage prediction rows.

    Labels are only taken from the fit-complement rows supplied by the caller.
    For determinism, each positive is paired with at most the first twenty
    negatives in candidate order.
    """

    if not rows:
        raise ValueError("fit rows must not be empty")
    if epochs < 1 or learning_rate <= 0 or l2 < 0:
        raise ValueError("invalid pairwise training hyperparameters")
    all_values: list[list[float]] = []
    case_candidates: dict[str, tuple[list[list[float]], list[list[float]]]] = {}
    case_count = 0
    pair_count = 0
    for row in rows:
        graph = row.get("graph")
        if not isinstance(graph, dict):
            continue
        feature_map = graph.get("features")
        relevant = {str(value) for value in row.get("relevant_ids", [])}
        ids = [str(value) for value in row.get("retrieved_ids", [])]
        if not isinstance(feature_map, dict) or not ids:
            continue
        positives: list[list[float]] = []
        negatives: list[list[float]] = []
        for chunk_id in ids:
            raw = feature_map.get(chunk_id)
            if not isinstance(raw, dict):
                continue
            feature = GraphRerankFeatures(
                base_rank=int(raw["base_rank"]),
                base_score=float(raw["base_score"]),
                graph_support=float(raw["graph_support"]),
                entity_support=float(raw["entity_support"]),
                section_support=float(raw["section_support"]),
                adjacency_support=float(raw["adjacency_support"]),
                final_score=float(raw["final_score"]),
                raw_score=(
                    None
                    if raw.get("raw_score") is None
                    else float(raw["raw_score"])
                ),
                raw_base_score=(
                    None
                    if raw.get("raw_base_score") is None
                    else float(raw["raw_base_score"])
                ),
                reranker_score=(
                    None
                    if raw.get("reranker_score") is None
                    else float(raw["reranker_score"])
                ),
            )
            values = _feature_vector(feature)
            (positives if chunk_id in relevant else negatives).append(values)
        if not positives or not negatives:
            continue
        case_count += 1
        bounded_negatives = negatives[:20]
        case_candidates[str(row["case_id"])] = (positives, bounded_negatives)
        all_values.extend(positives)
        all_values.extend(bounded_negatives)
        pair_count += len(positives) * len(bounded_negatives)
    if not all_values:
        raise ValueError("fit rows contain no positive/negative candidate pairs")

    dimensions = len(FEATURE_NAMES)
    means = [sum(values[index] for values in all_values) / len(all_values) for index in range(dimensions)]
    scales = []
    for index, mean in enumerate(means):
        variance = sum((values[index] - mean) ** 2 for values in all_values) / len(all_values)
        scales.append(math.sqrt(variance) or 1.0)

    # Build pair differences.  This is small (7 features per pair) even for
    # the full fit-complement and keeps the update order deterministic.
    pairs: list[list[float]] = []
    for positives, negatives in case_candidates.values():
        normalized_positives = [
            [
                (value - mean) / scale
                for value, mean, scale in zip(values, means, scales)
            ]
            for values in positives
        ]
        normalized_negatives = [
            [
                (value - mean) / scale
                for value, mean, scale in zip(values, means, scales)
            ]
            for values in negatives
        ]
        for positive in normalized_positives:
            for negative in normalized_negatives:
                pairs.append([left - right for left, right in zip(positive, negative)])

    weights = [0.0] * dimensions
    # Pairwise differences cancel any intercept.  Keeping a free bias would
    # monotonically increase the margin on every pair and prematurely drive
    # the sigmoid into saturation without learning useful feature weights.
    bias = 0.0
    for _ in range(epochs):
        for difference in pairs:
            margin = sum(weight * value for weight, value in zip(weights, difference))
            probability = _sigmoid(margin)
            error = 1.0 - probability
            for index, value in enumerate(difference):
                weights[index] += learning_rate * (error * value - l2 * weights[index])

    return LearnedGraphReranker(
        feature_names=FEATURE_NAMES,
        means=tuple(means),
        scales=tuple(scales),
        weights=tuple(weights),
        bias=bias,
        training_cases=case_count,
        positive_pairs=pair_count,
        fit_run_id=fit_run_id,
        dataset_sha256=dataset_sha256,
    )


def sha256_model(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = [
    "FEATURE_NAMES",
    "LearnedGraphReranker",
    "SCHEMA_VERSION",
    "sha256_model",
    "train_pairwise_graph_reranker",
]

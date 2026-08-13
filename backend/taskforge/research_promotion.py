"""Paired quality/latency gate for research retrieval profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ResearchPromotionPolicy:
    """Conservative defaults for promoting an expensive retrieval profile."""

    min_recall_at_10_gain: float = 0.05
    min_recall_at_50: float = 0.99
    # Recall is the primary objective. These are only sanity bounds to catch a
    # broken configuration (for example an accidental unbounded rerank loop),
    # not a normal performance promotion gate.
    max_p95_ratio: float = 20.0
    max_p95_ms: float = 5_000.0


@dataclass(frozen=True, slots=True)
class ResearchPromotionDecision:
    promoted: bool
    reasons: tuple[str, ...]
    metrics: dict[str, float]


def decide_research_promotion(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    policy: ResearchPromotionPolicy | None = None,
) -> ResearchPromotionDecision:
    """Return a fail-safe promotion decision from two report metric objects.

    The candidate must improve early ranking by a meaningful margin, preserve
    Candidate@50 recall, and stay within both an absolute and relative latency
    budget. Missing or malformed metrics fail closed.
    """

    gate = policy or ResearchPromotionPolicy()
    try:
        baseline_recall10 = float(baseline["recall_at_10"])
        candidate_recall10 = float(candidate["recall_at_10"])
        candidate_recall50 = float(candidate["recall_at_50"])
        baseline_p95 = float(baseline["p95_ms"])
        candidate_p95 = float(candidate["p95_ms"])
    except (KeyError, TypeError, ValueError) as exc:
        return ResearchPromotionDecision(
            promoted=False,
            reasons=(f"invalid_metrics:{exc}",),
            metrics={},
        )
    if baseline_p95 <= 0 or candidate_p95 < 0:
        return ResearchPromotionDecision(
            promoted=False,
            reasons=("invalid_latency",),
            metrics={},
        )
    gain = candidate_recall10 - baseline_recall10
    ratio = candidate_p95 / baseline_p95
    reasons: list[str] = []
    if gain < gate.min_recall_at_10_gain:
        reasons.append("recall_at_10_gain_below_gate")
    if candidate_recall50 < gate.min_recall_at_50:
        reasons.append("recall_at_50_below_gate")
    if ratio > gate.max_p95_ratio:
        reasons.append("p95_ratio_above_gate")
    if candidate_p95 > gate.max_p95_ms:
        reasons.append("p95_absolute_above_gate")
    return ResearchPromotionDecision(
        promoted=not reasons,
        reasons=tuple(reasons),
        metrics={
            "recall_at_10_gain": gain,
            "recall_at_50": candidate_recall50,
            "p95_ratio": ratio,
            "p95_ms": candidate_p95,
        },
    )


__all__ = [
    "ResearchPromotionDecision",
    "ResearchPromotionPolicy",
    "decide_research_promotion",
]

from taskforge.research_promotion import decide_research_promotion


def test_pathologically_expensive_candidate_is_not_promoted() -> None:
    decision = decide_research_promotion(
        {"recall_at_10": 0.5535, "p95_ms": 239.4},
        {"recall_at_10": 0.7022, "recall_at_50": 0.99, "p95_ms": 6000.0},
    )
    assert decision.promoted is False
    assert "p95_absolute_above_gate" in decision.reasons


def test_candidate_with_meaningful_gain_and_budget_is_promoted() -> None:
    decision = decide_research_promotion(
        {"recall_at_10": 0.70, "p95_ms": 300.0},
        {"recall_at_10": 0.78, "recall_at_50": 0.995, "p95_ms": 500.0},
    )
    assert decision.promoted is True
    assert decision.reasons == ()

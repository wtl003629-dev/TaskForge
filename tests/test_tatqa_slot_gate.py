from __future__ import annotations

from scripts.evaluate_tatqa_slot_reranker import _gate


def _score(seed: int, overall: float, table: float) -> dict[str, object]:
    return {
        "seed": seed,
        "macro_delta": overall - 0.8,
        "by_category": {
            "table": {"cases": 20, "recall": table},
            "count": {"cases": 10, "recall": 0.9},
        },
    }


def test_slot_gate_rejects_one_category_regression_across_stable_seeds() -> None:
    baseline = {
        "macro_recall": 0.8,
        "by_category": {
            "table": {"cases": 20, "recall": 1.0},
            "count": {"cases": 10, "recall": 0.5},
        },
    }
    candidates = [_score(seed, 0.9, 0.95) for seed in (1, 2, 3)]

    gate = _gate(
        baseline,
        candidates,
        min_macro_delta=0.03,
        max_category_drop=0.03,
    )

    assert gate["passed"] is False
    assert gate["checks"]["three_distinct_seeds"] is True
    assert gate["checks"]["category_non_regression"] is False
    assert len(gate["category_failures"]) == 3

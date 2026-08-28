from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from run_rag_profile_ab import _decision, _paired_ci  # noqa: E402


def _report(*, optimized: bool) -> dict[str, object]:
    gain = 0.2 if optimized else 0.0
    profile = "optimized" if optimized else "current"
    kb = (
        "research-scope:scope-1:v1:rag:optimized-e"
        if optimized
        else "research-scope:scope-1:v1"
    )
    return {
        "metrics": {
            "recall_at_5": 0.8,
            "recall_at_10": 0.9,
            "p95_ms": 110.0 if optimized else 100.0,
        },
        "agent_visible_metrics": {"recall_at_8": 0.85},
        "citation_metrics": {
            "localization_hit_rate_at_8": 0.9,
            "precision_at_8": 0.4,
            "roundtrip_verification_accuracy": 1.0,
        },
        "alignment_gate": {"passed": True},
        "ingestion": [
            {
                "status": "indexed",
                "index_statistics": {"knowledge_base_id": kb},
            }
        ],
        "rows": [
            {
                "query": "What result does the table report?",
                "mrr": 0.4 + gain,
                "ndcg_at_8": 0.5 + gain,
                "agent_visible_recall_at_8": 1.0,
                "failure_stage": "retrieval_success",
            }
            for _ in range(8)
        ],
        "rag_profile": {"name": profile, "ablation": "e" if optimized else "a"},
    }


def test_paired_ci_reports_a_strictly_positive_delta() -> None:
    result = _paired_ci([0.1, 0.2, 0.3], [0.2, 0.3, 0.4], samples=1_000)

    assert result["mean_delta"] > 0
    assert result["ci95_low"] > 0


def test_gate_keeps_current_active_until_answer_and_citation_eval_exists() -> None:
    decision = _decision(
        {"a": _report(optimized=False), "e": _report(optimized=True)},
        max_p95_ratio=1.25,
        current_answer_report=None,
        optimized_answer_report=None,
    )

    assert decision["retrieval_gate_passed"] is True
    assert decision["outcome"] == "no_go_pending_answer_and_citation_eval"
    assert decision["active_profile_must_remain"] == "current"

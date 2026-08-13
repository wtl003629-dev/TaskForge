from __future__ import annotations

from pathlib import Path

from scripts.run_research_retrieval_ab import run


def test_research_retrieval_ab_report_has_all_three_variants() -> None:
    cases = Path(__file__).parents[1] / "eval" / "research-retrieval-cases.json"
    report = run(cases)
    assert set(report["summary"]) == {"routed", "unified", "adaptive"}
    assert len(report["rows"]) == 4
    assert all("mean_recall_at_10" in report["summary"][name] for name in report["summary"])

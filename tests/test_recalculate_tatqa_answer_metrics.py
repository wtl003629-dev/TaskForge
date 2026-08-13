from __future__ import annotations

from scripts.recalculate_tatqa_answer_metrics import _failure_stage


def _metrics(*, full_recall: bool) -> dict[str, object]:
    return {"full_recall": full_recall, "exact_match": 0.0}


def test_offline_failure_stage_prefers_candidate_gap() -> None:
    stage = _failure_stage(
        {"generated_answer": "42"},
        candidate=_metrics(full_recall=False),
        top10=_metrics(full_recall=False),
        presented=_metrics(full_recall=False),
        answer_metrics={"exact_match": 0.0},
    )
    assert stage == "candidate_missing"


def test_offline_failure_stage_detects_format_error() -> None:
    stage = _failure_stage(
        {"generated_answer": "", "parse_error": "invalid_json"},
        candidate=_metrics(full_recall=True),
        top10=_metrics(full_recall=True),
        presented=_metrics(full_recall=True),
        answer_metrics={"exact_match": 1.0},
    )
    assert stage == "format_or_scale_failure"

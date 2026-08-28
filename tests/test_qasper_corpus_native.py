from __future__ import annotations

import json

from scripts.evaluate_qasper_corpus_native import run


def test_corpus_native_report_uses_strict_paragraph_recall_only(tmp_path) -> None:
    evidence = "The proposed method improves recall by four points."
    dataset_path = tmp_path / "qasper.json"
    dataset_path.write_text(
        json.dumps(
            {
                "paper-1": {
                    "title": "Paper",
                    "abstract": "Abstract.",
                    "full_text": [
                        {
                            "section_name": "Results",
                            "paragraphs": [evidence, "Unrelated paragraph."],
                        }
                    ],
                    "qas": [
                        {
                            "question": "What improves recall?",
                            "question_id": "q1",
                            "answers": [
                                {
                                    "annotation_id": "worker",
                                    "answer": {
                                        "unanswerable": False,
                                        "extractive_spans": ["method"],
                                        "free_form_answer": "",
                                        "yes_no": None,
                                        "evidence": [evidence],
                                    },
                                }
                            ],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    case_id = "qasper:paper-1:q1"
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps({"case_ids": [case_id]}),
        encoding="utf-8",
    )
    output_path = tmp_path / "report.json"

    report = run(
        dataset_path,
        split_path,
        output_path,
        limit=1,
        backend="bm25",
        candidate_k=10,
    )

    assert report["benchmark_track"] == "corpus_native_retrieval"
    assert report["metrics"]["recall_at_1"] == 1.0  # type: ignore[index]
    assert "ndcg_at_10" not in report["metrics"]  # type: ignore[operator]
    assert "thresholds" not in report
    assert "passed" not in report
    assert report["rows"][0]["gold_annotation_count"] == 1  # type: ignore[index]

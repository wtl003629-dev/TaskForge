from __future__ import annotations

import json
from pathlib import Path

import pytest

from taskforge.tatqa_mapping_eval import (
    TATQAMappingDiagnosticError,
    evaluate_tagop_mapping_retrieval,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _annotations() -> list[dict[str, object]]:
    return [
        {
            "table": {
                "uid": "table-1",
                "table": [["", "2021"], ["Revenue", "10"]],
            },
            "paragraphs": [],
            "questions": [
                {
                    "uid": "q1",
                    "question": "What was revenue in 2021?",
                    "answer_from": "table",
                    "answer_type": "arithmetic",
                    "answer": 10,
                    "derivation": "10",
                    "scale": "",
                    "mapping": {"table": [[1, 1], [1, 1]]},
                },
                {
                    "uid": "q2",
                    "question": "Which year is shown?",
                    "answer_from": "table",
                    "answer_type": "table",
                    "answer": "2021",
                    "derivation": "",
                    "scale": "",
                    "mapping": {"table": [[0, 1]]},
                },
                {
                    "uid": "q3",
                    "question": "What did the narrative say?",
                    "answer_from": "text",
                    "answer_type": "span",
                    "answer": "text",
                    "derivation": "",
                    "scale": "",
                    "mapping": {"paragraph": {"0": "text"}},
                },
            ],
        }
    ]


def _prediction(case_id: str, **updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "stage": "pair",
        "case_id": case_id,
        "category": "arithmetic",
        "retrieved_ids": ["tatqa:table-1:table"],
        "retrieved_row_ids": [],
        "retrieved_cell_ids": [],
        "retrieved_complete_table_ids": ["tatqa:table-1:table"],
        "retrieved_table_units_by_hit": [
            {
                "rank": 1,
                "document_id": "tatqa:table-1:table",
                "row_id": None,
                "cell_id": None,
                "table_complete": True,
            }
        ],
    }
    value.update(updates)
    return value


def test_diagnostic_separates_complete_table_context_from_explicit_units(
    tmp_path: Path,
) -> None:
    annotation_path = tmp_path / "tagop.json"
    prediction_path = tmp_path / "predictions.jsonl"
    _write_json(annotation_path, _annotations())
    prediction_path.write_text(
        "\n".join(
            json.dumps(value)
            for value in (
                _prediction("tatqa:q1"),
                _prediction(
                    "tatqa:q2",
                    retrieved_ids=["paragraph", "tatqa:table-1:table"],
                    retrieved_complete_table_ids=[],
                    retrieved_row_ids=["tatqa:table-1:table::row::0"],
                    retrieved_cell_ids=["tatqa:table-1:table::cell::0::1"],
                    retrieved_table_units_by_hit=[
                        {
                            "rank": 2,
                            "document_id": "tatqa:table-1:table",
                            "row_id": "tatqa:table-1:table::row::0",
                            "cell_id": "tatqa:table-1:table::cell::0::1",
                            "table_complete": False,
                        }
                    ],
                ),
                _prediction("tatqa:q3", category="text"),
            )
        ),
        encoding="utf-8",
    )

    report = evaluate_tagop_mapping_retrieval(
        annotation_path,
        prediction_path,
        expected_annotation_sha256=None,
    )

    assert report["promotion_eligible"] is False
    assert report["coverage"] == {
        "prediction_cases": 3,
        "mapping_eligible_cases": 2,
        "no_table_mapping_cases": 1,
        "header_row_mapping_cases": 1,
        "deduplicated_repeated_coordinates": 1,
        "hit_alignment_available_cases": 3,
    }
    aggregate = report["aggregate"]
    assert aggregate["tagop_heuristic_emitted_cell_unit_recall"] == 0.5
    assert aggregate["tagop_heuristic_context_cell_coverage"] == 1.0
    assert aggregate["complete_table_at_document_k"] == 0.5
    assert aggregate["tagop_heuristic_hit_aligned_context_cell_coverage_at_k"] == 1.0
    assert aggregate["tagop_heuristic_query_slot_cell_recall_at_k"] == 1.0
    assert report["program_oracle"]["eligible_cases"] == 1
    assert report["program_oracle"]["query_slots_gold_program_success"] == 1.0


def test_annotation_coordinates_are_bounds_checked(tmp_path: Path) -> None:
    annotations = _annotations()
    annotations[0]["questions"][0]["mapping"] = {"table": [[9, 1]]}  # type: ignore[index]
    annotation_path = tmp_path / "tagop.json"
    prediction_path = tmp_path / "predictions.jsonl"
    _write_json(annotation_path, annotations)
    prediction_path.write_text(json.dumps(_prediction("tatqa:q1")), encoding="utf-8")

    with pytest.raises(TATQAMappingDiagnosticError, match="out of bounds"):
        evaluate_tagop_mapping_retrieval(
            annotation_path,
            prediction_path,
            expected_annotation_sha256=None,
        )


def test_pinned_annotation_hash_is_enforced(tmp_path: Path) -> None:
    annotation_path = tmp_path / "tagop.json"
    prediction_path = tmp_path / "predictions.jsonl"
    _write_json(annotation_path, _annotations())
    prediction_path.write_text(json.dumps(_prediction("tatqa:q1")), encoding="utf-8")

    with pytest.raises(TATQAMappingDiagnosticError, match="SHA-256"):
        evaluate_tagop_mapping_retrieval(
            annotation_path,
            prediction_path,
            expected_annotation_sha256="0" * 64,
        )

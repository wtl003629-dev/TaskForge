from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from taskforge.qasper_data import normalize_qasper_parquet_row, prepare_qasper_data


def _row(paper_id: str) -> dict[str, object]:
    return {
        "id": paper_id,
        "title": "Paper title",
        "abstract": "Paper abstract",
        "full_text": {
            "section_name": ["Introduction", "Results"],
            "paragraphs": [["Intro paragraph"], ["Gold evidence"]],
        },
        "qas": {
            "question": ["What was found?"],
            "question_id": [f"q-{paper_id}"],
            "nlp_background": ["five"],
            "topic_background": ["familiar"],
            "paper_read": ["no"],
            "search_query": [""],
            "question_writer": ["writer"],
            "answers": [
                {
                    "answer": [
                        {
                            "unanswerable": False,
                            "extractive_spans": ["Gold"],
                            "yes_no": None,
                            "free_form_answer": "",
                            "evidence": ["Gold evidence"],
                            "highlighted_evidence": ["Gold evidence"],
                        }
                    ],
                    "annotation_id": ["annotation"],
                    "worker_id": ["worker"],
                }
            ],
        },
        "figures_and_tables": {"caption": [], "file": []},
    }


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_normalize_qasper_parquet_row_restores_original_shape() -> None:
    paper_id, paper = normalize_qasper_parquet_row(_row("paper-1"))
    assert paper_id == "paper-1"
    assert paper["full_text"][1] == {
        "section_name": "Results",
        "paragraphs": ["Gold evidence"],
    }
    assert paper["qas"][0]["answers"][0]["answer"]["evidence"] == [
        "Gold evidence"
    ]


def test_prepare_qasper_data_is_disjoint_deterministic_and_records_hashes(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.parquet"
    validation = tmp_path / "validation.parquet"
    _write_parquet(train, [_row("train-paper")])
    _write_parquet(validation, [_row("validation-paper")])
    train_json = tmp_path / "train.json"
    validation_json = tmp_path / "validation.json"
    manifest_path = tmp_path / "manifest.json"

    manifest = prepare_qasper_data(
        train_parquet=train,
        validation_parquet=validation,
        train_json=train_json,
        validation_json=validation_json,
        manifest_path=manifest_path,
    )
    assert manifest["final_test_downloaded"] is False
    assert manifest["partitions"]["train"]["paper_count"] == 1
    assert manifest["partitions"]["train"]["question_count"] == 1
    assert manifest["partitions"]["train"]["normalized_json"]["sha256"] == (
        hashlib.sha256(train_json.read_bytes()).hexdigest()
    )
    assert set(json.loads(train_json.read_text(encoding="utf-8"))) == {"train-paper"}
    first = manifest_path.read_bytes()
    prepare_qasper_data(
        train_parquet=train,
        validation_parquet=validation,
        train_json=train_json,
        validation_json=validation_json,
        manifest_path=manifest_path,
    )
    assert manifest_path.read_bytes() == first


def test_prepare_qasper_data_rejects_paper_overlap(tmp_path: Path) -> None:
    train = tmp_path / "train.parquet"
    validation = tmp_path / "validation.parquet"
    _write_parquet(train, [_row("same-paper")])
    _write_parquet(validation, [_row("same-paper")])
    with pytest.raises(ValueError, match="overlap"):
        prepare_qasper_data(
            train_parquet=train,
            validation_parquet=validation,
            train_json=tmp_path / "train.json",
            validation_json=tmp_path / "validation.json",
            manifest_path=tmp_path / "manifest.json",
        )

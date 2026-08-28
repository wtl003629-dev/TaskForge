from __future__ import annotations

import json

from scripts.validate_qasper_splits import validate_split_files


def _split(tmp_path, name: str, case_ids: list[str]):
    path = tmp_path / f"{name}.json"
    path.write_text(
        json.dumps({"split_id": name, "case_ids": case_ids}),
        encoding="utf-8",
    )
    return path


def test_split_validator_detects_same_paper_with_different_questions(
    tmp_path,
) -> None:
    first = _split(tmp_path, "first", ["qasper:paper-a:q1"])
    second = _split(tmp_path, "second", ["qasper:paper-a:q2"])

    report = validate_split_files([first, second])

    assert report["valid"] is False
    assert report["overlaps"] == [
        {
            "left": "first",
            "right": "second",
            "paper_ids": ["paper-a"],
        }
    ]


def test_split_validator_accepts_paper_disjoint_manifests(tmp_path) -> None:
    first = _split(tmp_path, "first", ["qasper:paper-a:q1"])
    second = _split(tmp_path, "second", ["qasper:paper-b:q2"])

    report = validate_split_files([first, second])

    assert report["valid"] is True
    assert report["overlaps"] == []

from __future__ import annotations

import json

from scripts.audit_tatqa_table_cleaning import build_audit


def test_cleaning_audit_records_source_and_safe_duplicate_contract(tmp_path) -> None:
    source = tmp_path / "tatqa.json"
    source.write_text(
        json.dumps(
            [
                {
                    "table": {
                        "uid": "t-1",
                        "table": [
                            ["Metric", "2025"],
                            ["Revenue", "10"],
                            ["", ""],
                            ["Revenue", "10"],
                        ],
                    },
                    "paragraphs": [],
                    "questions": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    report = build_audit(source)

    assert report["source"]["sha256"]
    assert report["contract"]["raw_rows_mutated"] is False
    assert report["contract"]["global_nonconsecutive_duplicates_removed"] is False
    assert report["totals"]["empty_rows_removed"] == 1
    assert report["totals"]["consecutive_duplicate_rows_folded"] == 1


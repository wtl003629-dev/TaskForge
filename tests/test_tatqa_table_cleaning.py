from __future__ import annotations

import json

from taskforge.rag_evaluation import load_tatqa_dataset
from taskforge.tatqa_table_cleaning import clean_tatqa_table


def _table() -> list[list[str]]:
    return [
        ["", "Years Ended", ""],
        ["", "2025", "2024"],
        [" Revenue\u00a0", "$1,000", "(900)"],
        ["", "", ""],
        ["", "Years Ended", ""],
        ["", "2025", "2024"],
        ["Revenue", "$1,000", "(900)"],
        ["Total", "-", "50%"],
        ["Total", "-", "50%"],
    ]


def test_cleaner_preserves_raw_coordinates_and_normalizes_financial_cells() -> None:
    raw = _table()

    cleaned = clean_tatqa_table(raw, context_text="Amounts in millions")

    assert raw == _table()
    assert cleaned.headers == (
        "row_label",
        "Years Ended | 2025",
        "Years Ended | 2024",
    )
    assert cleaned.rows == (
        ("", "Years Ended", ""),
        ("", "2025", "2024"),
        ("Revenue", "$1,000", "(900)"),
        ("Total", "", "50%"),
    )
    assert cleaned.row_source_indices == ((0,), (1,), (2, 6), (7, 8))
    assert cleaned.audit.empty_rows_removed == 1
    assert cleaned.audit.repeated_header_rows_removed == 2
    assert cleaned.audit.consecutive_duplicate_rows_folded == 2
    assert cleaned.audit.parenthesized_negative_cells == 1
    negative = cleaned.cell_metadata[2][2]
    assert negative["normalized"] == "-900"
    assert negative["scaled_numeric"] == -900_000_000.0
    assert negative["source_row_indices"] == [2, 6]
    percent = cleaned.cell_metadata[3][2]
    assert percent["unit"] == "percent"
    assert percent["normalized"] == "50"


def test_cleaner_preserves_nonconsecutive_duplicate_business_rows() -> None:
    cleaned = clean_tatqa_table(
        [
            ["Metric", "2025"],
            ["Total", "10"],
            ["Subtotal", "5"],
            ["Total", "10"],
        ]
    )

    assert len(cleaned.rows) == 4
    assert cleaned.row_source_indices == ((0,), (1,), (2,), (3,))
    assert cleaned.audit.nonconsecutive_duplicate_rows_preserved == 1


def test_unit_label_can_be_part_of_a_two_level_header() -> None:
    cleaned = clean_tatqa_table(
        [
            ["", "Payments Due by Year", ""],
            ["(In thousands)", "Total", "Beyond 5 Years"],
            ["Operating leases", "12,807", "0"],
        ]
    )

    assert cleaned.audit.header_depth == 2
    assert cleaned.headers == (
        "(In thousands)",
        "Payments Due by Year | Total",
        "Payments Due by Year | Beyond 5 Years",
    )


def test_tatqa_loader_exposes_clean_search_rows_without_replacing_raw_rows(
    tmp_path,
) -> None:
    payload = [
        {
            "table": {"uid": "table-1", "table": _table()},
            "paragraphs": [
                {"uid": "p-1", "order": 1, "text": "Amounts in millions."}
            ],
            "questions": [
                {
                    "uid": "q-1",
                    "question": "What was revenue in 2025?",
                    "answer": "1000",
                    "answer_type": "span",
                    "answer_from": "table",
                    "rel_paragraphs": [],
                    "derivation": "",
                    "scale": "million",
                }
            ],
        }
    ]
    path = tmp_path / "tatqa.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    raw_dataset = load_tatqa_dataset(path)
    cleaned_dataset = load_tatqa_dataset(path, table_cleaning=True)
    raw_document = raw_dataset.documents[0]
    cleaned_document = cleaned_dataset.documents[0]

    assert raw_document.metadata["table_rows"] == _table()
    assert "table_rows_cleaned" not in raw_document.metadata
    assert cleaned_document.metadata["table_rows"] == _table()
    assert cleaned_document.metadata["table_rows_cleaned"][2][0] == "Revenue"
    assert cleaned_document.metadata["table_cleaning"]["row_source_indices"] == [
        [0],
        [1],
        [2, 6],
        [7, 8],
    ]
    assert "Years Ended" in cleaned_document.text
    assert "2025 | 2024" in cleaned_document.text

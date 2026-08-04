from __future__ import annotations

from pathlib import Path

import pytest

from taskforge.synthetic_pdf_eval import (
    SyntheticPDFSuite,
    generate_synthetic_pdfs,
    load_generated_page_dataset,
)


ROOT = Path(__file__).resolve().parents[1]


def test_generates_real_pdfs_and_loads_page_evidence(tmp_path: Path) -> None:
    suite_path = ROOT / "eval" / "synthetic_pdf_suite.json"
    first = generate_synthetic_pdfs(suite_path, tmp_path / "first")
    second = generate_synthetic_pdfs(suite_path, tmp_path / "second")
    assert first.suite_sha256 == second.suite_sha256
    assert len(first.documents) == 3
    assert [item.sha256 for item in first.documents] == [
        item.sha256 for item in second.documents
    ]
    assert all(Path(item.path).read_bytes().startswith(b"%PDF-") for item in first.documents)

    dataset = load_generated_page_dataset(suite_path, first)
    assert len(dataset.documents) == 6
    assert len(dataset.cases) == 12
    combined = next(item for item in dataset.cases if item.case_id == "synthetic:change-combined")
    assert combined.relevant_ids == [
        "synthetic:change-control:page:1",
        "synthetic:change-control:page:2",
    ]
    table_page = next(
        item for item in dataset.documents
        if item.document_id == "synthetic:change-control:page:1"
    )
    assert "Change advisory board and security" in table_page.text


def test_suite_rejects_unknown_evidence_page() -> None:
    payload = {
        "suite_id": "bad",
        "documents": [
            {
                "document_id": "doc",
                "filename": "doc.pdf",
                "pages": [{"page": 1, "title": "Page"}],
            }
        ],
        "cases": [
            {
                "case_id": "case",
                "question": "Question?",
                "answer": "Answer",
                "category": "text",
                "evidence": [{"document_id": "doc", "pages": [2]}],
            }
        ],
    }
    with pytest.raises(ValueError, match="unknown synthetic page"):
        SyntheticPDFSuite.model_validate(payload)

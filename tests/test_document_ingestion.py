from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, StreamObject

from taskforge import document_ingestion
from taskforge.document_ingestion import (
    PDFDependencyError,
    PDFSafetyError,
    extract_pdf_document,
)


def _pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _write_pdf(
    path: Path,
    *,
    pages: int = 1,
    include_content: bool = True,
    include_table: bool = True,
    password: str | None = None,
) -> None:
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)
    for page_index in range(pages):
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): font_reference}
                )
            }
        )
        commands: list[str] = []
        if include_content:
            heading = f"{page_index + 1}. {'SECURITY CONTROLS' if page_index == 0 else 'APPROVALS'}"
            paragraph = (
                "Exports require audit logging."
                if page_index == 0
                else "High risk exports require human approval."
            )
            commands.extend(
                [
                    f"BT /F1 16 Tf 72 740 Td ({_pdf_text(heading)}) Tj ET",
                    f"BT /F1 11 Tf 72 710 Td ({_pdf_text(paragraph)}) Tj ET",
                ]
            )
            if include_table and page_index == 0:
                for y in (680, 650, 620):
                    commands.append(f"72 {y} m 360 {y} l S")
                for x in (72, 220, 360):
                    commands.append(f"{x} 620 m {x} 680 l S")
                commands.extend(
                    [
                        "BT /F1 10 Tf 82 660 Td (Control) Tj ET",
                        "BT /F1 10 Tf 230 660 Td (Status) Tj ET",
                        "BT /F1 10 Tf 82 630 Td (Audit) Tj ET",
                        "BT /F1 10 Tf 230 630 Td (Missing) Tj ET",
                    ]
                )
        if commands:
            stream = StreamObject()
            stream.set_data(("\n".join(commands) + "\n").encode("ascii"))
            page[NameObject("/Contents")] = writer._add_object(stream)
    if password is not None:
        writer.encrypt(password)
    with path.open("wb") as handle:
        writer.write(handle)


def test_extracts_pypdf_text_pdfplumber_table_and_provenance(tmp_path: Path) -> None:
    pdf = tmp_path / "review.pdf"
    _write_pdf(pdf)

    result = extract_pdf_document(
        pdf,
        source_uri="policies/review-v3.pdf",
        chunk_chars=256,
    )

    assert result.source_uri == "policies/review-v3.pdf"
    assert result.page_count == 1
    assert result.bytes_read == pdf.stat().st_size
    assert len(result.sha256) == 64
    paragraph_text = "\n".join(
        block.text for block in result.blocks if block.kind == "paragraph"
    )
    assert "Exports require audit logging." in paragraph_text

    table = next(block for block in result.blocks if block.kind == "table")
    assert table.page == 1
    assert table.table_rows == (("Control", "Status"), ("Audit", "Missing"))
    assert table.bbox != (0.0, 0.0, 612.0, 792.0)
    assert table.bbox[0] < table.bbox[2]
    assert table.bbox[1] < table.bbox[3]
    assert len(table.content_hash) == 64

    for index, block in enumerate(result.blocks):
        assert block.page == 1
        assert block.previous_block_id == (
            result.blocks[index - 1].block_id if index > 0 else None
        )
        assert block.next_block_id == (
            result.blocks[index + 1].block_id
            if index + 1 < len(result.blocks)
            else None
        )

    assert result.chunks
    first = result.chunks[0]
    assert first.heading == "1. SECURITY CONTROLS"
    assert first.pages == (1,)
    assert first.block_ids
    assert {item.block_id for item in first.provenance} == set(first.block_ids)
    assert all(item.page == 1 for item in first.provenance)


def test_heading_boundaries_and_chunk_neighbors_span_pages(tmp_path: Path) -> None:
    pdf = tmp_path / "two-pages.pdf"
    _write_pdf(pdf, pages=2, include_table=False)

    result = extract_pdf_document(pdf, chunk_chars=256)

    assert [chunk.heading for chunk in result.chunks] == [
        "1. SECURITY CONTROLS",
        "2. APPROVALS",
    ]
    assert [chunk.pages for chunk in result.chunks] == [(1,), (2,)]
    assert result.chunks[0].previous_chunk_id is None
    assert result.chunks[0].next_chunk_id == result.chunks[1].chunk_id
    assert result.chunks[1].previous_chunk_id == result.chunks[0].chunk_id
    assert result.chunks[1].next_chunk_id is None
    assert result.chunks[0].next_block_id == result.chunks[1].block_ids[0]
    assert result.chunks[1].previous_block_id == result.chunks[0].block_ids[-1]


def test_rejects_oversize_too_many_pages_encrypted_and_empty(tmp_path: Path) -> None:
    regular = tmp_path / "regular.pdf"
    _write_pdf(regular, pages=2, include_table=False)
    with pytest.raises(PDFSafetyError, match="byte limit"):
        extract_pdf_document(regular, max_bytes=regular.stat().st_size - 1)
    with pytest.raises(PDFSafetyError, match="page limit"):
        extract_pdf_document(regular, max_pages=1)

    encrypted = tmp_path / "encrypted.pdf"
    _write_pdf(encrypted, password="correct-horse-battery-staple")
    with pytest.raises(PDFSafetyError, match="encrypted"):
        extract_pdf_document(encrypted)

    empty = tmp_path / "empty.pdf"
    _write_pdf(empty, include_content=False)
    with pytest.raises(PDFSafetyError, match="no machine-readable"):
        extract_pdf_document(empty)


def test_rejects_non_pdf_and_reports_missing_dependency(tmp_path: Path, monkeypatch) -> None:
    not_pdf = tmp_path / "pretend.pdf"
    not_pdf.write_text("not a PDF", encoding="utf-8")
    with pytest.raises(PDFSafetyError, match="not a PDF"):
        extract_pdf_document(not_pdf)

    pdf = tmp_path / "dependency.pdf"
    _write_pdf(pdf)
    real_import = document_ingestion.importlib.import_module

    def import_with_pdfplumber_missing(name: str):
        if name == "pdfplumber":
            raise ImportError("simulated missing dependency")
        return real_import(name)

    monkeypatch.setattr(
        document_ingestion.importlib,
        "import_module",
        import_with_pdfplumber_missing,
    )
    with pytest.raises(PDFDependencyError, match="pdfplumber") as error:
        extract_pdf_document(pdf)
    assert "pypdf>=5,<7" in str(error.value)


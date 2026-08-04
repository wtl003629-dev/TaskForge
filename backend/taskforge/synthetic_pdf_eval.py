"""Reproducible, self-authored PDF retrieval suite used by offline CI.

The suite is deliberately small and transparent.  It verifies that a real PDF
is generated, parsed page by page, and evaluated without claiming to replace
licensed public benchmarks.  ReportLab is a generation-only dependency; the
runtime PDF ingestion path continues to use pypdf and pdfplumber.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from .domain import StrictModel
from .rag_evaluation import EvalCorpusDocument, RAGEvalCase, RAGEvalDataset


class SyntheticTable(StrictModel):
    headers: list[str] = Field(min_length=1)
    rows: list[list[str]] = Field(min_length=1)

    @model_validator(mode="after")
    def rectangular(self) -> "SyntheticTable":
        width = len(self.headers)
        if any(len(row) != width for row in self.rows):
            raise ValueError("synthetic table rows must match header width")
        return self


class SyntheticPage(StrictModel):
    page: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=160)
    paragraphs: list[str] = Field(default_factory=list)
    tables: list[SyntheticTable] = Field(default_factory=list)


class SyntheticDocument(StrictModel):
    document_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    filename: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*\.pdf$")
    pages: list[SyntheticPage] = Field(min_length=1)

    @model_validator(mode="after")
    def pages_are_contiguous(self) -> "SyntheticDocument":
        numbers = [item.page for item in self.pages]
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError("synthetic document pages must be contiguous and ordered")
        return self


class SyntheticEvidence(StrictModel):
    document_id: str = Field(min_length=1)
    pages: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_pages(self) -> "SyntheticEvidence":
        if len(self.pages) != len(set(self.pages)) or any(page < 1 for page in self.pages):
            raise ValueError("evidence pages must be unique positive integers")
        return self


class SyntheticCase(StrictModel):
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    category: str = Field(min_length=1)
    evidence: list[SyntheticEvidence] = Field(min_length=1)


class SyntheticPDFSuite(StrictModel):
    schema_version: str = "1.0"
    suite_id: str = Field(min_length=1)
    license: str = "CC0-1.0"
    documents: list[SyntheticDocument] = Field(min_length=1)
    cases: list[SyntheticCase] = Field(min_length=1)

    @model_validator(mode="after")
    def references_are_valid(self) -> "SyntheticPDFSuite":
        document_ids = [item.document_id for item in self.documents]
        case_ids = [item.case_id for item in self.cases]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("synthetic document IDs must be unique")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("synthetic case IDs must be unique")
        pages = {
            document.document_id: {page.page for page in document.pages}
            for document in self.documents
        }
        for case in self.cases:
            for evidence in case.evidence:
                if evidence.document_id not in pages:
                    raise ValueError("case references an unknown synthetic document")
                if not set(evidence.pages).issubset(pages[evidence.document_id]):
                    raise ValueError("case references an unknown synthetic page")
        return self


class GeneratedPDF(StrictModel):
    document_id: str
    filename: str
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes_written: int = Field(gt=0)
    pages: int = Field(gt=0)


class SyntheticGenerationManifest(StrictModel):
    schema_version: str = "1.0"
    suite_id: str
    suite_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    documents: list[GeneratedPDF]


def load_synthetic_suite(path: str | Path) -> SyntheticPDFSuite:
    return SyntheticPDFSuite.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _reportlab() -> tuple[Any, Any, Any, Any]:
    try:
        canvas = importlib.import_module("reportlab.pdfgen.canvas")
        pagesizes = importlib.import_module("reportlab.lib.pagesizes")
        colors = importlib.import_module("reportlab.lib.colors")
        table_module = importlib.import_module("reportlab.platypus")
    except ImportError as exc:
        raise RuntimeError(
            "synthetic PDF generation requires the evaluation extra: "
            "pip install -e '.[evaluation]'"
        ) from exc
    return canvas, pagesizes, colors, table_module


def generate_synthetic_pdfs(
    suite_path: str | Path,
    output_dir: str | Path,
) -> SyntheticGenerationManifest:
    suite_file = Path(suite_path).resolve(strict=True)
    suite_raw = suite_file.read_bytes()
    suite = SyntheticPDFSuite.model_validate_json(suite_raw)
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    canvas_module, pagesizes, colors, table_module = _reportlab()
    generated: list[GeneratedPDF] = []
    for document in suite.documents:
        target = (root / document.filename).resolve()
        if target.parent != root:
            raise ValueError("synthetic PDF target escapes output directory")
        temporary = root / f".{document.filename}.part"
        if temporary.exists():
            temporary.unlink()
        pdf = canvas_module.Canvas(
            str(temporary),
            pagesize=pagesizes.A4,
            pageCompression=0,
            invariant=1,
        )
        width, height = pagesizes.A4
        for page in document.pages:
            y = height - 54
            pdf.setFont("Helvetica-Bold", 18)
            pdf.drawString(54, y, page.title)
            y -= 34
            pdf.setFont("Helvetica", 10)
            for paragraph in page.paragraphs:
                for line in _wrap(paragraph, 92):
                    pdf.drawString(54, y, line)
                    y -= 15
                y -= 8
            for table_spec in page.tables:
                data = [table_spec.headers, *table_spec.rows]
                table = table_module.Table(
                    data,
                    colWidths=[(width - 108) / len(table_spec.headers)] * len(table_spec.headers),
                    repeatRows=1,
                )
                table.setStyle(
                    table_module.TableStyle(
                        [
                            ("GRID", (0, 0), (-1, -1), 0.75, colors.black),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                            ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 5),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                        ]
                    )
                )
                table_width, table_height = table.wrap(width - 108, y - 54)
                if table_height > y - 54:
                    raise ValueError(f"synthetic table does not fit page {page.page}")
                table.drawOn(pdf, 54, y - table_height)
                y -= table_height + 20
            pdf.setFont("Helvetica", 8)
            pdf.drawRightString(width - 54, 30, f"{document.document_id} / page {page.page}")
            pdf.showPage()
        pdf.save()
        temporary.replace(target)
        raw = target.read_bytes()
        generated.append(
            GeneratedPDF(
                document_id=document.document_id,
                filename=document.filename,
                path=str(target),
                sha256=hashlib.sha256(raw).hexdigest(),
                bytes_written=len(raw),
                pages=len(document.pages),
            )
        )
    return SyntheticGenerationManifest(
        suite_id=suite.suite_id,
        suite_sha256=hashlib.sha256(suite_raw).hexdigest(),
        documents=generated,
    )


def load_generated_page_dataset(
    suite_path: str | Path,
    manifest: SyntheticGenerationManifest,
) -> RAGEvalDataset:
    """Parse generated PDFs into page evidence IDs with the real pypdf reader."""

    try:
        pypdf = importlib.import_module("pypdf")
    except ImportError as exc:
        raise RuntimeError("pypdf is required to load generated evaluation PDFs") from exc
    suite = load_synthetic_suite(suite_path)
    if manifest.suite_id != suite.suite_id:
        raise ValueError("generation manifest belongs to another suite")
    manifest_documents = {item.document_id: item for item in manifest.documents}
    documents: list[EvalCorpusDocument] = []
    for source in suite.documents:
        generated = manifest_documents.get(source.document_id)
        if generated is None:
            raise ValueError("generation manifest is missing a document")
        path = Path(generated.path).resolve(strict=True)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != generated.sha256:
            raise ValueError("generated PDF checksum mismatch")
        reader = pypdf.PdfReader(path, strict=False)
        if len(reader.pages) != len(source.pages):
            raise ValueError("generated PDF page count mismatch")
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                raise ValueError("generated PDF page has no extractable text")
            evidence_id = f"synthetic:{source.document_id}:page:{page_number}"
            documents.append(
                EvalCorpusDocument(
                    document_id=evidence_id,
                    text=text,
                    source_uri=f"synthetic://{source.document_id}/page/{page_number}",
                    metadata={
                        "kind": "pdf-page",
                        "document_id": source.document_id,
                        "page": page_number,
                        "pdf_sha256": generated.sha256,
                    },
                )
            )
    cases = [
        RAGEvalCase(
            case_id=f"synthetic:{case.case_id}",
            dataset=suite.suite_id,
            query=case.question,
            relevant_ids=[
                f"synthetic:{evidence.document_id}:page:{page}"
                for evidence in case.evidence
                for page in evidence.pages
            ],
            category=case.category,
            answer=case.answer,
            metadata={"self_authored": True},
        )
        for case in suite.cases
    ]
    return RAGEvalDataset(
        dataset=suite.suite_id,
        license=suite.license,
        attribution_url="https://creativecommons.org/publicdomain/zero/1.0/",
        documents=documents,
        cases=cases,
    )


def _wrap(value: str, width: int) -> list[str]:
    words = re.sub(r"\s+", " ", value).strip().split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


__all__ = [
    "GeneratedPDF",
    "SyntheticCase",
    "SyntheticDocument",
    "SyntheticEvidence",
    "SyntheticGenerationManifest",
    "SyntheticPDFSuite",
    "SyntheticPage",
    "SyntheticTable",
    "generate_synthetic_pdfs",
    "load_generated_page_dataset",
    "load_synthetic_suite",
]

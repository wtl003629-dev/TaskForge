"""Safe, structure-preserving extraction for machine-generated PDF documents.

The module deliberately stops at deterministic document structure.  It does
not perform OCR, execute embedded content, or grant extracted text any
authority.  ``pypdf`` supplies the page text while ``pdfplumber`` supplies
machine-drawn table structure and positional bounding boxes.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

BoundingBox = tuple[float, float, float, float]
BlockKind = Literal["paragraph", "table"]


class DocumentIngestionError(ValueError):
    """Base error for a rejected or unreadable document."""


class PDFDependencyError(DocumentIngestionError):
    """Required PDF extraction libraries are unavailable."""


class PDFSafetyError(DocumentIngestionError):
    """A PDF violates a deterministic ingestion safety limit."""


class PDFExtractionError(DocumentIngestionError):
    """A PDF cannot be parsed into trustworthy machine-readable structure."""


@dataclass(frozen=True, slots=True)
class DocumentBlock:
    """One ordered, content-addressed paragraph or table from a PDF page.

    Bounding boxes use pdfplumber's page coordinate system:
    ``(x0, top, x1, bottom)`` with the origin at the top-left.  Page numbers
    are one-based for user-facing provenance.
    """

    block_id: str
    kind: BlockKind
    page: int
    bbox: BoundingBox
    text: str
    content_hash: str
    table_rows: tuple[tuple[str, ...], ...] = ()
    heading_level: int | None = None
    previous_block_id: str | None = None
    next_block_id: str | None = None

    @property
    def is_heading(self) -> bool:
        return self.kind == "paragraph" and self.heading_level is not None


@dataclass(frozen=True, slots=True)
class BlockProvenance:
    """Stable source locator copied into every structure-aware chunk."""

    block_id: str
    page: int
    bbox: BoundingBox
    content_hash: str
    previous_block_id: str | None
    next_block_id: str | None


@dataclass(frozen=True, slots=True)
class StructureChunk:
    """A chunk assembled only at heading, paragraph, or table boundaries."""

    chunk_id: str
    text: str
    content_hash: str
    pages: tuple[int, ...]
    block_ids: tuple[str, ...]
    provenance: tuple[BlockProvenance, ...]
    heading: str | None = None
    previous_block_id: str | None = None
    next_block_id: str | None = None
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None


@dataclass(frozen=True, slots=True)
class StructuredDocument:
    """Complete deterministic extraction result for one PDF version."""

    document_id: str
    source_uri: str
    sha256: str
    bytes_read: int
    page_count: int
    blocks: tuple[DocumentBlock, ...]
    chunks: tuple[StructureChunk, ...]


@dataclass(slots=True)
class _BlockDraft:
    kind: BlockKind
    page: int
    bbox: BoundingBox
    text: str
    table_rows: tuple[tuple[str, ...], ...] = ()
    heading_level: int | None = None


_HEADING_NUMBER = re.compile(
    r"^(?:(?:\d+(?:\.\d+){0,5})|(?:[一二三四五六七八九十]+))[\.、：:\s]+\S"
)
_SENTENCE_END = frozenset("。！？.!?；;")


def extract_pdf_document(
    path: str | Path,
    *,
    source_uri: str | None = None,
    max_bytes: int = 20_000_000,
    max_pages: int = 200,
    max_blocks: int = 20_000,
    chunk_chars: int = 2_000,
    preserve_page_boundaries: bool = False,
    table_settings: Mapping[str, Any] | None = None,
) -> StructuredDocument:
    """Extract a bounded, unencrypted, machine-generated PDF.

    ``chunk_chars`` is a target rather than a destructive hard split: a single
    paragraph or table is never cut merely to satisfy that target.  This keeps
    source provenance honest and makes long source blocks visible to callers.
    """

    _validate_limits(
        max_bytes=max_bytes,
        max_pages=max_pages,
        max_blocks=max_blocks,
        chunk_chars=chunk_chars,
    )
    pypdf, pdfplumber = _load_dependencies()
    pdf_path = Path(path).resolve(strict=True)
    if not pdf_path.is_file():
        raise PDFSafetyError("PDF ingestion target is not a regular file")
    if pdf_path.stat().st_size > max_bytes:
        raise PDFSafetyError(f"PDF exceeds the {max_bytes} byte limit")
    with pdf_path.open("rb") as handle:
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise PDFSafetyError(f"PDF exceeds the {max_bytes} byte limit")
    if not raw.startswith(b"%PDF-"):
        raise PDFSafetyError("ingestion target is not a PDF file")

    digest = hashlib.sha256(raw).hexdigest()
    document_id = f"pdf:{digest[:24]}"
    safe_source_uri = _safe_source_uri(source_uri, pdf_path)
    try:
        reader = pypdf.PdfReader(BytesIO(raw), strict=False)
    except Exception as exc:  # library exceptions vary by PDF defect
        raise PDFExtractionError("pypdf could not parse the PDF") from exc
    if bool(reader.is_encrypted):
        raise PDFSafetyError("encrypted PDFs cannot be ingested")
    try:
        page_count = len(reader.pages)
    except Exception as exc:
        raise PDFExtractionError("pypdf could not enumerate PDF pages") from exc
    if page_count < 1:
        raise PDFSafetyError("PDF must contain at least one page")
    if page_count > max_pages:
        raise PDFSafetyError(f"PDF exceeds the {max_pages} page limit")

    drafts: list[_BlockDraft] = []
    try:
        with pdfplumber.open(BytesIO(raw)) as plumber_document:
            if len(plumber_document.pages) != page_count:
                raise PDFExtractionError("PDF parsers disagree on the page count")
            for page_index, (reader_page, plumber_page) in enumerate(
                zip(reader.pages, plumber_document.pages, strict=True),
                start=1,
            ):
                page_drafts = _extract_page(
                    reader_page,
                    plumber_page,
                    page_number=page_index,
                    table_settings=table_settings,
                )
                drafts.extend(page_drafts)
                if len(drafts) > max_blocks:
                    raise PDFSafetyError(f"PDF exceeds the {max_blocks} block limit")
    except (PDFExtractionError, PDFSafetyError):
        raise
    except Exception as exc:  # pdfminer/pdfplumber expose several parser errors
        raise PDFExtractionError("pdfplumber could not extract PDF structure") from exc

    if not drafts or not any(item.text.strip() for item in drafts):
        raise PDFSafetyError(
            "PDF has no machine-readable text or tables; OCR is not performed"
        )
    blocks = _finalize_blocks(document_id, drafts)
    chunks = build_structure_chunks(
        blocks,
        document_id=document_id,
        chunk_chars=chunk_chars,
        preserve_page_boundaries=preserve_page_boundaries,
    )
    return StructuredDocument(
        document_id=document_id,
        source_uri=safe_source_uri,
        sha256=digest,
        bytes_read=len(raw),
        page_count=page_count,
        blocks=blocks,
        chunks=chunks,
    )


def build_structure_chunks(
    blocks: Sequence[DocumentBlock],
    *,
    document_id: str,
    chunk_chars: int = 2_000,
    preserve_page_boundaries: bool = False,
) -> tuple[StructureChunk, ...]:
    """Group blocks while optionally retaining page-level retrieval provenance."""

    if not document_id.strip():
        raise ValueError("document_id is required")
    if not 256 <= chunk_chars <= 50_000:
        raise ValueError("chunk_chars must be between 256 and 50000")
    if not blocks:
        return ()

    ordered = tuple(blocks)
    block_positions = {block.block_id: index for index, block in enumerate(ordered)}
    if len(block_positions) != len(ordered):
        raise ValueError("block IDs must be unique")

    groups: list[tuple[str | None, tuple[DocumentBlock, ...]]] = []
    current: list[DocumentBlock] = []
    active_heading: str | None = None
    current_heading: str | None = None
    current_chars = 0

    def flush() -> None:
        nonlocal current, current_chars, current_heading
        if current:
            groups.append((current_heading, tuple(current)))
        current = []
        current_chars = 0
        current_heading = active_heading

    for block in ordered:
        if current and preserve_page_boundaries and current[-1].page != block.page:
            flush()
        if block.is_heading:
            flush()
            active_heading = block.text
            current_heading = active_heading
        separator = 2 if current else 0
        if current and current_chars + separator + len(block.text) > chunk_chars:
            flush()
        if not current:
            current_heading = active_heading
        current.append(block)
        current_chars += separator + len(block.text)
    flush()

    chunks: list[StructureChunk] = []
    for chunk_index, (heading, group) in enumerate(groups):
        text = "\n\n".join(block.text for block in group)
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        chunk_id = hashlib.sha256(
            f"{document_id}\0{chunk_index}\0{content_hash}".encode()
        ).hexdigest()[:24]
        first_position = block_positions[group[0].block_id]
        last_position = block_positions[group[-1].block_id]
        provenance = tuple(
            BlockProvenance(
                block_id=block.block_id,
                page=block.page,
                bbox=block.bbox,
                content_hash=block.content_hash,
                previous_block_id=block.previous_block_id,
                next_block_id=block.next_block_id,
            )
            for block in group
        )
        chunks.append(
            StructureChunk(
                chunk_id=chunk_id,
                text=text,
                content_hash=content_hash,
                pages=tuple(dict.fromkeys(block.page for block in group)),
                block_ids=tuple(block.block_id for block in group),
                provenance=provenance,
                heading=heading,
                previous_block_id=(
                    ordered[first_position - 1].block_id if first_position > 0 else None
                ),
                next_block_id=(
                    ordered[last_position + 1].block_id
                    if last_position + 1 < len(ordered)
                    else None
                ),
            )
        )
    for index, chunk in enumerate(tuple(chunks)):
        chunks[index] = replace(
            chunk,
            previous_chunk_id=chunks[index - 1].chunk_id if index > 0 else None,
            next_chunk_id=chunks[index + 1].chunk_id if index + 1 < len(chunks) else None,
        )
    return tuple(chunks)


def _load_dependencies() -> tuple[Any, Any]:
    missing: list[str] = []
    modules: list[Any] = []
    for name in ("pypdf", "pdfplumber"):
        try:
            modules.append(importlib.import_module(name))
        except ImportError:
            missing.append(name)
    if missing:
        joined = ", ".join(missing)
        raise PDFDependencyError(
            f"PDF extraction requires missing dependencies: {joined}; "
            "install pypdf>=5,<7 and pdfplumber>=0.11,<1"
        )
    return modules[0], modules[1]


def _validate_limits(
    *, max_bytes: int, max_pages: int, max_blocks: int, chunk_chars: int
) -> None:
    if not 1 <= max_bytes <= 1_000_000_000:
        raise ValueError("max_bytes must be between 1 and 1000000000")
    if not 1 <= max_pages <= 10_000:
        raise ValueError("max_pages must be between 1 and 10000")
    if not 1 <= max_blocks <= 1_000_000:
        raise ValueError("max_blocks must be between 1 and 1000000")
    if not 256 <= chunk_chars <= 50_000:
        raise ValueError("chunk_chars must be between 256 and 50000")


def _safe_source_uri(source_uri: str | None, path: Path) -> str:
    value = source_uri if source_uri is not None else path.name
    value = value.strip()
    if not value or "\x00" in value or len(value) > 2_000:
        raise ValueError("source_uri must be a non-empty bounded string")
    return value


def _extract_page(
    reader_page: Any,
    plumber_page: Any,
    *,
    page_number: int,
    table_settings: Mapping[str, Any] | None,
) -> list[_BlockDraft]:
    try:
        page_text = reader_page.extract_text() or ""
    except Exception as exc:
        raise PDFExtractionError(f"pypdf could not extract page {page_number} text") from exc
    page_text = _normalise_text(page_text)

    tables: list[_BlockDraft] = []
    table_bboxes: list[BoundingBox] = []
    table_cells: set[str] = set()
    found_tables = plumber_page.find_tables(table_settings=dict(table_settings or {}))
    for table in found_tables:
        rows = _normalise_table_rows(table.extract())
        if not rows:
            continue
        bbox = _normalise_bbox(table.bbox, width=plumber_page.width, height=plumber_page.height)
        table_bboxes.append(bbox)
        table_cells.update(
            _comparison_text(cell) for row in rows for cell in row if cell.strip()
        )
        table_cells.update(
            _comparison_text(" ".join(cell for cell in row if cell.strip()))
            for row in rows
        )
        tables.append(
            _BlockDraft(
                kind="table",
                page=page_number,
                bbox=bbox,
                text=_render_table(rows),
                table_rows=rows,
            )
        )

    words = [
        word
        for word in plumber_page.extract_words(use_text_flow=True, keep_blank_chars=False)
        if not any(_word_inside_bbox(word, bbox) for bbox in table_bboxes)
    ]
    used_words: set[int] = set()
    paragraphs: list[_BlockDraft] = []
    for text, heading_level in _split_page_text(page_text, table_cells=table_cells):
        bbox = _locate_text_bbox(
            text,
            words,
            used_words,
            page_width=float(plumber_page.width),
            page_height=float(plumber_page.height),
        )
        paragraphs.append(
            _BlockDraft(
                kind="paragraph",
                page=page_number,
                bbox=bbox,
                text=text,
                heading_level=heading_level,
            )
        )

    combined = paragraphs + tables
    combined.sort(
        key=lambda item: (
            round(item.bbox[1], 3),
            round(item.bbox[0], 3),
            0 if item.kind == "paragraph" else 1,
        )
    )
    return combined


def _normalise_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def _comparison_text(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).casefold()


def _split_page_text(
    page_text: str,
    *,
    table_cells: set[str],
) -> list[tuple[str, int | None]]:
    if not page_text:
        return []
    groups: list[tuple[str, int | None]] = []
    current: list[str] = []

    def flush() -> None:
        if not current:
            return
        text = _normalise_text(" ".join(current))
        if text:
            groups.append((text, None))
        current.clear()

    for raw_line in page_text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            flush()
            continue
        if _comparison_text(line) in table_cells:
            flush()
            continue
        heading_level = _heading_level(line)
        if heading_level is not None:
            flush()
            groups.append((line.lstrip("# "), heading_level))
            continue
        current.append(line)
        if line[-1:] in _SENTENCE_END:
            flush()
    flush()
    return groups


def _heading_level(text: str) -> int | None:
    markdown = re.match(r"^(#{1,6})\s+\S", text)
    if markdown:
        return len(markdown.group(1))
    if _HEADING_NUMBER.match(text) and len(text) <= 160:
        number = re.match(r"^(\d+(?:\.\d+)*)", text)
        return min(6, number.group(1).count(".") + 1) if number else 1
    letters = [character for character in text if character.isalpha()]
    # ``str.islower`` is False for uncased scripts such as Chinese. Requiring
    # every alphabetic character to have a case prevents ordinary CJK body
    # lines from being mistaken for ALL-CAPS English headings.
    cased_letters = [
        character for character in letters if character.lower() != character.upper()
    ]
    if (
        1 <= len(text) <= 100
        and letters
        and len(cased_letters) == len(letters)
        and all(not character.islower() for character in cased_letters)
        and text[-1:] not in _SENTENCE_END
    ):
        return 1
    return None


def _normalise_table_rows(rows: Sequence[Sequence[Any] | None] | None) -> tuple[tuple[str, ...], ...]:
    cleaned: list[tuple[str, ...]] = []
    for raw_row in rows or ():
        if raw_row is None:
            continue
        row = tuple(
            re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(cell or ""))).strip()
            for cell in raw_row
        )
        if any(row):
            cleaned.append(row)
    if not cleaned:
        return ()
    width = max(len(row) for row in cleaned)
    return tuple(row + ("",) * (width - len(row)) for row in cleaned)


def _render_table(rows: tuple[tuple[str, ...], ...]) -> str:
    def cell(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    width = len(rows[0])
    lines = ["| " + " | ".join(cell(value) for value in rows[0]) + " |"]
    lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows[1:])
    return "\n".join(lines)


def _normalise_bbox(
    bbox: Sequence[float], *, width: float, height: float
) -> BoundingBox:
    if len(bbox) != 4:
        raise PDFExtractionError("pdfplumber returned an invalid bounding box")
    x0, top, x1, bottom = (float(value) for value in bbox)
    x0 = min(max(x0, 0.0), float(width))
    x1 = min(max(x1, x0), float(width))
    top = min(max(top, 0.0), float(height))
    bottom = min(max(bottom, top), float(height))
    return tuple(round(value, 3) for value in (x0, top, x1, bottom))  # type: ignore[return-value]


def _word_inside_bbox(word: Mapping[str, Any], bbox: BoundingBox) -> bool:
    center_x = (float(word["x0"]) + float(word["x1"])) / 2
    center_y = (float(word["top"]) + float(word["bottom"])) / 2
    return bbox[0] <= center_x <= bbox[2] and bbox[1] <= center_y <= bbox[3]


def _locate_text_bbox(
    text: str,
    words: Sequence[Mapping[str, Any]],
    used_words: set[int],
    *,
    page_width: float,
    page_height: float,
) -> BoundingBox:
    target = _comparison_text(text)
    if target:
        for start in range(len(words)):
            if start in used_words:
                continue
            candidate = ""
            indexes: list[int] = []
            for index in range(start, len(words)):
                if index in used_words:
                    break
                candidate += _comparison_text(str(words[index].get("text", "")))
                indexes.append(index)
                if candidate == target:
                    used_words.update(indexes)
                    return _bbox_for_words([words[item] for item in indexes])
                if len(candidate) >= len(target) or not target.startswith(candidate):
                    break
    # A page bbox is an honest containing locator when text tokenisation differs
    # between pypdf and pdfplumber; callers can see that it is page-level.
    return (0.0, 0.0, round(page_width, 3), round(page_height, 3))


def _bbox_for_words(words: Sequence[Mapping[str, Any]]) -> BoundingBox:
    return (
        round(min(float(word["x0"]) for word in words), 3),
        round(min(float(word["top"]) for word in words), 3),
        round(max(float(word["x1"]) for word in words), 3),
        round(max(float(word["bottom"]) for word in words), 3),
    )


def _finalize_blocks(
    document_id: str, drafts: Sequence[_BlockDraft]
) -> tuple[DocumentBlock, ...]:
    blocks: list[DocumentBlock] = []
    for index, draft in enumerate(drafts):
        canonical = (
            json.dumps(draft.table_rows, ensure_ascii=False, separators=(",", ":"))
            if draft.kind == "table"
            else draft.text
        )
        content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        block_id = hashlib.sha256(
            f"{document_id}\0{index}\0{draft.kind}\0{content_hash}".encode()
        ).hexdigest()[:24]
        blocks.append(
            DocumentBlock(
                block_id=block_id,
                kind=draft.kind,
                page=draft.page,
                bbox=draft.bbox,
                text=draft.text,
                content_hash=content_hash,
                table_rows=draft.table_rows,
                heading_level=draft.heading_level,
            )
        )
    original = tuple(blocks)
    for index, block in enumerate(original):
        blocks[index] = replace(
            block,
            previous_block_id=original[index - 1].block_id if index > 0 else None,
            next_block_id=original[index + 1].block_id if index + 1 < len(original) else None,
        )
    return tuple(blocks)


# A short, discoverable alias for callers that do not need to distinguish PDF
# extraction from subsequent knowledge-store ingestion.
extract_pdf = extract_pdf_document


__all__ = [
    "BlockProvenance",
    "BoundingBox",
    "DocumentBlock",
    "DocumentIngestionError",
    "PDFDependencyError",
    "PDFExtractionError",
    "PDFSafetyError",
    "StructureChunk",
    "StructuredDocument",
    "build_structure_chunks",
    "extract_pdf",
    "extract_pdf_document",
]

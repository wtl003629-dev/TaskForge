"""Adapter from the bounded pypdf/pdfplumber extractor to parser contracts."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path

from ..document_ingestion import PDFSafetyError, extract_pdf_document
from .contracts import DocumentBlock, ParsedDocument
from .quality_gate import ParseQualityPolicy, evaluate_parse_quality


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


class NativePDFParser:
    name = "native"

    def __init__(
        self,
        *,
        max_bytes: int = 25_000_000,
        max_pages: int = 300,
        max_blocks: int = 20_000,
        quality_policy: ParseQualityPolicy | None = None,
    ) -> None:
        self.max_bytes = max_bytes
        self.max_pages = max_pages
        self.max_blocks = max_blocks
        self.quality_policy = quality_policy
        self.parser_version = (
            f"pypdf-{_package_version('pypdf')}+"
            f"pdfplumber-{_package_version('pdfplumber')}"
        )

    async def parse(self, path: Path, *, source_uri: str) -> ParsedDocument:
        return await asyncio.to_thread(self._parse_sync, path, source_uri=source_uri)

    def _parse_sync(self, path: Path, *, source_uri: str) -> ParsedDocument:
        try:
            extracted = extract_pdf_document(
                path,
                source_uri=source_uri,
                max_bytes=self.max_bytes,
                max_pages=self.max_pages,
                max_blocks=self.max_blocks,
                # Blocks, not the legacy chunks, are the authoritative input to
                # the hierarchical projector. This value only bounds extraction.
                chunk_chars=6_000,
                preserve_page_boundaries=False,
            )
        except PDFSafetyError as exc:
            if "no machine-readable" not in str(exc):
                raise
            raw = path.read_bytes()
            pypdf = importlib.import_module("pypdf")
            reader = pypdf.PdfReader(BytesIO(raw), strict=False)
            page_count = len(reader.pages)
            digest = hashlib.sha256(raw).hexdigest()
            document_id = f"pdf:{digest[:24]}"
            quality = evaluate_parse_quality(
                (),
                page_count=page_count,
                parser=self.name,
                policy=self.quality_policy,
            )
            return ParsedDocument(
                document_id=document_id,
                source_uri=source_uri,
                sha256=digest,
                bytes_read=len(raw),
                page_count=page_count,
                parser=self.name,
                parser_version=self.parser_version,
                parser_backend="pypdf+pdfplumber",
                blocks=(),
                quality=quality,
            )
        page_orders: dict[int, int] = {}
        blocks: list[DocumentBlock] = []
        for block in extracted.blocks:
            order = page_orders.get(block.page, 0)
            page_orders[block.page] = order + 1
            block_type = (
                "title"
                if block.is_heading
                else "table"
                if block.kind == "table"
                else "paragraph"
            )
            structured: dict[str, object] = {
                "coordinate_space": "pdf_points",
                "source_block_kind": block.kind,
            }
            if block.table_rows:
                structured["table_rows"] = [list(row) for row in block.table_rows]
                structured["textual_rendering"] = block.text
            blocks.append(
                DocumentBlock(
                    block_id=block.block_id,
                    document_id=extracted.document_id,
                    parser=self.name,
                    parser_version=self.parser_version,
                    page=block.page,
                    bbox=block.bbox,
                    reading_order=order,
                    block_type=block_type,
                    text=block.text,
                    structured_content=structured,
                    content_hash=block.content_hash,
                    heading_level=block.heading_level,
                    previous_block_id=block.previous_block_id,
                    next_block_id=block.next_block_id,
                )
            )
        pypdf = importlib.import_module("pypdf")
        raw = path.read_bytes()
        reader = pypdf.PdfReader(BytesIO(raw), strict=False)
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                resources = page.get("/Resources") or {}
                resources = (
                    resources.get_object()
                    if hasattr(resources, "get_object")
                    else resources
                )
                xobjects = resources.get("/XObject") or {}
                xobjects = (
                    xobjects.get_object()
                    if hasattr(xobjects, "get_object")
                    else xobjects
                )
                image_names = [
                    str(name)
                    for name, reference in xobjects.items()
                    if str(reference.get_object().get("/Subtype")) == "/Image"
                ]
            except Exception:
                image_names = []
            for image_name in image_names:
                order = page_orders.get(page_number, 0)
                page_orders[page_number] = order + 1
                canonical = (
                    f"{extracted.document_id}\0{page_number}\0{image_name}\0image"
                )
                content_hash = hashlib.sha256(canonical.encode()).hexdigest()
                block_id = hashlib.sha256(
                    f"{canonical}\0{content_hash}".encode()
                ).hexdigest()[:24]
                media_box = page.mediabox
                blocks.append(
                    DocumentBlock(
                        block_id=block_id,
                        document_id=extracted.document_id,
                        parser=self.name,
                        parser_version=self.parser_version,
                        page=page_number,
                        bbox=(
                            0.0,
                            0.0,
                            float(media_box.width),
                            float(media_box.height),
                        ),
                        reading_order=order,
                        block_type="image",
                        structured_content={
                            "coordinate_space": "pdf_points",
                            "xobject_name": image_name,
                            "visual_detection": "pypdf_image_xobject",
                        },
                        content_hash=content_hash,
                    )
                )
        blocks.sort(key=lambda item: (item.page, item.reading_order, item.block_id))
        for index, block in enumerate(tuple(blocks)):
            blocks[index] = block.model_copy(
                update={
                    "previous_block_id": (
                        blocks[index - 1].block_id if index else None
                    ),
                    "next_block_id": (
                        blocks[index + 1].block_id
                        if index + 1 < len(blocks)
                        else None
                    ),
                }
            )
        quality = evaluate_parse_quality(
            blocks,
            page_count=extracted.page_count,
            parser=self.name,
            policy=self.quality_policy,
        )
        return ParsedDocument(
            document_id=extracted.document_id,
            source_uri=extracted.source_uri,
            sha256=extracted.sha256,
            bytes_read=extracted.bytes_read,
            page_count=extracted.page_count,
            parser=self.name,
            parser_version=self.parser_version,
            parser_backend="pypdf+pdfplumber",
            blocks=tuple(blocks),
            quality=quality,
        )


__all__ = ["NativePDFParser"]

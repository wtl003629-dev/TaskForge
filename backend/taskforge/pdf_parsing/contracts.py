"""Stable parser-neutral contracts for PDF ingestion.

Parser-specific output must be normalized into these models before chunking or
retrieval.  This keeps MinerU schema changes and native extractor details out
of the evidence layer.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from ..domain import StrictModel

DocumentBlockType = Literal[
    "title",
    "paragraph",
    "list",
    "table",
    "chart",
    "equation",
    "image",
    "caption",
    "footnote",
    "reference",
    "code",
    "algorithm",
    "header",
    "footer",
    "page_number",
    "aside",
]
ParseQualityStatus = Literal[
    "ready",
    "degraded",
    "visual_pending",
    "ocr_required",
    "layout_failed",
    "table_failed",
    "no_machine_text",
    "failed",
]
VisualType = Literal["chart", "diagram", "figure", "table"]


class DocumentBlock(StrictModel):
    block_id: str = Field(min_length=1, max_length=256)
    document_id: str = Field(min_length=1, max_length=512)
    parser: str = Field(min_length=1, max_length=128)
    parser_version: str = Field(min_length=1, max_length=128)
    page: int = Field(ge=1)
    bbox: tuple[float, float, float, float]
    reading_order: int = Field(ge=0)
    block_type: DocumentBlockType
    text: str = ""
    structured_content: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    image_artifact_id: str | None = Field(default=None, max_length=1_024)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    heading_level: int | None = Field(default=None, ge=1, le=12)
    previous_block_id: str | None = Field(default=None, max_length=256)
    next_block_id: str | None = Field(default=None, max_length=256)
    indexable: bool = True

    @field_validator("bbox")
    @classmethod
    def bbox_is_finite_and_ordered(
        cls,
        value: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = (float(item) for item in value)
        if not all(item == item and abs(item) != float("inf") for item in (x0, y0, x1, y1)):
            raise ValueError("block bbox values must be finite")
        if x1 < x0 or y1 < y0:
            raise ValueError("block bbox must be ordered")
        return x0, y0, x1, y1

    @model_validator(mode="after")
    def usable_content_is_present(self) -> DocumentBlock:
        if self.indexable and not self.text.strip() and not self.structured_content:
            raise ValueError("indexable blocks require text or structured content")
        return self


class VisualEvidence(StrictModel):
    """Text-model-safe rendering of one original visual artifact."""

    visual_id: str = Field(min_length=1, max_length=256)
    page: int = Field(ge=1)
    bbox: tuple[float, float, float, float]
    visual_type: VisualType
    caption: str = Field(default="", max_length=4_000)
    axes: dict[str, Any] | None = None
    legends: list[str] = Field(default_factory=list, max_length=128)
    data_points: list[dict[str, Any]] = Field(default_factory=list, max_length=5_000)
    nodes: list[dict[str, Any]] = Field(default_factory=list, max_length=2_000)
    edges: list[dict[str, Any]] = Field(default_factory=list, max_length=5_000)
    textual_rendering: str = Field(min_length=1, max_length=32_000)
    confidence: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list, max_length=32)
    image_artifact_id: str = Field(min_length=1, max_length=1_024)
    extractor: str = Field(min_length=1, max_length=128)
    extractor_version: str = Field(min_length=1, max_length=256)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("bbox")
    @classmethod
    def visual_bbox_is_finite_and_ordered(
        cls,
        value: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        return DocumentBlock.bbox_is_finite_and_ordered(value)


class ParseQualityReport(StrictModel):
    page_count: int = Field(ge=1)
    parsed_page_count: int = Field(ge=0)
    text_coverage: float = Field(ge=0.0, le=1.0)
    garbled_character_ratio: float = Field(ge=0.0, le=1.0)
    repeated_header_ratio: float = Field(ge=0.0, le=1.0)
    orphan_caption_count: int = Field(ge=0)
    empty_table_count: int = Field(ge=0)
    reading_order_warning_count: int = Field(ge=0)
    visual_unparsed_count: int = Field(ge=0)
    ocr_used: bool = False
    status: ParseQualityStatus
    recommended_parser: Literal["native", "mineru", "ocr", "none"]
    reasons: list[str] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def parsed_pages_fit_document(self) -> ParseQualityReport:
        if self.parsed_page_count > self.page_count:
            raise ValueError("parsed pages cannot exceed PDF page count")
        return self


class ParserAttempt(StrictModel):
    parser: str = Field(min_length=1, max_length=128)
    parser_version: str = Field(min_length=1, max_length=128)
    outcome: Literal["accepted", "rejected", "failed"]
    elapsed_ms: float = Field(ge=0.0)
    quality_status: ParseQualityStatus | None = None
    error: str | None = Field(default=None, max_length=2_000)


class ParsedDocument(StrictModel):
    document_id: str = Field(min_length=1, max_length=512)
    source_uri: str = Field(min_length=1, max_length=2_048)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes_read: int = Field(ge=1)
    page_count: int = Field(ge=1)
    parser: str = Field(min_length=1, max_length=128)
    parser_version: str = Field(min_length=1, max_length=128)
    parser_backend: str = Field(min_length=1, max_length=128)
    blocks: tuple[DocumentBlock, ...]
    quality: ParseQualityReport
    attempts: tuple[ParserAttempt, ...] = ()
    raw_output_artifact: str | None = Field(default=None, max_length=2_048)

    @model_validator(mode="after")
    def document_is_consistent(self) -> ParsedDocument:
        block_ids = [block.block_id for block in self.blocks]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("parsed document block IDs must be unique")
        if any(block.document_id != self.document_id for block in self.blocks):
            raise ValueError("parsed block document IDs must match their document")
        if any(block.page > self.page_count for block in self.blocks):
            raise ValueError("parsed block page exceeds PDF page count")
        if self.quality.page_count != self.page_count:
            raise ValueError("parse quality page count must match the document")
        return self


__all__ = [
    "DocumentBlock",
    "DocumentBlockType",
    "ParsedDocument",
    "ParseQualityReport",
    "ParseQualityStatus",
    "ParserAttempt",
    "VisualEvidence",
    "VisualType",
]

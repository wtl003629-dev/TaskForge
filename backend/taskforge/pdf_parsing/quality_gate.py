"""Deterministic parse diagnostics kept separate from retrieval scoring."""

from __future__ import annotations

import math
import unicodedata
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .contracts import DocumentBlock, ParseQualityReport


@dataclass(frozen=True, slots=True)
class ParseQualityPolicy:
    """Development defaults; locked-run values must be separately frozen."""

    minimum_text_coverage: float = 0.80
    maximum_garbled_character_ratio: float = 0.03
    maximum_repeated_header_ratio: float = 0.20

    def __post_init__(self) -> None:
        for value in (
            self.minimum_text_coverage,
            self.maximum_garbled_character_ratio,
            self.maximum_repeated_header_ratio,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("parse quality thresholds must be in [0, 1]")


def _garbled_ratio(text: str) -> float:
    if not text:
        return 0.0
    suspicious = 0
    visible = 0
    for character in text:
        if character.isspace():
            continue
        visible += 1
        category = unicodedata.category(character)
        if (
            character == "\ufffd"
            or category in {"Co", "Cs", "Cn"}
            or (category == "Cc" and character not in "\t\n\r")
        ):
            suspicious += 1
    return suspicious / visible if visible else 0.0


def _repeated_header_ratio(
    blocks: Sequence[DocumentBlock],
    page_count: int,
) -> float:
    candidates: dict[str, set[int]] = defaultdict(set)
    for block in blocks:
        # Explicit header/footer/page-number blocks have already been removed
        # from the retrieval index. This metric detects repeated edge text
        # that leaked into indexable evidence, not successful classification.
        if not block.indexable:
            continue
        compact = " ".join(block.text.casefold().split())
        if not compact or len(compact) > 160:
            continue
        coordinate_space = str(block.structured_content.get("coordinate_space") or "")
        y0, y1 = block.bbox[1], block.bbox[3]
        explicit = block.block_type in {"header", "footer", "page_number"}
        near_edge = (
            coordinate_space in {"normalized", "mineru_1000"}
            and (y0 <= (0.08 if coordinate_space == "normalized" else 80) or y1 >= (0.92 if coordinate_space == "normalized" else 920))
        )
        if explicit or near_edge:
            candidates[compact].add(block.page)
    repeated = sum(
        len(pages)
        for pages in candidates.values()
        if len(pages) >= max(2, math.ceil(page_count * 0.4))
    )
    return min(1.0, repeated / page_count) if page_count else 0.0


def _table_has_material(block: DocumentBlock) -> bool:
    if block.text.strip():
        return True
    content_keys = (
        "textual_rendering",
        "table_rows",
        "rows",
        "cells",
        "table_body",
        "table_content",
        "html",
    )
    if any(
        block.structured_content.get(key) not in (None, "", (), [], {})
        for key in content_keys
    ):
        return True
    nested = block.structured_content.get("content")
    return isinstance(nested, dict) and any(
        nested.get(key) not in (None, "", (), [], {}) for key in content_keys
    )


def evaluate_parse_quality(
    blocks: Sequence[DocumentBlock],
    *,
    page_count: int,
    ocr_used: bool = False,
    parser: str,
    policy: ParseQualityPolicy | None = None,
) -> ParseQualityReport:
    if page_count < 1:
        raise ValueError("page_count must be positive")
    selected_policy = policy or ParseQualityPolicy()
    indexable = [block for block in blocks if block.indexable]
    text = "\n".join(block.text for block in indexable if block.text.strip())
    pages = {block.page for block in indexable if block.text.strip() or block.structured_content}
    text_coverage = len(pages) / page_count
    garbled = _garbled_ratio(text)
    repeated = _repeated_header_ratio(blocks, page_count)
    by_page: dict[int, list[DocumentBlock]] = defaultdict(list)
    for block in indexable:
        by_page[block.page].append(block)
    orphan_captions = sum(
        1
        for block in indexable
        if block.block_type == "caption"
        and not any(
            peer.block_type in {"image", "chart", "table"}
            for peer in by_page[block.page]
        )
    )
    empty_tables = sum(
        block.block_type == "table"
        and not _table_has_material(block)
        for block in indexable
    )
    visual_unparsed = sum(
        block.block_type in {"image", "chart"}
        and (
            block.structured_content.get("visual_analysis_status") == "pending"
            or (
                not block.text.strip()
                and not block.structured_content.get("textual_rendering")
            )
        )
        for block in indexable
    )
    order_warnings = 0
    for page_blocks in by_page.values():
        orders = [block.reading_order for block in page_blocks]
        order_warnings += len(orders) - len(set(orders))
        order_warnings += sum(
            current < previous
            for previous, current in zip(orders, orders[1:])
        )
    reasons: list[str] = []
    if not indexable or not text.strip():
        status = "ocr_required" if parser == "native" else "no_machine_text"
        recommended = "ocr" if parser == "native" else "none"
        reasons.append("no machine-readable indexable text was produced")
    elif empty_tables:
        status = "table_failed"
        recommended = "mineru" if parser == "native" else "none"
        reasons.append("one or more table blocks contain no structured content")
    elif text_coverage < selected_policy.minimum_text_coverage:
        status = "degraded"
        recommended = "mineru" if parser == "native" else "none"
        reasons.append("parsed page coverage is below the configured development threshold")
    elif garbled > selected_policy.maximum_garbled_character_ratio:
        status = "degraded"
        recommended = "ocr" if parser == "native" else "none"
        reasons.append("garbled character ratio is above the configured development threshold")
    elif visual_unparsed:
        status = "visual_pending"
        recommended = "mineru" if parser == "native" else "none"
        reasons.append("one or more visual blocks lack a textual rendering")
    elif order_warnings:
        status = "layout_failed"
        recommended = "mineru" if parser == "native" else "none"
        reasons.append("reading order contains duplicate or non-monotonic positions")
    elif repeated > selected_policy.maximum_repeated_header_ratio:
        status = "degraded"
        recommended = "mineru" if parser == "native" else "none"
        reasons.append("repeated header/footer ratio exceeds the development threshold")
    else:
        status = "ready"
        recommended = "native" if parser == "native" else "mineru"
    return ParseQualityReport(
        page_count=page_count,
        parsed_page_count=len(pages),
        text_coverage=text_coverage,
        garbled_character_ratio=garbled,
        repeated_header_ratio=repeated,
        orphan_caption_count=orphan_captions,
        empty_table_count=int(empty_tables),
        reading_order_warning_count=order_warnings,
        visual_unparsed_count=visual_unparsed,
        ocr_used=ocr_used,
        status=status,
        recommended_parser=recommended,
        reasons=reasons,
    )


__all__ = ["ParseQualityPolicy", "evaluate_parse_quality"]

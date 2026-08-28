"""Normalize MinerU v2/v3 content lists into stable TaskForge blocks."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import DocumentBlock, DocumentBlockType, ParsedDocument
from .quality_gate import ParseQualityPolicy, evaluate_parse_quality


class MinerUNormalizationError(ValueError):
    pass


_TYPE_MAP: dict[str, DocumentBlockType] = {
    "text": "paragraph",
    "paragraph": "paragraph",
    "title": "title",
    "list": "list",
    "index": "list",
    "table": "table",
    "chart": "chart",
    "equation": "equation",
    "equation_interline": "equation",
    "interline_equation": "equation",
    "image": "image",
    "image_caption": "caption",
    "table_caption": "caption",
    "code_caption": "caption",
    "image_footnote": "footnote",
    "table_footnote": "footnote",
    "page_footnote": "footnote",
    "ref_text": "reference",
    "code": "code",
    "algorithm": "algorithm",
    "header": "header",
    "page_header": "header",
    "footer": "footer",
    "page_footer": "footer",
    "page_number": "page_number",
    "aside_text": "aside",
    "page_aside_text": "aside",
}
_NON_INDEXABLE = frozenset({"header", "footer", "page_number", "aside"})


def _json_value(value: object) -> object:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "{")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _flatten_text(value: object) -> str:
    value = _json_value(value)
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in (
            "textual_rendering",
            "paragraph_content",
            "title_content",
            "math_content",
            "table_body",
            "table_content",
            "code_body",
            "algorithm_content",
            "content",
            "text",
        ):
            if key in value:
                rendered = _flatten_text(value[key])
                if rendered:
                    return rendered
        pieces = [_flatten_text(item) for item in value.values()]
        return "\n".join(item for item in pieces if item)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        pieces = [_flatten_text(item) for item in value]
        return "\n".join(item for item in pieces if item)
    return str(value).strip()


def _payload(raw: Mapping[str, Any], filename: str) -> Mapping[str, Any]:
    results = raw.get("results")
    if isinstance(results, Mapping):
        exact = results.get(filename)
        if isinstance(exact, Mapping):
            return exact
        stem = filename.rsplit(".", 1)[0]
        for key, value in results.items():
            if isinstance(value, Mapping) and str(key).rsplit(".", 1)[0] == stem:
                return value
        values = [value for value in results.values() if isinstance(value, Mapping)]
        if len(values) == 1:
            return values[0]
    data = raw.get("data")
    if isinstance(data, Mapping):
        return data
    return raw


def _content_list(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in (
        "content_list_v2",
        "content_list_v2.json",
        "content_list",
        "content_list.json",
    ):
        value = _json_value(payload.get(key))
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    raise MinerUNormalizationError("MinerU response contains no content list")


def _page_items(
    values: Sequence[Mapping[str, Any]],
) -> list[tuple[int, Mapping[str, Any]]]:
    flattened: list[tuple[int, Mapping[str, Any]]] = []
    for page_position, value in enumerate(values):
        nested: object | None = None
        for key in ("items", "blocks", "content_list", "page_content"):
            if isinstance(value.get(key), list):
                nested = value[key]
                break
        if value.get("page_idx") is not None:
            page = int(value["page_idx"]) + 1
        elif value.get("page_index") is not None:
            page = int(value["page_index"]) + 1
        elif value.get("page_number") is not None:
            page = int(value["page_number"])
        else:
            page = page_position + 1
        page = max(1, page)
        if isinstance(nested, list):
            flattened.extend(
                (page, item) for item in nested if isinstance(item, Mapping)
            )
        else:
            flattened.append((page, value))
    return flattened


def _semantic_items(
    values: Sequence[tuple[int, Mapping[str, Any]]],
) -> list[tuple[int, Mapping[str, Any], str | None, int | None, int | None]]:
    """Expand MinerU list groups at their existing semantic item boundaries."""

    expanded: list[
        tuple[int, Mapping[str, Any], str | None, int | None, int | None]
    ] = []
    for page, item in values:
        raw_type = str(item.get("type") or item.get("block_type") or "text").casefold()
        list_items = item.get("list_items")
        if (
            raw_type != "list"
            or not isinstance(list_items, Sequence)
            or isinstance(list_items, (str, bytes, bytearray))
            or len(list_items) <= 1
        ):
            expanded.append((page, item, None, None, None))
            continue
        origin = hashlib.sha256(
            json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        count = len(list_items)
        for index, list_item in enumerate(list_items):
            synthetic = dict(item)
            synthetic["list_items"] = [list_item]
            # A nested content object, when present, takes precedence during
            # text rendering and must reflect the same semantic item.
            nested = item.get("content")
            if isinstance(nested, Mapping):
                synthetic["content"] = dict(nested)
                synthetic["content"]["list_items"] = [list_item]
            expanded.append((page, synthetic, origin, index, count))
    return expanded


def _bbox(value: object) -> tuple[float, float, float, float]:
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) == 4
    ):
        try:
            x0, y0, x1, y1 = (float(item) for item in value)
        except (TypeError, ValueError):
            pass
        else:
            if x1 >= x0 and y1 >= y0:
                return x0, y0, x1, y1
    return 0.0, 0.0, 0.0, 0.0


def _heading_level(value: object) -> int | None:
    """Return MinerU's positive heading level without accepting booleans."""

    if isinstance(value, bool):
        return None
    try:
        level = int(value) if value is not None else 0
    except (TypeError, ValueError):
        return None
    return level if level >= 1 else None


_NUMBERED_HEADING = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:\.|\s)\s*\S")


def _semantic_heading_level(text: str, mineru_level: int | None) -> int | None:
    """Recover numbered paper-section depth when MinerU flattens headings."""

    if mineru_level is None:
        return None
    match = _NUMBERED_HEADING.match(text)
    if match is None:
        return mineru_level
    # Level 1 is conventionally the paper title. A numbered top-level section
    # starts at level 2, `2.1` at level 3, and so on.
    numbered_depth = match.group(1).count(".") + 2
    return min(12, max(mineru_level, numbered_depth))


def _demote_dense_visual_labels(
    blocks: list[DocumentBlock],
    *,
    document_id: str,
) -> list[DocumentBlock]:
    """Prevent chart/diagram labels from becoming false section boundaries.

    MinerU can classify every blue label in a full-page diagram as a level-2
    title. We retain those strings as indexable paragraph evidence, with their
    original type and coordinates, but do not let them split Parent units.
    """

    by_page: dict[int, list[int]] = defaultdict(list)
    for index, block in enumerate(blocks):
        by_page[block.page].append(index)
    result = list(blocks)
    for indexes in by_page.values():
        page_blocks = [blocks[index] for index in indexes]
        titles = [block for block in page_blocks if block.block_type == "title"]
        indexable_count = sum(block.indexable for block in page_blocks)
        has_visual = any(
            block.block_type in {"image", "chart"} for block in page_blocks
        )
        if (
            not has_visual
            or len(titles) < 8
            or len(titles) / max(1, indexable_count) < 0.35
        ):
            continue
        title_ids = {block.block_id for block in titles}
        for index in indexes:
            block = blocks[index]
            if block.block_id not in title_ids:
                continue
            structured = dict(block.structured_content)
            structured["heading_demoted_reason"] = "dense_visual_labels"
            canonical = json.dumps(
                {
                    "type": "paragraph",
                    "text": block.text,
                    "content": structured.get("content"),
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            block_id = hashlib.sha256(
                (
                    f"{document_id}\0{block.page}\0{block.reading_order}\0"
                    f"paragraph\0{content_hash}"
                ).encode()
            ).hexdigest()[:24]
            result[index] = block.model_copy(
                update={
                    "block_id": block_id,
                    "block_type": "paragraph",
                    "heading_level": None,
                    "structured_content": structured,
                    "content_hash": content_hash,
                }
            )
    return result


def _semantic_text(
    block_type: DocumentBlockType,
    content: object,
    item: Mapping[str, Any],
) -> str:
    if block_type in {"image", "chart"}:
        fields = (
            "image_caption",
            "chart_caption",
            "content",
            "image_footnote",
            "chart_footnote",
        )
    elif block_type == "table":
        fields = (
            "table_caption",
            "table_body",
            "table_content",
            "content",
            "table_footnote",
        )
    elif block_type in {"code", "algorithm"}:
        fields = ("code_caption", "code_body", "content", "code_footnote")
    elif block_type == "list":
        fields = ("list_items", "content", "text")
    else:
        return _flatten_text(content)
    source = content if isinstance(content, Mapping) else item
    pieces = [_flatten_text(source.get(field)) for field in fields]
    value = "\n\n".join(dict.fromkeys(piece for piece in pieces if piece))
    return value or _flatten_text(content)


def normalize_mineru_response(
    raw: Mapping[str, Any],
    *,
    filename: str,
    source_uri: str,
    sha256: str,
    bytes_read: int,
    page_count: int,
    parser_version: str,
    parser_backend: str,
    effective_parse_method: str | None = None,
    raw_output_artifact: str | None = None,
    image_artifacts: Mapping[str, str] | None = None,
    quality_policy: ParseQualityPolicy | None = None,
) -> ParsedDocument:
    payload = _payload(raw, filename)
    values = _content_list(payload)
    document_id = f"pdf:{sha256[:24]}"
    page_orders: dict[int, int] = {}
    blocks: list[DocumentBlock] = []
    for page, item, list_origin, list_item_index, list_item_count in _semantic_items(
        _page_items(values)
    ):
        if page > page_count:
            raise MinerUNormalizationError(
                f"MinerU block page {page} exceeds PDF page count {page_count}"
            )
        raw_type = str(item.get("type") or item.get("block_type") or "text").casefold()
        sub_type = str(item.get("sub_type") or "").casefold()
        block_type = _TYPE_MAP.get(raw_type, _TYPE_MAP.get(sub_type, "paragraph"))
        content = item.get("content")
        if content is None:
            content = {
                key: item[key]
                for key in (
                    "paragraph_content",
                    "title_content",
                    "text",
                    "table_body",
                    "table_content",
                    "math_content",
                    "code_body",
                    "algorithm_content",
                    "list_items",
                    "image_caption",
                    "image_footnote",
                    "chart_caption",
                    "chart_footnote",
                    "table_caption",
                    "table_footnote",
                    "code_caption",
                    "code_footnote",
                )
                if item.get(key) not in (None, "", (), [], {})
            }
        level = item.get("level")
        if isinstance(content, Mapping) and level is None:
            level = content.get("level")
        heading_level = _heading_level(level or item.get("text_level"))
        # MinerU 3.4.x's flat content_list represents headings as `text`
        # records with `text_level`; only its nested/v2 form necessarily uses
        # `type=title`. Normalize both forms to the same parser-neutral type so
        # section boundaries survive into Parent–Child construction.
        if block_type == "paragraph" and heading_level is not None:
            block_type = "title"
        text = _semantic_text(block_type, content, item)
        if block_type == "title":
            heading_level = _semantic_heading_level(text, heading_level)
        if block_type == "caption" and not text:
            text = _flatten_text(
                item.get("image_caption")
                or item.get("table_caption")
                or item.get("chart_caption")
                or item.get("code_caption")
            )
        order = page_orders.get(page, 0)
        page_orders[page] = order + 1
        canonical = json.dumps(
            {"type": block_type, "text": text, "content": content},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        block_id = hashlib.sha256(
            f"{document_id}\0{page}\0{order}\0{block_type}\0{content_hash}".encode()
        ).hexdigest()[:24]
        confidence = item.get("score", item.get("confidence"))
        try:
            parsed_confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            parsed_confidence = None
        parsed_bbox = _bbox(item.get("bbox"))
        coordinate_space = (
            "normalized" if max(parsed_bbox, default=0.0) <= 1.0 else "mineru_1000"
        )
        structured = {
            "coordinate_space": coordinate_space,
            "mineru_type": raw_type,
            "mineru_sub_type": sub_type or None,
            "content": content,
        }
        if list_origin is not None:
            structured.update(
                {
                    "mineru_list_group_hash": list_origin,
                    "mineru_list_item_index": list_item_index,
                    "mineru_list_item_count": list_item_count,
                }
            )
        if text and block_type in {"table", "chart", "equation", "image"}:
            structured["textual_rendering"] = text
        if block_type in {"chart", "image"}:
            caption = _flatten_text(
                item.get("image_caption") or item.get("chart_caption")
            )
            content_text = _flatten_text(content)
            structured_signal = any(
                item.get(key) not in (None, "", (), [], {})
                for key in (
                    "axes",
                    "legends",
                    "data_points",
                    "nodes",
                    "edges",
                    "chart_data",
                    "table_body",
                )
            ) or bool(content_text and content_text != caption)
            structured["visual_analysis_status"] = (
                "ready" if structured_signal else "pending"
            )
        image_path = item.get("img_path", item.get("image_path"))
        image_path_text = str(image_path) if image_path else None
        blocks.append(
            DocumentBlock(
                block_id=block_id,
                document_id=document_id,
                parser="mineru",
                parser_version=parser_version,
                page=page,
                bbox=parsed_bbox,
                reading_order=order,
                block_type=block_type,
                text=text,
                structured_content=structured,
                confidence=(
                    min(1.0, max(0.0, parsed_confidence))
                    if parsed_confidence is not None
                    else None
                ),
                image_artifact_id=(
                    (image_artifacts or {}).get(image_path_text)
                    or f"mineru:{sha256[:16]}:{image_path_text}"
                    if image_path_text
                    else None
                ),
                content_hash=content_hash,
                heading_level=heading_level,
                indexable=block_type not in _NON_INDEXABLE,
            )
        )
    blocks = _demote_dense_visual_labels(blocks, document_id=document_id)
    for index, block in enumerate(tuple(blocks)):
        blocks[index] = block.model_copy(
            update={
                "previous_block_id": blocks[index - 1].block_id if index else None,
                "next_block_id": (
                    blocks[index + 1].block_id if index + 1 < len(blocks) else None
                ),
            }
        )
    ocr_used = (
        str(
            effective_parse_method
            or payload.get("parse_method", raw.get("parse_method", ""))
        ).casefold()
        == "ocr"
        or bool(payload.get("ocr_used"))
    )
    quality = evaluate_parse_quality(
        blocks,
        page_count=page_count,
        ocr_used=ocr_used,
        parser="mineru",
        policy=quality_policy,
    )
    return ParsedDocument(
        document_id=document_id,
        source_uri=source_uri,
        sha256=sha256,
        bytes_read=bytes_read,
        page_count=page_count,
        parser="mineru",
        parser_version=parser_version,
        parser_backend=parser_backend,
        blocks=tuple(blocks),
        quality=quality,
        raw_output_artifact=raw_output_artifact,
    )


__all__ = ["MinerUNormalizationError", "normalize_mineru_response"]

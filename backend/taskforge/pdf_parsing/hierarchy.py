"""Structure-aware Block to Parent/Child projection for paper retrieval."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace

from ..knowledge import tokenise
from .contracts import DocumentBlock, ParsedDocument

_ATOMIC_TYPES = frozenset({"table", "chart", "equation", "image", "code", "algorithm"})
_TEXT_TYPES = frozenset(
    {
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
    }
)


@dataclass(frozen=True, slots=True)
class HierarchicalUnit:
    unit_id: str
    role: str
    parent_id: str
    text: str
    heading_path: tuple[str, ...]
    block_ids: tuple[str, ...]
    pages: tuple[int, ...]
    block_types: tuple[str, ...]
    order: int
    previous_unit_id: str | None = None
    next_unit_id: str | None = None
    oversized_atomic: bool = False


def _block_text(block: DocumentBlock) -> str:
    if block.text.strip():
        return block.text.strip()
    rendered = block.structured_content.get("textual_rendering")
    return str(rendered).strip() if rendered is not None else ""


def _tokens(blocks: list[DocumentBlock]) -> int:
    return sum(len(tokenise(_block_text(block))) for block in blocks)


def _bounded_text_parts(text: str, max_tokens: int) -> list[str]:
    """Split oversized prose on semantic boundaries, then bounded words."""

    cleaned = text.strip()
    if not cleaned or len(tokenise(cleaned)) <= max_tokens:
        return [cleaned] if cleaned else []
    semantic_units = [
        value.strip()
        for value in re.split(r"(?<=[.!?。！？])\s+|\n+", cleaned)
        if value.strip()
    ]
    atomic_units: list[str] = []
    for unit in semantic_units:
        if len(tokenise(unit)) <= max_tokens:
            atomic_units.append(unit)
            continue
        words = unit.split()
        if len(words) <= 1:
            start = 0
            while start < len(unit):
                low, high = start + 1, len(unit)
                best = start + 1
                while low <= high:
                    middle = (low + high) // 2
                    if len(tokenise(unit[start:middle])) <= max_tokens:
                        best = middle
                        low = middle + 1
                    else:
                        high = middle - 1
                atomic_units.append(unit[start:best])
                start = best
            continue
        current_words: list[str] = []
        for word in words:
            candidate = " ".join((*current_words, word))
            if current_words and len(tokenise(candidate)) > max_tokens:
                atomic_units.append(" ".join(current_words))
                current_words = [word]
            else:
                current_words.append(word)
        if current_words:
            atomic_units.append(" ".join(current_words))
    parts: list[str] = []
    current: list[str] = []
    for unit in atomic_units:
        candidate = " ".join((*current, unit))
        if current and len(tokenise(candidate)) > max_tokens:
            parts.append(" ".join(current))
            current = [unit]
        else:
            current.append(unit)
    if current:
        parts.append(" ".join(current))
    return parts


def _expand_oversized_blocks(
    blocks: list[DocumentBlock],
    *,
    max_tokens: int,
) -> list[DocumentBlock]:
    expanded: list[DocumentBlock] = []
    for block in blocks:
        text = _block_text(block)
        if block.block_type in _ATOMIC_TYPES or len(tokenise(text)) <= max_tokens:
            expanded.append(block)
            continue
        if block.block_type == "list":
            list_items = [line.strip() for line in text.splitlines() if line.strip()]
            parts: list[str] = []
            current_items: list[str] = []
            for item in list_items:
                candidate = "\n".join((*current_items, item))
                if current_items and len(tokenise(candidate)) > max_tokens:
                    parts.append("\n".join(current_items))
                    current_items = [item]
                else:
                    current_items.append(item)
            if current_items:
                parts.append("\n".join(current_items))
            if not parts or any(len(tokenise(part)) > max_tokens for part in parts):
                parts = _bounded_text_parts(text, max_tokens)
        else:
            parts = _bounded_text_parts(text, max_tokens)
        for index, part in enumerate(parts):
            structured = dict(block.structured_content)
            structured.update(
                {
                    "chunk_fragment_index": index,
                    "chunk_fragment_count": len(parts),
                    "source_block_id": block.block_id,
                }
            )
            expanded.append(
                block.model_copy(
                    update={
                        "text": part,
                        "structured_content": structured,
                        "content_hash": hashlib.sha256(part.encode()).hexdigest(),
                    }
                )
            )
    return expanded


def _unit_id(
    document_id: str,
    role: str,
    index: int,
    blocks: list[DocumentBlock],
) -> str:
    digest = hashlib.sha256(
        "\0".join(
            (
                document_id,
                role,
                str(index),
                *(block.block_id for block in blocks),
            )
        ).encode()
    ).hexdigest()[:24]
    return f"{role}:{digest}"


def _heading_path(
    blocks: list[DocumentBlock],
    inherited: tuple[str, ...],
) -> tuple[str, ...]:
    path = list(inherited)
    for block in blocks:
        if block.block_type != "title" or not block.text.strip():
            continue
        level = block.heading_level or 1
        path = path[: max(0, level - 1)]
        path.append(block.text.strip())
    return tuple(path)


def _parents(
    document: ParsedDocument,
    *,
    parent_target_tokens: int,
    parent_max_tokens: int,
    hybrid_policy: bool = False,
) -> list[tuple[tuple[str, ...], list[DocumentBlock]]]:
    source = [
        block
        for block in document.blocks
        if block.indexable
        and block.block_type in _TEXT_TYPES
        and (_block_text(block) or block.block_type in _ATOMIC_TYPES)
    ]
    if hybrid_policy:
        source = _expand_oversized_blocks(source, max_tokens=parent_max_tokens)
    groups: list[tuple[tuple[str, ...], list[DocumentBlock]]] = []
    current: list[DocumentBlock] = []
    active_path: tuple[str, ...] = ()
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            groups.append((active_path, current))
        current = []
        current_tokens = 0

    for block in source:
        block_tokens = max(1, len(tokenise(_block_text(block))))
        starts_section = block.block_type == "title"
        if current and starts_section:
            flush()
        if current and (
            current_tokens >= parent_target_tokens
            or current_tokens + block_tokens > parent_max_tokens
        ):
            flush()
        if starts_section:
            active_path = _heading_path([block], active_path)
        current.append(block)
        current_tokens += block_tokens
    flush()
    return groups


def _child_segments(
    blocks: list[DocumentBlock],
    *,
    child_target_tokens: int,
    child_max_tokens: int,
    child_overlap_tokens: int,
    hybrid_policy: bool = False,
) -> list[list[DocumentBlock]]:
    if hybrid_policy:
        blocks = _expand_oversized_blocks(blocks, max_tokens=child_max_tokens)
    atomic_bound: list[list[DocumentBlock]] = []
    index = 0
    while index < len(blocks):
        block = blocks[index]
        if (
            block.block_type == "caption"
            and index + 1 < len(blocks)
            and blocks[index + 1].block_type in _ATOMIC_TYPES
        ):
            segment = [block, blocks[index + 1]]
            index += 2
            if hybrid_policy:
                while index < len(blocks) and blocks[index].block_type in {
                    "caption",
                    "footnote",
                }:
                    segment.append(blocks[index])
                    index += 1
            atomic_bound.append(segment)
            continue
        if block.block_type in _ATOMIC_TYPES:
            segment = [block]
            allowed_following = (
                {"caption", "footnote"} if hybrid_policy else {"caption"}
            )
            while index + 1 < len(blocks) and (
                blocks[index + 1].block_type in allowed_following
            ):
                index += 1
                segment.append(blocks[index])
            atomic_bound.append(segment)
        else:
            atomic_bound.append([block])
        index += 1

    children: list[list[DocumentBlock]] = []
    current: list[DocumentBlock] = []
    current_tokens = 0

    def overlap_suffix(segment: list[DocumentBlock]) -> list[DocumentBlock]:
        """Reuse complete trailing blocks without cutting semantic units."""

        suffix: list[DocumentBlock] = []
        suffix_tokens = 0
        for candidate in reversed(segment):
            candidate_tokens = max(1, len(tokenise(_block_text(candidate))))
            if candidate_tokens > child_overlap_tokens:
                break
            if suffix_tokens + candidate_tokens > child_overlap_tokens:
                break
            suffix.insert(0, candidate)
            suffix_tokens += candidate_tokens
        return suffix

    for segment in atomic_bound:
        atomic = any(block.block_type in _ATOMIC_TYPES for block in segment)
        segment_tokens = _tokens(segment)
        if atomic:
            if current:
                children.append(current)
                current = []
                current_tokens = 0
            children.append(segment)
            continue
        if current and (
            current_tokens >= child_target_tokens
            or current_tokens + segment_tokens > child_max_tokens
        ):
            children.append(current)
            current = overlap_suffix(current)
            current_tokens = _tokens(current)
            if current and current_tokens + segment_tokens > child_max_tokens:
                current = []
                current_tokens = 0
        current.extend(segment)
        current_tokens += segment_tokens
    if current:
        children.append(current)
    return children


def build_parent_child_units(
    document: ParsedDocument,
    *,
    parent_target_tokens: int = 2_000,
    parent_max_tokens: int = 3_000,
    child_target_tokens: int = 400,
    child_max_tokens: int = 500,
    child_overlap_tokens: int = 60,
    hybrid_policy: bool = False,
) -> tuple[HierarchicalUnit, ...]:
    if not 500 <= parent_target_tokens <= parent_max_tokens <= 8_000:
        raise ValueError("invalid Parent token budget")
    if not 100 <= child_target_tokens <= child_max_tokens <= parent_max_tokens:
        raise ValueError("invalid Child token budget")
    if not 0 <= child_overlap_tokens < child_target_tokens:
        raise ValueError("invalid Child overlap budget")
    parent_groups = _parents(
        document,
        parent_target_tokens=parent_target_tokens,
        parent_max_tokens=parent_max_tokens,
        hybrid_policy=hybrid_policy,
    )
    units: list[HierarchicalUnit] = []
    child_order = 0
    for parent_index, (heading_path, blocks) in enumerate(parent_groups):
        parent_id = _unit_id(document.document_id, "parent", parent_index, blocks)
        parent_text = "\n\n".join(
            _block_text(block) for block in blocks if _block_text(block)
        )
        if not parent_text:
            continue
        units.append(
            HierarchicalUnit(
                unit_id=parent_id,
                role="parent",
                parent_id=parent_id,
                text=parent_text,
                heading_path=heading_path,
                block_ids=tuple(block.block_id for block in blocks),
                pages=tuple(dict.fromkeys(block.page for block in blocks)),
                block_types=tuple(
                    dict.fromkeys(block.block_type for block in blocks)
                ),
                order=parent_index,
                oversized_atomic=(
                    _tokens(blocks) > parent_max_tokens and len(blocks) == 1
                ),
            )
        )
        for segment in _child_segments(
            blocks,
            child_target_tokens=child_target_tokens,
            child_max_tokens=child_max_tokens,
            child_overlap_tokens=child_overlap_tokens,
            hybrid_policy=hybrid_policy,
        ):
            child_id = _unit_id(document.document_id, "child", child_order, segment)
            child_text = "\n\n".join(
                _block_text(block) for block in segment if _block_text(block)
            )
            if not child_text:
                continue
            units.append(
                HierarchicalUnit(
                    unit_id=child_id,
                    role="child",
                    parent_id=parent_id,
                    text=child_text,
                    heading_path=heading_path,
                    block_ids=tuple(block.block_id for block in segment),
                    pages=tuple(dict.fromkeys(block.page for block in segment)),
                    block_types=tuple(
                        dict.fromkeys(block.block_type for block in segment)
                    ),
                    order=child_order,
                    oversized_atomic=(
                        _tokens(segment) > child_max_tokens
                        and any(
                            block.block_type in _ATOMIC_TYPES for block in segment
                        )
                    ),
                )
            )
            child_order += 1
    child_positions = [
        index for index, unit in enumerate(units) if unit.role == "child"
    ]
    original = tuple(units)
    for position, unit_index in enumerate(child_positions):
        units[unit_index] = replace(
            original[unit_index],
            previous_unit_id=(
                original[child_positions[position - 1]].unit_id if position else None
            ),
            next_unit_id=(
                original[child_positions[position + 1]].unit_id
                if position + 1 < len(child_positions)
                else None
            ),
        )
    return tuple(units)


def build_flat_units(
    document: ParsedDocument,
    *,
    target_chars: int = 2_000,
    overlap_chars: int = 0,
) -> tuple[HierarchicalUnit, ...]:
    """Build page-bounded flat units with optional whole-block overlap.

    Overlap is deliberately applied only between chunks on the same page and
    only by reusing complete trailing blocks.  This keeps page provenance and
    paragraph identity stable while allowing a controlled overlap ablation;
    arbitrary character overlap would create partial/duplicate paragraphs and
    make the strict Gold-evidence metric harder to interpret.
    """

    if not 256 <= target_chars <= 50_000:
        raise ValueError("flat chunk target must be between 256 and 50000 characters")
    if not 0 <= overlap_chars < target_chars:
        raise ValueError("flat chunk overlap must be between 0 and target_chars")
    source = [
        block
        for block in document.blocks
        if block.indexable and block.block_type in _TEXT_TYPES and _block_text(block)
    ]
    segments: list[list[DocumentBlock]] = []
    page_blocks: list[list[DocumentBlock]] = []
    for block in source:
        if not page_blocks or block.page != page_blocks[-1][-1].page:
            page_blocks.append([])
        page_blocks[-1].append(block)

    for blocks_on_page in page_blocks:
        current: list[DocumentBlock] = []
        current_chars = 0

        def suffix(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
            if not overlap_chars:
                return []
            selected: list[DocumentBlock] = []
            selected_chars = 0
            for candidate in reversed(blocks):
                candidate_chars = len(_block_text(candidate))
                separator = 2 if selected else 0
                if selected_chars + separator + candidate_chars > overlap_chars:
                    break
                selected.insert(0, candidate)
                selected_chars += separator + candidate_chars
            return selected

        for block in blocks_on_page:
            text = _block_text(block)
            separator = 2 if current else 0
            if current and current_chars + separator + len(text) > target_chars:
                previous = current
                segments.append(previous)
                current = suffix(previous)
                current_chars = sum(
                    len(_block_text(item)) + (2 if index else 0)
                    for index, item in enumerate(current)
                )
                # A carried suffix must not become a chunk by itself when the
                # next block cannot fit beside it; discard it and restart.
                if current and current_chars + 2 + len(text) > target_chars:
                    current = []
                    current_chars = 0
                separator = 2 if current else 0
            current.append(block)
            current_chars += separator + len(text)
        if current:
            segments.append(current)
    units: list[HierarchicalUnit] = []
    heading_path: tuple[str, ...] = ()
    for index, blocks in enumerate(segments):
        heading_path = _heading_path(blocks, heading_path)
        unit_id = _unit_id(document.document_id, "flat", index, blocks)
        text = "\n\n".join(_block_text(block) for block in blocks)
        units.append(
            HierarchicalUnit(
                unit_id=unit_id,
                role="child",
                parent_id=unit_id,
                text=text,
                heading_path=heading_path,
                block_ids=tuple(block.block_id for block in blocks),
                pages=tuple(dict.fromkeys(block.page for block in blocks)),
                block_types=tuple(
                    dict.fromkeys(block.block_type for block in blocks)
                ),
                order=index,
                previous_unit_id=(units[-1].unit_id if units else None),
                oversized_atomic=(
                    len(text) > target_chars
                    and any(block.block_type in _ATOMIC_TYPES for block in blocks)
                ),
            )
        )
        if len(units) > 1:
            units[-2] = replace(units[-2], next_unit_id=unit_id)
    return tuple(units)


def _split_text_char_parts(text: str, *, max_chars: int) -> list[str]:
    """Split one oversized prose block without dropping sentence text.

    The boundary-aware chunker treats a parser block as the smallest normal
    unit.  A very long paragraph is the exception: it is split at sentence or
    line boundaries first and only falls back to a hard character boundary
    when a single sentence has no usable boundary.  The returned pieces are
    intentionally plain strings; callers keep the original block id on every
    piece so evidence alignment still points at the authoritative parser
    block.
    """

    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]
    sentences = [
        value.strip()
        for value in re.split(r"(?<=[.!?。！？])\s+|\n+", cleaned)
        if value.strip()
    ]
    if not sentences:
        sentences = [cleaned]
    parts: list[str] = []
    current: list[str] = []
    current_chars = 0

    def flush() -> None:
        nonlocal current, current_chars
        if current:
            parts.append(" ".join(current).strip())
            current = []
            current_chars = 0

    for sentence in sentences:
        if len(sentence) > max_chars:
            flush()
            start = 0
            while start < len(sentence):
                parts.append(sentence[start : start + max_chars].strip())
                start += max_chars
            continue
        separator = 1 if current else 0
        if current and current_chars + separator + len(sentence) > max_chars:
            flush()
            separator = 0
        current.append(sentence)
        current_chars += separator + len(sentence)
    flush()
    return [part for part in parts if part]


def _boundary_groups(
    blocks: list[DocumentBlock],
    *,
    max_chars: int,
) -> list[list[DocumentBlock]]:
    """Create unsplittable scientific-document elements for boundary search."""

    expanded: list[DocumentBlock] = []
    for block in blocks:
        text = _block_text(block)
        if not text:
            continue
        # Tables, figures, equations, code and algorithms are authoritative
        # atomic regions.  Keep an oversized region intact and record the
        # oversize on the eventual unit instead of corrupting its structure.
        if block.block_type in _ATOMIC_TYPES or block.block_type == "list":
            expanded.append(block)
            continue
        parts = _split_text_char_parts(text, max_chars=max_chars)
        if len(parts) == 1:
            expanded.append(block)
            continue
        for index, part in enumerate(parts):
            structured = dict(block.structured_content)
            structured.update(
                {
                    "chunk_fragment_index": index,
                    "chunk_fragment_count": len(parts),
                    "source_block_id": block.block_id,
                }
            )
            expanded.append(
                block.model_copy(
                    update={
                        "text": part,
                        "structured_content": structured,
                        "content_hash": hashlib.sha256(part.encode()).hexdigest(),
                    }
                )
            )

    groups: list[list[DocumentBlock]] = []
    index = 0
    while index < len(expanded):
        block = expanded[index]
        if (
            block.block_type == "caption"
            and index + 1 < len(expanded)
            and expanded[index + 1].block_type in _ATOMIC_TYPES
        ):
            group = [block, expanded[index + 1]]
            index += 2
            while index < len(expanded) and expanded[index].block_type in {
                "caption",
                "footnote",
            }:
                group.append(expanded[index])
                index += 1
            groups.append(group)
            continue
        if block.block_type in _ATOMIC_TYPES:
            group = [block]
            index += 1
            while index < len(expanded) and expanded[index].block_type in {
                "caption",
                "footnote",
            }:
                group.append(expanded[index])
                index += 1
            groups.append(group)
            continue
        # Lists are kept as one group.  A heading is also a group so it can
        # act as a hard section boundary while remaining attached to the next
        # content group when the section is materialized.
        groups.append([block])
        index += 1
    return groups


def _group_chars(group: list[DocumentBlock]) -> int:
    return sum(
        len(_block_text(block)) + (2 if block_index else 0)
        for block_index, block in enumerate(group)
    )


_COMMON_SECTION_HEADINGS = frozenset(
    {
        "abstract",
        "introduction",
        "background",
        "method",
        "methods",
        "approach",
        "experiments",
        "experimental results",
        "results",
        "discussion",
        "related work",
        "conclusion",
        "conclusions",
        "references",
        "致谢",
        "摘要",
        "引言",
        "相关工作",
        "方法",
        "实验",
        "实验结果",
        "结果",
        "讨论",
        "结论",
        "参考文献",
    }
)


def _is_hard_heading(block: DocumentBlock) -> bool:
    """Conservatively identify section boundaries from parser title blocks.

    MinerU's flat 3.4.x output can mark short prose (for example, ``in Figure
    3.``) as a title.  Treating every title block as a hard boundary fragments
    papers and was the main failure mode of the first boundary experiment.
    Only top-level/numbered/common section headings become hard boundaries;
    deeper subsection labels remain valid metadata but may share a retrieval
    chunk with adjacent prose.
    """

    if block.block_type != "title":
        return False
    text = " ".join(block.text.split())
    if not text or len(text) > 180:
        return False
    if re.search(r"[.!?。！？,:;，：；]$", text):
        return False
    level = block.heading_level or 1
    if level > 2:
        return False
    normalized = text.casefold()
    if normalized in _COMMON_SECTION_HEADINGS:
        return True
    if re.match(r"^(?:\d+)(?:\.\d+)?\s+\S", text):
        return True
    if re.match(r"^第[一二三四五六七八九十百零0-9]+[章节部分]\s*\S+", text):
        return True
    # A short, punctuation-free heading at level 1 is usually the paper title
    # or a top-level section.  Do not promote level-2/3 parser guesses without
    # a numbering or known-section signal.
    return level == 1 and len(text.split()) <= 12


def _boundary_section_segments(
    groups: list[list[DocumentBlock]],
    *,
    target_chars: int,
    min_chars: int,
    max_chars: int,
    search_chars: int,
) -> list[list[DocumentBlock]]:
    """Choose natural group boundaries near a fixed-length target."""

    segments: list[list[DocumentBlock]] = []
    start = 0
    while start < len(groups):
        chars = 0
        candidate_ends: list[tuple[int, int]] = []
        end = start
        while end < len(groups):
            group_chars = _group_chars(groups[end])
            next_chars = chars + group_chars + (2 if chars else 0)
            if end > start and _is_hard_heading(groups[end][0]):
                break
            if next_chars > max_chars:
                break
            chars = next_chars
            end += 1
            if chars >= min_chars:
                candidate_ends.append((end, chars))
            if chars >= target_chars + search_chars:
                break
        if candidate_ends:
            selected_end, _ = min(
                candidate_ends,
                key=lambda item: (abs(item[1] - target_chars), -item[0]),
            )
            segments.append([block for group in groups[start:selected_end] for block in group])
            start = selected_end
            continue
        # A short section or a protected element that cannot reach min_chars
        # is retained rather than merged across a heading boundary.
        if end > start:
            segments.append([block for group in groups[start:end] for block in group])
            start = end
            continue
        # The first group itself exceeds max_chars.  It is either an atomic
        # group (handled above) or a parser block that must be retained whole.
        segments.append([block for group in groups[start : start + 1] for block in group])
        start += 1
    return segments


def build_boundary_aware_flat_units(
    document: ParsedDocument,
    *,
    target_chars: int = 2_000,
    min_chars: int = 1_000,
    max_chars: int = 2_600,
    search_chars: int = 400,
) -> tuple[HierarchicalUnit, ...]:
    """Build page-bounded Flat units with structure-safe boundary correction.

    This is an isolated experimental strategy.  It keeps the Flat retrieval
    shape (one evidence unit per chunk, no overlap, no Parent expansion) while
    moving boundaries to complete parser blocks near ``target_chars``.  Titles
    are hard section boundaries; tables, lists, figures, equations and other
    atomic regions are never cut internally.
    """

    if not 256 <= min_chars <= target_chars <= max_chars <= 50_000:
        raise ValueError("invalid boundary-aware chunk character budgets")
    if not 0 <= search_chars <= max_chars:
        raise ValueError("invalid boundary search window")
    source = [
        block
        for block in document.blocks
        if block.indexable and block.block_type in _TEXT_TYPES and _block_text(block)
    ]
    pages: list[list[DocumentBlock]] = []
    for block in source:
        if not pages or block.page != pages[-1][-1].page:
            pages.append([])
        pages[-1].append(block)

    segments: list[list[DocumentBlock]] = []
    for page_blocks in pages:
        groups = _boundary_groups(page_blocks, max_chars=max_chars)
        section_groups: list[list[list[DocumentBlock]]] = []
        current: list[list[DocumentBlock]] = []
        for group in groups:
            if current and _is_hard_heading(group[0]):
                section_groups.append(current)
                current = []
            current.append(group)
        if current:
            section_groups.append(current)
        for section in section_groups:
            segments.extend(
                _boundary_section_segments(
                    section,
                    target_chars=target_chars,
                    min_chars=min_chars,
                    max_chars=max_chars,
                    search_chars=search_chars,
                )
            )

    units: list[HierarchicalUnit] = []
    heading_path: tuple[str, ...] = ()
    for index, blocks in enumerate(segments):
        heading_path = _heading_path(blocks, heading_path)
        unit_id = _unit_id(document.document_id, "boundary", index, blocks)
        text = "\n\n".join(_block_text(block) for block in blocks if _block_text(block))
        if not text:
            continue
        units.append(
            HierarchicalUnit(
                unit_id=unit_id,
                role="child",
                parent_id=unit_id,
                text=text,
                heading_path=heading_path,
                block_ids=tuple(block.block_id for block in blocks),
                pages=tuple(dict.fromkeys(block.page for block in blocks)),
                block_types=tuple(dict.fromkeys(block.block_type for block in blocks)),
                order=len(units),
                previous_unit_id=(units[-1].unit_id if units else None),
                oversized_atomic=(
                    len(text) > max_chars
                    and any(block.block_type in _ATOMIC_TYPES for block in blocks)
                ),
            )
        )
        if len(units) > 1:
            units[-2] = replace(units[-2], next_unit_id=unit_id)
    return tuple(units)


def build_structure_region_units(
    document: ParsedDocument,
    *,
    target_chars: int = 1_800,
    min_chars: int = 900,
    max_chars: int = 2_400,
    search_chars: int = 300,
) -> tuple[HierarchicalUnit, ...]:
    """Build sparse, context-rich auxiliary units around document structure.

    Unlike the original 400/500-token Child lane, this experimental projection
    does not index every short Child.  It emits only section-leading or
    structured regions and keeps their surrounding prose near the Flat chunk
    size.  Flat remains the primary lane, so papers without reliable titles,
    lists, tables, equations, figures, code, or algorithms simply contribute no
    auxiliary units.

    Regions may cross a page boundary when the section remains continuous.
    Tables, lists, figures, equations, code, and algorithms stay atomic, with
    captions and footnotes attached by ``_boundary_groups``.
    """

    if not 256 <= min_chars <= target_chars <= max_chars <= 50_000:
        raise ValueError("invalid structure-region chunk character budgets")
    if not 0 <= search_chars <= max_chars:
        raise ValueError("invalid structure-region boundary search window")
    source = [
        block
        for block in document.blocks
        if block.indexable and block.block_type in _TEXT_TYPES and _block_text(block)
    ]
    if not source:
        return ()

    groups = _boundary_groups(source, max_chars=max_chars)
    sections: list[list[list[DocumentBlock]]] = []
    current: list[list[DocumentBlock]] = []
    for group in groups:
        if current and _is_hard_heading(group[0]):
            sections.append(current)
            current = []
        current.append(group)
    if current:
        sections.append(current)

    signal_types = _ATOMIC_TYPES | {"title", "list", "caption"}
    segments: list[list[DocumentBlock]] = []
    for section in sections:
        for segment in _boundary_section_segments(
            section,
            target_chars=target_chars,
            min_chars=min_chars,
            max_chars=max_chars,
            search_chars=search_chars,
        ):
            if any(block.block_type in signal_types for block in segment):
                segments.append(segment)

    units: list[HierarchicalUnit] = []
    heading_path: tuple[str, ...] = ()
    for segment in segments:
        heading_path = _heading_path(segment, heading_path)
        unit_id = _unit_id(
            document.document_id, "structure_region", len(units), segment
        )
        text = "\n\n".join(
            _block_text(block) for block in segment if _block_text(block)
        )
        if not text:
            continue
        units.append(
            HierarchicalUnit(
                unit_id=unit_id,
                role="child",
                parent_id=unit_id,
                text=text,
                heading_path=heading_path,
                block_ids=tuple(block.block_id for block in segment),
                pages=tuple(dict.fromkeys(block.page for block in segment)),
                block_types=tuple(dict.fromkeys(block.block_type for block in segment)),
                order=len(units),
                previous_unit_id=(units[-1].unit_id if units else None),
                oversized_atomic=(
                    len(text) > max_chars
                    and any(block.block_type in _ATOMIC_TYPES for block in segment)
                ),
            )
        )
        if len(units) > 1:
            units[-2] = replace(units[-2], next_unit_id=unit_id)
    return tuple(units)


def build_sliding_window_units(
    document: ParsedDocument,
    *,
    window_chars: int = 500,
    overlap_chars: int = 100,
) -> tuple[HierarchicalUnit, ...]:
    """Build same-page character windows while retaining Block provenance.

    This is an explicit ablation for cases where a paragraph-aware block
    chunk is not enough.  Windows never cross pages, and every window records
    all complete/partial Blocks intersecting its character range.  The mode
    is intentionally not the default: a raw window can split a long paragraph
    and therefore must pass the strict Gold-alignment gate before promotion.
    """

    if not 256 <= window_chars <= 50_000:
        raise ValueError("sliding window must be between 256 and 50000 characters")
    if not 0 <= overlap_chars < window_chars:
        raise ValueError("sliding window overlap must be between 0 and window_chars")
    source = [
        block
        for block in document.blocks
        if block.indexable and block.block_type in _TEXT_TYPES and _block_text(block)
    ]
    pages: list[list[DocumentBlock]] = []
    for block in source:
        if not pages or block.page != pages[-1][-1].page:
            pages.append([])
        pages[-1].append(block)

    units: list[HierarchicalUnit] = []
    heading_path: tuple[str, ...] = ()
    stride = max(1, window_chars - overlap_chars)
    for page_blocks in pages:
        pieces: list[str] = []
        ranges: list[tuple[int, int, DocumentBlock]] = []
        cursor = 0
        for block in page_blocks:
            if pieces:
                pieces.append("\n\n")
                cursor += 2
            text = _block_text(block)
            start = cursor
            cursor += len(text)
            ranges.append((start, cursor, block))
            pieces.append(text)
        page_text = "".join(pieces)
        if not page_text:
            continue
        final_start = max(0, len(page_text) - window_chars)
        starts = list(range(0, final_start + 1, stride))
        if not starts or starts[-1] != final_start:
            starts.append(final_start)
        for start in starts:
            end = min(len(page_text), start + window_chars)
            selected_blocks = [
                block
                for block_start, block_end, block in ranges
                if block_end > start and block_start < end
            ]
            text = page_text[start:end].strip()
            if not text or not selected_blocks:
                continue
            heading_path = _heading_path(selected_blocks, heading_path)
            unit_id = _unit_id(document.document_id, "sliding", len(units), selected_blocks)
            units.append(
                HierarchicalUnit(
                    unit_id=unit_id,
                    role="child",
                    parent_id=unit_id,
                    text=text,
                    heading_path=heading_path,
                    block_ids=tuple(block.block_id for block in selected_blocks),
                    pages=(page_blocks[0].page,),
                    block_types=tuple(
                        dict.fromkeys(block.block_type for block in selected_blocks)
                    ),
                    order=len(units),
                    previous_unit_id=(units[-1].unit_id if units else None),
                    oversized_atomic=(
                        len(text) > window_chars
                        and any(
                            block.block_type in _ATOMIC_TYPES
                            for block in selected_blocks
                        )
                    ),
                )
            )
            if len(units) > 1:
                units[-2] = replace(units[-2], next_unit_id=unit_id)
    return tuple(units)


__all__ = [
    "HierarchicalUnit",
    "build_boundary_aware_flat_units",
    "build_flat_units",
    "build_parent_child_units",
    "build_sliding_window_units",
    "build_structure_region_units",
]

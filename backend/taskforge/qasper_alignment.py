"""Strict paragraph alignment and Recall scoring for QASPER PDF retrieval.

The QASPER source labels evidence by paragraph text while the product retrieves
PDF-derived chunks.  This module keeps alignment separate from ranking: a
frozen text-only mapping is built first, then ranked child IDs are projected
back to complete gold evidence units for Recall@K.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from math import ceil
from typing import Literal

from pydantic import Field, model_validator

from .domain import StrictModel
from .rag_evaluation import GoldEvidenceUnit, QasperGoldLabels

_LIGATURES = str.maketrans(
    {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "st",
        "\ufb06": "st",
        "\u00ad": "",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201b": "'",
        "\u2032": "'",
    }
)
_TOKEN_RE = re.compile(
    r"[-+]?\d+(?:[.,]\d+)*(?:%|[a-z]+)?|[a-z]+(?:'[a-z]+)?|"
    r"[\u3400-\u4dbf\u4e00-\u9fff]",
    re.IGNORECASE,
)
_DEHYPHENATE_RE = re.compile(r"(?<=[A-Za-z])-\s+(?=[A-Za-z])")
_FLOAT_MARKER_RE = re.compile(r"^\s*float\s+selected\s*:\s*", re.IGNORECASE)
_STRUCTURE_REFERENCE_RE = re.compile(
    r"\b(table|tab\.?|figure|fig\.?|section|sec\.?)\s+"
    r"(?:TABREF|FIGREF|SECREF)?\d+(?:\.\d+)*(?:[A-Za-z])?\b",
    re.IGNORECASE,
)
_BIBLIOGRAPHY_REFERENCE_RE = re.compile(r"\bBIBREF\d+\b", re.IGNORECASE)
_PARENTHETICAL_CITATION_RE = re.compile(
    r"\((?=[^()]{0,120}\b(?:19|20)\d{2}[a-z]?\b)[^()]{1,120}\)",
    re.IGNORECASE,
)
_HTML_SUPERSCRIPT_RE = re.compile(
    r"<sup\b[^>]*>\s*(?:\d+|[*\u2020\u2021]+)\s*</sup>",
    re.IGNORECASE,
)
_HTML_TAG_RE = re.compile(r"</?[a-z][^>]*>", re.IGNORECASE)
_CORPUS_FORM_PLACEHOLDER_RE = re.compile(
    r"\b(?:INLINEFORM|DISPLAYFORM)\d+\b",
    re.IGNORECASE,
)
_LATEX_ARROW_RE = re.compile(
    r"\\(?:long)?(?:left|right)?arrow\b|[\u2190-\u21ff]",
    re.IGNORECASE,
)
_NON_PREFIX_RE = re.compile(r"\bnon[-\s]+(?=[a-z])", re.IGNORECASE)


class AlignmentChunk(StrictModel):
    child_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    order: int = Field(default=0, ge=0)
    section: str | None = None


class GoldAlignedSpan(StrictModel):
    child_id: str = Field(min_length=1)
    gold_token_start: int = Field(ge=0)
    gold_token_end: int = Field(gt=0)
    matched_tokens: int = Field(gt=0)
    score: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def span_is_ordered(self) -> GoldAlignedSpan:
        if self.gold_token_end <= self.gold_token_start:
            raise ValueError("gold alignment span must be non-empty")
        return self


class GoldAlignment(StrictModel):
    gold_unit_id: str = Field(min_length=1)
    status: Literal["exact", "fuzzy", "ambiguous", "unaligned"]
    aligned_child_spans: list[GoldAlignedSpan] = Field(default_factory=list)
    normalized_coverage: float = Field(ge=0.0, le=1.0)
    alignment_score: float = Field(ge=0.0, le=1.0)
    gold_token_count: int = Field(ge=1)


class QasperRecallAtK(StrictModel):
    k: int = Field(gt=0)
    recall: float = Field(ge=0.0, le=1.0)
    selected_annotation_id: str = Field(min_length=1)
    hit_unit_ids: list[str] = Field(default_factory=list)
    total_units: int = Field(gt=0)


class AlignmentDiagnostics(StrictModel):
    total_units: int = Field(ge=0)
    exact_units: int = Field(ge=0)
    fuzzy_units: int = Field(ge=0)
    ambiguous_units: int = Field(ge=0)
    unaligned_units: int = Field(ge=0)
    alignment_coverage: float = Field(ge=0.0, le=1.0)


def normalize_alignment_text(value: str) -> str:
    """Normalize PDF representation noise without dropping semantic tokens."""

    text = unicodedata.normalize("NFKC", str(value)).translate(_LIGATURES)
    # This is a QASPER annotation locator, not text that appears in the paper.
    # Removing only the anchored marker maps the label to the real caption
    # without injecting benchmark-specific words into the retrieval corpus.
    text = _FLOAT_MARKER_RE.sub("", text)
    text = _HTML_SUPERSCRIPT_RE.sub("", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _STRUCTURE_REFERENCE_RE.sub(lambda match: match.group(1), text)
    text = _BIBLIOGRAPHY_REFERENCE_RE.sub("", text)
    text = _CORPUS_FORM_PLACEHOLDER_RE.sub("", text)
    text = _LATEX_ARROW_RE.sub(" ", text)
    text = _PARENTHETICAL_CITATION_RE.sub("", text)
    text = _DEHYPHENATE_RE.sub("", text)
    # PDF parsers and source TeX disagree on whether lexical ``non`` prefixes
    # are joined (``nontargeted``) or hyphenated (``non-targeted``). Preserve
    # the negation while canonicalising that representation difference.
    text = _NON_PREFIX_RE.sub("non", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def alignment_tokens(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(normalize_alignment_text(value)))


def _subsequence_starts(
    needle: Sequence[str], haystack: Sequence[str]
) -> list[int]:
    if not needle or len(needle) > len(haystack):
        return []
    first = needle[0]
    return [
        index
        for index in range(len(haystack) - len(needle) + 1)
        if haystack[index] == first
        and tuple(haystack[index : index + len(needle)]) == tuple(needle)
    ]


def _covered_tokens(spans: Sequence[GoldAlignedSpan]) -> int:
    ranges = sorted(
        (span.gold_token_start, span.gold_token_end) for span in spans
    )
    if not ranges:
        return 0
    covered = 0
    current_start, current_end = ranges[0]
    for start, end in ranges[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        covered += current_end - current_start
        current_start, current_end = start, end
    return covered + current_end - current_start


def _candidate_spans(
    unit: GoldEvidenceUnit,
    chunks: Sequence[AlignmentChunk],
    *,
    minimum_fragment_tokens: int,
    minimum_fragment_precision: float,
    minimum_fragment_coverage: float,
) -> tuple[list[GoldAlignedSpan], bool]:
    gold = alignment_tokens(unit.text)
    spans: list[GoldAlignedSpan] = []
    complete_exact = False
    # A fixed six-token fragment makes short list items impossible to align
    # when a citation or abbreviation is inserted in the PDF. Require two or
    # more meaningful local fragments for short Gold units while retaining
    # the original threshold for normal paragraphs.
    effective_fragment_tokens = min(
        minimum_fragment_tokens,
        max(2, ceil(len(gold) * 0.30)),
    )
    for chunk in chunks:
        candidate = alignment_tokens(chunk.text)
        if not candidate:
            continue
        if _subsequence_starts(gold, candidate):
            complete_exact = True
            spans.append(
                GoldAlignedSpan(
                    child_id=chunk.child_id,
                    gold_token_start=0,
                    gold_token_end=len(gold),
                    matched_tokens=len(gold),
                    score=1.0,
                )
            )
            continue
        contained_starts = _subsequence_starts(candidate, gold)
        if contained_starts and len(candidate) >= effective_fragment_tokens:
            coverage = len(candidate) / len(gold)
            if coverage >= minimum_fragment_coverage:
                start = contained_starts[0]
                spans.append(
                    GoldAlignedSpan(
                        child_id=chunk.child_id,
                        gold_token_start=start,
                        gold_token_end=start + len(candidate),
                        matched_tokens=len(candidate),
                        score=min(1.0, 2.0 * coverage / (1.0 + coverage)),
                    )
                )
                continue
        matcher = SequenceMatcher(None, gold, candidate, autojunk=False)
        blocks = [
            block
            for block in matcher.get_matching_blocks()
            if block.size >= effective_fragment_tokens
        ]
        matched = sum(block.size for block in blocks)
        if not matched:
            continue
        # Child chunks intentionally carry local context. Alignment quality
        # must not punish unmatched prefix/suffix text outside the Gold
        # paragraph. Measure representation noise only inside the smallest
        # candidate window spanning the matched fragments (for example PDF
        # footnote URLs or expanded citation strings).
        candidate_start = min(block.b for block in blocks)
        candidate_end = max(block.b + block.size for block in blocks)
        local_candidate_tokens = max(1, candidate_end - candidate_start)
        precision = matched / local_candidate_tokens
        coverage = matched / len(gold)
        if (
            precision < minimum_fragment_precision
            or coverage < minimum_fragment_coverage
        ):
            continue
        score = 2.0 * precision * coverage / (precision + coverage)
        spans.extend(
            GoldAlignedSpan(
                child_id=chunk.child_id,
                gold_token_start=block.a,
                gold_token_end=block.a + block.size,
                matched_tokens=block.size,
                score=score,
            )
            for block in blocks
        )
    return spans, complete_exact


def align_gold_unit(
    unit: GoldEvidenceUnit,
    chunks: Sequence[AlignmentChunk],
    *,
    minimum_complete_coverage: float = 0.80,
    minimum_fragment_tokens: int = 6,
    minimum_fragment_precision: float = 0.65,
    minimum_fragment_coverage: float = 0.15,
) -> GoldAlignment:
    """Align one gold paragraph without consulting query or retrieval ranks."""

    if not 0.0 < minimum_complete_coverage <= 1.0:
        raise ValueError("minimum complete coverage must be in (0, 1]")
    gold_token_count = len(alignment_tokens(unit.text))
    if gold_token_count < 1:
        raise ValueError("gold evidence must contain at least one alignment token")
    spans, complete_exact = _candidate_spans(
        unit,
        chunks,
        minimum_fragment_tokens=min(minimum_fragment_tokens, gold_token_count),
        minimum_fragment_precision=minimum_fragment_precision,
        minimum_fragment_coverage=minimum_fragment_coverage,
    )
    coverage = min(1.0, _covered_tokens(spans) / gold_token_count)
    if complete_exact:
        status: Literal["exact", "fuzzy", "ambiguous", "unaligned"] = "exact"
    elif coverage >= minimum_complete_coverage:
        status = "fuzzy"
    elif spans:
        status = "ambiguous"
    else:
        status = "unaligned"
    return GoldAlignment(
        gold_unit_id=unit.unit_id,
        status=status,
        aligned_child_spans=spans,
        normalized_coverage=coverage,
        alignment_score=max((span.score for span in spans), default=0.0),
        gold_token_count=gold_token_count,
    )


def align_qasper_gold(
    labels: QasperGoldLabels,
    chunks: Sequence[AlignmentChunk],
    **kwargs: float | int,
) -> dict[str, GoldAlignment]:
    """Build one rank-independent alignment for every distinct gold unit."""

    by_id: dict[str, GoldEvidenceUnit] = {}
    for evidence_set in labels.evidence_sets:
        for unit in evidence_set.units:
            by_id.setdefault(unit.unit_id, unit)
    return {
        unit_id: align_gold_unit(unit, chunks, **kwargs)
        for unit_id, unit in by_id.items()
    }


def alignment_diagnostics(
    alignments: Mapping[str, GoldAlignment],
) -> AlignmentDiagnostics:
    values = list(alignments.values())
    aligned = sum(item.status in {"exact", "fuzzy"} for item in values)
    return AlignmentDiagnostics(
        total_units=len(values),
        exact_units=sum(item.status == "exact" for item in values),
        fuzzy_units=sum(item.status == "fuzzy" for item in values),
        ambiguous_units=sum(item.status == "ambiguous" for item in values),
        unaligned_units=sum(item.status == "unaligned" for item in values),
        alignment_coverage=aligned / len(values) if values else 0.0,
    )


def paragraph_recall_at_k(
    labels: QasperGoldLabels,
    retrieved_paragraph_ids: Sequence[str],
    k: int,
) -> QasperRecallAtK:
    """Score corpus-native paragraph IDs against alternative annotation sets."""

    if k <= 0:
        raise ValueError("k must be positive")
    head = set(retrieved_paragraph_ids[:k])
    scored: list[tuple[float, str, list[str], int]] = []
    for evidence_set in labels.evidence_sets:
        hits = [
            unit.unit_id
            for unit in evidence_set.units
            if head.intersection(unit.alternative_paragraph_ids)
        ]
        scored.append(
            (
                len(hits) / len(evidence_set.units),
                evidence_set.annotation_id,
                hits,
                len(evidence_set.units),
            )
        )
    recall, annotation_id, hits, total = max(
        scored, key=lambda item: (item[0], -item[3], item[1])
    )
    return QasperRecallAtK(
        k=k,
        recall=recall,
        selected_annotation_id=annotation_id,
        hit_unit_ids=hits,
        total_units=total,
    )


def aligned_recall_at_k(
    labels: QasperGoldLabels,
    alignments: Mapping[str, GoldAlignment],
    retrieved_child_ids: Sequence[str],
    k: int,
    *,
    minimum_complete_coverage: float = 0.80,
) -> QasperRecallAtK:
    """Score actual retrieved child content; page overlap is never consulted."""

    if k <= 0:
        raise ValueError("k must be positive")
    if not 0.0 < minimum_complete_coverage <= 1.0:
        raise ValueError("minimum complete coverage must be in (0, 1]")
    head = set(retrieved_child_ids[:k])
    scored: list[tuple[float, str, list[str], int]] = []
    for evidence_set in labels.evidence_sets:
        hits: list[str] = []
        for unit in evidence_set.units:
            alignment = alignments.get(unit.unit_id)
            if alignment is None or alignment.status not in {"exact", "fuzzy"}:
                continue
            presented = [
                span
                for span in alignment.aligned_child_spans
                if span.child_id in head
            ]
            coverage = _covered_tokens(presented) / alignment.gold_token_count
            if coverage >= minimum_complete_coverage:
                hits.append(unit.unit_id)
        scored.append(
            (
                len(hits) / len(evidence_set.units),
                evidence_set.annotation_id,
                hits,
                len(evidence_set.units),
            )
        )
    recall, annotation_id, hits, total = max(
        scored, key=lambda item: (item[0], -item[3], item[1])
    )
    return QasperRecallAtK(
        k=k,
        recall=recall,
        selected_annotation_id=annotation_id,
        hit_unit_ids=hits,
        total_units=total,
    )


def alignment_coverage_for_children(
    alignment: GoldAlignment,
    child_ids: Sequence[str] | set[str],
) -> float:
    """Return non-overlapping Gold-token coverage for selected child chunks."""

    selected = set(child_ids)
    spans = [
        span
        for span in alignment.aligned_child_spans
        if span.child_id in selected
    ]
    return min(1.0, _covered_tokens(spans) / alignment.gold_token_count)


__all__ = [
    "AlignmentChunk",
    "AlignmentDiagnostics",
    "GoldAlignedSpan",
    "GoldAlignment",
    "QasperRecallAtK",
    "align_gold_unit",
    "align_qasper_gold",
    "alignment_coverage_for_children",
    "aligned_recall_at_k",
    "alignment_diagnostics",
    "alignment_tokens",
    "normalize_alignment_text",
    "paragraph_recall_at_k",
]

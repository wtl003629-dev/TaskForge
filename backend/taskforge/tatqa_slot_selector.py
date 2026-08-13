"""Label-free query planning and table-slot selection for TAT-QA-style QA.

This module only consumes a question and the supplied table.  It deliberately
has no annotation, answer, ``facts`` or ``mapping`` input, so it can be used as
a production-side selector and evaluated independently against heuristic
coordinate annotations.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

TATQAOperator = Literal[
    "lookup",
    "multi_span",
    "count",
    "average",
    "sum",
    "subtract",
    "percentage_change",
    "compare",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_STOPWORDS = frozenset(
    "what was were is are the a an of in on for to from between and by how "
    "much many did does do over under with as at part respective all used "
    "table amount which year years company different".split()
)
TATQA_SLOT_FEATURE_NAMES = (
    "row_overlap",
    "header_overlap",
    "value_overlap",
    "query_year_header_match",
    "is_header_row",
    "is_label_column",
    "exact_value_mention",
    "numeric_cell",
    "query_has_year",
    "log_table_rows",
    "log_table_width",
    "operator_percentage",
    "operator_average",
    "operator_count",
    "operator_subtract",
    "operator_compare",
    "operator_multi_span",
    "operator_lookup_or_sum",
    "count_row_overlap",
    "count_header_overlap",
    "count_label_column",
    "count_numeric_cell",
    "multi_span_row_overlap",
    "multi_span_label_column",
    "numeric_operator_row_overlap",
    "numeric_operator_header_overlap",
    "numeric_operator_year_match",
)


@dataclass(frozen=True)
class TATQATableSlot:
    row_index: int
    column_index: int
    value: str
    row_label: str
    column_header: str
    score: float
    signals: tuple[str, ...]


@dataclass(frozen=True)
class TATQASlotPlan:
    operator: TATQAOperator
    query_terms: tuple[str, ...]
    years: tuple[str, ...]
    slots: tuple[TATQATableSlot, ...]


def _stem(token: str) -> str:
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _terms(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(
        _stem(token)
        for token in _TOKEN_RE.findall(normalized)
        if token not in _STOPWORDS
    )


def _overlap(query_terms: tuple[str, ...], value: str) -> float:
    candidate = set(_terms(value))
    query = set(query_terms)
    if not candidate:
        return 0.0
    intersection = candidate.intersection(query)
    return len(intersection) / len(candidate) + 0.25 * len(intersection) / max(
        1, len(query)
    )


def classify_tatqa_operator(question: str) -> TATQAOperator:
    """Classify a constrained numerical/table operator from query language."""

    normalized = unicodedata.normalize("NFKC", question).casefold()
    if (
        "percentage change" in normalized
        or "percentage increase" in normalized
        or "percentage" in normalized
        or re.search(r"\bpercent\b", normalized) is not None
    ):
        return "percentage_change"
    if "average" in normalized or "mean" in normalized:
        return "average"
    if "how many" in normalized or "number of" in normalized or "count" in normalized:
        return "count"
    if any(term in normalized for term in ("difference", "change in", "increase /")):
        return "subtract"
    if any(term in normalized for term in ("larger", "smaller", "higher", "lower")):
        return "compare"
    if any(term in normalized for term in ("total of", "sum of", "combined")):
        return "sum"
    if normalized.startswith("what are") or any(
        term in normalized for term in ("components", "categories", "types of")
    ):
        return "multi_span"
    return "lookup"


def _feature_operator(operator: TATQAOperator) -> str:
    return {
        "percentage_change": "percentage",
        "average": "average",
        "count": "count",
        "subtract": "subtract",
        "compare": "compare",
        "multi_span": "multi_span",
    }.get(operator, "lookup_or_sum")


def tatqa_slot_feature_vector(
    question: str,
    table: list[list[str]],
    row_index: int,
    column_index: int,
) -> tuple[float, ...]:
    """Return the frozen query/table feature contract for learned slot ranking."""

    if not table or any(not isinstance(row, list) for row in table):
        raise ValueError("table must contain at least one row")
    width = max((len(row) for row in table), default=0)
    if (
        width == 0
        or row_index < 0
        or row_index >= len(table)
        or column_index < 0
        or column_index >= len(table[row_index])
    ):
        raise ValueError("slot coordinate is outside the table")
    query_terms = _terms(question)
    header = [str(value) for value in table[0]] + [""] * (width - len(table[0]))
    row = [str(value) for value in table[row_index]] + [""] * (
        width - len(table[row_index])
    )
    value = row[column_index]
    row_overlap = _overlap(query_terms, row[0])
    header_overlap = _overlap(query_terms, header[column_index])
    value_overlap = _overlap(query_terms, value)
    years = set(_YEAR_RE.findall(question))
    year_match = float(
        bool(years.intersection(_YEAR_RE.findall(header[column_index])))
    )
    exact_value = float(
        bool(value.strip())
        and value.strip().casefold()
        in unicodedata.normalize("NFKC", question).casefold()
    )
    numeric_cell = float(bool(re.search(r"[-+]?\d", value)))
    operator = _feature_operator(classify_tatqa_operator(question))
    numeric_operator = operator in {"percentage", "average", "subtract"}
    values = (
        row_overlap,
        header_overlap,
        value_overlap,
        year_match,
        float(row_index == 0),
        float(column_index == 0),
        exact_value,
        numeric_cell,
        float(bool(years)),
        math.log1p(len(table)),
        math.log1p(width),
        *(float(operator == name) for name in (
            "percentage",
            "average",
            "count",
            "subtract",
            "compare",
            "multi_span",
            "lookup_or_sum",
        )),
        float(operator == "count") * row_overlap,
        float(operator == "count") * header_overlap,
        float(operator == "count" and column_index == 0),
        float(operator == "count") * numeric_cell,
        float(operator == "multi_span") * row_overlap,
        float(operator == "multi_span" and column_index == 0),
        float(numeric_operator) * row_overlap,
        float(numeric_operator) * header_overlap,
        float(numeric_operator) * year_match,
    )
    assert len(values) == len(TATQA_SLOT_FEATURE_NAMES)
    return tuple(float(value) for value in values)


def tatqa_slot_heuristic_score(
    question: str,
    table: list[list[str]],
    row_index: int,
    column_index: int,
) -> float:
    """Return the frozen label-free score used by the initial slot selector."""

    features = tatqa_slot_feature_vector(
        question, table, row_index, column_index
    )
    row_overlap, header_overlap, value_overlap, year_match = features[:4]
    score = 2.5 * row_overlap + 2.0 * header_overlap + 0.5 * value_overlap
    if year_match:
        score += 3.0
    if row_index == 0:
        score = 2.5 * header_overlap + 1.5 * value_overlap
    if column_index == 0:
        score += 0.4 * row_overlap
    if features[6]:
        score += 1.0
    return score


def select_tatqa_table_slots(
    question: str,
    table: list[list[str]],
    *,
    budget: int = 10,
) -> TATQASlotPlan:
    """Select query-relevant cells using only table structure and query text."""

    if not question.strip():
        raise ValueError("question must not be empty")
    if budget <= 0:
        raise ValueError("slot budget must be positive")
    if not table or any(not isinstance(row, list) for row in table):
        raise ValueError("table must contain at least one row")
    width = max((len(row) for row in table), default=0)
    if width == 0:
        raise ValueError("table must contain at least one cell")

    query_terms = _terms(question)
    years = tuple(dict.fromkeys(_YEAR_RE.findall(question)))
    year_set = set(years)
    normalized_question = unicodedata.normalize("NFKC", question).casefold()
    header = [str(value) for value in table[0]] + [""] * (width - len(table[0]))
    candidates: list[TATQATableSlot] = []
    for row_index, raw_row in enumerate(table):
        row = [str(value) for value in raw_row] + [""] * (width - len(raw_row))
        row_label = row[0] if row else ""
        row_overlap = _overlap(query_terms, row_label)
        for column_index, value in enumerate(row):
            if not value.strip():
                continue
            column_header = header[column_index]
            header_overlap = _overlap(query_terms, column_header)
            value_overlap = _overlap(query_terms, value)
            score = 2.5 * row_overlap + 2.0 * header_overlap + 0.5 * value_overlap
            signals: list[str] = []
            if row_overlap:
                signals.append("row_label_overlap")
            if header_overlap:
                signals.append("column_header_overlap")
            if value_overlap:
                signals.append("cell_value_overlap")
            if year_set.intersection(_YEAR_RE.findall(column_header)):
                score += 3.0
                signals.append("query_year_header_match")
            if row_index == 0:
                score = 2.5 * header_overlap + 1.5 * value_overlap
                signals.append("header_coordinate")
            if column_index == 0:
                score += 0.4 * row_overlap
                signals.append("row_label_coordinate")
            if value.strip().casefold() in normalized_question:
                score += 1.0
                signals.append("exact_value_mention")
            candidates.append(
                TATQATableSlot(
                    row_index=row_index,
                    column_index=column_index,
                    value=value,
                    row_label=row_label,
                    column_header=column_header,
                    score=score,
                    signals=tuple(signals),
                )
            )
    candidates.sort(
        key=lambda slot: (-slot.score, slot.row_index, slot.column_index)
    )
    return TATQASlotPlan(
        operator=classify_tatqa_operator(question),
        query_terms=tuple(dict.fromkeys(query_terms)),
        years=years,
        slots=tuple(candidates[:budget]),
    )


def render_tatqa_slot_context(plan: TATQASlotPlan) -> str:
    """Render selected slots ahead of a full table without changing evidence IDs."""

    lines = [
        "TAT-QA query slot plan (label-free; verify against the full evidence):",
        f"operator: {plan.operator}",
        "selected_cells:",
    ]
    for slot in plan.slots:
        lines.append(
            "- "
            f"row_index={slot.row_index} column_index={slot.column_index} "
            f"row_label={slot.row_label!r} column_header={slot.column_header!r} "
            f"value={slot.value!r}"
        )
    return "\n".join(lines)

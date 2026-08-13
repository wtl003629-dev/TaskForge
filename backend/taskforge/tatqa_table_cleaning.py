"""Coordinate-preserving table cleaning for the official TAT-QA JSON.

The cleaner builds a search representation and never mutates the raw table.
Every cleaned row records the original row coordinates that produced it so
TagOp/TAT-QA annotations remain comparable after empty rows, repeated headers,
and safe consecutive duplicates are removed.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

TATQA_MISSING_VALUES = frozenset(
    {
        "",
        "-",
        "--",
        "—",
        "–",
        "−",
        "n/a",
        "na",
        "n.a.",
        "nm",
        "n.m.",
        "not applicable",
        "not available",
    }
)

_WHITESPACE = re.compile(r"\s+")
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_NUMBER = re.compile(r"[-+\u2212]?\d[\d,]*(?:\.\d+)?")
_HEADER_SIGNAL = re.compile(
    r"(?:\b(?:19|20)\d{2}\b|\byears?\b|\bmonths?\b|\bquarters?\b|"
    r"\bup to\b|\bmore than\b|\btotal\b|%|january|february|march|april|"
    r"may|june|july|august|september|october|november|december)",
    re.IGNORECASE,
)
_SUPER_HEADER = re.compile(
    r"\b(years? ended|as of|payments? due|december 31|months? ended)\b",
    re.IGNORECASE,
)
_SCALE = re.compile(r"\b(thousands?|millions?|billions?)\b", re.IGNORECASE)
_UNIT_HEADER_LABEL = re.compile(
    r"^\(?\s*(?:in\s+)?(?:thousands?|millions?|billions?|percent|%)\s*\)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TATQATableCleaningAudit:
    original_rows: int
    cleaned_rows: int
    original_cells: int
    normalized_unicode_cells: int
    collapsed_whitespace_cells: int
    empty_rows_removed: int
    repeated_header_rows_removed: int
    consecutive_duplicate_rows_folded: int
    nonconsecutive_duplicate_rows_preserved: int
    header_depth: int
    missing_cells_normalized: int
    numeric_cells_normalized: int
    parenthesized_negative_cells: int
    percent_cells: int
    currency_cells: int
    scale_aware_cells: int

    def as_dict(self) -> dict[str, int]:
        return {
            field: int(getattr(self, field))
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class TATQACleanedTable:
    rows: tuple[tuple[str, ...], ...]
    row_source_indices: tuple[tuple[int, ...], ...]
    headers: tuple[str, ...]
    cell_metadata: tuple[tuple[Mapping[str, Any], ...], ...]
    audit: TATQATableCleaningAudit

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "coordinate_contract": "original_zero_based_rows_and_columns",
            "row_source_indices": [list(indices) for indices in self.row_source_indices],
            "headers": list(self.headers),
            "cell_metadata": [
                [dict(cell) for cell in row]
                for row in self.cell_metadata
            ],
            "audit": self.audit.as_dict(),
        }


def normalize_tatqa_cell(value: object) -> str:
    """Normalize Unicode and whitespace without changing the cell's meaning."""

    normalized = unicodedata.normalize("NFKC", str(value)).replace("\u00a0", " ")
    return _WHITESPACE.sub(" ", normalized).strip()


def _row_key(row: Sequence[str]) -> tuple[str, ...]:
    return tuple(value.casefold() for value in row)


def infer_tatqa_header_depth(rows: Sequence[Sequence[str]]) -> int:
    """Infer a conservative one/two-row financial-table header."""

    if len(rows) < 2:
        return 1
    first = list(rows[0])
    second = list(rows[1])
    first_label = first[0] if first else ""
    second_label = second[0] if second else ""
    second_fields = [value for value in second[1:] if value]
    second_label_is_unit = _UNIT_HEADER_LABEL.fullmatch(second_label) is not None
    if first_label or (second_label and not second_label_is_unit) or not second_fields:
        return 1
    first_super_header = any(
        _SUPER_HEADER.search(value) is not None
        for value in first[1:]
        if value
    )
    signalled = sum(_HEADER_SIGNAL.search(value) is not None for value in second_fields)
    return 2 if first_super_header or signalled >= max(1, len(second_fields) // 2) else 1


def merge_tatqa_headers(
    header_rows: Sequence[Sequence[str]],
    width: int,
) -> tuple[str, ...]:
    """Forward-fill hierarchical super-headers within their column spans."""

    if width <= 0:
        return ()
    padded = [list(row[:width]) + [""] * (width - len(row)) for row in header_rows]
    filled_levels: list[list[str]] = []
    for row in padded:
        filled = [row[0] if row else ""]
        current = ""
        for value in row[1:]:
            if value:
                current = value
            filled.append(current)
        filled_levels.append(filled)
    headers: list[str] = []
    for column_index in range(width):
        pieces: list[str] = []
        for level in filled_levels:
            value = level[column_index]
            if value and value not in pieces:
                pieces.append(value)
        if column_index == 0:
            headers.append(" | ".join(pieces) or "row_label")
        else:
            headers.append(" | ".join(pieces) or f"column_{column_index + 1}")
    return tuple(headers)


def _scale(value: str) -> str | None:
    match = _SCALE.search(value)
    if match is None:
        return None
    return {
        "thousands": "thousand",
        "millions": "million",
        "billions": "billion",
    }.get(match.group(1).casefold(), match.group(1).casefold())


def _numeric_metadata(value: str, header: str, context: str) -> dict[str, Any]:
    missing = value.casefold() in TATQA_MISSING_VALUES
    combined = f"{header} {context} {value}"
    unit_context = f"{header} {value}".casefold()
    scale = _scale(combined)
    unit = "number"
    if "%" in value or "percent" in unit_context:
        unit = "percent"
    elif any(symbol in f"{header} {value}" for symbol in "$\u20ac\u00a3\u00a5"):
        unit = "currency"
    elif "share" in combined.casefold():
        unit = "shares"
    output: dict[str, Any] = {
        "display": "" if missing else value,
        "missing": missing,
        "normalized": None,
        "numeric": None,
        "scaled_numeric": None,
        "sign": "missing" if missing else "non_numeric",
        "unit": unit,
        "scale": scale,
        "years": _YEAR.findall(header),
    }
    if missing:
        return output
    match = _NUMBER.search(value)
    if match is None:
        return output
    token = match.group(0).replace(",", "").replace("−", "-")
    try:
        numeric = float(token)
    except ValueError:
        return output
    stripped = value.strip()
    if stripped.startswith("(") and stripped.endswith(")"):
        numeric = -abs(numeric)
    multiplier = (
        {
            "thousand": 1_000.0,
            "million": 1_000_000.0,
            "billion": 1_000_000_000.0,
        }.get(scale)
        if unit != "percent"
        else None
    )
    output.update(
        {
            "normalized": format(numeric, ".15g"),
            "numeric": numeric,
            "scaled_numeric": numeric * multiplier if multiplier is not None else None,
            "sign": "negative" if numeric < 0 else "positive" if numeric > 0 else "zero",
        }
    )
    return output


def clean_tatqa_table(
    rows: Sequence[Sequence[object]],
    *,
    context_text: str = "",
) -> TATQACleanedTable:
    """Return a compact search table plus immutable raw-coordinate lineage."""

    if not rows:
        raise ValueError("TAT-QA table must contain at least one row")
    if any(isinstance(row, (str, bytes)) or not isinstance(row, Sequence) for row in rows):
        raise ValueError("TAT-QA table rows must be sequences")
    width = max((len(row) for row in rows), default=0)
    if width == 0:
        raise ValueError("TAT-QA table must contain at least one cell")

    normalized_rows: list[list[str]] = []
    unicode_changes = 0
    whitespace_changes = 0
    for row in rows:
        normalized_row: list[str] = []
        for value in row:
            raw = str(value)
            nfkc = unicodedata.normalize("NFKC", raw).replace("\u00a0", " ")
            normalized = normalize_tatqa_cell(value)
            unicode_changes += nfkc != raw
            whitespace_changes += normalized != nfkc.strip()
            normalized_row.append(normalized)
        normalized_rows.append(normalized_row + [""] * (width - len(normalized_row)))

    nonempty: list[tuple[int, list[str]]] = [
        (index, row)
        for index, row in enumerate(normalized_rows)
        if any(row)
    ]
    if not nonempty:
        raise ValueError("TAT-QA table contains no searchable cells")
    header_depth = infer_tatqa_header_depth([row for _, row in nonempty])
    header_items = nonempty[:header_depth]
    header_rows = [row for _, row in header_items]
    headers = merge_tatqa_headers(header_rows, width)
    header_keys = {_row_key(row) for row in header_rows}

    observed: dict[tuple[str, ...], int] = {}
    for _, row in nonempty:
        observed[_row_key(row)] = observed.get(_row_key(row), 0) + 1
    duplicate_total = sum(count - 1 for count in observed.values() if count > 1)

    body: list[tuple[list[int], list[str]]] = []
    repeated_headers = 0
    folded = 0
    for source_index, row in nonempty[header_depth:]:
        key = _row_key(row)
        if key in header_keys:
            repeated_headers += 1
            continue
        if body and _row_key(body[-1][1]) == key:
            body[-1][0].append(source_index)
            folded += 1
            continue
        body.append(([source_index], row))

    compact_rows: list[tuple[str, ...]] = [
        tuple(
            "" if value.casefold() in TATQA_MISSING_VALUES else value
            for value in row
        )
        for _, row in header_items
    ]
    lineage: list[tuple[int, ...]] = [(index,) for index, _ in header_items]
    for source_indices, row in body:
        display = tuple(
            "" if value.casefold() in TATQA_MISSING_VALUES else value
            for value in row
        )
        compact_rows.append(display)
        lineage.append(tuple(source_indices))

    context = normalize_tatqa_cell(context_text)
    metadata_rows: list[tuple[Mapping[str, Any], ...]] = []
    missing_count = 0
    numeric_count = 0
    negative_count = 0
    percent_count = 0
    currency_count = 0
    scale_count = 0
    for clean_row_index, row in enumerate(compact_rows):
        cell_row: list[Mapping[str, Any]] = []
        for column_index, value in enumerate(row):
            metadata = _numeric_metadata(value, headers[column_index], context)
            metadata.update(
                {
                    "clean_row_index": clean_row_index,
                    "column_index": column_index,
                    "source_row_indices": list(lineage[clean_row_index]),
                }
            )
            if clean_row_index >= header_depth:
                missing_count += bool(metadata["missing"])
                numeric_count += metadata["numeric"] is not None
                negative_count += (
                    metadata["sign"] == "negative"
                    and any(
                        normalized_rows[source][column_index].strip().startswith("(")
                        for source in lineage[clean_row_index]
                    )
                )
                percent_count += metadata["unit"] == "percent"
                currency_count += metadata["unit"] == "currency"
                scale_count += metadata["scale"] is not None
            cell_row.append(metadata)
        metadata_rows.append(tuple(cell_row))

    empty_removed = len(rows) - len(nonempty)
    audit = TATQATableCleaningAudit(
        original_rows=len(rows),
        cleaned_rows=len(compact_rows),
        original_cells=sum(len(row) for row in rows),
        normalized_unicode_cells=unicode_changes,
        collapsed_whitespace_cells=whitespace_changes,
        empty_rows_removed=empty_removed,
        repeated_header_rows_removed=repeated_headers,
        consecutive_duplicate_rows_folded=folded,
        nonconsecutive_duplicate_rows_preserved=max(
            0, duplicate_total - repeated_headers - folded
        ),
        header_depth=header_depth,
        missing_cells_normalized=missing_count,
        numeric_cells_normalized=numeric_count,
        parenthesized_negative_cells=negative_count,
        percent_cells=percent_count,
        currency_cells=currency_count,
        scale_aware_cells=scale_count,
    )
    return TATQACleanedTable(
        rows=tuple(compact_rows),
        row_source_indices=tuple(lineage),
        headers=headers,
        cell_metadata=tuple(metadata_rows),
        audit=audit,
    )


__all__ = [
    "TATQA_MISSING_VALUES",
    "TATQACleanedTable",
    "TATQATableCleaningAudit",
    "clean_tatqa_table",
    "infer_tatqa_header_depth",
    "merge_tatqa_headers",
    "normalize_tatqa_cell",
]

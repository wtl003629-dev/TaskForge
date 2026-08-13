"""Audit TaskForge's coordinate-preserving cleaning on a TAT-QA JSON file."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.tatqa_table_cleaning import clean_tatqa_table


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def build_audit(path: Path) -> dict[str, Any]:
    source = path.resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("TAT-QA payload must be a JSON array")
    totals: Counter[str] = Counter()
    tables_changed = 0
    examples: list[dict[str, Any]] = []
    for context in payload:
        if not isinstance(context, Mapping):
            raise ValueError("TAT-QA context must be an object")
        table = context.get("table")
        paragraphs = context.get("paragraphs", [])
        if not isinstance(table, Mapping) or not isinstance(paragraphs, list):
            raise ValueError("TAT-QA context is missing table or paragraphs")
        rows = table.get("table")
        uid = str(table.get("uid", "")).strip()
        if not uid or not isinstance(rows, list):
            raise ValueError("TAT-QA table requires uid and rows")
        paragraph_context = " ".join(
            str(paragraph.get("text", ""))
            for paragraph in paragraphs
            if isinstance(paragraph, Mapping)
        )
        cleaned = clean_tatqa_table(rows, context_text=paragraph_context)
        audit = cleaned.audit.as_dict()
        totals.update(audit)
        structural_changes = (
            audit["empty_rows_removed"]
            + audit["repeated_header_rows_removed"]
            + audit["consecutive_duplicate_rows_folded"]
        )
        representation_changes = (
            structural_changes
            + audit["normalized_unicode_cells"]
            + audit["collapsed_whitespace_cells"]
            + int(audit["header_depth"] > 1)
            + audit["missing_cells_normalized"]
        )
        if representation_changes:
            tables_changed += 1
        if structural_changes and len(examples) < 12:
            examples.append(
                {
                    "table_uid": uid,
                    "original_rows": audit["original_rows"],
                    "cleaned_rows": audit["cleaned_rows"],
                    "row_source_indices": [
                        list(indices) for indices in cleaned.row_source_indices
                    ],
                    "structural_changes": {
                        "empty_rows_removed": audit["empty_rows_removed"],
                        "repeated_header_rows_removed": audit[
                            "repeated_header_rows_removed"
                        ],
                        "consecutive_duplicate_rows_folded": audit[
                            "consecutive_duplicate_rows_folded"
                        ],
                    },
                }
            )
    return {
        "schema_version": "1.0",
        "source": {
            "path": source.as_posix(),
            "sha256": _sha256(source),
            "size_bytes": source.stat().st_size,
        },
        "contract": {
            "label_free": True,
            "raw_rows_mutated": False,
            "coordinates": "original_zero_based_rows_and_columns",
            "global_nonconsecutive_duplicates_removed": False,
            "rules": [
                "NFKC and whitespace normalization",
                "empty-row removal",
                "repeated-header removal",
                "consecutive exact-duplicate folding",
                "hierarchical header forward-fill and merge",
                "missing-value normalization",
                "parenthesized negative, percent, currency, and scale metadata",
            ],
        },
        "tables": len(payload),
        "tables_with_search_representation_changes": tables_changed,
        "totals": dict(sorted(totals.items())),
        "structural_examples": examples,
    }


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--input", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    report = build_audit(args.input)
    _write_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


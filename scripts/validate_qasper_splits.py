"""Fail when QASPER evaluation manifests share papers across split boundaries."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path


def qasper_papers(case_ids: Sequence[object]) -> set[str]:
    papers: set[str] = set()
    for raw in case_ids:
        value = str(raw)
        parts = value.split(":", 2)
        if len(parts) != 3 or parts[0] != "qasper" or not parts[1].strip():
            raise ValueError(f"invalid QASPER case ID: {value!r}")
        papers.add(parts[1].strip())
    return papers


def validate_split_files(paths: Sequence[Path]) -> dict[str, object]:
    if len(paths) < 2:
        raise ValueError("at least two QASPER split files are required")
    papers_by_split: dict[str, set[str]] = {}
    case_counts: dict[str, int] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"split must contain a JSON object: {path}")
        raw_ids = payload.get("case_ids")
        if not isinstance(raw_ids, list):
            raise ValueError(f"split case_ids must be an array: {path}")
        split_id = str(payload.get("split_id") or path.stem)
        if split_id in papers_by_split:
            raise ValueError(f"duplicate split ID: {split_id}")
        papers_by_split[split_id] = qasper_papers(raw_ids)
        case_counts[split_id] = len(raw_ids)
    overlaps: list[dict[str, object]] = []
    split_ids = sorted(papers_by_split)
    for index, left in enumerate(split_ids):
        for right in split_ids[index + 1 :]:
            shared = sorted(papers_by_split[left] & papers_by_split[right])
            if shared:
                overlaps.append(
                    {"left": left, "right": right, "paper_ids": shared}
                )
    return {
        "valid": not overlaps,
        "splits": {
            split_id: {
                "case_count": case_counts[split_id],
                "paper_count": len(papers_by_split[split_id]),
            }
            for split_id in split_ids
        },
        "overlaps": overlaps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("splits", nargs="+", type=Path)
    args = parser.parse_args()
    report = validate_split_files(args.splits)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

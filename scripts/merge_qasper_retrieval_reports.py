"""Merge deterministic, non-overlapping QASPER retrieval shards.

The direct-upload evaluator is intentionally shardable so a long local run can
be resumed without re-parsing or re-ranking completed cases.  This utility
merges only reports with identical frozen inputs and recomputes aggregate
metrics from rows; it never averages already-averaged shard metrics.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RECALL_KS = (1, 5, 10, 50)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row.get(key) or 0.0) for row in rows) if rows else 0.0


def _mean_nested(rows: list[dict[str, Any]], parent: str, key: int) -> float:
    return statistics.fmean(
        float((row.get(parent) or {}).get(str(key)) or 0.0) for row in rows
    ) if rows else 0.0


def _merge_alignment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("total_units", "exact_units", "fuzzy_units", "ambiguous_units", "unaligned_units")
    totals = {key: sum(int(row["alignment"][key]) for row in rows) for key in keys}
    total = totals["total_units"]
    totals["alignment_coverage"] = (
        (totals["exact_units"] + totals["fuzzy_units"]) / total if total else 0.0
    )
    totals["fully_aligned_cases"] = sum(bool(row.get("alignment_eligible")) for row in rows)
    totals["alignment_eligible_case_ratio"] = (
        totals["fully_aligned_cases"] / len(rows) if rows else 0.0
    )
    by_type: dict[str, dict[str, int | float]] = {}
    for row in rows:
        for kind, raw in (row.get("alignment_by_evidence_type") or {}).items():
            values = by_type.setdefault(
                str(kind),
                {"total": 0, "exact": 0, "fuzzy": 0, "ambiguous": 0, "unaligned": 0},
            )
            for key in ("total", "exact", "fuzzy", "ambiguous", "unaligned"):
                values[key] = int(values[key]) + int(raw.get(key) or 0)
    for values in by_type.values():
        total = int(values["total"])
        values["alignment_coverage"] = (
            (int(values["exact"]) + int(values["fuzzy"])) / total if total else 0.0
        )
    totals["by_evidence_type"] = by_type
    return totals


def merge(paths: list[Path], output: Path) -> dict[str, Any]:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not reports:
        raise ValueError("at least one shard is required")
    frozen_keys = (
        "schema_version",
        "evaluation_type",
        "benchmark_track",
        "dataset",
        "source_dataset_sha256",
        "split_sha256",
        "pdf_manifest",
        "retrieval",
        "parser",
    )
    for report in reports[1:]:
        for key in frozen_keys:
            left = report.get(key)
            right = reports[0].get(key)
            if key == "pdf_manifest" and isinstance(left, dict) and isinstance(right, dict):
                left = {name: left.get(name) for name in ("path", "sha256", "cohort_id")}
                right = {name: right.get(name) for name in ("path", "sha256", "cohort_id")}
            if key == "retrieval" and isinstance(left, dict) and isinstance(right, dict):
                left = {name: value for name, value in left.items() if name != "retrieval_route_counts"}
                right = {name: value for name, value in right.items() if name != "retrieval_route_counts"}
            if left != right:
                raise ValueError(f"shard frozen input mismatch at {key}")
    rows = [row for report in reports for row in report.get("rows", [])]
    offsets = [int(report.get("case_offset", 0)) for report in reports]
    if len(rows) != len({str(row["case_id"]) for row in rows}):
        raise ValueError("shards contain duplicate case IDs")
    rows.sort(key=lambda row: str(row["case_id"]))
    alignment = _merge_alignment(rows)
    gate = reports[0]["alignment_gate"]
    gate_passed = (
        alignment["alignment_coverage"] >= float(gate["minimum_unit_coverage"])
        and alignment["alignment_eligible_case_ratio"]
        >= float(gate["minimum_eligible_case_ratio"])
    )
    latencies = [float(row.get("latency_ms") or 0.0) for row in rows]
    metrics = {f"recall_at_{k}": _mean_nested(rows, "recall_at_k", k) for k in RECALL_KS}
    metrics.update(
        {
            "p50_ms": sorted(latencies)[round((len(latencies) - 1) * 0.50)] if latencies else 0.0,
            "p95_ms": sorted(latencies)[round((len(latencies) - 1) * 0.95)] if latencies else 0.0,
        }
    )
    candidate = {f"recall_at_{k}": _mean_nested(rows, "candidate_child_recall_at_k", k) for k in RECALL_KS}
    ingestion_by_paper = {
        str(item["paper_id"]): item
        for report in reports
        for item in report.get("ingestion", [])
    }
    manifest_by_paper = {
        str(item["paper_id"]): item
        for report in reports
        for item in (report.get("pdf_manifest") or {}).get("papers", [])
    }
    route_counts = Counter(str(row.get("retrieval_route") or "english") for row in rows)
    parser_diag = {
        "papers": len(ingestion_by_paper),
        "failed_papers": sum(item.get("status") == "failed" for item in ingestion_by_paper.values()),
        "parser_failure_rate": (
            sum(item.get("status") == "failed" for item in ingestion_by_paper.values())
            / len(ingestion_by_paper) if ingestion_by_paper else 0.0
        ),
        "ocr_used_papers": sum(bool(item.get("parse_quality", {}).get("ocr_used")) for item in ingestion_by_paper.values()),
        "visual_pending_blocks": sum(int(item.get("parse_quality", {}).get("visual_unparsed_count") or 0) for item in ingestion_by_paper.values()),
        "mean_page_coverage": statistics.fmean(float(item.get("parse_quality", {}).get("text_coverage") or 0.0) for item in ingestion_by_paper.values()) if ingestion_by_paper else 0.0,
        "pdf_bytes": sum(int(item.get("pdf_bytes") or 0) for item in ingestion_by_paper.values()),
        "indexed_characters": sum(int(item.get("indexed_characters") or 0) for item in ingestion_by_paper.values()),
    }
    report = dict(reports[0])
    report.update(
        {
            "created_at": datetime.now(UTC).isoformat(),
            "status": "complete" if gate_passed else "alignment_gate_failed",
            "cases": len(rows),
            "papers": len(ingestion_by_paper),
            "case_offset": min(offsets),
            "metrics": metrics if gate_passed else {**{f"recall_at_{k}": None for k in RECALL_KS}, "p50_ms": metrics["p50_ms"], "p95_ms": metrics["p95_ms"]},
            "candidate_child_metrics": {**candidate, "definition": reports[0]["candidate_child_metrics"]["definition"]},
            "agent_visible_metrics": {"recall_at_8": _mean(rows, "agent_visible_recall_at_8")},
            "reranked_visible_metrics": {"recall_at_8": _mean(rows, "reranked_visible_recall_at_8")},
            "diagnostic_lower_bound_metrics": metrics,
            "conditional_retrieval_metrics": {
                f"recall_at_{k}": statistics.fmean(
                    float(row["recall_at_k"][str(k)])
                    for row in rows if bool(row.get("alignment_eligible"))
                ) if any(bool(row.get("alignment_eligible")) for row in rows) else 0.0
                for k in RECALL_KS
            },
            "alignment_diagnostics": alignment,
            "retrieval": {
                **reports[0]["retrieval"],
                "retrieval_route_counts": dict(route_counts),
            },
            "pdf_manifest": {
                **(reports[0].get("pdf_manifest") or {}),
                "papers": list(manifest_by_paper.values()),
            },
            "ingestion": list(ingestion_by_paper.values()),
            "parser_diagnostics": parser_diag,
            "rows": rows,
            "elapsed_ms": sum(float(report.get("elapsed_ms") or 0.0) for report in reports),
        }
    )
    report["limitations"] = [
        item for item in report.get("limitations", [])
        if "shard" not in str(item).casefold()
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("shards", type=Path, nargs="+")
    args = parser.parse_args()
    report = merge(args.shards, args.output)
    print(json.dumps({"output": str(args.output), "cases": report["cases"], "metrics": report["metrics"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()

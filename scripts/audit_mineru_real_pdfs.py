"""Audit MinerU parsing and Parent–Child projection on checksum-pinned PDFs.

This is a parser-development audit, not a retrieval benchmark. It checkpoints
after every paper so a long GPU run can be resumed from MinerU's SHA cache.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from taskforge.pdf_parsing.hierarchy import build_parent_child_units
from taskforge.pdf_parsing.mineru_client import MinerUClient
from taskforge.pdf_parsing.quality_gate import ParseQualityPolicy

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    succeeded = [row for row in rows if row["status"] == "success"]
    failed = [row for row in rows if row["status"] == "failed"]
    latencies = [float(row["latency_ms"]) for row in succeeded]
    return {
        "paper_count": len(rows),
        "success_count": len(succeeded),
        "failure_count": len(failed),
        "quality_statuses": dict(
            Counter(str(row["quality"]["status"]) for row in succeeded)
        ),
        "total_pages": sum(int(row["page_count"]) for row in succeeded),
        "total_blocks": sum(int(row["block_count"]) for row in succeeded),
        "total_parents": sum(int(row["parent_count"]) for row in succeeded),
        "total_children": sum(int(row["child_count"]) for row in succeeded),
        "total_unparsed_visuals": sum(
            int(row["quality"]["visual_unparsed_count"]) for row in succeeded
        ),
        "ocr_paper_count": sum(bool(row["quality"]["ocr_used"]) for row in succeeded),
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies, default=None),
        },
    }


async def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requested = [value for value in args.paper_ids.split(",") if value]
    papers = {
        str(item["paper_id"]): item
        for item in manifest.get("papers", [])
        if isinstance(item, dict) and item.get("paper_id")
    }
    unknown = [paper_id for paper_id in requested if paper_id not in papers]
    if unknown:
        raise ValueError(f"paper IDs absent from manifest: {unknown}")
    selected = requested or list(papers)
    policy = ParseQualityPolicy(
        minimum_text_coverage=args.minimum_text_coverage,
        maximum_garbled_character_ratio=args.maximum_garbled_character_ratio,
        maximum_repeated_header_ratio=args.maximum_repeated_header_ratio,
    )
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "mineru_real_pdf_parse_audit",
        "status": "in_progress",
        "started_at": datetime.now(UTC).isoformat(),
        "completed_at": None,
        "manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
            "cohort_id": manifest.get("cohort_id"),
        },
        "configuration": {
            "base_url": args.mineru_base_url,
            "expected_version": args.mineru_version,
            "backend": args.mineru_backend,
            "parse_method": args.parse_method,
            "effort": args.effort,
            "quality_policy": {
                "minimum_text_coverage": policy.minimum_text_coverage,
                "maximum_garbled_character_ratio": policy.maximum_garbled_character_ratio,
                "maximum_repeated_header_ratio": policy.maximum_repeated_header_ratio,
            },
            "parent_target_tokens": args.parent_target_tokens,
            "parent_max_tokens": args.parent_max_tokens,
            "child_target_tokens": args.child_target_tokens,
            "child_max_tokens": args.child_max_tokens,
            "child_overlap_tokens": args.child_overlap_tokens,
        },
        "selected_paper_ids": selected,
        "papers": [],
        "summary": {},
    }
    output = args.output.resolve()
    client = MinerUClient(
        args.mineru_base_url,
        args.cache_root.resolve(),
        backend=args.mineru_backend,
        parse_method=args.parse_method,
        effort=args.effort,
        expected_version=args.mineru_version,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        concurrency=1,
        quality_policy=policy,
    )
    try:
        for position, paper_id in enumerate(selected, 1):
            item = papers[paper_id]
            pdf_path = (manifest_path.parent / str(item["path"])).resolve(strict=True)
            actual_sha = _sha256(pdf_path)
            if actual_sha != item.get("sha256"):
                raise ValueError(f"PDF checksum mismatch for {paper_id}")
            cache_hit = any(
                args.cache_root.resolve().glob(f"{actual_sha}.*.mineru.json")
            )
            started = perf_counter()
            try:
                document = await client.parse(
                    pdf_path,
                    source_uri=f"qasper://{paper_id}",
                )
                units = build_parent_child_units(
                    document,
                    parent_target_tokens=args.parent_target_tokens,
                    parent_max_tokens=args.parent_max_tokens,
                    child_target_tokens=args.child_target_tokens,
                    child_max_tokens=args.child_max_tokens,
                    child_overlap_tokens=args.child_overlap_tokens,
                )
                parents = [unit for unit in units if unit.role == "parent"]
                children = [unit for unit in units if unit.role == "child"]
                headings = [
                    {"level": block.heading_level, "text": block.text}
                    for block in document.blocks
                    if block.block_type == "title"
                ]
                row: dict[str, Any] = {
                    "paper_id": paper_id,
                    "position": position,
                    "status": "success",
                    "path": str(pdf_path),
                    "sha256": actual_sha,
                    "bytes": pdf_path.stat().st_size,
                    "page_count": document.page_count,
                    "parser": document.parser,
                    "parser_version": document.parser_version,
                    "parser_backend": document.parser_backend,
                    "cache_hit": cache_hit,
                    "latency_ms": (perf_counter() - started) * 1000,
                    "quality": document.quality.model_dump(mode="json"),
                    "block_count": len(document.blocks),
                    "block_types": dict(
                        Counter(block.block_type for block in document.blocks)
                    ),
                    "heading_count": len(headings),
                    "heading_level_counts": dict(
                        Counter(str(item["level"]) for item in headings)
                    ),
                    "heading_sample": headings[:12],
                    "image_artifact_count": sum(
                        bool(block.image_artifact_id) for block in document.blocks
                    ),
                    "parent_count": len(parents),
                    "child_count": len(children),
                    "oversized_atomic_child_count": sum(
                        unit.oversized_atomic for unit in children
                    ),
                    "max_child_characters": max(
                        (len(unit.text) for unit in children), default=0
                    ),
                    "raw_output_artifact": document.raw_output_artifact,
                }
            except Exception as exc:  # keep failures visible in the audit
                row = {
                    "paper_id": paper_id,
                    "position": position,
                    "status": "failed",
                    "path": str(pdf_path),
                    "sha256": actual_sha,
                    "bytes": pdf_path.stat().st_size,
                    "cache_hit": cache_hit,
                    "latency_ms": (perf_counter() - started) * 1000,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            report["papers"].append(row)
            report["summary"] = _summary(report["papers"])
            _write_report(output, report)
            print(
                f"[{position}/{len(selected)}] {paper_id} {row['status']} "
                f"{row['latency_ms'] / 1000:.1f}s",
                flush=True,
            )
    finally:
        await client.aclose()
    report["status"] = "complete" if not report["summary"]["failure_count"] else "complete_with_failures"
    report["completed_at"] = datetime.now(UTC).isoformat()
    _write_report(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--paper-ids", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=PROJECT_ROOT / ".taskforge" / "eval-cache" / "mineru-real-pdf-audit",
    )
    parser.add_argument("--mineru-base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--mineru-version", default="3.4.4")
    parser.add_argument("--mineru-backend", default="pipeline")
    parser.add_argument("--parse-method", choices=("auto", "txt", "ocr"), default="auto")
    parser.add_argument("--effort", choices=("medium", "high"), default="high")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--minimum-text-coverage", type=float, default=0.80)
    parser.add_argument("--maximum-garbled-character-ratio", type=float, default=0.03)
    parser.add_argument("--maximum-repeated-header-ratio", type=float, default=0.20)
    parser.add_argument("--parent-target-tokens", type=int, default=2_000)
    parser.add_argument("--parent-max-tokens", type=int, default=3_000)
    parser.add_argument("--child-target-tokens", type=int, default=400)
    parser.add_argument("--child-max-tokens", type=int, default=500)
    parser.add_argument("--child-overlap-tokens", type=int, default=60)
    args = parser.parse_args()
    report = asyncio.run(audit(args))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

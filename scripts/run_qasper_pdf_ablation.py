"""Run the frozen A0-A7 real-PDF retrieval ablation matrix.

This orchestrator never generates queries or downloads PDFs during a scored
run. The caller must supply a checksum-pinned PDF manifest and, from A3 onward,
a previously frozen query-variant manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class AblationSpec:
    ablation_id: str
    parser: str
    chunking: str
    query_mode: str
    rerank: bool
    operator_budget: int
    visual: bool
    change: str


ABLATIONS = (
    AblationSpec("A0", "native", "flat", "original", False, 0, False, "control"),
    AblationSpec("A1", "mineru", "flat", "original", False, 0, False, "MinerU"),
    AblationSpec(
        "A2",
        "mineru",
        "parent_child",
        "original",
        False,
        0,
        False,
        "Parent-Child",
    ),
    AblationSpec(
        "A3",
        "mineru",
        "parent_child",
        "synonym",
        False,
        0,
        False,
        "+ constrained synonym query",
    ),
    AblationSpec(
        "A4",
        "mineru",
        "parent_child",
        "full",
        False,
        0,
        False,
        "+ keyword/entity query",
    ),
    AblationSpec(
        "A5",
        "mineru",
        "parent_child",
        "full",
        True,
        0,
        False,
        "+ full Candidate@50 Cross-Encoder",
    ),
    AblationSpec(
        "A6",
        "mineru",
        "parent_child",
        "full",
        True,
        2,
        False,
        "+ one directed supplementary round",
    ),
    AblationSpec(
        "A7",
        "mineru",
        "parent_child",
        "full",
        True,
        2,
        True,
        "+ separate visual extractor",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_command(
    spec: AblationSpec,
    *,
    dataset: Path,
    split: Path,
    pdf_manifest: Path,
    query_variants: Path | None,
    mineru_base_url: str,
    mineru_version: str,
    mineru_cache_root: Path,
    reranker_backend: str,
    reranker_model: str | None,
    output: Path,
    limit: int,
    confirm_visual_calls: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_qasper_direct_upload.py"),
        "--dataset",
        str(dataset),
        "--split",
        str(split),
        "--pdf-manifest",
        str(pdf_manifest),
        "--output",
        str(output),
        "--limit",
        str(limit),
        "--backend",
        "fastembed",
        "--candidate-k",
        "50",
        "--no-graph",
        "--pdf-parser-backend",
        spec.parser,
        "--pdf-chunking-mode",
        spec.chunking,
        "--query-expansion-mode",
        spec.query_mode,
        "--operator-budget",
        str(spec.operator_budget),
    ]
    if spec.parser == "mineru":
        command.extend(
            [
                "--mineru-base-url",
                mineru_base_url,
                "--mineru-expected-version",
                mineru_version,
                "--mineru-cache-root",
                str(mineru_cache_root),
            ]
        )
    if spec.query_mode != "original":
        if query_variants is None:
            raise ValueError(f"{spec.ablation_id} requires --query-variants")
        command.extend(["--query-variants", str(query_variants)])
    if spec.rerank:
        if not reranker_model:
            raise ValueError(f"{spec.ablation_id} requires --reranker-model")
        command.extend(
            [
                "--reranker-backend",
                reranker_backend,
                "--reranker-model",
                reranker_model,
            ]
        )
    if spec.visual:
        if not confirm_visual_calls:
            raise ValueError("A7 requires --confirm-visual-calls")
        command.extend(["--enable-visual-extractor", "--confirm-visual-calls"])
    return command


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset", type=Path, required=True)
    value.add_argument("--split", type=Path, required=True)
    value.add_argument("--pdf-manifest", type=Path, required=True)
    value.add_argument("--query-variants", type=Path, default=None)
    value.add_argument("--mineru-base-url", default="http://127.0.0.1:8001")
    value.add_argument("--mineru-version", default="3.4.4")
    value.add_argument(
        "--mineru-cache-root",
        type=Path,
        default=PROJECT_ROOT / ".taskforge" / "eval-cache" / "mineru-shared-v1",
    )
    value.add_argument(
        "--reranker-backend",
        choices=("fastembed", "fastembed_ensemble", "flagembedding", "transformers"),
        default="fastembed_ensemble",
    )
    value.add_argument("--reranker-model", default=None)
    value.add_argument(
        "--only",
        default="A0,A1,A2,A3,A4,A5,A6",
        help="Comma-separated ablations. A7 is opt-in because it may be billable.",
    )
    value.add_argument("--limit", type=int, default=50)
    value.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / ".taskforge" / "eval-runs" / "qasper-pdf-ablation",
    )
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--confirm-visual-calls", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    selected_ids = [value.strip().upper() for value in args.only.split(",") if value.strip()]
    selected = [spec for spec in ABLATIONS if spec.ablation_id in selected_ids]
    if not selected or set(selected_ids) != {spec.ablation_id for spec in selected}:
        raise SystemExit("--only must contain unique values from A0 through A7")
    if not 1 <= args.limit <= 100:
        raise SystemExit("--limit must be between 1 and 100")
    required_paths = [args.dataset, args.split, args.pdf_manifest]
    if any(spec.query_mode != "original" for spec in selected):
        if args.query_variants is None:
            raise SystemExit("A3-A7 require --query-variants")
        required_paths.append(args.query_variants)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"required frozen inputs are missing: {missing}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []
    for spec in selected:
        output = output_dir / f"{spec.ablation_id.lower()}.json"
        try:
            command = build_command(
                spec,
                dataset=args.dataset.resolve(),
                split=args.split.resolve(),
                pdf_manifest=args.pdf_manifest.resolve(),
                query_variants=(
                    args.query_variants.resolve() if args.query_variants else None
                ),
                mineru_base_url=args.mineru_base_url,
                mineru_version=args.mineru_version,
                mineru_cache_root=args.mineru_cache_root.resolve(),
                reranker_backend=args.reranker_backend,
                reranker_model=args.reranker_model,
                output=output,
                limit=args.limit,
                confirm_visual_calls=args.confirm_visual_calls,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        run_record: dict[str, object] = {
            "spec": asdict(spec),
            "output": str(output),
            "command": command,
            "status": "planned" if args.dry_run else "running",
        }
        runs.append(run_record)
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        run_record["return_code"] = completed.returncode
        run_record["status"] = "complete" if completed.returncode == 0 else "failed"
        if completed.returncode != 0:
            break
        report = json.loads(output.read_text(encoding="utf-8"))
        run_record["report_status"] = report.get("status")
        run_record["metrics"] = report.get("metrics")
        run_record["alignment_gate"] = report.get("alignment_gate")

    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "status": (
            "planned"
            if args.dry_run
            else "complete"
            if len(runs) == len(selected)
            and all(run.get("status") == "complete" for run in runs)
            else "failed"
        ),
        "frozen_inputs": {
            "dataset": str(args.dataset.resolve()),
            "dataset_sha256": _sha256(args.dataset),
            "split": str(args.split.resolve()),
            "split_sha256": _sha256(args.split),
            "pdf_manifest": str(args.pdf_manifest.resolve()),
            "pdf_manifest_sha256": _sha256(args.pdf_manifest),
            "query_variants": str(args.query_variants.resolve())
            if args.query_variants
            else None,
            "query_variants_sha256": _sha256(args.query_variants)
            if args.query_variants
            else None,
        },
        "runs": runs,
    }
    manifest_path = output_dir / "ablation-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest_path), "status": manifest["status"]}))
    return 0 if manifest["status"] in {"planned", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

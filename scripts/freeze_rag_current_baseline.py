"""Freeze the immutable current-chain RAG baseline manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from taskforge.config import Settings  # noqa: E402
from taskforge.rag_experiment_profile import (  # noqa: E402
    resolve_rag_experiment_profile,
)

DEFAULT_REPORT = (
    PROJECT_ROOT
    / "eval"
    / "reports"
    / "qasper-real-pdf-locked100-current-original-parent-child-v2-top8.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "eval" / "baselines" / "rag-current-a-v1.json"
DEFAULT_REPRODUCTION_REPORT = (
    PROJECT_ROOT
    / "eval"
    / "reports"
    / "rag-profile-ab-screen20-final-v2"
    / "rag-profile-a.json"
)
DEFAULT_ANSWER_REPORT = (
    PROJECT_ROOT
    / "eval"
    / "reports"
    / "qasper-answer-e2e-live-clean-holdout-100-v1.json"
)
SOURCE_FILES = (
    "backend/taskforge/config.py",
    "backend/taskforge/literature/ingestion.py",
    "backend/taskforge/pdf_parsing/hierarchy.py",
    "backend/taskforge/research_retrieval.py",
    "backend/taskforge/rag_experiment_profile.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def freeze(
    report_path: Path,
    reproduction_report_path: Path,
    answer_report_path: Path,
    output_path: Path,
) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or report.get("cases") != 100:
        raise ValueError("current baseline must be a complete locked 100-case report")
    # This manifest intentionally inspects the local SQLite control fixtures;
    # make the compatibility backend explicit now that PostgreSQL is default.
    settings = Settings(_env_file=None, database_backend="sqlite")
    profile = resolve_rag_experiment_profile("current")
    reproduction = json.loads(
        reproduction_report_path.read_text(encoding="utf-8")
    )
    if (
        reproduction.get("status") != "complete"
        or reproduction.get("rag_profile") != {"name": "current", "ablation": "a"}
    ):
        raise ValueError("reproduction report must be a complete current/A run")
    answer_report = json.loads(answer_report_path.read_text(encoding="utf-8"))
    dirty_paths = [
        line[3:]
        for line in _git("status", "--short").splitlines()
        if len(line) > 3
    ]
    manifest: dict[str, object] = {
        "schema_version": "taskforge.rag_baseline.v1",
        "status": "frozen",
        "generated_at": datetime.now(UTC).isoformat(),
        "profile": {
            "name": profile.name,
            "ablation": profile.ablation,
            "label": profile.label,
            "retrieval_text_enabled": profile.retrieval_text_enabled,
            "parent_aware_rerank_enabled": profile.parent_aware_rerank_enabled,
            "lineage_diversity_enabled": profile.lineage_diversity_enabled,
            "structure_aware_chunking_enabled": (
                profile.structure_aware_chunking_enabled
            ),
            "document_identity_template": "research-paper:{scope_id}:{paper_id}",
            "knowledge_base_identity_template": (
                "research-scope:{scope_id}:v{scope_version}"
            ),
        },
        "runtime_defaults": {
            "rag_active_profile": settings.rag_active_profile,
            "rag_experiment_profile": settings.rag_experiment_profile,
            "pdf_chunking_mode": settings.pdf_chunking_mode,
            "pdf_parent_target_tokens": settings.pdf_parent_target_tokens,
            "pdf_parent_max_tokens": settings.pdf_parent_max_tokens,
            "pdf_child_target_tokens": settings.pdf_child_target_tokens,
            "pdf_child_max_tokens": settings.pdf_child_max_tokens,
            "pdf_child_overlap_tokens": settings.pdf_child_overlap_tokens,
            "research_reranker_backend": settings.research_reranker_backend,
            "research_reranker_model": settings.research_reranker_model,
            "research_parent_aware_configured": (
                settings.research_parent_aware_rerank_enabled
            ),
            "research_parent_aware_effective": (
                profile.parent_aware_rerank_enabled
            ),
            "research_lineage_diversity_configured": (
                settings.research_lineage_diversity_enabled
            ),
            "research_lineage_diversity_effective": (
                profile.lineage_diversity_enabled
            ),
            "candidate_k": reproduction.get("retrieval", {}).get("candidate_k"),
            "top_k": reproduction.get("retrieval", {}).get("agent_visible_k"),
            "mineru_expected_version": reproduction.get("parser", {}).get(
                "mineru_expected_version"
            ),
            "mineru_backend": settings.mineru_backend,
            "mineru_parse_method": settings.mineru_parse_method,
            "mineru_effort": settings.mineru_effort,
            "python_packages": {
                "fastembed": _package_version("fastembed"),
                "pydantic": _package_version("pydantic"),
            },
        },
        "code": {
            "git_commit": _git("rev-parse", "HEAD"),
            "working_tree_dirty": bool(dirty_paths),
            "dirty_paths": dirty_paths,
            "source_sha256": {
                relative: _sha256(PROJECT_ROOT / relative)
                for relative in SOURCE_FILES
            },
        },
        "locked_inputs": {
            "dataset": report.get("source_dataset"),
            "dataset_sha256": report.get("source_dataset_sha256"),
            "split": report.get("split"),
            "split_sha256": report.get("split_sha256"),
            "pdf_manifest": report.get("pdf_manifest", {}).get("path"),
            "pdf_manifest_sha256": report.get("pdf_manifest", {}).get("sha256"),
        },
        "baseline_report": {
            "path": str(report_path.resolve()),
            "sha256": _sha256(report_path),
            "created_at": report.get("created_at"),
            "cases": report.get("cases"),
            "retrieval": report.get("retrieval"),
            "parser": report.get("parser"),
            "metrics": report.get("metrics"),
            "agent_visible_metrics": report.get("agent_visible_metrics"),
            "alignment_gate": report.get("alignment_gate"),
        },
        "reproduction_report": {
            "path": str(reproduction_report_path.resolve()),
            "sha256": _sha256(reproduction_report_path),
            "created_at": reproduction.get("created_at"),
            "cases": reproduction.get("cases"),
            "state": reproduction.get("state"),
            "retrieval": reproduction.get("retrieval"),
            "parser": reproduction.get("parser"),
            "metrics": reproduction.get("metrics"),
            "agent_visible_metrics": reproduction.get("agent_visible_metrics"),
            "citation_metrics": reproduction.get("citation_metrics"),
            "alignment_gate": reproduction.get("alignment_gate"),
            "matches_historical_screen20": (
                math.isclose(
                    float(reproduction.get("metrics", {}).get("recall_at_1")),
                    0.12666666666666665,
                )
                and math.isclose(
                    float(reproduction.get("metrics", {}).get("recall_at_5")),
                    0.6566666666666666,
                )
                and math.isclose(
                    float(reproduction.get("metrics", {}).get("recall_at_10")),
                    0.79,
                )
                and math.isclose(
                    float(reproduction.get("metrics", {}).get("recall_at_50")),
                    0.9400000000000001,
                )
            ),
        },
        "answer_quality_anchor": {
            "path": str(answer_report_path.resolve()),
            "sha256": _sha256(answer_report_path),
            "created_at": answer_report.get("created_at"),
            "model": answer_report.get("model"),
            "metrics": answer_report.get("metrics"),
            "limitations": answer_report.get("limitations"),
            "note": (
                "Historical locked-100 answer anchor; a paired optimized answer "
                "run is intentionally skipped while the retrieval gate is No-Go."
            ),
        },
        "metric_availability": {
            "recall_at_1_5_10_50": True,
            "agent_visible_recall_at_8": True,
            "mrr": True,
            "ndcg_at_8": True,
            "citation_metrics": True,
            "answer_quality": True,
            "note": (
                "MRR/NDCG/citation metrics are from the exact screen20 "
                "reproduction; answer quality is the locked-100 historical anchor."
            ),
        },
        "claim": (
            "This manifest freezes current behavior only. It does not claim that "
            "the optimized profile improves retrieval or answer quality."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--reproduction-report",
        type=Path,
        default=DEFAULT_REPRODUCTION_REPORT,
    )
    parser.add_argument(
        "--answer-report",
        type=Path,
        default=DEFAULT_ANSWER_REPORT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = freeze(
        args.report.resolve(),
        args.reproduction_report.resolve(),
        args.answer_report.resolve(),
        args.output.resolve(),
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "status": manifest["status"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

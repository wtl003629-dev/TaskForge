"""Run TaskForge current-vs-optimized A-E RAG evaluation and apply gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_qasper_direct_upload import run  # noqa: E402

STAGES = ("a", "b", "c", "d", "e")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paired_ci(
    control: Sequence[float],
    experiment: Sequence[float],
    *,
    samples: int = 10_000,
    seed: int = 20260826,
) -> dict[str, float]:
    if len(control) != len(experiment) or not control:
        raise ValueError("paired metric vectors must have the same non-zero length")
    differences = [right - left for left, right in zip(control, experiment, strict=True)]
    generator = random.Random(seed)
    bootstrapped = sorted(
        sum(generator.choice(differences) for _ in differences) / len(differences)
        for _ in range(samples)
    )
    return {
        "mean_delta": sum(differences) / len(differences),
        "ci95_low": bootstrapped[int(samples * 0.025)],
        "ci95_high": bootstrapped[min(samples - 1, int(samples * 0.975))],
    }


def _metric_rows(report: dict[str, Any], key: str) -> list[float]:
    return [float(row.get(key) or 0.0) for row in report.get("rows", [])]


def _subgroup(query: str) -> set[str]:
    lowered = query.casefold()
    groups: set[str] = set()
    if any(value in lowered for value in ("table", "score", "percent", "accuracy", "recall", "表格", "指标")):
        groups.add("table_numeric")
    if any(value in lowered for value in ("list", "which", "what types", "enumerate", "列出", "哪些")):
        groups.add("list")
    if any(value in lowered for value in ("section", "method", "experiment", "result", "章节", "方法", "实验")):
        groups.add("section")
    return groups


def _subgroup_agent_recall(report: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    values: dict[str, list[float]] = {}
    for row in report.get("rows", []):
        for group in _subgroup(str(row.get("query") or "")):
            values.setdefault(group, []).append(
                float(row.get("agent_visible_recall_at_8") or 0.0)
            )
    return {
        group: {"cases": len(scores), "recall_at_8": sum(scores) / len(scores)}
        for group, scores in sorted(values.items())
        if scores
    }


def _answer_gate(
    current_path: Path | None,
    optimized_path: Path | None,
) -> dict[str, Any]:
    if current_path is None or optimized_path is None:
        return {
            "passed": False,
            "status": "pending",
            "reason": "paired answer/citation reports were not supplied",
        }
    current = json.loads(current_path.read_text(encoding="utf-8"))
    optimized = json.loads(optimized_path.read_text(encoding="utf-8"))
    current_metrics = current.get("metrics", {})
    optimized_metrics = optimized.get("metrics", {})
    fields = (
        "semantic_answer_accuracy",
        "semantic_strict_grounded_accuracy",
        "avg_citation_validity",
        "avg_gold_page_citation_precision",
    )
    comparisons = {
        field: {
            "current": current_metrics.get(field),
            "optimized": optimized_metrics.get(field),
            "non_decreasing": (
                optimized_metrics.get(field) is not None
                and current_metrics.get(field) is not None
                and float(optimized_metrics[field]) >= float(current_metrics[field])
            ),
        }
        for field in fields
    }
    return {
        "passed": all(value["non_decreasing"] for value in comparisons.values()),
        "status": "complete",
        "current_report": str(current_path.resolve()),
        "current_sha256": _sha256(current_path),
        "optimized_report": str(optimized_path.resolve()),
        "optimized_sha256": _sha256(optimized_path),
        "comparisons": comparisons,
    }


def _decision(
    reports: dict[str, dict[str, Any]],
    *,
    max_p95_ratio: float,
    current_answer_report: Path | None,
    optimized_answer_report: Path | None,
) -> dict[str, Any]:
    control = reports["a"]
    optimized = reports["e"]
    control_metrics = control["metrics"]
    optimized_metrics = optimized["metrics"]
    control_agent = control["agent_visible_metrics"]
    optimized_agent = optimized["agent_visible_metrics"]
    control_citations = control["citation_metrics"]
    optimized_citations = optimized["citation_metrics"]
    mrr_ci = _paired_ci(_metric_rows(control, "mrr"), _metric_rows(optimized, "mrr"))
    ndcg_ci = _paired_ci(
        _metric_rows(control, "ndcg_at_8"),
        _metric_rows(optimized, "ndcg_at_8"),
    )
    control_subgroups = _subgroup_agent_recall(control)
    optimized_subgroups = _subgroup_agent_recall(optimized)
    subgroup_gate = {
        group: (
            group in optimized_subgroups
            and float(optimized_subgroups[group]["recall_at_8"])
            >= float(value["recall_at_8"])
        )
        for group, value in control_subgroups.items()
    }
    p95_current = float(control_metrics["p95_ms"])
    p95_optimized = float(optimized_metrics["p95_ms"])
    retrieval_gates = {
        "alignment": bool(control.get("alignment_gate", {}).get("passed"))
        and bool(optimized.get("alignment_gate", {}).get("passed")),
        "recall_at_5_non_decreasing": float(optimized_metrics["recall_at_5"])
        >= float(control_metrics["recall_at_5"]),
        "recall_at_10_non_decreasing": float(optimized_metrics["recall_at_10"])
        >= float(control_metrics["recall_at_10"]),
        "agent_visible_recall_at_8_non_decreasing": float(
            optimized_agent["recall_at_8"]
        )
        >= float(control_agent["recall_at_8"]),
        "citation_localization_non_decreasing": float(
            optimized_citations["localization_hit_rate_at_8"]
        )
        >= float(control_citations["localization_hit_rate_at_8"]),
        "citation_precision_non_decreasing": float(
            optimized_citations["precision_at_8"]
        )
        >= float(control_citations["precision_at_8"]),
        "citation_roundtrip_non_decreasing": float(
            optimized_citations["roundtrip_verification_accuracy"]
        )
        >= float(control_citations["roundtrip_verification_accuracy"]),
        "ranking_metric_clear_improvement": (
            mrr_ci["ci95_low"] > 0.0 or ndcg_ci["ci95_low"] > 0.0
        ),
        "subgroups_non_decreasing": all(subgroup_gate.values()),
        "p95_within_limit": p95_optimized
        <= p95_current * max_p95_ratio,
        "optimized_index_isolated": all(
            ":rag:optimized-e" in str(
                item.get("index_statistics", {}).get("knowledge_base_id", "")
            )
            for item in optimized.get("ingestion", [])
            if item.get("status") == "indexed"
        ),
    }
    answer_gate = _answer_gate(current_answer_report, optimized_answer_report)
    retrieval_passed = all(retrieval_gates.values())
    if retrieval_passed and answer_gate["passed"]:
        outcome = "eligible_for_canary"
    elif retrieval_passed and answer_gate["status"] == "pending":
        outcome = "no_go_pending_answer_and_citation_eval"
    else:
        outcome = "no_go_keep_current"
    return {
        "outcome": outcome,
        "active_profile_must_remain": "current",
        "retrieval_gates": retrieval_gates,
        "retrieval_gate_passed": retrieval_passed,
        "ranking_paired_ci": {"mrr": mrr_ci, "ndcg_at_8": ndcg_ci},
        "subgroups": {
            "current": control_subgroups,
            "optimized": optimized_subgroups,
            "non_decreasing": subgroup_gate,
        },
        "latency": {
            "current_p95_ms": p95_current,
            "optimized_p95_ms": p95_optimized,
            "ratio": p95_optimized / p95_current if p95_current else None,
            "maximum_ratio": max_p95_ratio,
        },
        "answer_and_citation_gate": answer_gate,
        "failure_counts": {
            stage: dict(Counter(str(row.get("failure_stage") or "unknown") for row in report.get("rows", [])))
            for stage, report in reports.items()
        },
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset", type=Path, required=True)
    value.add_argument("--split", type=Path, required=True)
    value.add_argument("--pdf-manifest", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument(
        "--state-root",
        type=Path,
        default=None,
        help="Persist one isolated index state directory per A-E stage.",
    )
    value.add_argument("--limit", type=int, default=100)
    value.add_argument("--offset", type=int, default=0)
    value.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    value.add_argument("--backend", choices=("bm25", "fastembed"), default="fastembed")
    value.add_argument(
        "--reranker-backend",
        choices=("fastembed", "fastembed_ensemble", "flagembedding", "transformers"),
        default="fastembed_ensemble",
    )
    value.add_argument(
        "--reranker-model",
        default="jinaai/jina-reranker-v1-tiny-en,Xenova/ms-marco-MiniLM-L-6-v2",
    )
    value.add_argument("--pdf-parser-backend", choices=("native", "mineru"), default="mineru")
    value.add_argument("--mineru-base-url", default=None)
    value.add_argument("--mineru-expected-version", default=None)
    value.add_argument(
        "--mineru-cache-root",
        type=Path,
        default=PROJECT_ROOT / ".taskforge" / "eval-cache" / "mineru-shared-v1",
    )
    value.add_argument("--parent-target-tokens", type=int, default=2_000)
    value.add_argument("--parent-max-tokens", type=int, default=3_000)
    value.add_argument("--child-target-tokens", type=int, default=400)
    value.add_argument("--child-max-tokens", type=int, default=500)
    value.add_argument("--child-overlap-tokens", type=int, default=60)
    value.add_argument("--parent-aware-candidate-k", type=int, default=20)
    value.add_argument("--parent-context-max-tokens", type=int, default=800)
    value.add_argument("--parent-child-score-weight", type=float, default=0.55)
    value.add_argument("--parent-context-score-weight", type=float, default=0.35)
    value.add_argument("--parent-retrieval-score-weight", type=float, default=0.10)
    value.add_argument("--max-p95-ratio", type=float, default=1.25)
    value.add_argument("--current-answer-report", type=Path, default=None)
    value.add_argument("--optimized-answer-report", type=Path, default=None)
    value.add_argument("--plan-only", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if "a" not in args.stages or "e" not in args.stages:
        raise SystemExit("A and E are required for the final gate")
    if args.pdf_parser_backend == "mineru" and (
        not args.mineru_base_url or not args.mineru_expected_version
    ):
        raise SystemExit("MinerU A/B requires --mineru-base-url and --mineru-expected-version")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_plan = [
        {
            "stage": stage,
            "profile": "current" if stage == "a" else "optimized",
            "output": str((args.output_dir / f"rag-profile-{stage}.json").resolve()),
        }
        for stage in args.stages
    ]
    if args.plan_only:
        print(json.dumps({"status": "planned", "runs": run_plan}, indent=2))
        return 0
    reports: dict[str, dict[str, Any]] = {}
    for item in run_plan:
        stage = str(item["stage"])
        output = Path(str(item["output"]))
        reports[stage] = run(
            args.dataset.resolve(),
            args.split.resolve(),
            output,
            limit=args.limit,
            offset=args.offset,
            backend=args.backend,
            graph_enabled=False,
            reranker_model=args.reranker_model,
            reranker_backend=args.reranker_backend,
            query_expansion_mode="original",
            candidate_k=50,
            agent_visible_k=8,
            pdf_manifest_path=args.pdf_manifest.resolve(),
            pdf_parser_backend=args.pdf_parser_backend,
            mineru_base_url=args.mineru_base_url,
            mineru_expected_version=args.mineru_expected_version,
            mineru_cache_root=args.mineru_cache_root.resolve(),
            pdf_chunking_mode="parent_child",
            pdf_flat_chunk_chars=2_000,
            pdf_flat_overlap_chars=0,
            pdf_parent_target_tokens=args.parent_target_tokens,
            pdf_parent_max_tokens=args.parent_max_tokens,
            pdf_child_target_tokens=args.child_target_tokens,
            pdf_child_max_tokens=args.child_max_tokens,
            pdf_child_overlap_tokens=args.child_overlap_tokens,
            operator_budget=0,
            visual_extractor_enabled=False,
            rag_profile=str(item["profile"]),
            rag_ablation=stage,
            parent_aware_candidate_k=args.parent_aware_candidate_k,
            parent_context_max_tokens=args.parent_context_max_tokens,
            parent_child_score_weight=args.parent_child_score_weight,
            parent_context_score_weight=args.parent_context_score_weight,
            parent_retrieval_score_weight=args.parent_retrieval_score_weight,
            state_dir=(
                args.state_root.resolve() / stage
                if args.state_root is not None
                else None
            ),
        )
    decision = _decision(
        reports,
        max_p95_ratio=args.max_p95_ratio,
        current_answer_report=args.current_answer_report,
        optimized_answer_report=args.optimized_answer_report,
    )
    manifest = {
        "schema_version": "taskforge.rag_profile_ab.v1",
        "status": "complete",
        "created_at": datetime.now(UTC).isoformat(),
        "locked_inputs": {
            "dataset": str(args.dataset.resolve()),
            "dataset_sha256": _sha256(args.dataset),
            "split": str(args.split.resolve()),
            "split_sha256": _sha256(args.split),
            "pdf_manifest": str(args.pdf_manifest.resolve()),
            "pdf_manifest_sha256": _sha256(args.pdf_manifest),
        },
        "fixed_conditions": {
            "backend": args.backend,
            "reranker_backend": args.reranker_backend,
            "reranker_model": args.reranker_model,
            "parser_backend": args.pdf_parser_backend,
            "mineru_expected_version": args.mineru_expected_version,
            "candidate_k": 50,
            "top_k": 8,
            "query_mode": "original",
            "graph_enabled": False,
            "operator_budget": 0,
            "parent_aware_candidate_k": args.parent_aware_candidate_k,
            "parent_context_max_tokens": args.parent_context_max_tokens,
            "parent_score_weights": {
                "child": args.parent_child_score_weight,
                "context": args.parent_context_score_weight,
                "retrieval": args.parent_retrieval_score_weight,
            },
        },
        "state_root": (
            str(args.state_root.resolve()) if args.state_root is not None else None
        ),
        "runs": {
            stage: {
                "profile": report["rag_profile"],
                "path": str((args.output_dir / f"rag-profile-{stage}.json").resolve()),
                "sha256": _sha256(args.output_dir / f"rag-profile-{stage}.json"),
                "status": report["status"],
                "metrics": report["metrics"],
                "agent_visible_metrics": report["agent_visible_metrics"],
                "citation_metrics": report["citation_metrics"],
                "index_statistics": report["index_statistics"],
            }
            for stage, report in reports.items()
        },
        "decision": decision,
    }
    manifest_path = args.output_dir / "rag-profile-ab-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest_path.resolve()), **decision}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

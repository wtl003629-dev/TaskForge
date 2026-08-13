"""Re-score an existing TAT-QA answer run without calling a model.

The live answer evaluator historically exposed one ``evidence_recall`` field
computed over the complete candidate list.  This report keeps that candidate
metric separate from true Top-10 retrieval and the evidence actually shown to
the model, and adds the released TAT-QA answer metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = SCRIPT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.rag_evaluation import load_tatqa_dataset  # noqa: E402
from taskforge.tatqa_metrics import tatqa_answer_metrics  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _retrieval_metrics(relevant_ids: list[str], retrieved_ids: list[str]) -> dict[str, Any]:
    relevant = set(relevant_ids)
    retrieved = list(dict.fromkeys(retrieved_ids))
    matched = relevant.intersection(retrieved)
    precision = len(matched) / len(retrieved) if retrieved else 0.0
    recall = len(matched) / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "full_recall": recall == 1.0,
    }


def _failure_stage(
    row: dict[str, Any],
    *,
    candidate: dict[str, Any],
    top10: dict[str, Any],
    presented: dict[str, Any],
    answer_metrics: dict[str, Any],
) -> str:
    """Attribute an immutable prediction to its first observable failure."""

    if row.get("execution_error"):
        return "execution_error"
    if row.get("parse_error") or not str(row.get("generated_answer", "")).strip():
        return "format_or_scale_failure"
    if not candidate["full_recall"]:
        return "candidate_missing"
    if not top10["full_recall"]:
        return "top10_ranking_failure"
    if not presented["full_recall"]:
        return "context_coverage_failure"
    if answer_metrics["exact_match"] == 1.0:
        return "success"
    return "reasoning_failure"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"prediction row must be an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def recalculate(run_dir: Path, *, output: Path, dataset_override: Path | None = None) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    predictions_path = run_dir / "predictions.jsonl"
    if not manifest_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError("run must contain manifest.json and predictions.jsonl")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    dataset_config = manifest.get("dataset")
    if not isinstance(dataset_config, dict):
        raise ValueError("manifest.dataset is missing")
    dataset_path = dataset_override or SCRIPT_ROOT / str(dataset_config.get("input_path", ""))
    if not dataset_path.is_file():
        raise FileNotFoundError(f"TAT-QA input does not exist: {dataset_path}")
    expected_sha = str(dataset_config.get("input_sha256", ""))
    actual_sha = _sha256_file(dataset_path)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError("TAT-QA input SHA-256 does not match the run manifest")

    dataset = load_tatqa_dataset(dataset_path)
    cases = {case.case_id: case for case in dataset.cases}
    rows = _load_jsonl(predictions_path)
    case_ids = [str(row.get("case_id", "")) for row in rows]
    if not case_ids or any(not case_id for case_id in case_ids):
        raise ValueError("predictions must contain non-empty case_id values")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("predictions contain duplicate case IDs")
    missing = [case_id for case_id in case_ids if case_id not in cases]
    if missing:
        raise ValueError(f"predictions contain unknown case IDs: {missing[:3]}")

    budgets = manifest.get("budgets") if isinstance(manifest.get("budgets"), dict) else {}
    candidate_k = int(budgets.get("candidate_k", 50))
    top_k = int(budgets.get("evidence_top_k", 10))
    if candidate_k < top_k or top_k <= 0:
        raise ValueError("manifest budgets must have candidate_k >= evidence_top_k > 0")

    details: list[dict[str, Any]] = []
    for row in rows:
        case = cases[row["case_id"]]
        relevant_ids = [str(value) for value in case.relevant_ids]
        candidate_ids = [str(value) for value in row.get("retrieved_ids", [])][:candidate_k]
        top_ids = candidate_ids[:top_k]
        presented_ids = [str(value) for value in row.get("presented_evidence_ids", [])]
        answer_metrics = tatqa_answer_metrics(
            row.get("generated_answer", ""),
            case.answer,
            answer_type=str(case.metadata.get("answer_type", "")),
            gold_scale=str(case.metadata.get("scale", "")),
        )
        candidate_metrics = _retrieval_metrics(relevant_ids, candidate_ids)
        top_metrics = _retrieval_metrics(relevant_ids, top_ids)
        presented_metrics = _retrieval_metrics(relevant_ids, presented_ids)
        failure_stage = _failure_stage(
            row,
            candidate=candidate_metrics,
            top10=top_metrics,
            presented=presented_metrics,
            answer_metrics=answer_metrics,
        )
        details.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "answer_type": case.metadata.get("answer_type", ""),
                "candidate_recall_at_50": candidate_metrics["recall"],
                "retrieval_recall_at_10": top_metrics["recall"],
                "presented_context_recall": presented_metrics["recall"],
                "candidate_precision_at_50": candidate_metrics["precision"],
                "retrieval_precision_at_10": top_metrics["precision"],
                "presented_context_precision": presented_metrics["precision"],
                "candidate_full_recall": candidate_metrics["full_recall"],
                "retrieval_full_recall_at_10": top_metrics["full_recall"],
                "presented_context_full_recall": presented_metrics["full_recall"],
                "generic_exact_match": float(row.get("exact_match", 0.0)),
                "generic_token_f1": float(row.get("token_f1", 0.0)),
                "tatqa_exact_match": answer_metrics["exact_match"],
                "tatqa_f1": answer_metrics["f1"],
                "tatqa_scale_match": answer_metrics["scale_match"],
                "prediction_scale": answer_metrics["prediction_scale"],
                "gold_scale": answer_metrics["gold_scale"],
                "failure_stage": failure_stage,
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        grouped[detail["category"]].append(detail)

    def summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "cases": len(items),
            "candidate_recall_at_50": _mean([item["candidate_recall_at_50"] for item in items]),
            "retrieval_recall_at_10": _mean([item["retrieval_recall_at_10"] for item in items]),
            "presented_context_recall": _mean([item["presented_context_recall"] for item in items]),
            "candidate_full_recall_rate": _mean([float(item["candidate_full_recall"]) for item in items]),
            "retrieval_full_recall_at_10_rate": _mean(
                [float(item["retrieval_full_recall_at_10"]) for item in items]
            ),
            "tatqa_exact_match": _mean([item["tatqa_exact_match"] for item in items]),
            "tatqa_f1": _mean([item["tatqa_f1"] for item in items]),
            "tatqa_scale_match": _mean([item["tatqa_scale_match"] for item in items]),
            "generic_exact_match": _mean([item["generic_exact_match"] for item in items]),
            "generic_token_f1": _mean([item["generic_token_f1"] for item in items]),
            "failure_stage_counts": dict(
                sorted(Counter(item["failure_stage"] for item in items).items())
            ),
        }

    report = {
        "schema_version": "1.1",
        "metric_source": "NExTplusplus/TAT-QA tatqa_metric.py semantics",
        "run_id": manifest.get("run_id"),
        "run_directory": str(run_dir),
        "dataset": {
            "path": str(dataset_path),
            "sha256": actual_sha,
            "case_count": len(details),
        },
        "budgets": {"candidate_k": candidate_k, "retrieval_top_k": top_k},
        "summary": summary(details),
        "by_category": {category: summary(items) for category, items in sorted(grouped.items())},
        "cases": details,
    }
    _write_atomic(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Existing answer-eval run directory")
    parser.add_argument("--output", required=True, type=Path, help="Offline report JSON path")
    parser.add_argument("--dataset", type=Path, help="Optional TAT-QA input override")
    args = parser.parse_args()
    report = recalculate(args.run.resolve(), output=args.output.resolve(), dataset_override=args.dataset)
    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

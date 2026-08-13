"""Gate three document-disjoint TAT-QA query-slot reranker seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.rag_baseline import load_locked_split  # noqa: E402
from taskforge.tatqa_mapping_eval import TAGOP_TRAIN_SHA256  # noqa: E402
from taskforge.tatqa_slot_reranker import TATQASlotReranker  # noqa: E402
from taskforge.tatqa_slot_selector import select_tatqa_table_slots  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--annotations", type=Path, required=True)
    value.add_argument("--validation-split", type=Path, required=True)
    value.add_argument("--tuning-split", type=Path, required=True)
    value.add_argument("--model", type=Path, action="append", required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--budget", type=int, default=10)
    value.add_argument("--rrf-k", type=int, default=60)
    value.add_argument("--learned-weight", type=float, default=1.0)
    value.add_argument("--heuristic-weight", type=float, default=0.5)
    value.add_argument("--min-macro-delta", type=float, default=0.03)
    value.add_argument("--max-category-drop", type=float, default=0.03)
    return value


def _score_rows(
    rows: list[tuple[str, str, list[list[str]], set[tuple[int, int]]]],
    model: TATQASlotReranker | None,
    *,
    budget: int,
    rrf_k: int,
    learned_weight: float,
    heuristic_weight: float,
) -> dict[str, Any]:
    values: list[float] = []
    by_category: dict[str, list[float]] = {}
    states = {"zero": 0, "partial": 0, "complete": 0}
    for question, category, table, gold in rows:
        plan = (
            select_tatqa_table_slots(question, table, budget=budget)
            if model is None
            else model.select(
                question,
                table,
                budget=budget,
                fusion="weighted_rrf",
                rrf_k=rrf_k,
                learned_weight=learned_weight,
                heuristic_weight=heuristic_weight,
            )
        )
        selected = {
            (slot.row_index, slot.column_index) for slot in plan.slots
        }
        recall = len(gold.intersection(selected)) / len(gold)
        values.append(recall)
        by_category.setdefault(category, []).append(recall)
        states[
            "zero" if recall == 0 else "complete" if recall == 1 else "partial"
        ] += 1
    return {
        "macro_recall": sum(values) / len(values),
        "state_counts": states,
        "by_category": {
            category: {
                "cases": len(scores),
                "recall": sum(scores) / len(scores),
            }
            for category, scores in sorted(by_category.items())
        },
    }


def _gate(
    baseline: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    min_macro_delta: float,
    max_category_drop: float,
) -> dict[str, Any]:
    category_failures: list[dict[str, Any]] = []
    for candidate in candidates:
        for category, baseline_row in baseline["by_category"].items():
            delta = (
                candidate["by_category"][category]["recall"]
                - baseline_row["recall"]
            )
            if delta < -max_category_drop:
                category_failures.append(
                    {
                        "seed": candidate["seed"],
                        "category": category,
                        "delta": delta,
                    }
                )
    checks = {
        "three_distinct_seeds": len({row["seed"] for row in candidates}) >= 3,
        "all_macro_deltas": all(
            row["macro_delta"] >= min_macro_delta for row in candidates
        ),
        "category_non_regression": not category_failures,
        "direction_consistent": all(row["macro_delta"] > 0 for row in candidates),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "category_failures": category_failures,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.budget <= 0 or args.rrf_k <= 0:
            raise ValueError("budgets must be positive")
        if _sha256(args.annotations) != TAGOP_TRAIN_SHA256:
            raise ValueError("annotation SHA-256 does not match pinned TagOp")
        validation = load_locked_split(args.validation_split)
        tuning = load_locked_split(args.tuning_split)
        if validation.dataset != "TAT-QA" or tuning.dataset != "TAT-QA":
            raise ValueError("splits must describe TAT-QA")
        if set(validation.case_ids).intersection(tuning.case_ids):
            raise ValueError("validation and tuning case IDs overlap")
        annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
        if not isinstance(annotations, list):
            raise ValueError("annotation root must be a list")
        validation_ids = set(validation.case_ids)
        rows: list[tuple[str, str, list[list[str]], set[tuple[int, int]]]] = []
        covered: set[str] = set()
        for context in annotations:
            table = [[str(value) for value in row] for row in context["table"]["table"]]
            for question in context["questions"]:
                case_id = f"tatqa:{question['uid']}"
                if case_id not in validation_ids:
                    continue
                covered.add(case_id)
                raw_mapping = question.get("mapping", {})
                gold = {
                    tuple(coordinate)
                    for coordinate in raw_mapping.get("table", [])
                }
                if gold:
                    rows.append(
                        (
                            str(question["question"]),
                            str(question["answer_type"]),
                            table,
                            gold,
                        )
                    )
        if covered != validation_ids:
            raise ValueError("annotations do not cover validation split")
        baseline = _score_rows(
            rows,
            None,
            budget=args.budget,
            rrf_k=args.rrf_k,
            learned_weight=args.learned_weight,
            heuristic_weight=args.heuristic_weight,
        )
        candidates: list[dict[str, Any]] = []
        training_split_hashes: set[str] = set()
        for model_path in args.model:
            model = TATQASlotReranker.load(model_path)
            seed = model.training.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError("model training seed is missing")
            training_split_hash = model.training.get("train_split_sha256")
            if not isinstance(training_split_hash, str):
                raise ValueError("model training split hash is missing")
            training_split_hashes.add(training_split_hash)
            score = _score_rows(
                rows,
                model,
                budget=args.budget,
                rrf_k=args.rrf_k,
                learned_weight=args.learned_weight,
                heuristic_weight=args.heuristic_weight,
            )
            score.update(
                {
                    "seed": seed,
                    "model_id": model.model_id,
                    "artifact_sha256": _sha256(model_path),
                    "macro_delta": score["macro_recall"]
                    - baseline["macro_recall"],
                    "category_deltas": {
                        category: row["recall"]
                        - baseline["by_category"][category]["recall"]
                        for category, row in score["by_category"].items()
                    },
                }
            )
            candidates.append(score)
        if len(training_split_hashes) != 1:
            raise ValueError("models do not share one training partition")
        gate = _gate(
            baseline,
            candidates,
            min_macro_delta=args.min_macro_delta,
            max_category_drop=args.max_category_drop,
        )
        report = {
            "schema_version": "1.0",
            "diagnostic_only": True,
            "promotion_eligible": gate["passed"],
            "validation_split": {
                "id": validation.split_id,
                "sha256": _sha256(args.validation_split),
                "mapping_eligible_cases": len(rows),
            },
            "tuning_split": {
                "id": tuning.split_id,
                "sha256": _sha256(args.tuning_split),
            },
            "configuration": {
                "budget": args.budget,
                "rrf_k": args.rrf_k,
                "learned_weight": args.learned_weight,
                "heuristic_weight": args.heuristic_weight,
                "min_macro_delta": args.min_macro_delta,
                "max_category_drop": args.max_category_drop,
            },
            "baseline": baseline,
            "candidates": candidates,
            "gate": gate,
        }
        payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        sys.stdout.write(payload)
        return 0 if gate["passed"] else 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        sys.stdout.write(json.dumps({"error": str(exc)}, ensure_ascii=False) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Train a document-disjoint, category-balanced TAT-QA cell reranker."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.rag_baseline import load_locked_split, sha256_file  # noqa: E402
from taskforge.tatqa_mapping_eval import TAGOP_TRAIN_SHA256  # noqa: E402
from taskforge.tatqa_slot_reranker import TATQASlotReranker  # noqa: E402
from taskforge.tatqa_slot_selector import (  # noqa: E402
    tatqa_slot_feature_vector,
    tatqa_slot_heuristic_score,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--annotations", type=Path, required=True)
    value.add_argument("--train-source", type=Path, required=True)
    value.add_argument("--train-split", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--seed", type=int, required=True)
    value.add_argument("--epochs", type=int, default=180)
    value.add_argument("--hard-negatives", type=int, default=12)
    value.add_argument("--random-negatives", type=int, default=8)
    return value


def _training_rows(
    annotations: list[object],
    case_ids: set[str],
    *,
    seed: int,
    hard_negatives: int,
    random_negatives: int,
) -> tuple[np.ndarray, np.ndarray, list[str], Counter[str], int]:
    rng = random.Random(seed)
    vectors: list[tuple[float, ...]] = []
    labels: list[float] = []
    categories: list[str] = []
    case_counts: Counter[str] = Counter()
    covered_case_ids: set[str] = set()
    for raw_context in annotations:
        if not isinstance(raw_context, dict):
            raise ValueError("annotation contexts must be objects")
        table_object = raw_context.get("table")
        questions = raw_context.get("questions")
        if not isinstance(table_object, dict) or not isinstance(questions, list):
            raise ValueError("annotation context has invalid table/questions")
        raw_table = table_object.get("table")
        if not isinstance(raw_table, list) or any(
            not isinstance(row, list) for row in raw_table
        ):
            raise ValueError("annotation table is invalid")
        table = [[str(value) for value in row] for row in raw_table]
        coordinates = [
            (row_index, column_index)
            for row_index, row in enumerate(table)
            for column_index, value in enumerate(row)
            if value.strip()
        ]
        for question in questions:
            if not isinstance(question, dict) or not isinstance(question.get("uid"), str):
                raise ValueError("annotation question is invalid")
            case_id = f"tatqa:{question['uid']}"
            if case_id not in case_ids:
                continue
            covered_case_ids.add(case_id)
            mapping = question.get("mapping", {})
            raw_gold = mapping.get("table", []) if isinstance(mapping, dict) else []
            if not raw_gold:
                continue
            gold = {tuple(coordinate) for coordinate in raw_gold}
            query = str(question.get("question", ""))
            category = str(question.get("answer_type", "unknown"))
            negatives = [coordinate for coordinate in coordinates if coordinate not in gold]
            negatives.sort(
                key=lambda coordinate: (
                    -tatqa_slot_heuristic_score(query, table, *coordinate),
                    coordinate[0],
                    coordinate[1],
                )
            )
            hard = negatives[:hard_negatives]
            remainder = negatives[hard_negatives:]
            random_count = min(random_negatives, len(remainder))
            sampled = rng.sample(remainder, random_count) if random_count else []
            for coordinate in [*sorted(gold), *hard, *sampled]:
                vectors.append(tatqa_slot_feature_vector(query, table, *coordinate))
                labels.append(float(coordinate in gold))
                categories.append(category)
            case_counts[category] += 1
    if covered_case_ids != case_ids:
        raise ValueError("annotations do not cover every training split case")
    if not vectors or not any(labels):
        raise ValueError("training partition has no mapped positive cells")
    return (
        np.asarray(vectors, dtype=np.float64),
        np.asarray(labels, dtype=np.float64),
        categories,
        case_counts,
        len(covered_case_ids),
    )


def _fit(
    matrix: np.ndarray,
    labels: np.ndarray,
    categories: list[str],
    case_counts: Counter[str],
    *,
    epochs: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (matrix - mean) / scale
    weights = np.zeros(normalized.shape[1], dtype=np.float64)
    bias = 0.0
    positive_count = float(labels.sum())
    negative_count = float(len(labels) - positive_count)
    class_weights = np.where(
        labels > 0,
        len(labels) / (2.0 * positive_count),
        len(labels) / (2.0 * negative_count),
    )
    category_weights = np.asarray(
        [
            sum(case_counts.values()) / (len(case_counts) * case_counts[category])
            for category in categories
        ],
        dtype=np.float64,
    )
    example_weights = class_weights * category_weights
    example_weights /= example_weights.mean()
    for epoch in range(epochs):
        logits = np.clip(normalized @ weights + bias, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        errors = (probabilities - labels) * example_weights
        learning_rate = 0.15 / (1.0 + epoch / 80.0)
        weights -= learning_rate * (
            normalized.T @ errors / len(labels) + 0.002 * weights
        )
        bias -= learning_rate * float(errors.mean())
    return weights, bias, mean, scale


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for label, value in (
        ("--hard-negatives", args.hard_negatives),
        ("--random-negatives", args.random_negatives),
    ):
        if value < 0:
            raise SystemExit(f"{label} must be non-negative")
    if args.hard_negatives + args.random_negatives == 0:
        raise SystemExit("at least one negative is required")
    if _sha256(args.annotations) != TAGOP_TRAIN_SHA256:
        raise SystemExit("annotation SHA-256 does not match the pinned TagOp artifact")
    split = load_locked_split(args.train_split)
    if split.dataset != "TAT-QA":
        raise SystemExit("training split must describe TAT-QA")
    if split.source_sha256 != sha256_file(args.train_source):
        raise SystemExit("training split/source SHA-256 mismatch")
    if split.selection.get("strategy") != "parent_document_disjoint_complement":
        raise SystemExit("training split must be a parent-document-disjoint complement")
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    if not isinstance(annotations, list):
        raise SystemExit("annotation root must be a list")
    matrix, labels, categories, case_counts, covered_cases = _training_rows(
        annotations,
        set(split.case_ids),
        seed=args.seed,
        hard_negatives=args.hard_negatives,
        random_negatives=args.random_negatives,
    )
    weights, bias, mean, scale = _fit(
        matrix,
        labels,
        categories,
        case_counts,
        epochs=args.epochs,
    )
    model = TATQASlotReranker(
        weights,
        bias,
        mean,
        scale,
        model_id=f"taskforge-tatqa-slot-logistic-seed-{args.seed}",
        training={
            "annotation_sha256": _sha256(args.annotations),
            "train_source_sha256": _sha256(args.train_source),
            "train_split_id": split.split_id,
            "train_split_sha256": _sha256(args.train_split),
            "seed": args.seed,
            "epochs": args.epochs,
            "hard_negatives_per_question": args.hard_negatives,
            "random_negatives_per_question": args.random_negatives,
            "category_balanced": True,
            "case_category_counts": dict(sorted(case_counts.items())),
            "training_cases": covered_cases,
            "mapped_training_cases": sum(case_counts.values()),
            "training_examples": len(labels),
            "positive_examples": int(labels.sum()),
            "negative_examples": int(len(labels) - labels.sum()),
        },
    )
    model.save(args.output)
    print(
        json.dumps(
            {
                "artifact": str(args.output.resolve()),
                "artifact_sha256": _sha256(args.output),
                **model.training,
                "model_id": model.model_id,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

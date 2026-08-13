"""Train/evaluate the paper-disjoint supervised QASPER ranker."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import fmean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.rag_evaluation import load_qasper_dataset  # noqa: E402
from taskforge.research_supervised_ranker import (  # noqa: E402
    row_features,
    train_supervised_ranker,
)


def _rows(run: Path, stage: str) -> list[dict]:
    source = run if run.is_file() else run / "predictions.jsonl"
    return [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip() and (stage == "*" or json.loads(line).get("stage") == stage)
    ]


def _metrics(rows: list[dict], model, documents: dict[str, dict]) -> dict[str, float]:
    values: dict[int, list[float]] = {key: [] for key in (1, 5, 10, 20, 50)}
    ndcg: list[float] = []
    for row in rows:
        ids = [str(value) for value in row.get("retrieved_ids", [])]
        relevant = {str(value) for value in row.get("relevant_ids", [])}
        features = row_features(row, documents)
        order = model.rerank(features)
        ranked = [ids[index] for index in order]
        for key in values:
            values[key].append(len(relevant.intersection(ranked[:key])) / len(relevant))
        gains = [int(item in relevant) for item in ranked[:10]]
        dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
        ideal = sum(1 / math.log2(index + 2) for index in range(min(10, len(relevant))))
        ndcg.append(dcg / ideal if ideal else 0.0)
    return {f"recall_at_{key}": fmean(values[key]) for key in values} | {
        "ndcg_at_10": fmean(ndcg)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fit-run", type=Path, required=True)
    parser.add_argument("--validation-run", type=Path, required=True)
    parser.add_argument("--stage", default="qdrant_qasper_dense_rerank")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.3)
    parser.add_argument("--max-rank", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = load_qasper_dataset(args.dataset)
    documents = {
        document.document_id: {"text": document.text, "metadata": dict(document.metadata)}
        for document in dataset.documents
    }
    fit_rows = _rows(args.fit_run, args.stage)
    validation_rows = _rows(args.validation_run, args.stage)
    model = train_supervised_ranker(
        fit_rows,
        documents,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        max_rank=args.max_rank,
    )
    result = {
        "schema_version": "1.0",
        "dataset": "QASPER",
        "fit_run": str(args.fit_run),
        "validation_run": str(args.validation_run),
        "stage": args.stage,
        "training_cases": model.training_cases,
        "positive_examples": model.positive_examples,
        "validation": _metrics(validation_rows, model, documents),
        "model": model.model_dump(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["validation"], ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

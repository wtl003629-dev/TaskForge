"""Train and evaluate QASPER paper-level feature fusion on disjoint runs."""

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
from taskforge.research_feature_reranker import (  # noqa: E402
    feature_vector,
    train_pairwise_feature_reranker,
)


def _rows(run: Path, stage: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (run / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("stage") == stage
    ]


def _metrics(rows: list[dict], documents: dict[str, dict]) -> dict[str, float]:
    values: dict[str, list[float]] = {"1": [], "5": [], "10": [], "50": [], "ndcg10": []}
    for row in rows:
        ids = [str(value) for value in row.get("retrieved_ids", [])]
        relevant = {str(value) for value in row.get("relevant_ids", [])}
        visible_ids = [chunk_id for chunk_id in ids if chunk_id in documents]
        order = list(range(len(visible_ids)))
        # The caller replaces this deterministic placeholder with a model order
        # by attaching ``_feature_order`` to each row.
        if "_feature_order" in row:
            order = list(row["_feature_order"])
        ranked = [visible_ids[index] for index in order]
        for k in (1, 5, 10, 50):
            values[str(k)].append(len(relevant.intersection(ranked[:k])) / len(relevant))
        gains = [1 if item in relevant else 0 for item in ranked[:10]]
        dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(10, len(relevant))))
        values["ndcg10"].append(dcg / ideal if ideal else 0.0)
    return {f"recall_at_{key}": fmean(values[key]) for key in ("1", "5", "10", "50")} | {
        "ndcg_at_10": fmean(values["ndcg10"])
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fit-run", type=Path, required=True)
    parser.add_argument("--validation-run", type=Path, required=True)
    parser.add_argument("--stage", default="qdrant_qasper_dense_rerank")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = load_qasper_dataset(args.dataset)
    documents = {
        document.document_id: {"text": document.text, "metadata": dict(document.metadata)}
        for document in dataset.documents
    }
    fit_rows = _rows(args.fit_run, args.stage)
    validation_rows = _rows(args.validation_run, args.stage)
    model = train_pairwise_feature_reranker(fit_rows, documents)
    for row in validation_rows:
        ids = [str(value) for value in row.get("retrieved_ids", [])]
        base_scores = list(row.get("base_scores", []))
        reranker_scores = list(row.get("reranker_scores", []))
        scored_count = sum(value is not None for value in reranker_scores)
        scored_count = min(len(ids), scored_count or len(ids))
        features = [
            feature_vector(
                str(row.get("query", "")),
                documents[chunk_id]["text"],
                documents[chunk_id]["metadata"],
                base_rank=index + 1,
                base_score=float(base_scores[index]) if index < len(base_scores) else 0.0,
                reranker_score=(
                    None
                    if index >= len(reranker_scores) or reranker_scores[index] is None
                    else float(reranker_scores[index])
                ),
            )
            for index, chunk_id in enumerate(ids[:scored_count])
            if chunk_id in documents
        ]
        row["_feature_order"] = model.rerank(features) + list(
            range(len(features), len(ids))
        )
    result = {
        "schema_version": "1.0",
        "dataset": "QASPER",
        "fit_run": str(args.fit_run),
        "validation_run": str(args.validation_run),
        "stage": args.stage,
        "training_cases": model.training_cases,
        "positive_pairs": model.positive_pairs,
        "feature_names": list(model.feature_names),
        "validation": _metrics(validation_rows, documents),
        "model": model.model_dump(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["validation"], ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Rerank every chunk in the selected QASPER paper (candidate-expansion A/B)."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from taskforge.hybrid_retrieval import FastEmbedCrossEncoderReranker  # noqa: E402
from taskforge.rag_evaluation import load_qasper_dataset  # noqa: E402


def _rows(run: Path, limit: int, offset: int) -> list[dict]:
    values = [
        json.loads(line)
        for line in (run / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("stage") == "qdrant_qasper_dense_rerank"
    ]
    return values[offset : offset + limit]


def _metrics(rows: list[dict]) -> dict[str, float]:
    values: dict[int, list[float]] = {key: [] for key in (1, 5, 10, 20, 50)}
    ndcg: list[float] = []
    for row in rows:
        relevant = set(row["relevant_ids"])
        ranked = row["reranked_ids"]
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
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = load_qasper_dataset(args.dataset)
    by_paper: dict[str, list] = defaultdict(list)
    for document in dataset.documents:
        by_paper[str(document.metadata.get("paper_id"))].append(document)
    rows = _rows(args.run, args.limit, args.offset)
    reranker = FastEmbedCrossEncoderReranker(args.model, batch_size=args.batch_size)
    started = time.perf_counter()
    for row in rows:
        paper_id = str(row["filter_request"]["parent_document_ids"][0]).removesuffix(":paper")
        if paper_id.startswith("qasper:"):
            paper_id = paper_id.removeprefix("qasper:")
        documents = by_paper[paper_id]
        ids = [document.document_id for document in documents]
        scores = reranker.score(str(row["query"]), [document.text for document in documents])
        order = sorted(range(len(ids)), key=lambda index: (-scores[index], index))
        row["reranked_ids"] = [ids[index] for index in order]
        row["reranker_scores"] = [scores[index] for index in order]
        row["scope_candidate_count"] = len(ids)
    report = {
        "schema_version": "1.0",
        "dataset": "QASPER",
        "run": str(args.run),
        "model": args.model,
        "cases": len(rows),
        "offset": args.offset,
        "candidate_scope": "all_chunks_in_parent_paper",
        "metrics": _metrics(rows),
        "telemetry": reranker.telemetry(),
        "elapsed_ms": (time.perf_counter() - started) * 1000,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("cases", "model", "metrics", "elapsed_ms")}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

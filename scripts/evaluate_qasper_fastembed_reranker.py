"""Evaluate a FastEmbed cross-encoder on cached QASPER candidates."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from taskforge.hybrid_retrieval import FastEmbedCrossEncoderReranker  # noqa: E402
from taskforge.rag_evaluation import load_qasper_dataset  # noqa: E402
from taskforge.research_reranking import TransformerCrossEncoderReranker  # noqa: E402


def _rows(run: Path, stage: str, limit: int, offset: int) -> list[dict]:
    source = run if run.is_file() else run / "predictions.jsonl"
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
        and (stage == "*" or json.loads(line).get("stage") == stage)
    ]
    return rows[offset : offset + limit]


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
    parser.add_argument(
        "--backend",
        choices=("fastembed", "transformers"),
        default="fastembed",
        help="Cross-encoder implementation used for scoring.",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = load_qasper_dataset(args.dataset)
    documents = {document.document_id: document.text for document in dataset.documents}
    rows = _rows(args.run, "qdrant_qasper_dense_rerank", args.limit, args.offset)
    if not rows:
        raise SystemExit("no cached rows selected")
    reranker = (
        TransformerCrossEncoderReranker(args.model, batch_size=args.batch_size)
        if args.backend == "transformers"
        else FastEmbedCrossEncoderReranker(args.model, batch_size=args.batch_size)
    )
    started = time.perf_counter()
    for row in rows:
        ids = [str(item) for item in row["retrieved_ids"][: args.top_n]]
        scores = reranker.score(str(row["query"]), [documents[item] for item in ids])
        order = sorted(range(len(ids)), key=lambda index: (-scores[index], index))
        row["reranked_ids"] = [ids[index] for index in order]
        row["reranker_scores"] = [scores[index] for index in order]
    report = {
        "schema_version": "1.0",
        "dataset": "QASPER",
        "run": str(args.run),
        "model": args.model,
        "backend": args.backend,
        "cases": len(rows),
        "offset": args.offset,
        "top_n": args.top_n,
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

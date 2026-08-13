"""Evaluate BGE-M3 on cached QASPER candidate lists without re-ingestion."""

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

from taskforge.rag_evaluation import load_qasper_dataset  # noqa: E402
from taskforge.research_reranking import BGEV2M3Reranker  # noqa: E402


def _rows(run: Path, limit: int, offset: int) -> list[dict]:
    rows = [
        json.loads(line)
        for line in (run / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
        and json.loads(line).get("stage") == "qdrant_qasper_dense_rerank"
    ]
    return rows[offset : offset + limit]


def _metrics(rows: list[dict]) -> dict[str, float]:
    output: dict[str, float] = {}
    for k in (1, 5, 10, 20, 50):
        output[f"recall_at_{k}"] = fmean(
            len(set(row["relevant_ids"]).intersection(row["reranked_ids"][:k]))
            / len(row["relevant_ids"])
            for row in rows
        )
    ndcg: list[float] = []
    for row in rows:
        relevant = set(row["relevant_ids"])
        gains = [int(item in relevant) for item in row["reranked_ids"][:10]]
        dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
        ideal = sum(1 / math.log2(index + 2) for index in range(min(10, len(relevant))))
        ndcg.append(dcg / ideal if ideal else 0.0)
    output["ndcg_at_10"] = fmean(ndcg)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = load_qasper_dataset(args.dataset)
    documents = {document.document_id: document.text for document in dataset.documents}
    rows = _rows(args.run, args.limit, args.offset)
    if not rows:
        raise SystemExit("no cached rows selected")
    reranker = BGEV2M3Reranker(str(args.model), device="cpu", batch_size=args.batch_size)
    started = time.perf_counter()
    for row in rows:
        ids = [str(item) for item in row["retrieved_ids"][: args.top_n]]
        texts = [documents[item] for item in ids]
        scores = reranker.score(str(row["query"]), texts)
        order = sorted(range(len(ids)), key=lambda index: (-scores[index], index))
        row["reranked_ids"] = [ids[index] for index in order]
        row["m3_scores"] = [scores[index] for index in order]
    report = {
        "schema_version": "1.0",
        "dataset": "QASPER",
        "run": str(args.run),
        "model": str(args.model),
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
    print(json.dumps({key: report[key] for key in ("cases", "top_n", "metrics", "elapsed_ms")}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

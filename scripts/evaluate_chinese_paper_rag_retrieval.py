"""Evaluate Chinese paper chunk retrieval with Bailian embedding and rerank.

The evaluator intentionally keeps the corpus-native unit as the annotated
Child chunk.  Each question is scoped to its source paper, while the
retriever still runs the same BM25 + dense + rerank path used by the live
research route.  This makes the Chinese run directly comparable with the
English QASPER run without turning paper-level labels into false chunk hits.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.config import Settings  # noqa: E402
from taskforge.knowledge import (  # noqa: E402
    AccessContext,
    InMemoryKnowledgeStore,
    KnowledgeChunk,
)
from taskforge.research_reranking import build_research_reranker  # noqa: E402
from taskforge.research_retrieval import (  # noqa: E402
    ResearchQuery,
    ResearchRetrievalService,
)
from taskforge.semantic_providers import BailianDenseEmbedder  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _recall(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]).intersection(relevant)) / len(relevant)


def _mrr(ranked: list[str], relevant: set[str], k: int) -> float:
    for rank, chunk_id in enumerate(ranked[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def _ndcg(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(ranked[:k], start=1)
        if chunk_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def _build_service(
    chunks: list[KnowledgeChunk],
    settings: Settings,
) -> ResearchRetrievalService:
    if settings.bailian_api_key is None:
        raise RuntimeError("TASKFORGE_BAILIAN_API_KEY is required")
    api_key = settings.bailian_api_key.get_secret_value()
    embedder = BailianDenseEmbedder(
        api_key=api_key,
        base_url=settings.bailian_base_url,
        model_name=settings.bailian_model,
        dimension=settings.bailian_embedding_dimension,
        batch_size=settings.bailian_batch_size,
        timeout_seconds=settings.bailian_timeout_seconds,
        max_retries=settings.bailian_max_retries,
        cache_path=str(settings.bailian_cache_path),
        index_name=settings.bailian_index_name,
    )
    reranker = build_research_reranker(
        "bailian",
        settings.bailian_rerank_model,
        bailian_api_key=api_key,
        bailian_base_url=settings.bailian_rerank_base_url,
        bailian_timeout_seconds=settings.bailian_rerank_timeout_seconds,
        bailian_max_retries=settings.bailian_rerank_max_retries,
    )
    return ResearchRetrievalService(
        InMemoryKnowledgeStore(chunks),
        dense_embedder=embedder,
        reranker=reranker,
        multilingual_dense_embedder=embedder,
        multilingual_reranker=reranker,
        graph_enabled=False,
        operator_budget_standard=0,
        operator_budget_rigorous=0,
    )


def run(
    dataset_dir: Path,
    output_path: Path,
    *,
    limit: int = 90,
    offset: int = 0,
    candidate_k: int = 50,
) -> dict[str, object]:
    if not 1 <= limit <= 10_000:
        raise ValueError("limit must be between 1 and 10000")
    if not 10 <= candidate_k <= 100:
        raise ValueError("candidate_k must be between 10 and 100")
    queries_path = dataset_dir / "queries.jsonl"
    qrels_path = dataset_dir / "qrels.jsonl"
    chunks_path = dataset_dir / "chunks.jsonl.gz"
    for path in (queries_path, qrels_path, chunks_path):
        if not path.exists():
            raise FileNotFoundError(path)

    query_rows = _read_jsonl(queries_path)[offset : offset + limit]
    qrels_by_query: dict[str, set[str]] = defaultdict(set)
    for row in _read_jsonl(qrels_path):
        if int(row.get("relevance", 0)) > 0:
            qrels_by_query[str(row["query_id"])].add(str(row["document_id"]))
    chunk_rows = _read_jsonl_gz(chunks_path)
    chunks: list[KnowledgeChunk] = []
    for row in chunk_rows:
        paper_id = str(row["paper_id"])
        chunk_id = str(row["chunk_id"])
        chunks.append(
            KnowledgeChunk(
                chunk_id=chunk_id,
                tenant_id="zhpaper-rag-30",
                text=str(row["text"]),
                source_uri=f"chinese-paper://{paper_id}",
                document_id=str(row.get("document_id") or paper_id),
                acl=frozenset({"user:evaluator"}),
                metadata={
                    "knowledge_base_id": f"zhpaper-paper:{paper_id}",
                    "paper_id": paper_id,
                    "document_id": str(row.get("document_id") or paper_id),
                    "title": row.get("title"),
                    "language": "zh",
                    "chunk_index": row.get("chunk_index"),
                    "retrieval_text": str(row["text"]),
                },
            )
        )

    settings = Settings()
    service = _build_service(chunks, settings)
    principal = AccessContext(tenant_id="zhpaper-rag-30", user_id="evaluator")
    rows: list[dict[str, object]] = []
    started = perf_counter()
    for query_row in query_rows:
        query_id = str(query_row["query_id"])
        paper_id = str(query_row["paper_id"])
        relevant = qrels_by_query.get(query_id, set())
        query_started = perf_counter()
        result = service.search(
            ResearchQuery(
                query=str(query_row["query"]),
                top_k=min(candidate_k, 50),
                candidate_k=candidate_k,
                knowledge_base_ids=(f"zhpaper-paper:{paper_id}",),
            ),
            principal,
        )
        latency_ms = (perf_counter() - query_started) * 1_000
        retrieved_ids = [item.chunk_id for item in result.evidence]
        rows.append(
            {
                "query_id": query_id,
                "paper_id": paper_id,
                "question_type": query_row.get("question_type"),
                "query": query_row["query"],
                "relevant_chunk_ids": sorted(relevant),
                "retrieved_ids": retrieved_ids,
                "recall_at_k": {
                    str(k): _recall(retrieved_ids, relevant, k)
                    for k in (1, 5, 10, 20, 50)
                },
                "mrr_at_k": {
                    str(k): _mrr(retrieved_ids, relevant, k)
                    for k in (1, 5, 10, 20, 50)
                },
                "ndcg_at_k": {
                    str(k): _ndcg(retrieved_ids, relevant, k)
                    for k in (1, 5, 10, 20, 50)
                },
                "candidate_count": result.candidate_count,
                "retrieval_route": result.retrieval_route,
                "latency_ms": latency_ms,
            }
        )

    def aggregate(subset: list[dict[str, object]]) -> dict[str, float]:
        if not subset:
            return {}
        return {
            **{
                f"recall_at_{k}": statistics.fmean(
                    float(row["recall_at_k"][str(k)])  # type: ignore[index]
                    for row in subset
                )
                for k in (1, 5, 10, 20, 50)
            },
            **{
                f"mrr_at_{k}": statistics.fmean(
                    float(row["mrr_at_k"][str(k)])  # type: ignore[index]
                    for row in subset
                )
                for k in (1, 5, 10, 20, 50)
            },
            **{
                f"ndcg_at_{k}": statistics.fmean(
                    float(row["ndcg_at_k"][str(k)])  # type: ignore[index]
                    for row in subset
                )
                for k in (1, 5, 10, 20, 50)
            },
        }

    latencies = [float(row["latency_ms"]) for row in rows]
    by_type: dict[str, dict[str, float]] = {}
    for question_type in sorted({str(row.get("question_type")) for row in rows}):
        by_type[question_type] = aggregate(
            [row for row in rows if str(row.get("question_type")) == question_type]
        )
    metrics = aggregate(rows)
    metrics.update({"p50_ms": _percentile(latencies, 0.50), "p95_ms": _percentile(latencies, 0.95)})
    report: dict[str, object] = {
        "schema_version": "1.0",
        "evaluation_type": "chinese_paper_chunk_retrieval",
        "benchmark_track": "annotated_chunk_native_retrieval",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": str(dataset_dir),
        "dataset_sha256": {
            "queries": _sha256(queries_path),
            "qrels": _sha256(qrels_path),
            "chunks": _sha256(chunks_path),
        },
        "cases": len(rows),
        "case_offset": offset,
        "pipeline": ["annotated_chinese_chunks", "bm25", "bailian_dense", "bailian_rerank", "search"],
        "retrieval": {
            "query_profile": "original",
            "backend": "bailian",
            "semantic_model": settings.bailian_model,
            "reranker_backend": "bailian",
            "reranker_model": settings.bailian_rerank_model,
            "candidate_k": candidate_k,
            "language": "zh",
        },
        "metrics": metrics,
        "metrics_by_question_type": by_type,
        "rows": rows,
        "elapsed_ms": (perf_counter() - started) * 1_000,
        "limitations": [
            "Chunk-level labels are silver-curated and source-verified, not human-reviewed.",
            "Each query is scoped to its source paper to measure chunk ranking rather than paper identification.",
            "MRR/NDCG use the curated relevant chunk set for each query.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "eval" / "queries" / "chinese-paper-rag-30-v2-precision",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=90)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--candidate-k", type=int, default=50)
    args = parser.parse_args()
    report = run(
        args.dataset_dir,
        args.output,
        limit=args.limit,
        offset=args.offset,
        candidate_k=args.candidate_k,
    )
    print(json.dumps({"output": str(args.output), "metrics": report["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

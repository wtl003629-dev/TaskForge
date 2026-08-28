"""Evaluate QASPER paragraph retrieval without PDF parsing or page proxies."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.hybrid_retrieval import FastEmbedEmbedder  # noqa: E402
from taskforge.knowledge import (  # noqa: E402
    AccessContext,
    InMemoryKnowledgeStore,
    KnowledgeChunk,
)
from taskforge.qasper_alignment import paragraph_recall_at_k  # noqa: E402
from taskforge.rag_evaluation import load_qasper_dataset  # noqa: E402
from taskforge.config import Settings  # noqa: E402
from taskforge.research_reranking import build_research_reranker  # noqa: E402
from taskforge.semantic_providers import BailianDenseEmbedder  # noqa: E402
from taskforge.research_retrieval import (  # noqa: E402
    ResearchQuery,
    ResearchRetrievalService,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[
        min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    ]


def run(
    dataset_path: Path,
    split_path: Path,
    output_path: Path,
    *,
    limit: int = 100,
    offset: int = 0,
    backend: str = "bm25",
    semantic_model: str = "BAAI/bge-small-en-v1.5",
    reranker_model: str | None = None,
    reranker_backend: str = "fastembed",
    candidate_k: int = 50,
) -> dict[str, object]:
    if backend not in {"bm25", "fastembed", "bailian"}:
        raise ValueError("backend must be bm25, fastembed, or bailian")
    if not 1 <= limit <= 10_000:
        raise ValueError("limit must be between 1 and 10000")
    if not 10 <= candidate_k <= 100:
        raise ValueError("candidate_k must be between 10 and 100")
    dataset = load_qasper_dataset(dataset_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    locked_ids = [str(item) for item in split["case_ids"]][
        offset : offset + limit
    ]
    case_by_id = {case.case_id: case for case in dataset.cases}
    missing = [case_id for case_id in locked_ids if case_id not in case_by_id]
    if missing:
        raise ValueError(f"locked QASPER cases are missing: {missing[:3]}")
    cases = [case_by_id[case_id] for case_id in locked_ids]
    paper_ids = {str(case.metadata["paper_id"]) for case in cases}
    document_by_id = {
        document.document_id: document
        for document in dataset.documents
        if str(document.metadata.get("paper_id")) in paper_ids
    }
    chunks = [
        KnowledgeChunk(
            chunk_id=document.document_id,
            tenant_id="qasper-corpus-native",
            text=document.text,
            source_uri=f"qasper://{document.metadata['paper_id']}",
            document_id=str(document.metadata["parent_document_id"]),
            acl=frozenset({"user:evaluator"}),
            metadata={
                **document.metadata,
                "knowledge_base_id": (
                    f"qasper-paper:{document.metadata['paper_id']}"
                ),
                "evidence_id": document.document_id,
            },
        )
        for document in document_by_id.values()
    ]
    settings = Settings() if backend == "bailian" or reranker_backend == "bailian" else None
    if backend == "fastembed":
        embedder = FastEmbedEmbedder(
            semantic_model,
            cache_path=str(
                PROJECT_ROOT
                / ".taskforge"
                / "eval-cache"
                / "qasper-corpus-native-embeddings.sqlite3"
            ),
        )
    elif backend == "bailian":
        if settings is None or settings.bailian_api_key is None:
            raise RuntimeError(
                "TASKFORGE_BAILIAN_API_KEY is required for --backend bailian"
            )
        embedder = BailianDenseEmbedder(
            api_key=settings.bailian_api_key.get_secret_value(),
            base_url=settings.bailian_base_url,
            model_name=settings.bailian_model,
            dimension=settings.bailian_embedding_dimension,
            batch_size=settings.bailian_batch_size,
            timeout_seconds=settings.bailian_timeout_seconds,
            max_retries=settings.bailian_max_retries,
            cache_path=str(settings.bailian_cache_path),
            index_name=settings.bailian_index_name,
        )
    else:
        embedder = None
    if reranker_backend == "bailian":
        settings = settings or Settings()
        api_key = (
            settings.bailian_api_key.get_secret_value()
            if settings.bailian_api_key is not None
            else None
        )
        if not api_key:
            raise RuntimeError(
                "TASKFORGE_BAILIAN_API_KEY is required for --reranker-backend bailian"
            )
        reranker = build_research_reranker(
            "bailian",
            reranker_model or settings.bailian_rerank_model,
            bailian_api_key=api_key,
            bailian_base_url=settings.bailian_rerank_base_url,
            bailian_timeout_seconds=settings.bailian_rerank_timeout_seconds,
            bailian_max_retries=settings.bailian_rerank_max_retries,
        )
    else:
        reranker = (
            build_research_reranker(
                reranker_backend,  # type: ignore[arg-type]
                reranker_model,
            )
            if reranker_model
            else None
        )
    service = ResearchRetrievalService(
        InMemoryKnowledgeStore(chunks),
        dense_embedder=embedder,
        reranker=reranker,
        graph_enabled=False,
        operator_budget_standard=0,
        operator_budget_rigorous=0,
    )
    principal = AccessContext(
        tenant_id="qasper-corpus-native",
        user_id="evaluator",
    )
    rows: list[dict[str, object]] = []
    started = perf_counter()
    for case in cases:
        if case.qasper_gold is None:
            raise RuntimeError(
                f"QASPER case lacks multi-annotation gold labels: {case.case_id}"
            )
        paper_id = str(case.metadata["paper_id"])
        query_started = perf_counter()
        result = service.search(
            ResearchQuery(
                query=case.query,
                top_k=min(candidate_k, 50),
                candidate_k=candidate_k,
                knowledge_base_ids=(f"qasper-paper:{paper_id}",),
            ),
            principal,
        )
        latency_ms = (perf_counter() - query_started) * 1_000
        retrieved_ids = [item.chunk_id for item in result.evidence]
        recall_results = {
            str(k): paragraph_recall_at_k(
                case.qasper_gold,
                retrieved_ids,
                k,
            )
            for k in (1, 5, 10, 50)
        }
        rows.append(
            {
                "case_id": case.case_id,
                "paper_id": paper_id,
                "query": case.query,
                "gold_annotation_count": len(
                    case.qasper_gold.evidence_sets
                ),
                "gold_evidence_set_sizes": [
                    len(item.units) for item in case.qasper_gold.evidence_sets
                ],
                "recall_at_k": {
                    key: value.recall
                    for key, value in recall_results.items()
                },
                "selected_annotation_at_k": {
                    key: value.selected_annotation_id
                    for key, value in recall_results.items()
                },
                "retrieved_ids": retrieved_ids,
                "retrieved_evidence": [
                    {
                        "document_id": item.chunk_id,
                        "score": item.score,
                        "text": document_by_id[item.chunk_id].text,
                    }
                    for item in result.evidence
                ],
                "candidate_count": result.candidate_count,
                "latency_ms": latency_ms,
            }
        )
    latencies = [float(row["latency_ms"]) for row in rows]
    metrics = {
        f"recall_at_{k}": statistics.fmean(
            float(row["recall_at_k"][str(k)])  # type: ignore[index]
            for row in rows
        )
        for k in (1, 5, 10, 50)
    }
    metrics.update(
        {
            "p50_ms": _percentile(latencies, 0.50),
            "p95_ms": _percentile(latencies, 0.95),
        }
    )
    report: dict[str, object] = {
        "schema_version": "1.0",
        "evaluation_type": "qasper_corpus_native_retrieval",
        "benchmark_track": "corpus_native_retrieval",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": "QASPER v0.3 official paragraphs",
        "source_dataset": str(dataset_path),
        "source_dataset_sha256": _sha256(dataset_path),
        "split": str(split_path),
        "split_sha256": _sha256(split_path),
        "cases": len(rows),
        "case_offset": offset,
        "pipeline": ["official_paragraphs", "index", "search"],
        "retrieval": {
            "query_profile": "original",
            "backend": backend,
            "semantic_model": (
                semantic_model
                if backend == "fastembed"
                else settings.bailian_model
                if backend == "bailian" and settings is not None
                else None
            ),
            "reranker_backend": reranker_backend if (reranker_model or reranker_backend == "bailian") else None,
            "reranker_model": (
                reranker_model
                if reranker_model
                else settings.bailian_rerank_model
                if reranker_backend == "bailian" and settings is not None
                else None
            ),
            "candidate_k": candidate_k,
        },
        "metrics": metrics,
        "rows": rows,
        "elapsed_ms": (perf_counter() - started) * 1_000,
        "limitations": [
            "This track bypasses PDF parsing and measures retrieval over official QASPER paragraphs.",
            "Only Recall@1/5/10/50 are retrieval quality metrics; no page proxy is used.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            PROJECT_ROOT / ".taskforge" / "eval-cache" / "qasper-dev-v0.3.json"
        ),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=(
            PROJECT_ROOT
            / "eval"
            / "splits"
            / "qasper-dev-general-100-v1.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--backend", choices=("bm25", "fastembed", "bailian"), default="bm25"
    )
    parser.add_argument(
        "--semantic-model", default="BAAI/bge-small-en-v1.5"
    )
    parser.add_argument("--reranker-model", default=None)
    parser.add_argument(
        "--reranker-backend",
        choices=("fastembed", "fastembed_ensemble", "flagembedding", "transformers", "bailian"),
        default="fastembed",
    )
    parser.add_argument("--candidate-k", type=int, default=50)
    args = parser.parse_args()
    report = run(
        args.dataset,
        args.split,
        args.output,
        limit=args.limit,
        offset=args.offset,
        backend=args.backend,
        semantic_model=args.semantic_model,
        reranker_model=args.reranker_model,
        reranker_backend=args.reranker_backend,
        candidate_k=args.candidate_k,
    )
    print(json.dumps({"output": str(args.output), "metrics": report["metrics"]}))


if __name__ == "__main__":
    main()

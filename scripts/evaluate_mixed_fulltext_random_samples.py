"""Evaluate RAG retrieval on balanced random samples of the mixed paper corpus.

This is a reproducible corpus-scale smoke test, not a replacement for a
human-judged Chinese benchmark.  English cases reuse one real QASPER question
per sampled paper when available.  Chinese cases use a title probe because the
new Chinese full-text corpus does not yet have Chinese qrels.  The target is
therefore the sampled paper, and metrics are reported at paper level after
collapsing ranked chunks to their first document occurrence.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

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

DEFAULT_DATASET = PROJECT_ROOT / ".taskforge" / "datasets" / "mixed-paper-fulltext-v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "eval" / "reports" / "mixed-fulltext-random-rag-v1"
DEFAULT_SEED = 20260827
DEFAULT_SIZES = (10, 20, 30, 40, 50)
RECALL_KS = (1, 5, 10, 20, 50)
TENANT_ID = "mixed-fulltext-random-eval"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl_gz(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            yield value


def _text_probe(row: Mapping[str, Any], queries_by_document: Mapping[str, list[dict[str, Any]]]) -> tuple[str, str, str | None]:
    document_id = str(row["document_id"])
    if row.get("language") == "en" and queries_by_document.get(document_id):
        query_row = queries_by_document[document_id][0]
        return str(query_row["query"]).strip(), "qasper_question", str(query_row["query_id"])
    title = " ".join(str(row.get("title") or "").split())
    if title:
        return title, "title_probe", None
    abstract = " ".join(str(row.get("abstract") or "").split())
    if abstract:
        return abstract[:240], "abstract_probe", None
    raise ValueError(f"document has no usable title or abstract: {document_id}")


def _sample_documents(
    documents: list[dict[str, Any]],
    *,
    size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if size < 2 or size % 2:
        raise ValueError("sample sizes must be even and at least 2 for balanced zh/en sampling")
    by_language: dict[str, list[dict[str, Any]]] = {"en": [], "zh": []}
    for row in documents:
        language = str(row.get("language") or "")
        if language in by_language:
            by_language[language].append(row)
    rng = random.Random(seed + size * 1009)
    en_count = size // 2
    zh_count = size - en_count
    if len(by_language["en"]) < en_count or len(by_language["zh"]) < zh_count:
        raise ValueError(f"not enough documents for balanced sample size {size}")
    selected = rng.sample(by_language["en"], en_count) + rng.sample(by_language["zh"], zh_count)
    return sorted(selected, key=lambda row: (str(row["language"]), str(row["document_id"])))


def _chunks_for_documents(
    documents: Iterable[Mapping[str, Any]],
    *,
    chunk_chars: int,
) -> list[KnowledgeChunk]:
    if not 256 <= chunk_chars <= 50_000:
        raise ValueError("chunk_chars must be between 256 and 50000")
    chunks: list[KnowledgeChunk] = []
    for row in documents:
        document_id = str(row["document_id"])
        title = " ".join(str(row.get("title") or "").split())
        text = str(row.get("text") or "").strip()
        if not text:
            raise ValueError(f"empty document text: {document_id}")
        for index, offset in enumerate(range(0, len(text), chunk_chars)):
            body = text[offset : offset + chunk_chars].strip()
            if not body:
                continue
            # Repeat the title in every retrieval chunk.  This is a bounded
            # retrieval projection; the authoritative body remains unchanged
            # in the mixed corpus and is only used here for indexing.
            chunk_text = f"TITLE: {title}\n\n{body}" if title else body
            chunk_id = f"{document_id}:flat-{index:05d}"
            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    tenant_id=TENANT_ID,
                    text=chunk_text,
                    source_uri=f"dataset://mixed-paper-fulltext/{document_id}",
                    document_id=document_id,
                    metadata={
                        "paper_id": str(row.get("paper_id") or document_id),
                        "title": title,
                        "language": str(row.get("language") or ""),
                        "source_dataset": str(row.get("source_dataset") or ""),
                        "chunk_index": index,
                        "retrieval_role": "child",
                    },
                )
            )
    return chunks


def _ranked_document_ids(result: Any) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for evidence in result.evidence:
        raw = evidence.chunk_id.rsplit(":flat-", 1)[0]
        if raw not in seen:
            seen.add(raw)
            ordered.append(raw)
    return ordered


def _metrics(ranked: list[str], target: str) -> dict[str, float]:
    try:
        rank = ranked.index(target) + 1
    except ValueError:
        rank = None
    values: dict[str, float] = {}
    for k in RECALL_KS:
        values[f"recall_at_{k}"] = 1.0 if rank is not None and rank <= k else 0.0
        values[f"mrr_at_{k}"] = 1.0 / rank if rank is not None and rank <= k else 0.0
        values[f"ndcg_at_{k}"] = (
            1.0 / math.log2(rank + 1) if rank is not None and rank <= k else 0.0
        )
    values["rank"] = float(rank or 0)
    return values


def _mean(rows: Iterable[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return statistics.fmean(values) if values else 0.0


def _load_inputs(dataset_dir: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    corpus_path = dataset_dir / "corpus.jsonl.gz"
    queries_path = dataset_dir / "queries.jsonl.gz"
    if not corpus_path.is_file():
        raise FileNotFoundError(corpus_path)
    documents = [row for row in _read_jsonl_gz(corpus_path) if row.get("document_type") == "full_text"]
    queries_by_document: dict[str, list[dict[str, Any]]] = {}
    if queries_path.is_file():
        for row in _read_jsonl_gz(queries_path):
            document_id = str(row.get("document_id") or "")
            if row.get("source_dataset") == "QASPER" and document_id:
                queries_by_document.setdefault(document_id, []).append(row)
        for values in queries_by_document.values():
            values.sort(key=lambda row: str(row.get("query_id") or ""))
    ids = [str(row.get("document_id") or "") for row in documents]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("mixed corpus must have unique non-empty document IDs")
    if not documents:
        raise ValueError("mixed corpus has no full-text documents")
    return documents, queries_by_document


def _make_embedder(backend: str):
    if backend == "bm25":
        return None
    if backend != "bailian":
        raise ValueError("backend must be bm25 or bailian")
    settings = Settings()
    if settings.bailian_api_key is None:
        raise ValueError("Bailian backend requires TASKFORGE_BAILIAN_API_KEY")
    return BailianDenseEmbedder(
        api_key=settings.bailian_api_key.get_secret_value(),
        base_url=settings.bailian_base_url,
        model_name=settings.bailian_model,
        dimension=settings.bailian_embedding_dimension,
        batch_size=settings.bailian_batch_size,
        timeout_seconds=settings.bailian_timeout_seconds,
        max_retries=settings.bailian_max_retries,
        cache_path=settings.bailian_cache_path,
        index_name=settings.bailian_index_name,
    )


def _make_reranker(enabled: bool):
    if not enabled:
        return None
    settings = Settings()
    model = settings.research_reranker_model
    if model is None:
        raise ValueError("reranker requested but research_reranker_model is not configured")
    return build_research_reranker(
        settings.research_reranker_backend,
        model,
        device=settings.research_reranker_device,
        batch_size=settings.research_reranker_batch_size,
        fastembed_cache_dir=settings.fastembed_model_cache_root,
    )


def _evaluate_sample(
    documents: list[dict[str, Any]],
    queries_by_document: Mapping[str, list[dict[str, Any]]],
    *,
    size: int,
    seed: int,
    chunk_chars: int,
    backend: str,
    embedder: Any | None,
    reranker: Any | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = _sample_documents(documents, size=size, seed=seed)
    selected_ids = {str(row["document_id"]) for row in selected}
    chunks = _chunks_for_documents(selected, chunk_chars=chunk_chars)
    store = InMemoryKnowledgeStore(chunks)
    service = ResearchRetrievalService(
        store,
        dense_embedder=embedder,
        reranker=reranker,
        graph_enabled=False,
        parent_aware_rerank_enabled=False,
        operator_budget_standard=0,
        operator_budget_rigorous=0,
        index_cache_size=2,
    )
    principal = AccessContext(tenant_id=TENANT_ID, user_id="random-eval")
    cases: list[dict[str, Any]] = []
    for row in selected:
        query, query_type, query_id = _text_probe(row, queries_by_document)
        cases.append(
            {
                "case_id": f"random:{size}:{row['document_id']}",
                "document_id": str(row["document_id"]),
                "language": str(row["language"]),
                "source_dataset": str(row["source_dataset"]),
                "query": query,
                "query_type": query_type,
                "query_id": query_id,
            }
        )

    # Build the index once before timing query latency.  The provider cache
    # also makes the subsequent measured pass independent of index warm-up.
    warmup_request = ResearchQuery(query=cases[0]["query"], top_k=8, candidate_k=50)
    service.search(warmup_request, principal)
    rows: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        result = service.search(
            ResearchQuery(query=case["query"], top_k=8, candidate_k=50),
            principal,
        )
        latency_ms = (time.perf_counter() - started) * 1_000
        ranked = _ranked_document_ids(result)
        if not set(ranked).issubset(selected_ids):
            raise RuntimeError("retriever returned a document outside the sample scope")
        values = _metrics(ranked, case["document_id"])
        rows.append(
            {
                **case,
                "ranked_document_ids": ranked,
                "retrieved_chunk_ids": [item.chunk_id for item in result.evidence],
                "retrieval_route": result.retrieval_route,
                "candidate_count": result.candidate_count,
                "retrieval_rounds": result.retrieval_rounds,
                "latency_ms": latency_ms,
                **values,
            }
        )

    metrics: dict[str, Any] = {
        "documents": size,
        "chunks": len(chunks),
        "queries": len(rows),
        "documents_by_language": dict(Counter(str(row["language"]) for row in selected)),
        "queries_by_type": dict(Counter(str(row["query_type"]) for row in rows)),
        "retrieval_route_counts": dict(Counter(str(row["retrieval_route"]) for row in rows)),
        "p50_ms": statistics.median(float(row["latency_ms"]) for row in rows),
        "p95_ms": sorted(float(row["latency_ms"]) for row in rows)[min(len(rows) - 1, math.ceil(len(rows) * 0.95) - 1)],
    }
    for k in RECALL_KS:
        metrics[f"recall_at_{k}"] = _mean(rows, f"recall_at_{k}")
        metrics[f"mrr_at_{k}"] = _mean(rows, f"mrr_at_{k}")
        metrics[f"ndcg_at_{k}"] = _mean(rows, f"ndcg_at_{k}")
    sample = {
        "sample_size": size,
        "seed": seed + size * 1009,
        "documents": [
            {
                "document_id": str(row["document_id"]),
                "paper_id": str(row.get("paper_id") or row["document_id"]),
                "language": str(row["language"]),
                "title": str(row.get("title") or ""),
                "source_dataset": str(row["source_dataset"]),
            }
            for row in selected
        ],
        "cases": cases,
    }
    report = {
        "sample_size": size,
        "backend": backend,
        "chunk_chars": chunk_chars,
        "metrics": metrics,
        "rows": rows,
    }
    return sample, report


def run(
    *,
    dataset_dir: Path,
    output_dir: Path,
    seed: int,
    sizes: tuple[int, ...],
    chunk_chars: int,
    backend: str,
    with_reranker: bool,
    confirm_external_calls: bool,
) -> dict[str, Any]:
    if backend == "bailian" and not confirm_external_calls:
        raise ValueError("Bailian evaluation requires --confirm-external-calls")
    documents, queries_by_document = _load_inputs(dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = dataset_dir / "corpus.jsonl.gz"
    all_samples: dict[str, Any] = {}
    all_reports: dict[str, Any] = {}
    embedder = _make_embedder(backend)
    reranker = _make_reranker(with_reranker)
    try:
        for size in sizes:
            sample, report = _evaluate_sample(
                documents,
                queries_by_document,
                size=size,
                seed=seed,
                chunk_chars=chunk_chars,
                backend=backend,
                embedder=embedder,
                reranker=reranker,
            )
            key = str(size)
            all_samples[key] = sample
            all_reports[key] = report
            (output_dir / f"sample-{size}.json").write_text(
                json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            (output_dir / f"report-{size}.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    finally:
        if embedder is not None:
            embedder.close()

    summary = {
        "schema_version": "1.0",
        "evaluation_type": "mixed_fulltext_rag_retrieval_random_probe",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": str(dataset_dir.resolve()),
        "dataset_corpus_sha256": _sha256(corpus_path),
        "seed": seed,
        "sizes": list(sizes),
        "backend": backend,
        "with_reranker": with_reranker,
        "chunk_chars": chunk_chars,
        "reports": all_reports,
        "samples": all_samples,
        "limitations": [
            "This validates paper-level retrieval on random sampled documents; it is not a human-judged Chinese QA benchmark.",
            "English samples use one QASPER question per paper when available; Chinese samples use a title probe because Chinese qrels are not available.",
            "Metrics collapse flat chunks to the first occurrence of a paper, so they measure paper Recall/MRR/nDCG rather than evidence-paragraph recall.",
            "A Bailian run measures the configured embedding route; it does not claim answer-generation quality.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--chunk-chars", type=int, default=2_000)
    parser.add_argument("--backend", choices=("bm25", "bailian"), default="bailian")
    parser.add_argument(
        "--with-reranker",
        action="store_true",
        help="Use the configured research reranker in addition to retrieval.",
    )
    parser.add_argument("--confirm-external-calls", action="store_true")
    args = parser.parse_args()
    if not args.sizes or any(size < 2 or size % 2 for size in args.sizes):
        raise SystemExit("--sizes must contain positive even sample sizes")
    summary = run(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        sizes=tuple(args.sizes),
        chunk_chars=args.chunk_chars,
        backend=args.backend,
        with_reranker=args.with_reranker,
        confirm_external_calls=args.confirm_external_calls,
    )
    compact_reports = {
        size: report["metrics"]
        for size, report in summary["reports"].items()
    }
    print(
        json.dumps(
            {"output_dir": str(args.output_dir), "reports": compact_reports},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

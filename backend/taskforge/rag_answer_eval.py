"""End-to-end RAG answer evaluation: retrieve, generate, score.

This is the "scale": for every locked case it runs a real retrieval step and a
real model answer generation, then scores the generated answer against the
dataset's gold answer with exact match and token F1.  Unlike the retrieval
ablation (which stops at recall@k), this measures whether the pipeline answers
questions correctly.  The provider is injected and the run is billable, so the
CLI requires ``--confirm-live-call``.
"""

from __future__ import annotations

import os
import platform
import shutil
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import Field

from . import __version__
from .domain import AgentProfile, StrictModel, Task, utc_now
from .hybrid_retrieval import (
    BM25Index,
    DeterministicHashEmbedder,
    FastEmbedEmbedder,
    LexicalOverlapFallbackReranker,
    QdrantHybridIndex,
)
from .providers import ModelProvider
from .rag_evaluation import (
    RAGEvalCase,
    answer_exact_match,
    answer_token_f1,
)
from .rag_experiment import (
    ExperimentDatasetConfig,
    ExperimentFilterConfig,
    ExperimentRetrievalConfig,
    _canonical_json,
    _deduped_document_ids,
    _hybrid_chunks,
    _prepare_dataset,
    _search_request,
    _sha256_bytes,
    _source_hashes,
    _write_new,
)

RetrieverName = Literal["bm25", "qdrant_rrf", "qdrant_rrf_rerank"]


class RAGAnswerEvalConfig(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset: ExperimentDatasetConfig = Field(default_factory=ExperimentDatasetConfig)
    retrieval: ExperimentRetrievalConfig = Field(
        default_factory=ExperimentRetrievalConfig
    )
    filters: ExperimentFilterConfig = Field(default_factory=ExperimentFilterConfig)
    retriever: RetrieverName = "bm25"
    model: str = Field(min_length=1)
    evidence_top_k: int = Field(default=5, ge=1, le=20)
    max_evidence_chars: int = Field(default=16_000, ge=500, le=80_000)
    max_cases: int | None = Field(default=None, ge=1)


async def _generate_answer(
    provider: ModelProvider,
    case: RAGEvalCase,
    evidence_texts: Sequence[str],
    *,
    model: str,
    max_evidence_chars: int,
) -> str | None:
    """Ask the model to answer the question from the retrieved evidence only."""

    budget = max_evidence_chars
    joined: list[str] = []
    used = 0
    for text in evidence_texts:
        if used >= budget:
            break
        take = text[: budget - used]
        joined.append(take)
        used += len(take)
    task = Task(
        tenant_id="tenant-answer-eval",
        user_id="user-answer-eval",
        goal=case.query,
    )
    profile = AgentProfile(
        id="answer-eval-agent",
        name="Answer eval agent",
        instructions=(
            "Answer the research question using ONLY the evidence provided under "
            "UNTRUSTED EVIDENCE CONTEXT. Do not use outside knowledge. If the "
            "evidence does not contain the answer, answer exactly UNKNOWN. "
            "Answer with the fact only; do not add reasoning."
        ),
        model=model,
        allowed_tools=[],
    )
    turn = await provider.complete(
        task=task,
        profile=profile,
        context={
            "assembled": {
                "evidence": joined,
                "question": case.query,
            },
            "trajectory": [],
        },
        tools=[],
    )
    if turn.kind != "final" or not turn.final_answer:
        return None
    return turn.final_answer


def _indexes(
    chunks: Sequence[Any],
    config: RAGAnswerEvalConfig,
) -> dict[str, Any]:
    lexical = BM25Index(
        chunks,
        k1=config.retrieval.bm25_k1,
        b=config.retrieval.bm25_b,
        field_weights=config.retrieval.bm25_field_weights,
    )
    if config.retrieval.semantic_embedding:
        embedder = FastEmbedEmbedder(config.retrieval.semantic_model)
    else:
        embedder = DeterministicHashEmbedder(config.retrieval.hash_dimension)
    qdrant = QdrantHybridIndex.in_memory(
        collection_name="taskforge-answer-eval",
        embedder=embedder,
        reranker=LexicalOverlapFallbackReranker(),
    )
    qdrant.upsert(chunks)
    return {
        "bm25": lexical,
        "qdrant_rrf": qdrant,
        "qdrant_rrf_rerank": qdrant,
    }


async def run_rag_answer_eval(
    *,
    output_dir: str | Path,
    config: RAGAnswerEvalConfig,
    provider: ModelProvider,
    repository_root: str | Path,
    created_at: datetime | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Run retrieve->generate->score and publish evidence artifacts."""

    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"answer eval output already exists: {target}")
    repository = Path(repository_root).resolve()
    if not repository.is_dir():
        raise FileNotFoundError(f"repository root does not exist: {repository}")
    timestamp = created_at or utc_now()
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("created_at must be timezone-aware")
    timestamp = timestamp.astimezone(UTC)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        prepared = _prepare_dataset(config, repository, staging)
        chunks = _hybrid_chunks(prepared.dataset, prepared.cases, config)
        indexes = _indexes(chunks, config)
        index = indexes[config.retriever]
        cases = prepared.cases
        if config.max_cases is not None:
            cases = cases[: config.max_cases]

        rows: list[dict[str, Any]] = []
        for case in cases:
            request = _search_request(
                case.query,
                config,
                rerank=config.retriever == "qdrant_rrf_rerank",
            )
            response = index.search(request)
            retrieved_ids = _deduped_document_ids(
                response.hits, max_documents=config.evidence_top_k
            )
            # Feed the model the top retrieved chunk texts directly (not a
            # whole-document truncation) so the answer sentence is not cut away.
            evidence = [
                hit.chunk.text
                for hit in response.hits[: config.evidence_top_k]
            ]
            answer = await _generate_answer(
                provider,
                case,
                evidence,
                model=config.model,
                max_evidence_chars=config.max_evidence_chars,
            )
            predicted = answer or ""
            exact = answer_exact_match(predicted, case.answer)
            f1 = answer_token_f1(predicted, case.answer)
            rows.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "query": case.query,
                    "gold_answer": case.answer,
                    "generated_answer": predicted,
                    "retrieved_ids": retrieved_ids,
                    "exact_match": exact,
                    "token_f1": f1,
                    "model": config.model,
                    "retriever": config.retriever,
                }
            )

        exact_scores = [row["exact_match"] for row in rows]
        f1_scores = [row["token_f1"] for row in rows]
        by_category: dict[str, dict[str, float]] = {}
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["category"], []).append(row)
        for category, group in grouped.items():
            by_category[category] = {
                "exact_match": sum(item["exact_match"] for item in group) / len(group),
                "token_f1": sum(item["token_f1"] for item in group) / len(group),
                "cases": len(group),
            }
        metrics: dict[str, Any] = {
            "schema_version": "1.0",
            "retriever": config.retriever,
            "model": config.model,
            "total_cases": len(rows),
            "exact_match_accuracy": (
                sum(exact_scores) / len(exact_scores) if exact_scores else 0.0
            ),
            "avg_token_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            "by_category": by_category,
            "category_counts": dict(
                sorted(Counter(case.category for case in cases).items())
            ),
        }

        predictions_payload = _canonical_json(rows) + b"\n"
        metrics_payload = _canonical_json(metrics) + b"\n"
        effective_config = config.model_dump(mode="json")
        effective_config_hash = _sha256_bytes(_canonical_json(effective_config))
        source_hashes = _source_hashes()
        code_hash = _sha256_bytes(_canonical_json(source_hashes))
        run_id = _sha256_bytes(
            "\0".join(
                (
                    str(prepared.provenance["normalized_sha256"]),
                    effective_config_hash,
                    code_hash,
                )
            ).encode("ascii")
        )[:20]
        manifest: dict[str, Any] = {
            "schema_version": "1.0",
            "run_id": run_id,
            "created_at": timestamp.isoformat().replace("+00:00", "Z"),
            "dataset": dict(prepared.provenance),
            "config": {
                "effective": effective_config,
                "sha256": effective_config_hash,
            },
            "code": {
                "package": "taskforge-agent",
                "package_version": __version__,
                "python": platform.python_version(),
                "source_sha256": source_hashes,
                "sha256": code_hash,
            },
            "artifacts": {
                "predictions.jsonl": {
                    "sha256": _sha256_bytes(predictions_payload),
                    "size_bytes": len(predictions_payload),
                },
                "metrics.json": {
                    "sha256": _sha256_bytes(metrics_payload),
                    "size_bytes": len(metrics_payload),
                },
            },
        }
        manifest_payload = _canonical_json(manifest) + b"\n"
        _write_new(staging / "predictions.jsonl", predictions_payload)
        _write_new(staging / "metrics.json", metrics_payload)
        _write_new(staging / "manifest.json", manifest_payload)
        if target.exists():
            raise FileExistsError(f"answer eval output already exists: {target}")
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return rows, metrics, manifest

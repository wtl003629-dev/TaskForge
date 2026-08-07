"""End-to-end RAG answer evaluation: retrieve, generate, score.

This is the "scale": for every locked case it runs a real retrieval step and a
real model answer generation, then scores the generated answer against the
dataset's gold answer with exact match and token F1.  Unlike the retrieval
ablation (which stops at recall@k), this measures whether the pipeline answers
questions correctly.  The provider is injected and the run is billable, so the
CLI requires ``--confirm-live-call``.
"""

from __future__ import annotations

import asyncio
import os
import platform
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import Field

from . import __version__
from .builtins import create_tool_registry
from .context import ContextAssembler
from .domain import (
    AgentProfile,
    ModelTurn,
    RunState,
    RunStatus,
    StrictModel,
    Task,
    utc_now,
)
from .hybrid_knowledge import HybridKnowledgeStore, knowledge_to_hybrid_chunk
from .hybrid_retrieval import (
    BM25Index,
    DeterministicHashEmbedder,
    FastEmbedEmbedder,
    LexicalOverlapFallbackReranker,
    QdrantHybridIndex,
)
from .knowledge import AccessContext, KnowledgeChunk, KnowledgeHit
from .memory import InMemoryMemoryStore
from .providers import ModelProvider, RetryableProviderError
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
    _PreparedDataset,
    _search_request,
    _sha256_bytes,
    _source_hashes,
    _write_new,
    chunk_text,
)
from .runtime import AgentRuntime
from .tooling import CapabilityPolicy

RetrieverName = Literal["bm25", "qdrant_rrf", "qdrant_rrf_rerank"]


ANSWER_EVAL_METADATA_FIELD_WEIGHTS = {
    "title": 2.0,
    "source": 2.0,
    "published_at": 1.0,
}


class RAGAnswerEvalConfig(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset: ExperimentDatasetConfig = Field(default_factory=ExperimentDatasetConfig)
    retrieval: ExperimentRetrievalConfig = Field(
        default_factory=lambda: ExperimentRetrievalConfig(
            bm25_field_weights=dict(ANSWER_EVAL_METADATA_FIELD_WEIGHTS)
        )
    )
    filters: ExperimentFilterConfig = Field(default_factory=ExperimentFilterConfig)
    retriever: RetrieverName = "bm25"
    model: str = Field(min_length=1)
    mode: Literal["naive", "agentic"] = "naive"
    agent_max_steps: int = Field(default=8, ge=1, le=20)
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
            "UNTRUSTED EVIDENCE CONTEXT. Do not use outside knowledge. Give your "
            "best determination from the evidence even if uncertain; do not refuse "
            "on uncertainty. For agreement/comparison questions, answer Yes only "
            "if every clause is supported, No if any clause is contradicted, and "
            "Different only when the evidence confirms the items differ; otherwise "
            "reply with only the bare fact. Keep the answer short and never leave "
            "it blank."
        ),
        model=model,
        allowed_tools=[],
    )
    # The host-evidence fallback runs outside the AgentRuntime step loop, so a
    # transient provider failure here would otherwise abort the whole eval.  A
    # small bounded retry keeps a single connection blip from discarding the
    # case.  The agentic path already has its own durable retry in the runtime.
    turn: ModelTurn | None = None
    for attempt in range(3):
        try:
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
            break
        except RetryableProviderError:
            if attempt == 2:
                raise
            await asyncio.sleep(0.5 * (2**attempt))
    if turn is None or turn.kind != "final" or not turn.final_answer:
        return None
    return turn.final_answer


class _InMemoryCheckpointStore:
    """Keeps run states in memory; the eval needs no durable checkpoints."""

    def __init__(self) -> None:
        self.saved: list[Any] = []

    def save(self, state: Any) -> None:
        self.saved.append(state.model_copy(deep=True))


class _CitationAwareContextAssembler(ContextAssembler):
    """Expands retrieval to quoted citations so the agent sees source-rich context."""

    def _retrieval_query(
        self,
        query: str | None,
        task: object | None,
        profile: object | None,
    ) -> str:
        base = super()._retrieval_query(query, task, profile)
        return " ".join(_retrieval_subqueries(base))


def _evidence_text(hit: Any) -> str:
    """Prepend citation metadata to a chunk body when the corpus provides it."""

    metadata = getattr(getattr(hit, "chunk", None), "metadata", None)
    header: list[str] = []
    if isinstance(metadata, Mapping):
        for key in ("title", "source", "published_at"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                header.append(f"{key}: {value.strip()}")
    body = hit.chunk.text
    return "\n".join([*header, body]) if header else body


def _evidence_from_state(state: RunState) -> tuple[list[str], list[str]]:
    """Collect deduplicated evidence texts/ids from every successful search."""

    texts: dict[str, str] = {}
    for step in state.steps:
        for result in step.tool_results:
            if not result.ok or not isinstance(result.output, Mapping):
                continue
            hits = result.output.get("hits")
            if not isinstance(hits, list):
                continue
            for hit in hits:
                if not isinstance(hit, Mapping):
                    continue
                evidence_id = hit.get("evidence_id")
                if not isinstance(evidence_id, str):
                    evidence_id = hit.get("source")
                text = hit.get("text")
                if not isinstance(evidence_id, str) or not evidence_id:
                    continue
                if not isinstance(text, str) or not text:
                    continue
                texts.setdefault(evidence_id, text)
    return list(texts.values()), list(texts.keys())


def _build_agentic_runtime(
    provider: ModelProvider,
    prepared: _PreparedDataset,
    *,
    staging: Path,
    config: RAGAnswerEvalConfig,
) -> tuple[AgentRuntime, HybridKnowledgeStore]:
    """Build a real AgentRuntime whose knowledge base is the evaluation corpus."""

    chunks: list[KnowledgeChunk] = []
    for document in sorted(
        prepared.dataset.documents, key=lambda value: value.document_id
    ):
        texts = (
            chunk_text(
                document.text,
                max_chars=config.retrieval.chunk_max_chars,
                overlap_chars=config.retrieval.chunk_overlap_chars,
            )
            if config.retrieval.chunking
            else [document.text]
        )
        for index, text in enumerate(texts):
            chunk_id = (
                f"{document.document_id}::chunk::{index}"
                if len(texts) > 1
                else document.document_id
            )
            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    tenant_id="local",
                    text=text,
                    source_uri=document.source_uri,
                    document_id=document.document_id,
                    acl=frozenset({"user:demo"}),
                    metadata={
                        "knowledge_base_id": "answer-eval",
                        # Evidence stays document-granular so host verification
                        # and the model's citations keep referring to the same
                        # retrieved document, not to one of its chunks.
                        "evidence_id": document.document_id,
                        **document.metadata,
                    },
                )
            )
    hybrid_chunks = [knowledge_to_hybrid_chunk(chunk) for chunk in chunks]
    if config.retriever == "bm25":
        index = BM25Index(
            hybrid_chunks,
            k1=config.retrieval.bm25_k1,
            b=config.retrieval.bm25_b,
            field_weights=config.retrieval.bm25_field_weights,
        )
        knowledge = _TimeAwareHybridKnowledgeStore(index, chunks)
    else:
        if config.retrieval.semantic_embedding:
            embedder = FastEmbedEmbedder(config.retrieval.semantic_model)
        else:
            embedder = DeterministicHashEmbedder(config.retrieval.hash_dimension)
        qdrant = QdrantHybridIndex.in_memory(
            collection_name="taskforge-answer-eval-agentic",
            embedder=embedder,
            reranker=LexicalOverlapFallbackReranker(),
        )
        qdrant.upsert(hybrid_chunks)
        knowledge = _TimeAwareHybridKnowledgeStore(
            qdrant,
            chunks,
            rerank=config.retriever == "qdrant_rrf_rerank",
        )
    memory = InMemoryMemoryStore()
    registry = create_tool_registry(
        workspace_root=staging,
        artifact_root=staging / "artifacts",
        knowledge_store=knowledge,
        memory_store=memory,
    )
    return (
        AgentRuntime(
            provider=provider,
            registry=registry,
            policy=CapabilityPolicy(registry),
            checkpoint=_InMemoryCheckpointStore(),
            context=_CitationAwareContextAssembler(knowledge, memory),
        ),
        knowledge,
    )


def _retrieval_subqueries(query: str) -> list[str]:
    """Deterministic citation-aware queries: the question plus every quoted phrase."""

    queries = [query]
    for phrase in re.findall(r"'([^']+)'", query):
        cleaned = " ".join(phrase.split())
        if cleaned and cleaned not in queries:
            queries.append(cleaned)
    return queries


_MONTH_NAMES = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December"
)
# "October 7, 2023" — full and abbreviated month names.
_DATE_RE = re.compile(
    r"\b(?:(?P<full>" + _MONTH_NAMES + r")|(?P<abbr>[A-Z][a-z]{2}))\.?\s+"
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?,?\s+(?P<year>\d{4})\b"
)
_DIRECTION_BEFORE = re.compile(
    r"\b(before|prior to|as of|no later than)\b", re.IGNORECASE
)
_DIRECTION_AFTER = re.compile(
    r"\b(after|since|following|subsequent to)\b", re.IGNORECASE
)
_BETWEEN = re.compile(r"\bbetween\b", re.IGNORECASE)


def _query_time_window(
    query: str,
) -> tuple[datetime | None, datetime | None]:
    """Best-effort ``(after, before)`` publication window from a question.

    Returns ``(None, None)`` when the query has no unambiguous date signal, so
    non-temporal retrieval is never accidentally constrained.  ``between A and
    B`` maps to ``(after=A, before=B)``; a single ``before``/``after`` bound
    applies to the first detected date.  This is an intentional heuristic, not
    a parser: host evidence retrieval stays deterministic and fails open.
    """

    matches = list(_DATE_RE.finditer(query))
    if not matches:
        return None, None
    def _parse(match: re.Match[str]) -> datetime | None:
        month = match.group("full") or match.group("abbr")
        try:
            return datetime(
                int(match.group("year")),
                int(datetime.strptime(month, "%b").month)
                if match.group("abbr")
                else int(datetime.strptime(match.group("full"), "%B").month),
                int(match.group("day")),
                tzinfo=UTC,
            )
        except ValueError:
            return None
    dates = [_parse(m) for m in matches]
    dates = [d for d in dates if d is not None]
    if not dates:
        return None, None
    if _BETWEEN.search(query):
        return dates[0], dates[-1]
    if _DIRECTION_BEFORE.search(query):
        return None, dates[0]
    if _DIRECTION_AFTER.search(query):
        return dates[0], None
    return None, None


class _TimeAwareHybridKnowledgeStore(HybridKnowledgeStore):
    """Wraps search so date phrases in the query bound publication time."""

    def search(
        self,
        query: str,
        principal: AccessContext,
        **kwargs: Any,
    ) -> list[KnowledgeHit]:
        after, before = _query_time_window(query)
        if after is not None:
            kwargs.setdefault("published_after", after)
        if before is not None:
            kwargs.setdefault("published_before", before)
        return super().search(query, principal, **kwargs)


def _host_evidence(
    knowledge: HybridKnowledgeStore,
    case: RAGEvalCase,
    config: RAGAnswerEvalConfig,
) -> tuple[list[str], list[str]]:
    """Host-side multi-query retrieval so a single weak search cannot starve evidence."""

    principal = AccessContext(tenant_id="local", user_id="demo")
    texts: dict[str, str] = {}
    for query in _retrieval_subqueries(case.query):
        hits = knowledge.search(
            query,
            principal,
            top_k=config.evidence_top_k + 3,
            knowledge_base_ids=["answer-eval"],
        )
        for hit in hits:
            evidence_id = str(
                hit.chunk.metadata.get("evidence_id") or hit.chunk.chunk_id
            )
            texts.setdefault(evidence_id, _evidence_text(hit))
    return list(texts.values()), list(texts.keys())


async def _agentic_answer(
    runtime: AgentRuntime,
    knowledge: HybridKnowledgeStore,
    case: RAGEvalCase,
    config: RAGAnswerEvalConfig,
) -> tuple[str | None, int, list[str]]:
    """Let the model drive retrieval, then force a bare answer if its budget ends."""

    task = Task(
        tenant_id="local",
        user_id="demo",
        goal=case.query,
    )
    profile = AgentProfile(
        id="answer-eval-agent",
        name="Answer eval agent",
        instructions=(
            "Answer the research question using ONLY the knowledge base. You must "
            "make at least one knowledge_search call before your final answer; "
            "never answer from memory. Plan one knowledge_search for every source, "
            "publication, date, or entity named in the question; request limit 10 "
            "and refine queries until each named article is covered. Stop "
            "retrieving as soon as the evidence covers every source, publication, "
            "date, and entity named in the question; do not search on when the "
            "needed articles are already in hand. When the question names a date "
            "or a period, pass published_before / published_after (ISO-8601) to "
            "knowledge_search so retrieval is confined to articles published in "
            "that window. The question's "
            "clauses are the claim to verify: answer Yes only if every clause is "
            "supported by the retrieved articles, No if any clause is contradicted, "
            "and Different only when the question asks whether the items differ and "
            "the evidence confirms a difference. Answer with exactly one of Yes, No, "
            "or Different for agreement/comparison questions; otherwise reply with "
            "only the bare fact. Never leave the answer blank; if evidence is "
            "ambiguous, choose the best-supported answer. Do not include reasoning, "
            "markdown, or explanations."
        ),
        model=config.model,
        allowed_tools=["knowledge_search"],
        knowledge_base_ids=["answer-eval"],
        max_steps=config.agent_max_steps,
    )
    state = await runtime.run(task, profile)
    steps = len(state.steps)
    agent_texts, agent_ids = _evidence_from_state(state)
    if state.status != RunStatus.COMPLETED or not state.final_answer:
        host_texts, host_ids = _host_evidence(knowledge, case, config)
        merged_texts = list(agent_texts)
        merged_ids = list(agent_ids)
        seen = set(agent_ids)
        for evidence_id, text in zip(host_ids, host_texts):
            if evidence_id not in seen:
                merged_ids.append(evidence_id)
                merged_texts.append(text)
                seen.add(evidence_id)
        forced = await _generate_answer(
            provider=runtime.provider,
            case=case,
            evidence_texts=merged_texts,
            model=config.model,
            max_evidence_chars=config.max_evidence_chars,
        )
        if forced is not None:
            return forced, steps, merged_ids
    return state.final_answer, steps, agent_ids


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
        if config.mode == "agentic":
            runtime, knowledge = _build_agentic_runtime(
                provider, prepared, staging=staging, config=config
            )
            index = None
        else:
            chunks = _hybrid_chunks(prepared.dataset, prepared.cases, config)
            indexes = _indexes(chunks, config)
            index = indexes[config.retriever]
        cases = prepared.cases
        if config.max_cases is not None:
            cases = cases[: config.max_cases]

        rows: list[dict[str, Any]] = []
        for case in cases:
            if config.mode == "agentic":
                answer, steps, retrieved_ids = await _agentic_answer(
                    runtime, knowledge, case, config
                )
                mode = "agentic"
            else:
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
                    _evidence_text(hit)
                    for hit in response.hits[: config.evidence_top_k]
                ]
                answer = await _generate_answer(
                    provider,
                    case,
                    evidence,
                    model=config.model,
                    max_evidence_chars=config.max_evidence_chars,
                )
                steps = 0
                mode = "naive"
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
                    "mode": mode,
                    "steps": steps,
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
            "mode": config.mode,
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

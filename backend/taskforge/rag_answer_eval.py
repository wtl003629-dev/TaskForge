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
import json
import math
import os
import platform
import re
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import Field, model_validator

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
    BM25DenseRRFIndex,
    BM25Index,
    DeterministicHashEmbedder,
    FastEmbedEmbedder,
    QdrantDenseIndex,
    QdrantHybridIndex,
    SourceCoverageRRFIndex,
)
from .knowledge import (
    AccessContext,
    InMemoryKnowledgeStore,
    KnowledgeChunk,
    KnowledgeHit,
)
from .memory import InMemoryMemoryStore
from .providers import ModelProvider, ProviderError, RetryableProviderError
from .rag_evaluation import (
    CitedAnswerPrediction,
    RAGEvalCase,
    answer_exact_match,
    answer_token_f1,
    evaluate_answer_grounding,
)
from .rag_experiment import (
    ExperimentDatasetConfig,
    ExperimentFilterConfig,
    ExperimentRetrievalConfig,
    RAGExperimentConfig,
    _build_reranker,
    _canonical_json,
    _chunk_document_text,
    _deduped_document_ids,
    _hybrid_chunks,
    _latency_summary,
    _prepare_dataset,
    _PreparedDataset,
    _run_stages,
    _search_request,
    _sha256_bytes,
    _write_new,
)
from .runtime import AgentRuntime
from .tatqa_slot_selector import (
    render_tatqa_slot_context,
    select_tatqa_table_slots,
)
from .tooling import CapabilityPolicy, ToolRegistry

RetrieverName = Literal[
    "bm25",
    "bm25_source_coverage_rrf",
    "qdrant_dense",
    "bm25_dense_rrf",
    "bm25_dense_rrf_rerank",
    "qdrant_rrf",
    "qdrant_rrf_rerank",
    "tatqa_frozen_bm25",
    "tatqa_frozen_pair_rerank",
]
AnswerContract = Literal["bare_v1", "cited_v1", "online_cited_v1"]

FROZEN_TATQA_BM25_STAGE = "lexical_bm25"
FROZEN_TATQA_PAIR_STAGE = (
    "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_pair_rerank_rrf"
)
FROZEN_TATQA_RETRIEVERS = frozenset(
    {"tatqa_frozen_bm25", "tatqa_frozen_pair_rerank"}
)


ANSWER_EVAL_METADATA_FIELD_WEIGHTS = {
    "title": 2.0,
    "source": 2.0,
    "published_at": 1.0,
}


_BARE_ANSWER_RULES = (
    "For agreement/comparison questions, answer Yes only if every clause is "
    "supported, No if any clause is contradicted, and Different only when the "
    "evidence confirms the items differ; otherwise reply with only the bare "
    "fact. Keep the answer short and never leave it blank."
)

_CITED_ANSWER_RULES = (
    "Return exactly one JSON object with exactly two keys: answer and "
    "citation_ids. answer must be a non-empty short string. citation_ids must "
    "be a JSON array of unique evidence_id strings copied from evidence that "
    "was actually provided. Do not invent IDs and do not wrap the JSON in "
    "markdown. For agreement/comparison questions, answer Yes only if every "
    "clause is supported, No if any clause is contradicted, and Different only "
    "when the evidence confirms the items differ."
)

_ONLINE_CITED_ANSWER_RULES = (
    "Return exactly one JSON object with exactly five keys: answer, derivation, "
    "cited_evidence_ids, calculator_used, and abstained. answer and derivation "
    "must be non-empty strings. cited_evidence_ids must be a JSON array of "
    "unique evidence_id strings copied from evidence actually provided. "
    "calculator_used and abstained must be JSON booleans. Use the calculator "
    "tool only when the question requires an actual arithmetic expression; do "
    "not call it merely to copy, count, or verify a value already stated in "
    "evidence. Use it for arithmetic instead of doing arithmetic mentally. Set "
    "calculator_used true if and only if the host calculator was called at any "
    "earlier turn, including a failed call or if you later abstain. derivation must "
    "be a concise, auditable calculation or evidence "
    "summary, not hidden chain-of-thought. If none of the supplied evidence "
    "supports an answer, explain the insufficiency in answer and derivation and "
    "set abstained to true. Citations may identify the evidence inspected. Keep "
    "calculator_used consistent with whether any calculator call was attempted. "
    "Otherwise abstained must be false and at least one "
    "citation is required. Never invent IDs, use outside knowledge, or wrap the "
    "JSON in markdown."
)


def _answer_instructions(*, mode: Literal["naive", "agentic"], contract: AnswerContract) -> str:
    if mode == "naive":
        base = (
            "Answer the research question using ONLY the evidence provided under "
            "UNTRUSTED EVIDENCE CONTEXT. Do not use outside knowledge. Give your "
            "best determination from the evidence even if uncertain; do not refuse "
            "on uncertainty. "
        )
    else:
        base = (
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
            "that window. "
        )
    if contract == "online_cited_v1":
        return base + _ONLINE_CITED_ANSWER_RULES
    return base + (_CITED_ANSWER_RULES if contract == "cited_v1" else _BARE_ANSWER_RULES)


@dataclass(frozen=True, slots=True)
class _GeneratedAnswer:
    raw_output: str | None
    answer: str
    citation_ids: list[str]
    parse_error: str | None
    presented_evidence_ids: list[str]
    derivation: str = ""
    calculator_used: bool = False
    abstained: bool = False
    calculator_receipts: tuple[Mapping[str, Any], ...] = ()
    contract_retry_count: int = 0
    usage: dict[str, int] | None = None
    provider_response_ids: tuple[str, ...] = ()
    provider_calls: int = 0
    retry_count: int = 0
    execution_error: str | None = None


def _parse_cited_answer(value: str | None) -> tuple[str, list[str], str | None]:
    if value is None or not value.strip():
        return "", [], "empty_output"
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return "", [], "invalid_json"
    if not isinstance(decoded, Mapping) or set(decoded) != {"answer", "citation_ids"}:
        return "", [], "invalid_shape"
    answer = decoded.get("answer")
    citations = decoded.get("citation_ids")
    if not isinstance(answer, str) or not answer.strip():
        return "", [], "invalid_answer"
    if not isinstance(citations, list) or len(citations) > 100:
        return "", [], "invalid_citation_ids"
    if any(
        not isinstance(item, str) or not item.strip() or len(item) > 512
        for item in citations
    ):
        return "", [], "invalid_citation_ids"
    cleaned = [item.strip() for item in citations]
    if len(cleaned) != len(set(cleaned)):
        return "", [], "duplicate_citation_ids"
    return answer.strip(), cleaned, None


def _parse_online_cited_answer(
    value: str | None,
) -> tuple[str, str, list[str], bool, bool, str | None]:
    if value is None or not value.strip():
        return "", "", [], False, False, "empty_output"
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return "", "", [], False, False, "invalid_json"
    required = {
        "answer",
        "derivation",
        "cited_evidence_ids",
        "calculator_used",
        "abstained",
    }
    if not isinstance(decoded, Mapping) or set(decoded) != required:
        return "", "", [], False, False, "invalid_shape"
    answer = decoded.get("answer")
    derivation = decoded.get("derivation")
    citations = decoded.get("cited_evidence_ids")
    calculator_used = decoded.get("calculator_used")
    abstained = decoded.get("abstained")
    if not isinstance(answer, str) or not answer.strip():
        return "", "", [], False, False, "invalid_answer"
    if not isinstance(derivation, str) or not derivation.strip():
        return "", "", [], False, False, "invalid_derivation"
    if not isinstance(calculator_used, bool) or not isinstance(abstained, bool):
        return "", "", [], False, False, "invalid_boolean_flags"
    if not isinstance(citations, list) or len(citations) > 100:
        return "", "", [], False, False, "invalid_cited_evidence_ids"
    if any(
        not isinstance(item, str) or not item.strip() or len(item) > 512
        for item in citations
    ):
        return "", "", [], False, False, "invalid_cited_evidence_ids"
    cleaned = [item.strip() for item in citations]
    if len(cleaned) != len(set(cleaned)):
        return "", "", [], False, False, "duplicate_cited_evidence_ids"
    if not abstained and not cleaned:
        return "", "", [], False, False, "missing_citation_ids"
    return (
        answer.strip(),
        derivation.strip(),
        cleaned,
        calculator_used,
        abstained,
        None,
    )


def _decode_answer(
    raw_output: str | None,
    *,
    contract: AnswerContract,
    presented_evidence_ids: Sequence[str],
    usage: dict[str, int] | None = None,
    provider_response_ids: Sequence[str] = (),
    provider_calls: int = 0,
    retry_count: int = 0,
    execution_error: str | None = None,
    calculator_receipts: Sequence[Mapping[str, Any]] = (),
    contract_retry_count: int = 0,
) -> _GeneratedAnswer:
    derivation = ""
    calculator_used = False
    abstained = False
    if contract == "online_cited_v1":
        (
            answer,
            derivation,
            citation_ids,
            calculator_used,
            abstained,
            parse_error,
        ) = _parse_online_cited_answer(raw_output)
        actual_calculator_used = bool(calculator_receipts)
        if parse_error is None and calculator_used != actual_calculator_used:
            parse_error = "calculator_usage_mismatch"
    elif contract == "cited_v1":
        answer, citation_ids, parse_error = _parse_cited_answer(raw_output)
    else:
        answer = raw_output.strip() if isinstance(raw_output, str) else ""
        citation_ids = []
        parse_error = None
    return _GeneratedAnswer(
        raw_output=raw_output,
        answer=answer,
        citation_ids=citation_ids,
        parse_error=parse_error,
        presented_evidence_ids=list(dict.fromkeys(presented_evidence_ids)),
        derivation=derivation,
        calculator_used=calculator_used,
        abstained=abstained,
        calculator_receipts=tuple(dict(item) for item in calculator_receipts),
        contract_retry_count=contract_retry_count,
        usage=usage,
        provider_response_ids=tuple(provider_response_ids),
        provider_calls=provider_calls,
        retry_count=retry_count,
        execution_error=execution_error,
    )


class OnlineModelPrice(StrictModel):
    """Versioned per-million-token prices used only for auditable estimates."""

    model: str = Field(min_length=1)
    currency: Literal["USD"] = "USD"
    input_cache_hit_per_million: float = Field(ge=0.0)
    input_cache_miss_per_million: float = Field(ge=0.0)
    output_per_million: float = Field(ge=0.0)
    source_url: str = Field(min_length=1)
    retrieved_at: str = Field(min_length=10)


class RAGAnswerEvalConfig(StrictModel):
    schema_version: Literal["1.1"] = "1.1"
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
    answer_contract: AnswerContract = "bare_v1"
    agent_max_steps: int = Field(default=8, ge=1, le=20)
    agentic_host_fallback: bool = False
    evidence_top_k: int = Field(default=5, ge=1, le=20)
    max_evidence_chars: int = Field(default=16_000, ge=500, le=80_000)
    tatqa_query_slot_context: bool = False
    tatqa_query_slot_k: int = Field(default=10, ge=1, le=50)
    calculator_max_calls: int = Field(default=8, ge=1, le=8)
    contract_max_retries: int = Field(default=1, ge=0, le=2)
    max_cases: int | None = Field(default=None, ge=1)
    execution_mode: Literal["contract_test", "live"] = "contract_test"
    thinking_mode: Literal["disabled", "enabled", "provider_default"] = (
        "provider_default"
    )
    json_mode: bool = False
    price_table: OnlineModelPrice | None = None

    @model_validator(mode="after")
    def online_baseline_is_strict(self) -> RAGAnswerEvalConfig:
        if self.tatqa_query_slot_context:
            if self.dataset.kind != "tatqa_locked":
                raise ValueError("TAT-QA query slot context requires a TAT-QA dataset")
            if self.dataset.tatqa_context_mode != "provided_hybrid_context":
                raise ValueError(
                    "TAT-QA query slot context requires provided_hybrid_context"
                )
            if self.mode != "naive":
                raise ValueError("TAT-QA query slot context requires naive mode")
        if self.answer_contract == "online_cited_v1":
            if self.mode != "naive":
                raise ValueError("online_cited_v1 requires the fixed host retrieval path")
            if self.retriever not in FROZEN_TATQA_RETRIEVERS:
                raise ValueError("online_cited_v1 requires a frozen TAT-QA retriever")
            if self.dataset.kind != "tatqa_locked":
                raise ValueError("online_cited_v1 requires a locked TAT-QA dataset")
            if self.evidence_top_k != 10:
                raise ValueError("online_cited_v1 requires Top-10 evidence")
            if self.thinking_mode != "disabled":
                raise ValueError("online_cited_v1 pins non-thinking mode for tool replay")
            if not self.json_mode:
                raise ValueError("online_cited_v1 requires provider JSON mode")
        if self.execution_mode == "live" and self.price_table is None:
            raise ValueError("live answer evaluation requires a versioned price table")
        return self


def _normalise_usage(metadata: Mapping[str, Any]) -> dict[str, int] | None:
    usage = metadata.get("usage")
    if not isinstance(usage, Mapping):
        return None

    def token_count(*names: str) -> int | None:
        for name in names:
            value = usage.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return None

    input_tokens = token_count("input_tokens", "prompt_tokens")
    output_tokens = token_count("output_tokens", "completion_tokens")
    total_tokens = token_count("total_tokens")
    cache_hit_tokens = token_count("prompt_cache_hit_tokens", "cache_read_input_tokens")
    cache_miss_tokens = token_count(
        "prompt_cache_miss_tokens", "cache_creation_input_tokens"
    )
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    if total_tokens is None:
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    normalized = {
        "input_tokens": input_tokens or 0,
        "output_tokens": output_tokens or 0,
        "total_tokens": total_tokens,
    }
    if cache_hit_tokens is not None:
        normalized["input_cache_hit_tokens"] = cache_hit_tokens
    if cache_miss_tokens is not None:
        normalized["input_cache_miss_tokens"] = cache_miss_tokens
    return normalized


def _merge_usage(*values: dict[str, int] | None) -> dict[str, int] | None:
    reported = [value for value in values if value is not None]
    if not reported:
        return None
    return {
        key: sum(value.get(key, 0) for value in reported)
        for key in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "input_cache_hit_tokens",
            "input_cache_miss_tokens",
        )
    }


def _estimated_cost(
    usage: Mapping[str, int] | None,
    price: OnlineModelPrice | None,
) -> dict[str, Any]:
    if usage is None:
        return {
            "status": "usage_unreported",
            "currency": price.currency if price is not None else None,
            "amount": None,
        }
    if price is None:
        return {
            "status": "price_table_not_configured",
            "currency": None,
            "amount": None,
        }
    input_tokens = int(usage.get("input_tokens", 0))
    hit_tokens = min(
        input_tokens, int(usage.get("input_cache_hit_tokens", 0))
    )
    reported_miss = int(usage.get("input_cache_miss_tokens", 0))
    miss_tokens = min(input_tokens - hit_tokens, reported_miss)
    # Any input not classified by the provider is conservatively priced as a
    # cache miss, so the estimate never understates the published list price.
    unclassified = max(0, input_tokens - hit_tokens - miss_tokens)
    billed_miss = miss_tokens + unclassified
    output_tokens = int(usage.get("output_tokens", 0))
    amount = (
        hit_tokens * price.input_cache_hit_per_million
        + billed_miss * price.input_cache_miss_per_million
        + output_tokens * price.output_per_million
    ) / 1_000_000
    return {
        "status": "estimated_from_reported_usage",
        "currency": price.currency,
        "amount": amount,
        "input_cache_hit_tokens": hit_tokens,
        "input_cache_miss_tokens": billed_miss,
        "output_tokens": output_tokens,
        "unclassified_input_priced_as_miss": unclassified,
    }


def _state_usage(state: RunState) -> tuple[dict[str, int] | None, tuple[str, ...]]:
    usages: list[dict[str, int] | None] = []
    response_ids: list[str] = []
    for step in state.steps:
        turn = step.model_turn
        if turn is None:
            continue
        usages.append(_normalise_usage(turn.metadata))
        if turn.provider_response_id:
            response_ids.append(turn.provider_response_id)
    return _merge_usage(*usages), tuple(response_ids)


async def _generate_answer(
    provider: ModelProvider,
    case: RAGEvalCase,
    evidence: Sequence[tuple[str, str]],
    *,
    model: str,
    max_evidence_chars: int,
    contract: AnswerContract,
    calculator_registry: ToolRegistry | None = None,
    calculator_max_calls: int = 8,
    contract_max_retries: int = 1,
    instructions_override: str | None = None,
) -> _GeneratedAnswer:
    """Ask the model to answer the question from the retrieved evidence only."""

    budget = max_evidence_chars
    joined: list[str] = []
    presented_ids: list[str] = []
    used = 0
    for evidence_id, text in evidence:
        if used >= budget:
            break
        take = text[: budget - used]
        if not take:
            continue
        joined.append(take)
        if evidence_id not in presented_ids:
            presented_ids.append(evidence_id)
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
            instructions_override
            if instructions_override is not None
            else _answer_instructions(mode="naive", contract=contract)
        ),
        model=model,
        allowed_tools=["calculator"] if contract == "online_cited_v1" else [],
    )
    if contract == "online_cited_v1" and calculator_registry is None:
        raise ValueError("online_cited_v1 requires a host calculator registry")
    tool_schemas = (
        list(calculator_registry.list_specs(profile))
        if calculator_registry is not None
        else []
    )
    policy = CapabilityPolicy(calculator_registry) if calculator_registry else None
    state = RunState(
        task_id=task.id,
        agent_profile_id=profile.id,
        status=RunStatus.RUNNING,
        step_budget=calculator_max_calls + 1,
    )
    trajectory: list[dict[str, Any]] = []
    usage_values: list[dict[str, int] | None] = []
    response_ids: list[str] = []
    calculator_receipts: list[Mapping[str, Any]] = []
    provider_calls = 0
    retry_count = 0
    calculator_calls = 0
    contract_retry_count = 0

    def failed(error: str) -> _GeneratedAnswer:
        return _decode_answer(
            None,
            contract=contract,
            presented_evidence_ids=presented_ids,
            usage=_merge_usage(*usage_values),
            provider_response_ids=response_ids,
            provider_calls=provider_calls,
            retry_count=retry_count,
            execution_error=error,
            calculator_receipts=calculator_receipts,
            contract_retry_count=contract_retry_count,
        )

    # One initial request plus at most calculator_max_calls tool turns, followed
    # by a final structured answer. Each network request has bounded retries.
    for _ in range(calculator_max_calls + contract_max_retries + 1):
        turn: ModelTurn | None = None
        for attempt in range(3):
            provider_calls += 1
            try:
                turn = await provider.complete(
                    task=task,
                    profile=profile,
                    context={
                        "assembled": {
                            "evidence": joined,
                            "question": case.query,
                        },
                        "trajectory": trajectory,
                    },
                    tools=tool_schemas,
                )
                break
            except RetryableProviderError as exc:
                if attempt == 2:
                    return failed(type(exc).__name__)
                retry_count += 1
                await asyncio.sleep(0.5 * (2**attempt))
            except ProviderError as exc:
                return failed(type(exc).__name__)
        if turn is None:
            return failed("invalid_model_turn")
        usage_values.append(_normalise_usage(turn.metadata))
        if turn.provider_response_id:
            response_ids.append(turn.provider_response_id)
        if turn.kind == "final":
            decoded = _decode_answer(
                turn.final_answer,
                contract=contract,
                presented_evidence_ids=presented_ids,
                usage=_merge_usage(*usage_values),
                provider_response_ids=response_ids,
                provider_calls=provider_calls,
                retry_count=retry_count,
                calculator_receipts=calculator_receipts,
                contract_retry_count=contract_retry_count,
            )
            if (
                decoded.parse_error is not None
                and contract == "online_cited_v1"
                and contract_retry_count < contract_max_retries
            ):
                contract_retry_count += 1
                observed_called = bool(calculator_receipts)
                correction = (
                    "\n\nHOST CONTRACT VALIDATION: Your previous final response failed "
                    f"with {decoded.parse_error}. Return the exact five-key JSON now. "
                    f"The host observed calculator_called={str(observed_called).lower()}, "
                    "so calculator_used must equal that value. Do not call another tool "
                    "unless a new arithmetic operation is genuinely required."
                )
                profile = profile.model_copy(
                    update={"instructions": profile.instructions + correction}
                )
                continue
            return decoded
        if contract != "online_cited_v1" or calculator_registry is None or policy is None:
            return failed("unexpected_tool_request")
        if calculator_calls + len(turn.tool_requests) > calculator_max_calls:
            return failed("calculator_call_limit")
        tool_results: list[Mapping[str, Any]] = []
        for request in turn.tool_requests:
            calculator_calls += 1
            if request.name != "calculator":
                return failed("unexpected_tool_request")
            decision = await policy.evaluate(task, profile, request)
            if not decision.allowed or decision.requires_approval:
                return failed("calculator_policy_denied")
            result = await calculator_registry.execute(request, task, profile, state)
            receipt = {
                **result.model_dump(mode="json"),
                "request": request.model_dump(mode="json"),
            }
            calculator_receipts.append(receipt)
            tool_results.append(receipt)
        trajectory.append(
            {
                "step": len(trajectory),
                "assistant_text": turn.assistant_text,
                "provider_response_id": turn.provider_response_id,
                "tool_requests": [
                    request.model_dump(mode="json") for request in turn.tool_requests
                ],
                "tool_results": tool_results,
            }
        )
    return failed("calculator_step_limit")


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

    chunk = getattr(hit, "chunk", None)
    metadata = getattr(chunk, "metadata", None)
    header: list[str] = []
    evidence_id: object | None = None
    if isinstance(metadata, Mapping):
        evidence_id = metadata.get("evidence_id")
        for key in ("title", "source", "published_at"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                header.append(f"{key}: {value.strip()}")
    if not isinstance(evidence_id, str) or not evidence_id.strip():
        evidence_id = getattr(chunk, "document_id", None)
    if isinstance(evidence_id, str) and evidence_id.strip():
        header.insert(0, f"evidence_id: {evidence_id.strip()}")
    body = hit.chunk.text
    return "\n".join([*header, body]) if header else body


def _answer_evidence_text(
    hit: Any,
    case: RAGEvalCase,
    config: RAGAnswerEvalConfig,
) -> str:
    """Optionally place label-free table slots before unchanged evidence."""

    evidence = _evidence_text(hit)
    if not config.tatqa_query_slot_context:
        return evidence
    chunk = getattr(hit, "chunk", None)
    metadata = getattr(chunk, "metadata", None)
    if not isinstance(metadata, Mapping) or metadata.get("kind") != "table":
        return evidence
    raw_rows = metadata.get("table_rows")
    if not isinstance(raw_rows, list) or any(
        not isinstance(row, list) for row in raw_rows
    ):
        return evidence
    table = [[str(value) for value in row] for row in raw_rows]
    plan = select_tatqa_table_slots(
        case.query,
        table,
        budget=config.tatqa_query_slot_k,
    )
    return render_tatqa_slot_context(plan) + "\n\nFull retrieved evidence:\n" + evidence


def _evidence_from_state(
    state: RunState,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Collect retrieved evidence and the subset seen by a later model turn."""

    retrieved: dict[str, str] = {}
    presented: dict[str, str] = {}
    final_step_index = len(state.steps) - 1
    for step_index, step in enumerate(state.steps):
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
                retrieved.setdefault(evidence_id, text)
                # A tool result enters the trajectory only on the next provider
                # request. Results from the last step of a step-limited run were
                # retrieved, but the model never received them.
                if step_index < final_step_index:
                    presented.setdefault(evidence_id, text)
    return (
        list(retrieved.values()),
        list(retrieved.keys()),
        list(presented.values()),
        list(presented.keys()),
    )


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
        texts = _chunk_document_text(
            document.text,
            document.metadata,
            config.retrieval,
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
                        "parent_document_id": document.document_id,
                        # Evidence stays document-granular so host verification
                        # and the model's citations keep referring to the same
                        # retrieved document, not to one of its chunks.
                        "evidence_id": document.document_id,
                        **document.metadata,
                    },
                )
            )
    hybrid_chunks = [knowledge_to_hybrid_chunk(chunk) for chunk in chunks]
    if config.retriever in {"bm25", "bm25_source_coverage_rrf"}:
        lexical = BM25Index(
            hybrid_chunks,
            k1=config.retrieval.bm25_k1,
            b=config.retrieval.bm25_b,
            field_weights=config.retrieval.bm25_field_weights,
        )
        index = (
            SourceCoverageRRFIndex(
                lexical,
                hybrid_chunks,
                rrf_k=config.retrieval.rrf_k,
            )
            if config.retriever == "bm25_source_coverage_rrf"
            else lexical
        )
        knowledge = _TimeAwareHybridKnowledgeStore(
            index,
            chunks,
            neighbor_window=0,
            max_chunks_per_document=config.retrieval.max_chunks_per_document,
        )
    else:
        if config.retrieval.semantic_embedding:
            embedder = FastEmbedEmbedder(config.retrieval.semantic_model)
        else:
            embedder = DeterministicHashEmbedder(config.retrieval.hash_dimension)
        reranker, _ = _build_reranker(config.retrieval)
        qdrant = QdrantHybridIndex.in_memory(
            collection_name="taskforge-answer-eval-agentic",
            embedder=embedder,
            reranker=reranker,
            embedding_metadata_fields=config.retrieval.bm25_field_weights,
        )
        qdrant.upsert(hybrid_chunks)
        composite = BM25DenseRRFIndex(
            BM25Index(
                hybrid_chunks,
                k1=config.retrieval.bm25_k1,
                b=config.retrieval.bm25_b,
                field_weights=config.retrieval.bm25_field_weights,
            ),
            qdrant,
            reranker=reranker,
            rrf_k=config.retrieval.rrf_k,
            bm25_weight=config.retrieval.rrf_bm25_weight,
            dense_weight=config.retrieval.rrf_dense_weight,
        )
        backends = {
            "qdrant_dense": QdrantDenseIndex(qdrant),
            "bm25_dense_rrf": composite,
            "bm25_dense_rrf_rerank": composite,
            "qdrant_rrf": qdrant,
            "qdrant_rrf_rerank": qdrant,
        }
        knowledge = _TimeAwareHybridKnowledgeStore(
            backends[config.retriever],
            chunks,
            rerank=config.retriever
            in {"bm25_dense_rrf_rerank", "qdrant_rrf_rerank"},
            neighbor_window=0,
            max_chunks_per_document=config.retrieval.max_chunks_per_document,
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
            # Agentic answer eval observes only explicit knowledge_search tool
            # receipts. Hidden automatic pre-retrieval would make R/P accounting
            # incomplete and give the agent evidence before its required tool call.
            context=_CitationAwareContextAssembler(
                knowledge,
                memory,
                knowledge_limit=0,
                memory_limit=0,
            ),
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
) -> tuple[_GeneratedAnswer, int, list[str], bool]:
    """Let the model drive explicit retrieval and return auditable evidence use."""

    task = Task(
        tenant_id="local",
        user_id="demo",
        goal=case.query,
    )
    profile = AgentProfile(
        id="answer-eval-agent",
        name="Answer eval agent",
        instructions=_answer_instructions(
            mode="agentic",
            contract=config.answer_contract,
        ),
        model=config.model,
        allowed_tools=["knowledge_search"],
        knowledge_base_ids=["answer-eval"],
        max_steps=config.agent_max_steps,
    )
    state = await runtime.run(task, profile)
    steps = len(state.steps)
    agent_texts, agent_ids, _, presented_agent_ids = _evidence_from_state(state)
    usage, response_ids = _state_usage(state)
    if state.status == RunStatus.COMPLETED and state.final_answer:
        return (
            _decode_answer(
                state.final_answer,
                contract=config.answer_contract,
                presented_evidence_ids=presented_agent_ids,
                usage=usage,
                provider_response_ids=response_ids,
                provider_calls=steps,
            ),
            steps,
            agent_ids,
            False,
        )

    state_error = state.error.code if state.error is not None else state.status.value
    if not config.agentic_host_fallback:
        return (
            _decode_answer(
                None,
                contract=config.answer_contract,
                presented_evidence_ids=presented_agent_ids,
                usage=usage,
                provider_response_ids=response_ids,
                provider_calls=steps,
                execution_error=state_error,
            ),
            steps,
            agent_ids,
            False,
        )

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
            evidence=list(zip(merged_ids, merged_texts)),
            model=config.model,
            max_evidence_chars=config.max_evidence_chars,
            contract=config.answer_contract,
        )
        return (
            _decode_answer(
                forced.raw_output,
                contract=config.answer_contract,
                presented_evidence_ids=forced.presented_evidence_ids,
                usage=_merge_usage(usage, forced.usage),
                provider_response_ids=(*response_ids, *forced.provider_response_ids),
                provider_calls=steps + forced.provider_calls,
                retry_count=forced.retry_count,
                execution_error=forced.execution_error or state_error,
            ),
            steps,
            merged_ids,
            True,
        )
    raise RuntimeError("unreachable agentic answer state")


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
    lexical_indexes = {
        "bm25": lexical,
        "bm25_source_coverage_rrf": SourceCoverageRRFIndex(
            lexical,
            chunks,
            rrf_k=config.retrieval.rrf_k,
        ),
    }
    if config.retriever in lexical_indexes:
        return lexical_indexes

    if config.retrieval.semantic_embedding:
        embedder = FastEmbedEmbedder(config.retrieval.semantic_model)
    else:
        embedder = DeterministicHashEmbedder(config.retrieval.hash_dimension)
    reranker, _ = _build_reranker(config.retrieval)
    qdrant = QdrantHybridIndex.in_memory(
        collection_name="taskforge-answer-eval",
        embedder=embedder,
        reranker=reranker,
        embedding_metadata_fields=config.retrieval.bm25_field_weights,
    )
    qdrant.upsert(chunks)
    composite = BM25DenseRRFIndex(
        lexical,
        qdrant,
        reranker=reranker,
        rrf_k=config.retrieval.rrf_k,
        bm25_weight=config.retrieval.rrf_bm25_weight,
        dense_weight=config.retrieval.rrf_dense_weight,
    )
    return {
        **lexical_indexes,
        "qdrant_dense": QdrantDenseIndex(qdrant),
        "bm25_dense_rrf": composite,
        "bm25_dense_rrf_rerank": composite,
        "qdrant_rrf": qdrant,
        "qdrant_rrf_rerank": qdrant,
    }


def _frozen_tatqa_retrieval_config(
    value: ExperimentRetrievalConfig,
    retriever: RetrieverName,
) -> ExperimentRetrievalConfig:
    """Apply the exact immutable retrieval settings promoted offline."""

    if retriever == "tatqa_frozen_bm25":
        stage = FROZEN_TATQA_BM25_STAGE
    elif retriever == "tatqa_frozen_pair_rerank":
        stage = FROZEN_TATQA_PAIR_STAGE
    else:
        return value
    return value.model_copy(
        update={
            "stages": [stage],
            "development_sweep": True,
            "top_k": [1, 5, 10],
            "candidate_k": 50,
            "parent_top_k": 5,
            "parent_sibling_coverage": True,
            "bm25_k1": 1.5,
            "bm25_b": 0.75,
            "bm25_field_weights": {},
            "semantic_embedding": False,
            "learned_sparse": False,
            "learned_reranker": False,
            "chunking": False,
            "table_aware_chunking": False,
            "max_chunks_per_document": None,
            "query_expansion": False,
            "context_seed_k": 1,
            "tatqa_lineage_seed_k": 20,
            "tatqa_lineage_closure_slots": 12,
            "tatqa_lineage_max_siblings_per_parent": 2,
            "tatqa_structured_candidate_slots": 10,
            "tatqa_lineage_pair_rerank_slots": 1,
            "tatqa_lineage_pair_min_score": 0.24,
            "tatqa_numeric_cell_weight": 0.25,
            "tatqa_numeric_scan_weight": 0.5,
        }
    )


def _run_frozen_tatqa_retrieval(
    prepared: _PreparedDataset,
    config: RAGAnswerEvalConfig,
    *,
    timer_ns: Callable[[], int],
) -> tuple[dict[str, dict[str, Any]], Mapping[str, Any]]:
    """Reuse the exact offline stage and capture the evidence seen online."""

    experiment_config = RAGExperimentConfig(
        dataset=config.dataset,
        retrieval=config.retrieval,
        filters=config.filters,
    )
    captured: dict[str, dict[str, Any]] = {}

    def observe(stage: str, case: RAGEvalCase, response: Any, duration_ms: float) -> None:
        if response is None:
            raise RuntimeError(f"frozen online retrieval stage {stage} returned no hits")
        retrieved_ids = _deduped_document_ids(
            response.hits,
            max_documents=config.retrieval.candidate_k,
        )
        evidence: list[tuple[str, str]] = []
        seen: set[str] = set()
        for hit in response.hits:
            identifiers = _deduped_document_ids([hit], max_documents=1)
            if not identifiers:
                continue
            evidence_id = identifiers[0]
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            evidence.append((evidence_id, _answer_evidence_text(hit, case, config)))
            if len(evidence) >= config.evidence_top_k:
                break
        captured[case.case_id] = {
            "stage": stage,
            "retrieved_ids": retrieved_ids,
            "evidence": evidence,
            "latency_ms": duration_ms,
            "backend": response.backend,
        }

    _, stage_metrics = _run_stages(
        prepared,
        experiment_config,
        timer_ns=timer_ns,
        response_observer=observe,
    )
    expected = {case.case_id for case in prepared.cases}
    if set(captured) != expected:
        raise RuntimeError("frozen online retrieval did not capture every locked case")
    return captured, stage_metrics


def _evidence_retrieval_metrics(
    relevant_ids: Sequence[str],
    retrieved_ids: Sequence[str],
) -> dict[str, Any]:
    relevant = set(relevant_ids)
    retrieved = set(retrieved_ids)
    matched = relevant.intersection(retrieved)
    precision = len(matched) / len(retrieved) if retrieved else 0.0
    recall = len(matched) / len(relevant)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "full_recall": recall == 1.0,
        "matched_relevant_ids": [
            evidence_id for evidence_id in relevant_ids if evidence_id in matched
        ],
        "missing_relevant_ids": [
            evidence_id for evidence_id in relevant_ids if evidence_id not in matched
        ],
        "irrelevant_retrieved_ids": [
            evidence_id for evidence_id in retrieved_ids if evidence_id not in relevant
        ],
    }


def _failure_bucket(
    *,
    exact_match: float,
    full_evidence_recall: bool,
    answer: str,
    parse_error: str | None,
    execution_error: str | None,
) -> str:
    if execution_error is not None:
        return "execution_error"
    if parse_error is not None:
        return "answer_contract_error"
    if not answer:
        return "empty_answer"
    if full_evidence_recall and exact_match == 1.0:
        return "success"
    if full_evidence_recall:
        return "generation_error_with_full_evidence"
    if exact_match == 1.0:
        return "correct_without_full_gold_evidence"
    return "retrieval_or_generation_error"


def _failure_stage(
    *,
    exact_match: float,
    candidate_full_recall: bool,
    top10_full_recall: bool,
    presented_full_recall: bool,
    answer: str,
    parse_error: str | None,
    execution_error: str | None,
) -> str:
    """Attribute a failed answer to the first observable pipeline stage."""

    if execution_error is not None:
        return "execution_error"
    if parse_error is not None or not answer:
        return "format_or_scale_failure"
    if not candidate_full_recall:
        return "candidate_missing"
    if not top10_full_recall:
        return "top10_ranking_failure"
    if not presented_full_recall:
        return "context_coverage_failure"
    if exact_match == 1.0:
        return "success"
    return "reasoning_failure"


_NUMERIC_ANSWER = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _numeric_answer_correct(prediction: object, expected: object) -> float:
    """Score arithmetic/count answers numerically without leaking derivations."""

    def values(value: object) -> list[float]:
        candidates = value if isinstance(value, list) else [value]
        numbers: list[float] = []
        for candidate in candidates:
            text = str(candidate).replace(",", "")
            for matched in _NUMERIC_ANSWER.findall(text):
                try:
                    numbers.append(float(matched))
                except ValueError:
                    continue
        return numbers

    predicted_values = values(prediction)
    expected_values = values(expected)
    if not predicted_values or not expected_values:
        return 0.0
    return float(
        any(
            math.isclose(predicted, gold, rel_tol=1e-6, abs_tol=1e-6)
            for predicted in predicted_values
            for gold in expected_values
        )
    )


def _answer_eval_source_hashes(repository: Path) -> dict[str, str]:
    package_root = Path(__file__).resolve().parent
    names = (
        "builtins.py",
        "context.py",
        "hybrid_knowledge.py",
        "hybrid_retrieval.py",
        "knowledge.py",
        "openai_provider.py",
        "providers.py",
        "rag_answer_eval.py",
        "rag_baseline.py",
        "rag_evaluation.py",
        "rag_experiment.py",
        "runtime.py",
        "tooling.py",
    )
    result = {
        f"taskforge.{Path(name).stem}": _sha256_bytes((package_root / name).read_bytes())
        for name in names
    }
    runner = repository / "scripts" / "run_rag_answer_eval.py"
    if not runner.is_file():
        # Tests may supply a fixture-only repository for dataset provenance;
        # code provenance must still point at the installed source tree.
        runner = package_root.parents[1] / "scripts" / "run_rag_answer_eval.py"
    if not runner.is_file():
        raise FileNotFoundError(f"RAG answer eval CLI source is missing: {runner}")
    result["scripts.run_rag_answer_eval"] = _sha256_bytes(runner.read_bytes())
    return result


def _prompt_descriptor(config: RAGAnswerEvalConfig) -> dict[str, str]:
    prompt = _answer_instructions(mode=config.mode, contract=config.answer_contract)
    return {
        "id": f"rag-answer-{config.mode}-{config.answer_contract}",
        "sha256": _sha256_bytes(prompt.encode("utf-8")),
    }


async def run_rag_answer_eval(
    *,
    output_dir: str | Path,
    config: RAGAnswerEvalConfig,
    provider: ModelProvider,
    repository_root: str | Path,
    created_at: datetime | None = None,
    timer_ns: Callable[[], int] = time.perf_counter_ns,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Run retrieve->generate->score and publish evidence artifacts."""

    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"answer eval output already exists: {target}")
    repository = Path(repository_root).resolve()
    if not repository.is_dir():
        raise FileNotFoundError(f"repository root does not exist: {repository}")
    if config.execution_mode == "live":
        provider_identity = (
            provider.__class__.__module__,
            provider.__class__.__name__,
        )
        if provider_identity not in {
            ("taskforge.openai_provider", "OpenAIChatCompletionsProvider"),
            ("taskforge.openai_provider", "OpenAIResponsesProvider"),
        }:
            raise ValueError("live evaluation requires a real network provider adapter")
        if getattr(provider, "_owns_client", False) is not True:
            raise ValueError("live evaluation rejects injected/mock HTTP clients")
    if config.retriever in FROZEN_TATQA_RETRIEVERS:
        config = config.model_copy(
            update={
                "retrieval": _frozen_tatqa_retrieval_config(
                    config.retrieval, config.retriever
                )
            }
        )
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
        cases = prepared.cases
        if config.max_cases is not None:
            cases = cases[: config.max_cases]
        evaluation_prepared = replace(prepared, cases=tuple(cases))
        runtime: AgentRuntime | None = None
        index: Any | None = None
        frozen_retrieval: dict[str, dict[str, Any]] | None = None
        retrieval_stage_metrics: Mapping[str, Any] | None = None
        if config.mode == "agentic":
            runtime, knowledge = _build_agentic_runtime(
                provider, evaluation_prepared, staging=staging, config=config
            )
        elif config.retriever in FROZEN_TATQA_RETRIEVERS:
            frozen_retrieval, retrieval_stage_metrics = _run_frozen_tatqa_retrieval(
                evaluation_prepared,
                config,
                timer_ns=timer_ns,
            )
        else:
            chunks = _hybrid_chunks(prepared.dataset, cases, config)
            indexes = _indexes(chunks, config)
            index = indexes[config.retriever]
        calculator_registry: ToolRegistry | None = None
        if config.answer_contract == "online_cited_v1":
            calculator_registry = create_tool_registry(
                workspace_root=repository,
                artifact_root=staging / "tool-artifacts",
                knowledge_store=InMemoryKnowledgeStore([]),
                memory_store=InMemoryMemoryStore(),
            )

        rows: list[dict[str, Any]] = []
        for case in cases:
            case_started_ns = timer_ns()
            retrieval_latency_ms: float | None = None
            frozen_latency_ms = 0.0
            fallback_used = False
            if config.mode == "agentic":
                assert runtime is not None
                generated, steps, retrieved_ids, fallback_used = await _agentic_answer(
                    runtime, knowledge, case, config
                )
                mode = "agentic"
            else:
                if frozen_retrieval is not None:
                    frozen = frozen_retrieval[case.case_id]
                    retrieved_ids = list(frozen["retrieved_ids"])
                    evidence = list(frozen["evidence"])
                    retrieval_latency_ms = float(frozen["latency_ms"])
                    frozen_latency_ms = retrieval_latency_ms
                else:
                    assert index is not None
                    request = _search_request(
                        case.query,
                        config,
                        rerank=config.retriever
                        in {"bm25_dense_rrf_rerank", "qdrant_rrf_rerank"},
                    )
                    response = index.search(request)
                    retrieval_finished_ns = timer_ns()
                    retrieval_latency_ms = max(
                        0.0,
                        (retrieval_finished_ns - case_started_ns) / 1_000_000,
                    )
                    retrieved_ids = _deduped_document_ids(response.hits)
                    # Feed the model the top retrieved chunk texts directly (not a
                    # whole-document truncation) so the answer sentence is not cut away.
                    evidence = []
                    seen_evidence_ids: set[str] = set()
                    for hit in response.hits:
                        identifiers = _deduped_document_ids([hit], max_documents=1)
                        if not identifiers or identifiers[0] in seen_evidence_ids:
                            continue
                        seen_evidence_ids.add(identifiers[0])
                        evidence.append(
                            (
                                identifiers[0],
                                _answer_evidence_text(hit, case, config),
                            )
                        )
                        if len(evidence) >= config.evidence_top_k:
                            break
                generated = await _generate_answer(
                    provider,
                    case,
                    evidence,
                    model=config.model,
                    max_evidence_chars=config.max_evidence_chars,
                    contract=config.answer_contract,
                    calculator_registry=calculator_registry,
                    calculator_max_calls=config.calculator_max_calls,
                    contract_max_retries=config.contract_max_retries,
                )
                steps = 0
                mode = "naive"
            total_latency_ms = frozen_latency_ms + max(
                0.0,
                (timer_ns() - case_started_ns) / 1_000_000,
            )
            predicted = generated.answer
            exact = answer_exact_match(predicted, case.answer)
            f1 = answer_token_f1(predicted, case.answer)
            numeric_correct = (
                _numeric_answer_correct(predicted, case.answer)
                if case.category in {"arithmetic", "count"}
                else None
            )
            candidate_metrics = _evidence_retrieval_metrics(
                case.relevant_ids,
                retrieved_ids,
            )
            retrieval_top10_metrics = _evidence_retrieval_metrics(
                case.relevant_ids,
                retrieved_ids[: config.evidence_top_k],
            )
            presented_context_metrics = _evidence_retrieval_metrics(
                case.relevant_ids,
                generated.presented_evidence_ids,
            )
            failure_bucket = _failure_bucket(
                exact_match=exact,
                full_evidence_recall=presented_context_metrics["full_recall"],
                answer=predicted,
                parse_error=generated.parse_error,
                execution_error=generated.execution_error,
            )
            failure_stage = _failure_stage(
                exact_match=exact,
                candidate_full_recall=candidate_metrics["full_recall"],
                top10_full_recall=retrieval_top10_metrics["full_recall"],
                presented_full_recall=presented_context_metrics["full_recall"],
                answer=predicted,
                parse_error=generated.parse_error,
                execution_error=generated.execution_error,
            )
            rows.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "query": case.query,
                    "gold_answer": case.answer,
                    "relevant_ids": list(case.relevant_ids),
                    "generated_answer": predicted,
                    "raw_model_output": generated.raw_output,
                    "retrieved_ids": retrieved_ids,
                    "presented_evidence_ids": generated.presented_evidence_ids,
                    "tatqa_query_slot_context": config.tatqa_query_slot_context,
                    "tatqa_query_slot_k": config.tatqa_query_slot_k,
                    "citation_ids": generated.citation_ids,
                    "derivation": generated.derivation,
                    "calculator_used": generated.calculator_used,
                    "calculator_receipts": list(generated.calculator_receipts),
                    "calculator_failure_count": sum(
                        receipt.get("ok") is not True
                        for receipt in generated.calculator_receipts
                    ),
                    "contract_retry_count": generated.contract_retry_count,
                    "abstained": generated.abstained,
                    "parse_error": generated.parse_error,
                    "exact_match": exact,
                    "token_f1": f1,
                    "numeric_correct": numeric_correct,
                    # ``evidence_*`` is the actual Top-K retrieval metric.  The
                    # complete candidate list is recorded separately so a
                    # Candidate@50 value cannot be mistaken for Recall@10.
                    "evidence_precision": retrieval_top10_metrics["precision"],
                    "evidence_recall": retrieval_top10_metrics["recall"],
                    "evidence_f1": retrieval_top10_metrics["f1"],
                    "full_evidence_recall": presented_context_metrics["full_recall"],
                    "matched_relevant_ids": retrieval_top10_metrics[
                        "matched_relevant_ids"
                    ],
                    "missing_relevant_ids": retrieval_top10_metrics[
                        "missing_relevant_ids"
                    ],
                    "irrelevant_retrieved_ids": retrieval_top10_metrics[
                        "irrelevant_retrieved_ids"
                    ],
                    "candidate_precision_at_k": candidate_metrics["precision"],
                    "candidate_recall_at_k": candidate_metrics["recall"],
                    "candidate_f1_at_k": candidate_metrics["f1"],
                    "candidate_full_recall": candidate_metrics["full_recall"],
                    "retrieval_top10_ids": list(retrieved_ids[: config.evidence_top_k]),
                    "retrieval_top10_precision": retrieval_top10_metrics["precision"],
                    "retrieval_top10_recall": retrieval_top10_metrics["recall"],
                    "retrieval_top10_f1": retrieval_top10_metrics["f1"],
                    "retrieval_top10_full_recall": retrieval_top10_metrics["full_recall"],
                    "presented_context_precision": presented_context_metrics["precision"],
                    "presented_context_recall": presented_context_metrics["recall"],
                    "presented_context_f1": presented_context_metrics["f1"],
                    "presented_context_full_recall": presented_context_metrics["full_recall"],
                    "model": config.model,
                    "retriever": config.retriever,
                    "mode": mode,
                    "answer_contract": config.answer_contract,
                    "steps": steps,
                    "model_calls": generated.provider_calls,
                    "provider_call_attempted": generated.provider_calls > 0,
                    "retry_count": generated.retry_count,
                    "model_usage": generated.usage,
                    "estimated_cost": _estimated_cost(
                        generated.usage, config.price_table
                    ),
                    "provider_response_ids": list(
                        generated.provider_response_ids
                    ),
                    "latency_ms": {
                        "retrieval": retrieval_latency_ms,
                        "total": total_latency_ms,
                    },
                    "execution_error": generated.execution_error,
                    "fallback_used": fallback_used,
                    "failure_bucket": failure_bucket,
                    "failure_stage": failure_stage,
                }
            )

        cited_contract = config.answer_contract in {"cited_v1", "online_cited_v1"}
        if cited_contract:
            grounding_report = evaluate_answer_grounding(
                cases,
                [
                    CitedAnswerPrediction(
                        case_id=row["case_id"],
                        answer=row["generated_answer"],
                        retrieved_ids=row["retrieved_ids"],
                        presented_evidence_ids=row["presented_evidence_ids"],
                        citation_ids=row["citation_ids"],
                        parse_error=row["parse_error"],
                    )
                    for row in rows
                ],
            )
            grounding_by_case = {
                item.case_id: item for item in grounding_report.cases
            }
            for row in rows:
                item = grounding_by_case[row["case_id"]]
                row["grounding"] = item.model_dump(mode="json")
                presented = set(row["presented_evidence_ids"])
                row["invalid_evidence_ids"] = [
                    evidence_id
                    for evidence_id in row["citation_ids"]
                    if evidence_id not in presented
                ]
                has_presented_gold = bool(
                    set(row["relevant_ids"]).intersection(presented)
                )
                row["expected_abstention"] = not has_presented_gold
                row["abstention_correct"] = (
                    row["abstained"] == row["expected_abstention"]
                )
                row["unsupported_answer"] = (
                    not row["abstained"] and not item.valid_citation_ids
                )
            grounding_metrics: dict[str, Any] = {
                "status": "measured_strict_gold_evidence",
                "scope": (
                    "short-answer exact match plus gold document IDs; this is "
                    "not semantic entailment"
                ),
                "summary": grounding_report.summary.model_dump(mode="json"),
            }
        else:
            for row in rows:
                row["grounding"] = None
                row["invalid_evidence_ids"] = []
                row["expected_abstention"] = None
                row["abstention_correct"] = None
                row["unsupported_answer"] = None
            grounding_metrics = {
                "status": "not_measured",
                "reason": "bare_v1 has no model citation_ids",
            }

        exact_scores = [row["exact_match"] for row in rows]
        f1_scores = [row["token_f1"] for row in rows]
        evidence_precision_scores = [row["evidence_precision"] for row in rows]
        evidence_recall_scores = [row["evidence_recall"] for row in rows]
        evidence_f1_scores = [row["evidence_f1"] for row in rows]
        candidate_precision_scores = [row["candidate_precision_at_k"] for row in rows]
        candidate_recall_scores = [row["candidate_recall_at_k"] for row in rows]
        candidate_f1_scores = [row["candidate_f1_at_k"] for row in rows]
        presented_precision_scores = [row["presented_context_precision"] for row in rows]
        presented_recall_scores = [row["presented_context_recall"] for row in rows]
        presented_f1_scores = [row["presented_context_f1"] for row in rows]
        by_category: dict[str, dict[str, Any]] = {}
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["category"], []).append(row)
        for category, group in grouped.items():
            numeric_group = [
                item["numeric_correct"]
                for item in group
                if item["numeric_correct"] is not None
            ]
            by_category[category] = {
                "exact_match": sum(item["exact_match"] for item in group) / len(group),
                "token_f1": sum(item["token_f1"] for item in group) / len(group),
                "evidence_precision": sum(
                    item["evidence_precision"] for item in group
                )
                / len(group),
                "evidence_recall": sum(item["evidence_recall"] for item in group)
                / len(group),
                "evidence_f1": sum(item["evidence_f1"] for item in group)
                / len(group),
                "candidate_recall_at_k": sum(
                    item["candidate_recall_at_k"] for item in group
                )
                / len(group),
                "presented_context_recall": sum(
                    item["presented_context_recall"] for item in group
                )
                / len(group),
                "full_evidence_recall_rate": sum(
                    item["full_evidence_recall"] for item in group
                )
                / len(group),
                "failure_counts": dict(
                    sorted(Counter(item["failure_bucket"] for item in group).items())
                ),
                "failure_stage_counts": dict(
                    sorted(Counter(item["failure_stage"] for item in group).items())
                ),
                "cases": len(group),
                "numeric_accuracy": (
                    sum(numeric_group) / len(numeric_group)
                    if numeric_group
                    else None
                ),
            }
            if cited_contract:
                by_category[category]["grounding"] = {
                    "citation_precision": sum(
                        item["grounding"]["citation_precision"] for item in group
                    )
                    / len(group),
                    "citation_recall": sum(
                        item["grounding"]["citation_recall"] for item in group
                    )
                    / len(group),
                    "strict_supported_claim_rate": sum(
                        item["grounding"]["strict_supported_claim"]
                        for item in group
                    )
                    / len(group),
                }
                by_category[category]["unsupported_answer_rate"] = sum(
                    bool(item["unsupported_answer"]) for item in group
                ) / len(group)
                by_category[category]["abstention_accuracy"] = sum(
                    bool(item["abstention_correct"]) for item in group
                ) / len(group)

        reported_usage = [
            row["model_usage"] for row in rows if row["model_usage"] is not None
        ]
        total_latencies = [row["latency_ms"]["total"] for row in rows]
        retrieval_latencies = [
            row["latency_ms"]["retrieval"]
            for row in rows
            if row["latency_ms"]["retrieval"] is not None
        ]
        numeric_scores = [
            row["numeric_correct"]
            for row in rows
            if row["numeric_correct"] is not None
        ]
        estimated_amounts = [
            row["estimated_cost"]["amount"]
            for row in rows
            if row["estimated_cost"]["amount"] is not None
        ]
        provider_attempted = sum(row["provider_call_attempted"] for row in rows)
        provider_succeeded = sum(
            row["provider_call_attempted"] and row["execution_error"] is None
            for row in rows
        )
        metrics: dict[str, Any] = {
            "schema_version": "1.3",
            "mode": config.mode,
            "retriever": config.retriever,
            "model": config.model,
            "answer_contract": config.answer_contract,
            "total_cases": len(rows),
            "exact_match_accuracy": (
                sum(exact_scores) / len(exact_scores) if exact_scores else 0.0
            ),
            "avg_token_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            "numeric_accuracy": (
                sum(numeric_scores) / len(numeric_scores)
                if numeric_scores
                else None
            ),
            "numeric_eligible_cases": len(numeric_scores),
            "evidence_retrieval": {
                "definition": "retrieved Top-K evidence IDs, where K=evidence_top_k",
                "avg_precision": (
                    sum(evidence_precision_scores) / len(evidence_precision_scores)
                    if evidence_precision_scores
                    else 0.0
                ),
                "avg_recall": (
                    sum(evidence_recall_scores) / len(evidence_recall_scores)
                    if evidence_recall_scores
                    else 0.0
                ),
                "avg_f1": (
                    sum(evidence_f1_scores) / len(evidence_f1_scores)
                    if evidence_f1_scores
                    else 0.0
                ),
                "full_recall_rate": (
                    sum(row["full_evidence_recall"] for row in rows) / len(rows)
                    if rows
                    else 0.0
                ),
            },
            "candidate_retrieval": {
                "definition": "complete candidate list, where K=candidate_k; not final context",
                "avg_precision": sum(candidate_precision_scores) / len(candidate_precision_scores)
                if candidate_precision_scores
                else 0.0,
                "avg_recall": sum(candidate_recall_scores) / len(candidate_recall_scores)
                if candidate_recall_scores
                else 0.0,
                "avg_f1": sum(candidate_f1_scores) / len(candidate_f1_scores)
                if candidate_f1_scores
                else 0.0,
                "full_recall_rate": (
                    sum(row["candidate_full_recall"] for row in rows) / len(rows)
                    if rows
                    else 0.0
                ),
            },
            "presented_context": {
                "definition": "evidence IDs actually included in the model context",
                "avg_precision": sum(presented_precision_scores) / len(presented_precision_scores)
                if presented_precision_scores
                else 0.0,
                "avg_recall": sum(presented_recall_scores) / len(presented_recall_scores)
                if presented_recall_scores
                else 0.0,
                "avg_f1": sum(presented_f1_scores) / len(presented_f1_scores)
                if presented_f1_scores
                else 0.0,
                "full_recall_rate": (
                    sum(row["presented_context_full_recall"] for row in rows) / len(rows)
                    if rows
                    else 0.0
                ),
            },
            "grounding": grounding_metrics,
            "online_safety": {
                "execution_mode": config.execution_mode,
                "provider_attempted_cases": provider_attempted,
                "provider_successful_cases": provider_succeeded,
                "real_api_call_rate": (
                    provider_attempted / len(rows)
                    if rows and config.execution_mode == "live"
                    else None
                ),
                "provider_success_rate": (
                    provider_succeeded / provider_attempted
                    if provider_attempted
                    else 0.0
                ),
                "invalid_evidence_id_cases": sum(
                    bool(row["invalid_evidence_ids"]) for row in rows
                ),
                "invalid_evidence_ids": sum(
                    len(row["invalid_evidence_ids"]) for row in rows
                ),
                "acl_tenant_violation_cases": 0,
                "unsupported_answer_rate": (
                    sum(bool(row["unsupported_answer"]) for row in rows) / len(rows)
                    if rows and cited_contract
                    else None
                ),
                "abstention_rate": (
                    sum(row["abstained"] for row in rows) / len(rows)
                    if rows and config.answer_contract == "online_cited_v1"
                    else None
                ),
                "abstention_accuracy": (
                    sum(bool(row["abstention_correct"]) for row in rows) / len(rows)
                    if rows and cited_contract
                    else None
                ),
                "calculator_used_cases": sum(row["calculator_used"] for row in rows),
                "calculator_tool_calls": sum(
                    len(row["calculator_receipts"]) for row in rows
                ),
                "calculator_tool_failures": sum(
                    row["calculator_failure_count"] for row in rows
                ),
                "contract_retries": sum(
                    row["contract_retry_count"] for row in rows
                ),
                "fallback_cases": sum(row["fallback_used"] for row in rows),
            },
            "failure_counts": dict(
                sorted(Counter(row["failure_bucket"] for row in rows).items())
            ),
            "failure_stage_counts": dict(
                sorted(Counter(row["failure_stage"] for row in rows).items())
            ),
            "execution_error_cases": sum(
                row["execution_error"] is not None for row in rows
            ),
            "execution_error_types": dict(
                sorted(
                    Counter(
                        row["execution_error"]
                        for row in rows
                        if row["execution_error"] is not None
                    ).items()
                )
            ),
            "parse_error_types": dict(
                sorted(
                    Counter(
                        row["parse_error"]
                        for row in rows
                        if row["parse_error"] is not None
                    ).items()
                )
            ),
            "fallback_cases": sum(row["fallback_used"] for row in rows),
            "latency_ms": {
                "total": _latency_summary(total_latencies) if rows else None,
                "retrieval": (
                    _latency_summary(retrieval_latencies)
                    if retrieval_latencies
                    else {
                        "status": "not_separately_observed",
                        "reason": "agentic retrieval occurs inside tool execution",
                    }
                ),
            },
            "model_usage": {
                "reported_cases": len(reported_usage),
                "unreported_cases": len(rows) - len(reported_usage),
                "input_tokens": sum(
                    usage["input_tokens"] for usage in reported_usage
                ),
                "output_tokens": sum(
                    usage["output_tokens"] for usage in reported_usage
                ),
                "total_tokens": sum(
                    usage["total_tokens"] for usage in reported_usage
                ),
                "model_calls": sum(row["model_calls"] for row in rows),
                "retries": sum(row["retry_count"] for row in rows),
                "input_cache_hit_tokens": sum(
                    usage.get("input_cache_hit_tokens", 0)
                    for usage in reported_usage
                ),
                "input_cache_miss_tokens": sum(
                    usage.get("input_cache_miss_tokens", 0)
                    for usage in reported_usage
                ),
                "estimated_cost": (
                    sum(estimated_amounts)
                    if len(estimated_amounts) == len(rows)
                    else None
                ),
                "estimated_cost_status": (
                    "complete"
                    if rows and len(estimated_amounts) == len(rows)
                    else "incomplete_usage_or_price_table"
                ),
                "currency": (
                    config.price_table.currency if config.price_table is not None else None
                ),
                "price_table": (
                    config.price_table.model_dump(mode="json")
                    if config.price_table is not None
                    else None
                ),
            },
            "by_category": by_category,
            "category_counts": dict(
                sorted(Counter(case.category for case in cases).items())
            ),
        }

        predictions_payload = b"\n".join(_canonical_json(row) for row in rows) + b"\n"
        failure_rows = [
            row
            for row in rows
            if row["failure_bucket"] != "success"
            or bool(row["unsupported_answer"])
            or bool(row["invalid_evidence_ids"])
            or row["calculator_failure_count"] > 0
        ]
        failures_payload = (
            b"\n".join(_canonical_json(row) for row in failure_rows) + b"\n"
            if failure_rows
            else b""
        )
        cost_rows = [
            {
                "case_id": row["case_id"],
                "model_calls": row["model_calls"],
                "retry_count": row["retry_count"],
                "usage": row["model_usage"],
                "estimated_cost": row["estimated_cost"],
            }
            for row in rows
        ]
        costs_payload = b"\n".join(_canonical_json(row) for row in cost_rows) + b"\n"
        metrics_payload = _canonical_json(metrics) + b"\n"
        effective_config = config.model_dump(mode="json")
        effective_config_hash = _sha256_bytes(_canonical_json(effective_config))
        source_hashes = _answer_eval_source_hashes(repository)
        code_hash = _sha256_bytes(_canonical_json(source_hashes))
        prompt = _prompt_descriptor(config)
        tool_schemas: list[Mapping[str, Any]] = []
        if runtime is not None:
            knowledge_search = runtime.registry.spec("knowledge_search")
            if knowledge_search is None:
                raise RuntimeError("agentic answer eval is missing knowledge_search")
            tool_schemas.append(knowledge_search.provider_schema())
        if calculator_registry is not None:
            calculator = calculator_registry.spec("calculator")
            if calculator is None:
                raise RuntimeError("online answer eval is missing calculator")
            tool_schemas.append(calculator.provider_schema())
        tool_schema_hash = _sha256_bytes(_canonical_json(tool_schemas))
        index_identity = {
            "dataset_sha256": prepared.provenance["normalized_sha256"],
            "retriever": config.retriever,
            "retrieval": config.retrieval.model_dump(mode="json"),
            "filters": config.filters.model_dump(mode="json"),
            "offline_stage_metrics": retrieval_stage_metrics,
        }
        index_identity_hash = _sha256_bytes(_canonical_json(index_identity))
        experiment_id = _sha256_bytes(
            "\0".join(
                (
                    str(prepared.provenance["normalized_sha256"]),
                    effective_config_hash,
                    code_hash,
                )
            ).encode("ascii")
        )[:20]
        trial_id = _sha256_bytes(
            f"{experiment_id}\0{timestamp.isoformat()}".encode("ascii")
        )[:20]
        manifest: dict[str, Any] = {
            "schema_version": "1.3",
            "run_id": trial_id,
            "experiment_id": experiment_id,
            "trial_id": trial_id,
            "created_at": timestamp.isoformat().replace("+00:00", "Z"),
            "answer_contract": config.answer_contract,
            "dataset": dict(prepared.provenance),
            "sample": {
                "case_ids": [case.case_id for case in cases],
                "selected_cases": len(cases),
                "category_counts": dict(
                    sorted(Counter(case.category for case in cases).items())
                ),
                "is_full_locked_split": config.max_cases is None,
            },
            "config": {
                "effective": effective_config,
                "sha256": effective_config_hash,
            },
            "budgets": {
                "top_k": list(config.retrieval.top_k),
                "candidate_k": config.retrieval.candidate_k,
                "evidence_top_k": config.evidence_top_k,
                "max_evidence_chars": config.max_evidence_chars,
                "tatqa_query_slot_context": config.tatqa_query_slot_context,
                "tatqa_query_slot_k": config.tatqa_query_slot_k,
                "agent_max_steps": config.agent_max_steps,
                "calculator_max_calls": config.calculator_max_calls,
                "contract_max_retries": config.contract_max_retries,
                "max_cases": config.max_cases,
            },
            "prompt": prompt,
            "tools": {
                "names": [str(schema["name"]) for schema in tool_schemas],
                "schema_sha256": tool_schema_hash,
            },
            "provider": {
                "execution_mode": config.execution_mode,
                "adapter_module": provider.__class__.__module__,
                "adapter_class": provider.__class__.__name__,
                "model": config.model,
                "price_table": (
                    config.price_table.model_dump(mode="json")
                    if config.price_table is not None
                    else None
                ),
            },
            "index": {
                "identity": index_identity,
                "sha256": index_identity_hash,
            },
            "code": {
                "package": "taskforge-agent",
                "package_version": __version__,
                "python": platform.python_version(),
                "source_sha256": source_hashes,
                "sha256": code_hash,
            },
            "limitations": [
                "single-run model scores do not establish variance or statistical stability",
                *(
                    [
                        "strict gold-evidence grounding is document-ID support, not semantic entailment"
                    ]
                    if cited_contract
                    else ["bare_v1 does not measure model citations or grounding"]
                ),
                *(
                    ["agentic host fallback was enabled and is disclosed per case"]
                    if config.agentic_host_fallback
                    else []
                ),
            ],
            "artifacts": {
                "predictions.jsonl": {
                    "sha256": _sha256_bytes(predictions_payload),
                    "size_bytes": len(predictions_payload),
                },
                "metrics.json": {
                    "sha256": _sha256_bytes(metrics_payload),
                    "size_bytes": len(metrics_payload),
                },
                "failures.jsonl": {
                    "sha256": _sha256_bytes(failures_payload),
                    "size_bytes": len(failures_payload),
                },
                "costs.jsonl": {
                    "sha256": _sha256_bytes(costs_payload),
                    "size_bytes": len(costs_payload),
                },
            },
        }
        manifest_payload = _canonical_json(manifest) + b"\n"
        _write_new(staging / "predictions.jsonl", predictions_payload)
        _write_new(staging / "metrics.json", metrics_payload)
        _write_new(staging / "failures.jsonl", failures_payload)
        _write_new(staging / "costs.jsonl", costs_payload)
        _write_new(staging / "manifest.json", manifest_payload)
        if target.exists():
            raise FileExistsError(f"answer eval output already exists: {target}")
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return rows, metrics, manifest

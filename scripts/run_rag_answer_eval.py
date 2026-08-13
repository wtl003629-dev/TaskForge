"""Run TaskForge's end-to-end RAG answer evaluation (retrieve -> generate -> score).

For every locked case the eval retrieves evidence, asks a real model to answer
from that evidence alone, and scores answer quality plus host-verified evidence
use. This makes billable model calls and requires ``--confirm-live-call``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.config import Settings
from taskforge.openai_provider import OpenAIChatCompletionsProvider
from taskforge.rag_answer_eval import (
    ANSWER_EVAL_METADATA_FIELD_WEIGHTS,
    OnlineModelPrice,
    RAGAnswerEvalConfig,
    run_rag_answer_eval,
)
from taskforge.rag_experiment import (
    ExperimentDatasetConfig,
    ExperimentRetrievalConfig,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--dataset",
        choices=("tatqa", "multihop-rag"),
        default="multihop-rag",
    )
    value.add_argument(
        "--retriever",
        choices=(
            "bm25",
            "bm25_source_coverage_rrf",
            "qdrant_dense",
            "bm25_dense_rrf",
            "bm25_dense_rrf_rerank",
            "qdrant_rrf",
            "qdrant_rrf_rerank",
            "tatqa_frozen_bm25",
            "tatqa_frozen_pair_rerank",
        ),
        default=None,
        help=(
            "Defaults to bm25, bm25_dense_rrf with --semantic, or its rerank "
            "variant with --learned-reranker. qdrant_rrf is the legacy control."
        ),
    )
    value.add_argument(
        "--semantic",
        action="store_true",
        help="Use real semantic dense embeddings for the qdrant retriever.",
    )
    value.add_argument(
        "--agentic",
        action="store_true",
        help=(
            "Agentic mode: the model drives multi-turn knowledge_search through "
            "the real AgentRuntime before answering (vs a single retrieve+answer)."
        ),
    )
    value.add_argument(
        "--answer-contract",
        choices=("bare-v1", "cited-v1", "online-cited-v1"),
        default="cited-v1",
        help=(
            "Model output contract. cited-v1 is the auditable default; bare-v1 "
            "is retained only for historical compatibility."
        ),
    )
    value.add_argument(
        "--allow-agentic-host-fallback",
        action="store_true",
        help=(
            "After an agentic failure/step limit, allow host retrieval plus a "
            "forced answer. Fallback cases are disclosed and fail promotion gates."
        ),
    )
    value.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit to the first N locked cases (for a quick smoke).",
    )
    value.add_argument(
        "--multihop-split",
        type=str,
        default=None,
        help="Repository-relative MultiHop-RAG locked split manifest.",
    )
    value.add_argument(
        "--model",
        default=None,
        help="Model name; defaults to TASKFORGE_DEEPSEEK_MODEL.",
    )
    value.add_argument(
        "--agent-max-steps",
        type=int,
        default=None,
        help="Override the agentic answer loop's step budget.",
    )
    value.add_argument(
        "--chunking",
        action="store_true",
        help="Enable document chunking (chunk_max_chars=1500) in the answer-eval index.",
    )
    value.add_argument(
        "--tatqa-input",
        default=".taskforge/eval-cache/tatqa_dataset_train.json",
        help="Repository-relative pinned TAT-QA source artifact.",
    )
    value.add_argument(
        "--tatqa-split",
        default="eval/splits/tatqa-train-online-heldout-100-v1.json",
        help="Repository-relative locked TAT-QA split manifest.",
    )
    value.add_argument(
        "--tatqa-context-mode",
        choices=("provided-hybrid-context", "global-discovery"),
        default="provided-hybrid-context",
        help="Official supplied-context task or the global-discovery stress test.",
    )
    value.add_argument(
        "--tatqa-query-slot-context",
        action="store_true",
        help=(
            "Prepend a label-free metric/year table-slot plan to TAT-QA evidence; "
            "requires provided-hybrid-context and does not change retrieval IDs."
        ),
    )
    value.add_argument(
        "--tatqa-query-slot-k",
        type=int,
        default=10,
        help="Fixed selected-cell budget for --tatqa-query-slot-context.",
    )
    value.add_argument(
        "--evidence-top-k",
        type=int,
        default=None,
        help="Evidence documents shown to the model; online-cited-v1 requires 10.",
    )
    value.add_argument(
        "--table-aware-chunking",
        action="store_true",
        help="Index tables as schema, row, and column chunks; also enables chunking.",
    )
    value.add_argument(
        "--max-chunks-per-document",
        type=int,
        default=None,
        help="Cap retrieved chunks per document to protect cross-document coverage.",
    )
    value.add_argument(
        "--learned-reranker",
        action="store_true",
        help="Use a FastEmbed ONNX cross-encoder for rerank retrievers.",
    )
    value.add_argument(
        "--reranker-model",
        default="Xenova/ms-marco-MiniLM-L-6-v2",
        help="FastEmbed cross-encoder model name.",
    )
    value.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Exact output run directory; an existing directory is never overwritten.",
    )
    value.add_argument(
        "--confirm-live-call",
        action="store_true",
        help="Required acknowledgement that this makes billable model calls.",
    )
    return value


def _default_output(dataset_kind: str, now: datetime) -> Path:
    stamp = now.astimezone(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    label = "tatqa" if dataset_kind == "tatqa_locked" else "multihop-rag"
    return REPOSITORY_ROOT / ".taskforge" / "eval-runs" / f"answer-{label}-{stamp}"


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.confirm_live_call:
        raise SystemExit("refusing billable answer eval without --confirm-live-call")
    settings = Settings(_env_file=REPOSITORY_ROOT / ".env")
    api_key = os.environ.get("TASKFORGE_DEEPSEEK_API_KEY", "").strip() or (
        settings.deepseek_api_key.get_secret_value()
        if settings.deepseek_api_key is not None
        else ""
    )
    if not api_key:
        raise SystemExit("TASKFORGE_DEEPSEEK_API_KEY is required")
    model = args.model or settings.deepseek_model or "deepseek-chat"
    base_url = os.environ.get(
        "TASKFORGE_DEEPSEEK_BASE_URL", settings.deepseek_base_url
    ).strip()

    dataset_kind = "tatqa_locked" if args.dataset == "tatqa" else "multihop_rag_locked"
    dataset_kwargs: dict[str, str] = {}
    if args.dataset == "tatqa":
        dataset_kwargs["tatqa_input_path"] = args.tatqa_input.strip().replace("\\", "/")
        dataset_kwargs["tatqa_locked_split_path"] = args.tatqa_split.strip().replace("\\", "/")
        dataset_kwargs["tatqa_context_mode"] = args.tatqa_context_mode.replace(
            "-", "_"
        )
    if args.multihop_split is not None:
        dataset_kwargs["multihop_rag_locked_split_path"] = (
            args.multihop_split.strip().replace("\\", "/")
        )
    retriever = args.retriever or (
        (
            "bm25_dense_rrf_rerank"
            if args.learned_reranker
            else "bm25_dense_rrf"
        )
        if args.semantic
        else (
            "bm25_source_coverage_rrf"
            if args.dataset == "multihop-rag"
            else "bm25"
        )
    )
    answer_contract = args.answer_contract.replace("-", "_")
    if retriever in {"tatqa_frozen_bm25", "tatqa_frozen_pair_rerank"} and args.dataset != "tatqa":
        raise SystemExit("frozen TAT-QA retrievers require --dataset tatqa")
    if answer_contract == "online_cited_v1" and retriever not in {
        "tatqa_frozen_bm25",
        "tatqa_frozen_pair_rerank",
    }:
        raise SystemExit(
            "online-cited-v1 requires --retriever tatqa_frozen_bm25 or "
            "tatqa_frozen_pair_rerank"
        )
    if model != "deepseek-v4-flash":
        raise SystemExit(
            "this versioned live-cost baseline is pinned to deepseek-v4-flash"
        )
    max_chunks_per_document = args.max_chunks_per_document
    if max_chunks_per_document is None and args.dataset == "multihop-rag":
        max_chunks_per_document = 1
    config = RAGAnswerEvalConfig(
        dataset=ExperimentDatasetConfig(kind=dataset_kind, **dataset_kwargs),
        retrieval=ExperimentRetrievalConfig(
            semantic_embedding=args.semantic,
            chunking=args.chunking or args.table_aware_chunking,
            table_aware_chunking=args.table_aware_chunking,
            max_chunks_per_document=max_chunks_per_document,
            learned_reranker=args.learned_reranker,
            reranker_model=args.reranker_model,
            bm25_field_weights=dict(ANSWER_EVAL_METADATA_FIELD_WEIGHTS),
        ),
        retriever=retriever,
        mode="agentic" if args.agentic else "naive",
        answer_contract=answer_contract,
        model=model,
        agent_max_steps=(
            args.agent_max_steps if args.agent_max_steps is not None else 8
        ),
        agentic_host_fallback=args.allow_agentic_host_fallback,
        evidence_top_k=(
            args.evidence_top_k
            if args.evidence_top_k is not None
            else (10 if answer_contract == "online_cited_v1" else 5)
        ),
        tatqa_query_slot_context=args.tatqa_query_slot_context,
        tatqa_query_slot_k=args.tatqa_query_slot_k,
        max_cases=args.max_cases,
        execution_mode="live",
        thinking_mode="disabled",
        json_mode=True,
        price_table=OnlineModelPrice(
            model="deepseek-v4-flash",
            input_cache_hit_per_million=0.0028,
            input_cache_miss_per_million=0.14,
            output_per_million=0.28,
            source_url="https://api-docs.deepseek.com/quick_start/pricing/",
            retrieved_at="2026-08-10",
        ),
    )
    output = args.output or _default_output(dataset_kind, datetime.now(UTC))

    async def run() -> dict[str, object]:
        provider = OpenAIChatCompletionsProvider(
            api_key=api_key,
            enabled=True,
            model=model,
            base_url=base_url,
            timeout_seconds=120,
            thinking_mode="disabled",
            json_mode=True,
        )
        try:
            rows, metrics, manifest = await run_rag_answer_eval(
                output_dir=output,
                config=config,
                provider=provider,
                repository_root=REPOSITORY_ROOT,
            )
        finally:
            await provider.aclose()
        return {
            "run_id": manifest["run_id"],
            "output": str(output),
            "dataset": dataset_kind,
            "retriever": retriever,
            "model": model,
            "cases": metrics["total_cases"],
            "exact_match_accuracy": metrics["exact_match_accuracy"],
            "avg_token_f1": metrics["avg_token_f1"],
            "evidence_retrieval": metrics["evidence_retrieval"],
            "grounding": metrics["grounding"],
            "failure_counts": metrics["failure_counts"],
            "latency_ms": metrics["latency_ms"],
            "model_usage": metrics["model_usage"],
            "by_category": metrics["by_category"],
            "rows": rows,
        }

    report = asyncio.run(run())
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"\npredictions: {output / 'predictions.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

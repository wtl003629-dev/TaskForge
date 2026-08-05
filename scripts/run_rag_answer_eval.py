"""Run TaskForge's end-to-end RAG answer evaluation (retrieve -> generate -> score).

For every locked case the eval retrieves evidence, asks a real model to answer
from that evidence alone, and scores exact match + token F1 against the gold
answer.  This makes billable model calls and requires ``--confirm-live-call``.
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
from taskforge.rag_answer_eval import RAGAnswerEvalConfig, run_rag_answer_eval
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
        choices=("bm25", "qdrant_rrf", "qdrant_rrf_rerank"),
        default=None,
        help="Retriever stage; defaults to qdrant_rrf_rerank with --semantic else bm25.",
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
        "--max-cases",
        type=int,
        default=None,
        help="Limit to the first N locked cases (for a quick smoke).",
    )
    value.add_argument(
        "--model",
        default=None,
        help="Model name; defaults to TASKFORGE_DEEPSEEK_MODEL.",
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
    retriever = args.retriever or (
        "qdrant_rrf_rerank" if args.semantic else "bm25"
    )
    config = RAGAnswerEvalConfig(
        dataset=ExperimentDatasetConfig(kind=dataset_kind),
        retrieval=ExperimentRetrievalConfig(semantic_embedding=args.semantic),
        retriever=retriever,
        mode="agentic" if args.agentic else "naive",
        model=model,
        max_cases=args.max_cases,
    )
    output = args.output or _default_output(dataset_kind, datetime.now(UTC))

    async def run() -> dict[str, object]:
        provider = OpenAIChatCompletionsProvider(
            api_key=api_key,
            enabled=True,
            model=model,
            base_url=base_url,
            timeout_seconds=120,
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
            "by_category": metrics["by_category"],
            "rows": rows,
        }

    report = asyncio.run(run())
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"\npredictions: {output / 'predictions.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run TaskForge's offline three-stage RAG retrieval ablation."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.rag_experiment import (
    RAGExperimentConfig,
    load_experiment_config,
    run_rag_experiment,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run lexical BM25, local-Qdrant RRF, and local-Qdrant RRF + "
            "fallback rerank on identical cases. No model download is used."
        )
    )
    value.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "eval" / "rag_experiment_config.json",
        help="Versioned experiment configuration.",
    )
    value.add_argument(
        "--dataset",
        choices=("synthetic", "tatqa", "multihop-rag"),
        help="Override the configured dataset; default config uses synthetic PDFs.",
    )
    value.add_argument(
        "--suite",
        type=Path,
        help="Synthetic suite path inside the repository.",
    )
    value.add_argument(
        "--tatqa-input",
        type=Path,
        help="Pinned TAT-QA cache path inside the repository.",
    )
    value.add_argument(
        "--locked-split",
        type=Path,
        help="TAT-QA locked split manifest inside the repository.",
    )
    value.add_argument(
        "--multihop-queries",
        type=Path,
        help="Pinned MultiHop-RAG query cache path inside the repository.",
    )
    value.add_argument(
        "--multihop-corpus",
        type=Path,
        help="Pinned MultiHop-RAG corpus cache path inside the repository.",
    )
    value.add_argument(
        "--multihop-locked-split",
        type=Path,
        help="MultiHop-RAG locked split manifest inside the repository.",
    )
    value.add_argument(
        "--semantic",
        action="store_true",
        help=(
            "Use a real semantic dense embedder (downloads an ONNX model). "
            "Off by default so the offline M1 gate never downloads a model."
        ),
    )
    value.add_argument(
        "--semantic-model",
        type=str,
        default=None,
        help="fastembed model name for --semantic (default BAAI/bge-small-en-v1.5).",
    )
    value.add_argument(
        "--chunking",
        action="store_true",
        help=(
            "Split long documents into paragraph-aware, overlapping chunks so "
            "dense embeddings stop truncating long evidence. Off by default."
        ),
    )
    value.add_argument(
        "--query-expansion",
        action="store_true",
        help=(
            "Deterministic pseudo-relevance feedback: expand the lexical query "
            "with terms from a first-pass retrieval. Off by default."
        ),
    )
    value.add_argument(
        "--bm25-field-weights",
        type=str,
        default=None,
        help=(
            "BM25 metadata field weights as field=weight pairs, e.g. "
            "title=3.0,source=2.0. Off by default."
        ),
    )
    value.add_argument(
        "--output",
        type=Path,
        help="Exact output run directory; an existing directory is never overwritten.",
    )
    return value


def _repository_relative(path: Path, label: str) -> str:
    candidate = path if path.is_absolute() else REPOSITORY_ROOT / path
    candidate = candidate.resolve()
    try:
        return candidate.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the TaskForge repository") from exc


def _with_overrides(
    config: RAGExperimentConfig,
    args: argparse.Namespace,
) -> RAGExperimentConfig:
    payload = config.model_dump(mode="json")
    dataset = dict(payload["dataset"])
    if args.dataset is not None:
        dataset["kind"] = {
            "synthetic": "synthetic_pdf",
            "tatqa": "tatqa_locked",
            "multihop-rag": "multihop_rag_locked",
        }[args.dataset]
    if args.suite is not None:
        dataset["synthetic_suite_path"] = _repository_relative(args.suite, "--suite")
    if args.tatqa_input is not None:
        dataset["tatqa_input_path"] = _repository_relative(
            args.tatqa_input, "--tatqa-input"
        )
    if args.locked_split is not None:
        dataset["tatqa_locked_split_path"] = _repository_relative(
            args.locked_split, "--locked-split"
        )
    if args.multihop_queries is not None:
        dataset["multihop_rag_queries_path"] = _repository_relative(
            args.multihop_queries, "--multihop-queries"
        )
    if args.multihop_corpus is not None:
        dataset["multihop_rag_corpus_path"] = _repository_relative(
            args.multihop_corpus, "--multihop-corpus"
        )
    if args.multihop_locked_split is not None:
        dataset["multihop_rag_locked_split_path"] = _repository_relative(
            args.multihop_locked_split, "--multihop-locked-split"
        )
    payload["dataset"] = dataset
    retrieval = dict(payload["retrieval"])
    if args.semantic:
        retrieval["semantic_embedding"] = True
    if args.semantic_model is not None:
        retrieval["semantic_model"] = args.semantic_model
    if args.chunking:
        retrieval["chunking"] = True
    if args.query_expansion:
        retrieval["query_expansion"] = True
    if args.bm25_field_weights is not None:
        weights: dict[str, float] = {}
        for pair in args.bm25_field_weights.split(","):
            pair = pair.strip()
            if not pair:
                continue
            field, _, raw = pair.partition("=")
            field = field.strip()
            if not field or not raw.strip():
                raise ValueError(
                    "--bm25-field-weights must be field=weight pairs"
                )
            weights[field] = float(raw)
        retrieval["bm25_field_weights"] = weights
    payload["retrieval"] = retrieval
    return RAGExperimentConfig.model_validate(payload)


def _default_output(now: datetime, dataset_kind: str) -> Path:
    stamp = now.astimezone(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    label = {
        "synthetic_pdf": "synthetic-pdf",
        "tatqa_locked": "tatqa-locked",
        "multihop_rag_locked": "multihop-rag-locked",
    }[dataset_kind]
    return REPOSITORY_ROOT / ".taskforge" / "eval-runs" / f"rag-{label}-{stamp}"


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = _with_overrides(load_experiment_config(args.config), args)
        now = datetime.now(UTC)
        output = args.output or _default_output(now, config.dataset.kind)
        result = run_rag_experiment(
            output_dir=output,
            config=config,
            repository_root=REPOSITORY_ROOT,
            config_source_path=args.config,
            created_at=now,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"RAG experiment failed: {exc}", file=sys.stderr)
        return 2

    print(f"run_id={result.manifest['run_id']}")
    print(f"output={result.output_dir}")
    print(f"mode={result.manifest['experiment_mode']}")
    print(f"cases={len(result.manifest['sample']['case_ids'])}")
    for stage, details in result.metrics["stages"].items():
        summary = details["retrieval"]["summary"]
        latency = details["latency"]
        largest_k = str(max(result.metrics["top_k"]))
        print(
            f"stage={stage} backend={details['backend']} "
            f"recall@{largest_k}={summary['recall_at_k'][largest_k]:.6f} "
            f"p50_ms={latency['p50']:.6f} p95_ms={latency['p95']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

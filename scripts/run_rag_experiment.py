"""Run TaskForge's configured RAG retrieval ablation."""

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
            "Run configured BM25, dense, RRF and rerank stages on identical cases. "
            "A model download occurs only when --semantic is explicit."
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
        choices=("synthetic", "tatqa", "multihop-rag", "qasper"),
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
        "--tatqa-context-mode",
        choices=("global_discovery", "provided_hybrid_context"),
        default=None,
        help=(
            "TAT-QA task contract: search the global evaluation KB or constrain "
            "retrieval to the hybrid context supplied with each question."
        ),
    )
    value.add_argument(
        "--tatqa-table-cleaning",
        action="store_true",
        help=(
            "Use the coordinate-preserving TAT-QA table search representation. "
            "Raw table rows and annotation coordinates remain unchanged."
        ),
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
        "--qasper-input",
        type=Path,
        help="Pinned QASPER JSON cache path inside the repository.",
    )
    value.add_argument(
        "--qasper-locked-split",
        type=Path,
        help="QASPER locked split manifest inside the repository.",
    )
    value.add_argument(
        "--qasper-context-mode",
        choices=("global_discovery", "provided_document_context"),
        default=None,
        help=(
            "QASPER task contract: search the global evaluation KB or constrain "
            "evidence selection to the paper supplied with each question."
        ),
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
        "--learned-sparse",
        action="store_true",
        help="Use an explicit FastEmbed learned sparse encoder for SPLADE stages.",
    )
    value.add_argument(
        "--sparse-model",
        type=str,
        default=None,
        help="FastEmbed learned sparse model name.",
    )
    value.add_argument(
        "--tatqa-sparse-weight",
        type=float,
        default=None,
        help="RRF branch weight for the table-profile learned sparse candidate.",
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
        "--table-aware-chunking",
        action="store_true",
        help="Index tables as schema, row, and column chunks; also enables chunking.",
    )
    value.add_argument(
        "--max-chunks-per-document",
        type=int,
        default=None,
        help="Cap ranked chunks per document to protect cross-document coverage.",
    )
    value.add_argument(
        "--parent-top-k",
        type=int,
        default=None,
        help="Override the bounded Parent-Child routing budget for an ablation.",
    )
    value.add_argument(
        "--no-parent-sibling-coverage",
        action="store_true",
        help="Disable same-parent evidence supplementation for a negative ablation.",
    )
    value.add_argument(
        "--tatqa-parent-query-expansion",
        action="store_true",
        help=(
            "Use deterministic TAT-QA subqueries only for parent-context routing; "
            "child evidence ranking keeps the original query."
        ),
    )
    value.add_argument(
        "--learned-reranker",
        action="store_true",
        help="Use an explicit FastEmbed ONNX cross-encoder for rerank stages.",
    )
    value.add_argument(
        "--development-sweep",
        action="store_true",
        help="Allow an explicitly incomplete stage matrix; artifacts cannot be promoted.",
    )
    value.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Comma-separated retrieval stages for --development-sweep.",
    )
    value.add_argument(
        "--reranker-model",
        type=str,
        default=None,
        help="FastEmbed cross-encoder model name.",
    )
    value.add_argument(
        "--domain-reranker",
        type=Path,
        default=None,
        help="Repository-relative TAT-QA domain reranker JSON artifact.",
    )
    value.add_argument(
        "--rerank-top-k",
        type=int,
        default=None,
        help="Number of candidates sent to a learned reranker (default 20).",
    )
    value.add_argument(
        "--rrf-k",
        type=int,
        default=None,
        help="Override reciprocal-rank-fusion k for an explicit ablation.",
    )
    value.add_argument(
        "--context-seed-k",
        type=int,
        default=None,
        help="Number of fused candidates used for same-context coverage.",
    )
    value.add_argument(
        "--tatqa-lineage-seed-k",
        type=int,
        default=None,
        help="Top ranked table-profile candidates whose parent lineage is inspected.",
    )
    value.add_argument(
        "--tatqa-lineage-closure-slots",
        type=int,
        default=None,
        help="Candidate-tail slots reserved for query-relevant same-parent evidence.",
    )
    value.add_argument(
        "--tatqa-lineage-max-siblings",
        type=int,
        default=None,
        help="Maximum query-relevant sibling evidence units added per seed parent.",
    )
    value.add_argument(
        "--tatqa-structured-candidate-slots",
        type=int,
        default=None,
        help="Candidate-tail slots reserved for query-typed structured table facts.",
    )
    value.add_argument(
        "--tatqa-lineage-pair-rerank-slots",
        type=int,
        default=None,
        help="Existing same-parent candidates promoted into the stable ranking head.",
    )
    value.add_argument(
        "--tatqa-lineage-pair-min-score",
        type=float,
        default=None,
        help="Minimum lineage-closure score for the isolated pair reranker.",
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
        "--graph-fusion",
        action="store_true",
        help=(
            "Fuse the lexical ranking with a local document co-occurrence graph "
            "via RRF as an extra graph_fused stage. Off by default."
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
            "qasper": "qasper_locked",
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
    if args.tatqa_context_mode is not None:
        dataset["tatqa_context_mode"] = args.tatqa_context_mode
    if args.tatqa_table_cleaning:
        dataset["tatqa_table_cleaning"] = True
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
    if args.qasper_input is not None:
        dataset["qasper_input_path"] = _repository_relative(
            args.qasper_input, "--qasper-input"
        )
    if args.qasper_locked_split is not None:
        dataset["qasper_locked_split_path"] = _repository_relative(
            args.qasper_locked_split, "--qasper-locked-split"
        )
    if args.qasper_context_mode is not None:
        dataset["qasper_context_mode"] = args.qasper_context_mode
    payload["dataset"] = dataset
    retrieval = dict(payload["retrieval"])
    if args.semantic:
        retrieval["semantic_embedding"] = True
    if args.semantic_model is not None:
        retrieval["semantic_model"] = args.semantic_model
    if args.learned_sparse:
        retrieval["learned_sparse"] = True
    if args.sparse_model is not None:
        retrieval["sparse_model"] = args.sparse_model
    if args.tatqa_sparse_weight is not None:
        retrieval["tatqa_sparse_weight"] = args.tatqa_sparse_weight
    if args.chunking:
        retrieval["chunking"] = True
    if args.table_aware_chunking:
        retrieval["chunking"] = True
        retrieval["table_aware_chunking"] = True
    if args.max_chunks_per_document is not None:
        retrieval["max_chunks_per_document"] = args.max_chunks_per_document
    if args.parent_top_k is not None:
        retrieval["parent_top_k"] = args.parent_top_k
    if args.no_parent_sibling_coverage:
        retrieval["parent_sibling_coverage"] = False
    if args.tatqa_parent_query_expansion:
        retrieval["tatqa_parent_query_expansion"] = True
    if args.learned_reranker:
        retrieval["learned_reranker"] = True
    if args.reranker_model is not None:
        retrieval["reranker_model"] = args.reranker_model
    if args.domain_reranker is not None:
        retrieval["domain_reranker_path"] = _repository_relative(
            args.domain_reranker,
            "--domain-reranker",
        )
    if args.rerank_top_k is not None:
        retrieval["rerank_top_k"] = args.rerank_top_k
    if args.rrf_k is not None:
        retrieval["rrf_k"] = args.rrf_k
    if args.context_seed_k is not None:
        retrieval["context_seed_k"] = args.context_seed_k
    if args.tatqa_lineage_seed_k is not None:
        retrieval["tatqa_lineage_seed_k"] = args.tatqa_lineage_seed_k
    if args.tatqa_lineage_closure_slots is not None:
        retrieval["tatqa_lineage_closure_slots"] = args.tatqa_lineage_closure_slots
    if args.tatqa_lineage_max_siblings is not None:
        retrieval["tatqa_lineage_max_siblings_per_parent"] = (
            args.tatqa_lineage_max_siblings
        )
    if args.tatqa_structured_candidate_slots is not None:
        retrieval["tatqa_structured_candidate_slots"] = (
            args.tatqa_structured_candidate_slots
        )
    if args.tatqa_lineage_pair_rerank_slots is not None:
        retrieval["tatqa_lineage_pair_rerank_slots"] = (
            args.tatqa_lineage_pair_rerank_slots
        )
    if args.tatqa_lineage_pair_min_score is not None:
        retrieval["tatqa_lineage_pair_min_score"] = (
            args.tatqa_lineage_pair_min_score
        )
    if args.development_sweep:
        retrieval["development_sweep"] = True
    if args.stages is not None:
        retrieval["stages"] = [
            stage.strip() for stage in args.stages.split(",") if stage.strip()
        ]
    if args.query_expansion:
        retrieval["query_expansion"] = True
    if args.graph_fusion:
        retrieval["graph_fusion"] = True
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
        "qasper_locked": "qasper-locked",
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

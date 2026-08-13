"""Offline, reproducible retrieval ablations for TaskForge.

The experiment deliberately uses a real qdrant-client local ``:memory:``
collection while keeping model downloads and provider calls out of the M1
gate.  Its dense branch is deterministic feature hashing, not a semantic
embedding model, and every artifact records that degraded limitation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from . import __version__
from .domain import StrictModel
from .evidence_graph import EvidenceQueryPlan, LocalEvidenceGraph
from .graph_reranker import LearnedGraphReranker, sha256_model
from .hybrid_retrieval import (
    AppliedRetrievalFilters,
    BM25DenseRRFIndex,
    BM25Index,
    CandidateTailUnionIndex,
    DenseEmbedder,
    DeterministicHashEmbedder,
    FastEmbedCrossEncoderReranker,
    FastEmbedEmbedder,
    FastEmbedSparseIndex,
    HybridChunk,
    HybridSearchHit,
    HybridSearchRequest,
    HybridSearchResponse,
    InMemoryDenseIndex,
    LexicalOverlapFallbackReranker,
    MultiQueryRRFIndex,
    ParentChildIndex,
    QdrantDenseIndex,
    QdrantHybridIndex,
    RepresentationRRFIndex,
    SearchRepresentationIndex,
    SourceCoverageRRFIndex,
    TATQADomainReranker,
    TATQAFeatureReranker,
    _matches_scope,
)
from .knowledge import tokenise
from .local_graph import LocalDocumentGraph
from .rag_baseline import (
    load_locked_split,
    select_locked_cases,
    sha256_file,
)
from .rag_evaluation import (
    EvalCorpusDocument,
    RAGEvalCase,
    RAGEvalDataset,
    RetrievalPrediction,
    evaluate_hierarchical_retrieval,
    evaluate_retrieval,
    load_multihop_rag_dataset,
    load_qasper_dataset,
    load_tatqa_dataset,
)
from .rag_profiles import (
    corpus_metadata,
    profile_metadata,
    query_features,
    select_retrieval_profile,
)
from .rag_tatqa_diagnostics import build_tatqa_query_plan_from_text
from .synthetic_pdf_eval import (
    SyntheticGenerationManifest,
    generate_synthetic_pdfs,
    load_generated_page_dataset,
)

StageName = Literal[
    "lexical_bm25",
    "lexical_bm25_rerank",
    "bm25_table_router",
    "bm25_table_multi_rep_rrf",
    "bm25_table_multi_rep_max",
    "bm25_table_multi_rep_adaptive",
    "bm25_table_row_cell_rrf",
    "bm25_dense_table_row_cell_rrf",
    "bm25_multi_query_rrf",
    "bm25_tatqa_query_rrf",
    "bm25_tatqa_query_plan_rrf",
    "bm25_tatqa_query_plan_scan_rrf",
    "bm25_tatqa_query_plan_scan_context_rrf",
    "bm25_tatqa_query_plan_parent_scan_rrf",
    "bm25_tatqa_query_plan_context_scan_rrf",
    "bm25_dense_tatqa_query_plan_parent_scan_rrf",
    "bm25_tatqa_query_plan_parent_scan_feature_rerank",
    "bm25_tatqa_query_plan_parent_scan_closure_rrf",
    "bm25_tatqa_query_plan_parent_scan_closure_all_profile_rrf",
    "bm25_tatqa_query_plan_parent_scan_closure_table_profile_rrf",
    "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_rrf",
    "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_structured_rrf",
    "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_pair_rerank_rrf",
    "bm25_dense_tatqa_query_plan_parent_scan_closure_table_profile_lineage_pair_rerank_candidate_union",
    "bm25_tatqa_query_plan_compact_parent_scan_closure_table_profile_lineage_pair_rerank_rrf",
    "bm25_tatqa_query_plan_passage_parent_scan_closure_table_profile_lineage_pair_rerank_rrf",
    "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_splade_rrf",
    "bm25_tatqa_query_plan_parent_scan_closure_table_profile_feature_rerank",
    "bm25_tatqa_query_plan_parent_scan_closure_table_profile_feature_rerank_blend05",
    "bm25_tatqa_query_plan_parent_scan_closure_table_profile_domain_rerank",
    "bm25_tatqa_query_plan_parent_scan_closure_table_profile_cross_encoder_rerank",
    "bm25_tatqa_query_plan_parent_scan_closure_feature_rerank",
    "bm25_dense_tatqa_query_rrf",
    "bm25_dense_tatqa_query_context_rrf",
    "bm25_dense_tatqa_query_context_query_weighted_rrf",
    "bm25_dense_tatqa_query_context_dense_weighted_rrf",
    "bm25_dense_tatqa_query_table_candidate_rrf",
    "bm25_dense_tatqa_query_context_rerank",
    "bm25_dense_tatqa_dual_query_rrf",
    "bm25_dense_tatqa_dual_query_context_rrf",
    "bm25_dense_tatqa_table_rrf",
    "bm25_dense_tatqa_table_context_rrf",
    "bm25_dense_tatqa_query_feature_rerank",
    "bm25_source_coverage_rrf",
    "bm25_source_coverage_anchor_rrf",
    "bm25_parent_child",
    "bm25_dense_parent_child",
    "bm25_dense_parent_child_rrf",
    "qdrant_dense",
    "bm25_dense_rrf",
    "bm25_dense_table_profile_rrf",
    "bm25_dense_rrf_coverage",
    "bm25_dense_max_coverage",
    "bm25_dense_rrf_rerank",
    "bm25_qasper_hierarchical",
    "qdrant_qasper_dense",
    "qdrant_qasper_dense_rerank",
    "bm25_dense_qasper_candidate_union",
    "bm25_dense_qasper_section_parent",
    "bm25_dense_qasper_section_parent_rrf",
    "qdrant_rrf",
    "qdrant_rrf_rerank",
    "graph_fused",
    "graph_feature_rerank",
]
DatasetKind = Literal[
    "synthetic_pdf",
    "tatqa_locked",
    "multihop_rag_locked",
    "qasper_locked",
]
REQUIRED_STAGES: tuple[StageName, ...] = (
    "lexical_bm25",
    "qdrant_rrf",
    "qdrant_rrf_rerank",
)
EXPERIMENT_MODE = "degraded_nonsemantic"
_DENIED_PRINCIPAL = "principal:taskforge-denied-probe"
_PROBE_PREFIX = "__taskforge_filter_probe__"
_CHUNK_SEP = "::chunk::"
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
_TATQA_COMPACT_PARENT_PAIR_STAGE = (
    "bm25_tatqa_query_plan_compact_parent_scan_closure_table_profile_"
    "lineage_pair_rerank_rrf"
)
_TATQA_PASSAGE_PARENT_PAIR_STAGE = (
    "bm25_tatqa_query_plan_passage_parent_scan_closure_table_profile_"
    "lineage_pair_rerank_rrf"
)
_TATQA_DENSE_CANDIDATE_UNION_STAGE = (
    "bm25_dense_tatqa_query_plan_parent_scan_closure_table_profile_"
    "lineage_pair_rerank_candidate_union"
)
_SEMANTIC_EMBEDDING_BATCH_SIZE = 64
_QDRANT_UPSERT_BATCH_SIZE = 128
_SEMANTIC_DENSE_CANDIDATE_SLOTS = 10


def _safe_repository_path(value: object, field_name: str) -> str:
    candidate = str(value).strip().replace("\\", "/")
    if (
        not candidate
        or candidate.startswith("/")
        or ":" in candidate
        or ".." in candidate.split("/")
    ):
        raise ValueError(f"{field_name} must be a safe repository-relative path")
    return candidate


def _hard_split(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split an over-budget block on character boundaries with tail overlap."""

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        pieces.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return pieces


def chunk_text(
    text: str,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Paragraph-aware, bounded chunking with tail overlap.

    Paragraphs are packed greedily into chunks of at most ``max_chars``; when a
    chunk closes, the next chunk re-opens with the previous chunk's tail so
    evidence straddling a boundary stays contiguous.  A single paragraph larger
    than the budget is hard-split on character boundaries.
    """

    if max_chars < 200:
        raise ValueError("max_chars must be at least 200")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be between 0 and max_chars")
    paragraphs = [paragraph.strip() for paragraph in _PARAGRAPH_SPLIT.split(text)]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    if not paragraphs:
        return [text]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_split(paragraph, max_chars, overlap_chars))
            continue
        if current and len(current) + 1 + len(paragraph) > max_chars:
            chunks.append(current)
            current = current[-overlap_chars:] if overlap_chars else ""
        current = f"{current}\n{paragraph}" if current else paragraph
    if current:
        chunks.append(current)
    return chunks


def table_aware_chunks(
    text: str,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Create schema, row, and column representations with repeated headers."""

    rows = [
        [cell.strip() for cell in line.split("|")]
        for line in text.splitlines()
        if line.strip()
    ]
    if len(rows) < 2 or len(rows[0]) < 2:
        return chunk_text(text, max_chars=max_chars, overlap_chars=overlap_chars)
    width = max(len(row) for row in rows)
    header = [
        rows[0][index] if index < len(rows[0]) and rows[0][index] else f"column_{index + 1}"
        for index in range(width)
    ]
    body = [row + [""] * (width - len(row)) for row in rows[1:]]
    representations: list[str] = []
    row_labels = [row[0] for row in body if row[0]]
    schema = "Table columns: " + " | ".join(header)
    if row_labels:
        schema += "\nTable row labels: " + " | ".join(row_labels)
    representations.extend(
        chunk_text(schema, max_chars=max_chars, overlap_chars=overlap_chars)
    )
    for row in body:
        fields = [
            f"{column}: {value}"
            for column, value in zip(header, row, strict=True)
            if value
        ]
        if fields:
            row_text = "Table columns: " + " | ".join(header) + "\nTable row: " + " | ".join(fields)
            representations.extend(
                chunk_text(row_text, max_chars=max_chars, overlap_chars=overlap_chars)
            )
    label_header = header[0]
    for column_index in range(1, width):
        values = [
            f"{label_header}={row[0]} | {header[column_index]}={row[column_index]}"
            for row in body
            if row[0] and row[column_index]
        ]
        if values:
            column_text = f"Table column: {header[column_index]}\n" + "\n".join(values)
            representations.extend(
                chunk_text(column_text, max_chars=max_chars, overlap_chars=overlap_chars)
            )
    return list(dict.fromkeys(representations))


def _chunk_document_text(
    text: str,
    metadata: Mapping[str, Any],
    config: ExperimentRetrievalConfig,
) -> list[str]:
    if not config.chunking:
        return [text]
    if config.table_aware_chunking and metadata.get("kind") == "table":
        return table_aware_chunks(
            text,
            max_chars=config.chunk_max_chars,
            overlap_chars=config.chunk_overlap_chars,
        )
    return chunk_text(
        text,
        max_chars=config.chunk_max_chars,
        overlap_chars=config.chunk_overlap_chars,
    )


def _document_id_from_chunk_id(chunk_id: str) -> str:
    return chunk_id.split(_CHUNK_SEP, 1)[0]


def _expand_with_prf(
    index: BM25Index,
    request: HybridSearchRequest,
    *,
    first_pass_k: int = 5,
    added_terms: int = 6,
) -> str | None:
    """Deterministic pseudo-relevance feedback query expansion.

    A first-pass retrieval surfaces the top chunks; the most frequent terms in
    them that are not already in the query are appended to the query.  This adds
    new terms without an LLM, and is honest: the added terms are grounded in
    retrieved corpus text, and the extra latency is measured in the stage.
    """

    probe = request.model_copy(update={"top_k": first_pass_k})
    response = index.search(probe)
    if not response.hits:
        return None
    query_terms = set(tokenise(request.query))
    term_scores: Counter[str] = Counter()
    for hit in response.hits:
        for term, count in Counter(tokenise(hit.chunk.text)).items():
            if term not in query_terms:
                term_scores[term] += count
    selected = [term for term, _ in term_scores.most_common(added_terms)]
    if not selected:
        return None
    return f"{request.query} {' '.join(selected)}"


def _fuse_rrf(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
    limit: int,
) -> list[str]:
    """Weighted reciprocal-rank fusion of ranked document lists."""

    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, document_id in enumerate(ranking, start=1):
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [document_id for document_id, _ in ordered[:limit]]


def _graph_fused_search(
    graph: LocalDocumentGraph,
    lexical: BM25Index,
    config: RAGExperimentConfig,
    case_query: str,
) -> tuple[list[str], str]:
    """Fuse the lexical document ranking with one-hop graph neighbors by RRF."""

    seed_request = _search_request(case_query, config, rerank=False).model_copy(
        update={"top_k": config.retrieval.candidate_k}
    )
    response = lexical.search(seed_request)
    lexical_documents = _deduped_document_ids(
        response.hits, max_documents=config.retrieval.candidate_k
    )
    graph_documents = graph.search(
        case_query, max_results=config.retrieval.graph_max_neighbors
    )
    fused = _fuse_rrf(
        [lexical_documents, graph_documents],
        limit=max(config.retrieval.top_k),
    )
    return fused, "local_graph_rrf"


def _deduped_document_ids(
    hits: Sequence[Any],
    *,
    max_documents: int | None = None,
) -> list[str]:
    """Map ranked chunk hits to top-N deduplicated document identifiers.

    Chunks of one document may crowd a fixed top-k, so when the index was built
    with chunking the caller retrieves more chunks and groups them here, taking
    at most ``max_documents`` documents in hit order.
    """

    seen: set[str] = set()
    result: list[str] = []
    for hit in hits:
        document_id = str(
            getattr(hit.chunk, "document_id", "")
            or _document_id_from_chunk_id(hit.chunk.chunk_id)
        )
        if document_id not in seen:
            seen.add(document_id)
            result.append(document_id)
            if max_documents is not None and len(result) >= max_documents:
                break
    return result


def _deduped_parent_ids(
    hits: Sequence[Any],
    *,
    max_documents: int | None = None,
) -> list[str]:
    """Map ranked hits to stable parent-document identifiers."""

    seen: set[str] = set()
    result: list[str] = []
    for hit in hits:
        metadata = hit.chunk.metadata
        parent_id = str(
            metadata.get("parent_document_id", hit.chunk.document_id)
        )
        if parent_id not in seen:
            seen.add(parent_id)
            result.append(parent_id)
            if max_documents is not None and len(result) >= max_documents:
                break
    return result


def _retrieved_table_units(
    hits: Sequence[Any],
) -> tuple[list[str], list[str], list[str], list[dict[str, Any]]]:
    """Extract table units plus their original evidence-hit ranks.

    The flattened row/cell lists are retained for backward compatibility.  The
    aligned records prevent a de-duplicated document ranking from being
    mistaken for row/cell hit rank in coordinate-level diagnostics.
    """

    row_ids: list[str] = []
    cell_ids: list[str] = []
    complete_table_ids: list[str] = []
    seen_rows: set[str] = set()
    seen_cells: set[str] = set()
    seen_tables: set[str] = set()
    units_by_hit: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        metadata = chunk.metadata
        document_id = chunk.document_id
        if metadata.get("table_complete") is True and document_id not in seen_tables:
            seen_tables.add(document_id)
            complete_table_ids.append(document_id)
        raw_row = metadata.get("table_row_index")
        if not isinstance(raw_row, int):
            raw_row = metadata.get("structured_row_index")
        raw_column = metadata.get("table_column_index")
        row_id: str | None = None
        cell_id: str | None = None
        if isinstance(raw_row, int) and raw_row >= 0:
            row_id = f"{document_id}::row::{raw_row}"
            if row_id not in seen_rows:
                seen_rows.add(row_id)
                row_ids.append(row_id)
            if isinstance(raw_column, int) and raw_column >= 0:
                cell_id = f"{document_id}::cell::{raw_row}::{raw_column}"
                if cell_id not in seen_cells:
                    seen_cells.add(cell_id)
                    cell_ids.append(cell_id)
        if metadata.get("kind") == "table":
            units_by_hit.append(
                {
                    "rank": rank,
                    "chunk_id": chunk.chunk_id,
                    "document_id": document_id,
                    "representation": metadata.get("representation"),
                    "table_complete": metadata.get("table_complete") is True,
                    "row_index": raw_row if isinstance(raw_row, int) else None,
                    "column_index": (
                        raw_column if isinstance(raw_column, int) else None
                    ),
                    "row_id": row_id,
                    "cell_id": cell_id,
                }
            )
    return row_ids, cell_ids, complete_table_ids, units_by_hit


class ExperimentDatasetConfig(StrictModel):
    kind: DatasetKind = "synthetic_pdf"
    tatqa_context_mode: Literal[
        "global_discovery", "provided_hybrid_context"
    ] = "global_discovery"
    tatqa_table_cleaning: bool = False
    qasper_context_mode: Literal[
        "global_discovery", "provided_document_context"
    ] = "global_discovery"
    synthetic_suite_path: str = "eval/synthetic_pdf_suite.json"
    tatqa_input_path: str = ".taskforge/eval-cache/tatqa_dataset_dev.json"
    tatqa_locked_split_path: str = "eval/splits/tatqa-dev-m0-100-v1.json"
    multihop_rag_queries_path: str = ".taskforge/eval-cache/MultiHopRAG.json"
    multihop_rag_corpus_path: str = ".taskforge/eval-cache/corpus.json"
    multihop_rag_locked_split_path: str = (
        "eval/splits/multihop-rag-dev-m0-100-v1.json"
    )
    qasper_input_path: str = ".taskforge/eval-cache/qasper-dev-v0.3.json"
    qasper_locked_split_path: str = "eval/splits/qasper-dev-general-100-v1.json"

    @field_validator(
        "synthetic_suite_path",
        "tatqa_input_path",
        "tatqa_locked_split_path",
        "multihop_rag_queries_path",
        "multihop_rag_corpus_path",
        "multihop_rag_locked_split_path",
        "qasper_input_path",
        "qasper_locked_split_path",
        mode="before",
    )
    @classmethod
    def paths_are_repository_relative(cls, value: object, info: Any) -> str:
        return _safe_repository_path(value, info.field_name)

    @model_validator(mode="after")
    def context_mode_matches_dataset(self) -> ExperimentDatasetConfig:
        if (
            self.tatqa_context_mode == "provided_hybrid_context"
            and self.kind != "tatqa_locked"
        ):
            raise ValueError(
                "provided_hybrid_context is only valid for the TAT-QA dataset"
            )
        if self.tatqa_table_cleaning and self.kind != "tatqa_locked":
            raise ValueError("tatqa_table_cleaning is only valid for TAT-QA")
        if (
            self.qasper_context_mode == "provided_document_context"
            and self.kind != "qasper_locked"
        ):
            raise ValueError(
                "provided_document_context is only valid for the QASPER dataset"
            )
        return self


class ExperimentRetrievalConfig(StrictModel):
    stages: list[StageName] = Field(default_factory=lambda: list(REQUIRED_STAGES))
    development_sweep: bool = False
    top_k: list[int] = Field(default_factory=lambda: [1, 5, 10])
    candidate_k: int = Field(default=25, ge=1, le=500)
    # Parent routing is an explicit budget, separate from child candidate_k.
    # TAT-QA ablations use Parent Top-5 unless an experiment opts in to
    # another value on the optimization split.
    parent_top_k: int = Field(default=5, ge=1, le=100)
    parent_sibling_coverage: bool = True
    tatqa_parent_query_expansion: bool = False
    bm25_k1: float = Field(default=1.5, gt=0.0, le=10.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1, le=10_000)
    rrf_bm25_weight: float = Field(default=1.0, gt=0.0, le=100.0)
    rrf_dense_weight: float = Field(default=1.0, gt=0.0, le=100.0)
    tatqa_numeric_cell_weight: float = Field(default=0.25, gt=0.0, le=100.0)
    tatqa_numeric_scan_weight: float = Field(default=0.5, gt=0.0, le=100.0)
    hash_dimension: int = Field(default=64, ge=8, le=65_536)
    semantic_embedding: bool = False
    semantic_model: str = Field(default="BAAI/bge-small-en-v1.5", min_length=1)
    learned_sparse: bool = False
    sparse_model: str = Field(default="prithivida/Splade_PP_en_v1", min_length=1)
    tatqa_sparse_weight: float = Field(default=0.5, gt=0.0, le=100.0)
    chunking: bool = False
    table_aware_chunking: bool = False
    chunk_max_chars: int = Field(default=1500, ge=200, le=20_000)
    chunk_overlap_chars: int = Field(default=150, ge=0, le=10_000)
    query_expansion: bool = False
    bm25_field_weights: dict[str, float] = Field(default_factory=dict)
    graph_fusion: bool = False
    graph_max_neighbors: int = Field(default=12, ge=1, le=100)
    graph_feature_rerank: bool = False
    graph_rerank_base_stage: str = "qdrant_qasper_dense_rerank"
    graph_rerank_seed_k: int = Field(default=10, ge=1, le=100)
    graph_rerank_graph_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    graph_rerank_entity_weight: float = Field(default=0.08, ge=0.0, le=1.0)
    graph_rerank_section_weight: float = Field(default=0.02, ge=0.0, le=1.0)
    graph_rerank_adjacency_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    graph_rerank_ppr_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    graph_candidate_expansion: bool = False
    graph_expansion_hops: int = Field(default=1, ge=1, le=2)
    graph_expansion_slots: int = Field(default=0, ge=0, le=50)
    graph_learned_reranker_path: str | None = Field(default=None, max_length=2_048)
    max_chunks_per_document: int | None = Field(default=None, ge=1, le=100)
    learned_reranker: bool = False
    reranker_model: str = Field(
        default="Xenova/ms-marco-MiniLM-L-6-v2",
        min_length=1,
    )
    reranker_batch_size: int = Field(default=32, ge=1, le=128)
    domain_reranker_path: str | None = Field(default=None, max_length=2_048)
    rerank_top_k: int = Field(default=20, ge=1, le=100)
    adaptive_rerank_enabled: bool = False
    adaptive_rerank_min_k: int = Field(default=20, ge=1, le=99)
    adaptive_rerank_margin_threshold: float = Field(default=0.7, ge=0.0)
    context_seed_k: int = Field(default=20, ge=1, le=100)
    tatqa_lineage_seed_k: int = Field(default=20, ge=1, le=100)
    tatqa_lineage_closure_slots: int = Field(default=12, ge=1, le=100)
    tatqa_lineage_max_siblings_per_parent: int = Field(default=2, ge=1, le=10)
    tatqa_structured_candidate_slots: int = Field(default=10, ge=1, le=100)
    tatqa_lineage_pair_rerank_slots: int = Field(default=1, ge=1, le=10)
    tatqa_lineage_pair_min_score: float = Field(default=0.24, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def chunk_overlap_is_smaller_than_budget(self) -> ExperimentRetrievalConfig:
        if self.chunking and self.chunk_overlap_chars >= self.chunk_max_chars:
            raise ValueError("chunk_overlap_chars must be smaller than chunk_max_chars")
        if self.table_aware_chunking and not self.chunking:
            raise ValueError("table_aware_chunking requires chunking")
        if self.domain_reranker_path is not None:
            cleaned_path = self.domain_reranker_path.strip()
            if not cleaned_path:
                raise ValueError("domain_reranker_path must not be blank")
            object.__setattr__(self, "domain_reranker_path", cleaned_path)
        if self.graph_learned_reranker_path is not None:
            cleaned_path = _safe_repository_path(
                self.graph_learned_reranker_path,
                "graph_learned_reranker_path",
            )
            object.__setattr__(self, "graph_learned_reranker_path", cleaned_path)
        if not self.graph_rerank_base_stage.strip():
            raise ValueError("graph_rerank_base_stage must not be blank")
        if self.graph_rerank_base_stage in {"graph_fused", "graph_feature_rerank"}:
            raise ValueError("graph_rerank_base_stage must be a non-graph stage")
        graph_weights = (
            self.graph_rerank_graph_weight,
            self.graph_rerank_entity_weight,
            self.graph_rerank_section_weight,
            self.graph_rerank_adjacency_weight,
        )
        if sum(graph_weights) > 1.0:
            raise ValueError("graph rerank feature weights must sum to at most 1")
        if self.graph_learned_reranker_path is not None and not self.graph_feature_rerank:
            raise ValueError("graph_learned_reranker_path requires graph_feature_rerank")
        if self.adaptive_rerank_enabled:
            if not self.learned_reranker:
                raise ValueError("adaptive rerank requires learned_reranker")
            if "qdrant_qasper_dense_rerank" not in self.stages:
                raise ValueError(
                    "adaptive rerank is isolated to qdrant_qasper_dense_rerank"
                )
            if self.adaptive_rerank_min_k >= self.rerank_top_k:
                raise ValueError(
                    "adaptive_rerank_min_k must be smaller than rerank_top_k"
                )
            if self.rerank_top_k > self.candidate_k:
                raise ValueError("adaptive rerank max budget must not exceed candidate_k")
            if not math.isfinite(self.adaptive_rerank_margin_threshold):
                raise ValueError(
                    "adaptive_rerank_margin_threshold must be finite"
                )
        for field, weight in self.bm25_field_weights.items():
            if not isinstance(weight, (int, float)) or weight <= 0:
                raise ValueError("bm25_field_weights must be finite positive numbers")
        return self

    @model_validator(mode="after")
    def stages_and_budgets_are_comparable(self) -> ExperimentRetrievalConfig:
        if len(self.stages) != len(set(self.stages)):
            raise ValueError("retrieval stages must not contain duplicates")
        missing = [stage for stage in REQUIRED_STAGES if stage not in self.stages]
        if missing and not self.development_sweep:
            raise ValueError(
                "retrieval stages must include the complete M1 ablation: "
                + ", ".join(missing)
            )
        if not self.top_k:
            raise ValueError("top_k must not be empty")
        if len(self.top_k) != len(set(self.top_k)):
            raise ValueError("top_k must not contain duplicates")
        if any(value < 1 or value > 100 for value in self.top_k):
            raise ValueError("top_k values must be between 1 and 100")
        ordered = sorted(self.top_k)
        if self.candidate_k < max(ordered):
            raise ValueError("candidate_k must be greater than or equal to max(top_k)")
        lineage_stages = {
            "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_rrf",
            "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_structured_rrf",
            "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_pair_rerank_rrf",
            _TATQA_DENSE_CANDIDATE_UNION_STAGE,
            "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_splade_rrf",
            _TATQA_COMPACT_PARENT_PAIR_STAGE,
            _TATQA_PASSAGE_PARENT_PAIR_STAGE,
        }
        if lineage_stages.intersection(self.stages) and (
            self.candidate_k - self.tatqa_lineage_closure_slots < max(ordered)
        ):
            raise ValueError(
                "tatqa lineage closure must preserve max(top_k) ranked candidates"
            )
        structured_stages = {
            "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_structured_rrf",
            "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_pair_rerank_rrf",
            _TATQA_DENSE_CANDIDATE_UNION_STAGE,
            _TATQA_COMPACT_PARENT_PAIR_STAGE,
            _TATQA_PASSAGE_PARENT_PAIR_STAGE,
        }
        if structured_stages.intersection(self.stages) and (
            self.candidate_k - self.tatqa_structured_candidate_slots < max(ordered)
        ):
            raise ValueError(
                "tatqa structured candidates must preserve max(top_k) ranked candidates"
            )
        pair_stages = {
            "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_pair_rerank_rrf",
            _TATQA_DENSE_CANDIDATE_UNION_STAGE,
            _TATQA_COMPACT_PARENT_PAIR_STAGE,
            _TATQA_PASSAGE_PARENT_PAIR_STAGE,
        }
        if pair_stages.intersection(self.stages) and (
            max(ordered) < 2
            or self.tatqa_lineage_pair_rerank_slots >= max(ordered)
        ):
            raise ValueError(
                "tatqa lineage pair rerank slots must be smaller than max(top_k)"
            )
        object.__setattr__(self, "top_k", ordered)
        return self


class ExperimentFilterConfig(StrictModel):
    tenant_id: str = Field(default="tenant-evaluation", min_length=1, max_length=128)
    request_principals: list[str] = Field(
        default_factory=lambda: ["user:evaluator"], min_length=1, max_length=64
    )
    indexed_acl_principals: list[str] = Field(
        default_factory=lambda: ["user:evaluator", "role:rag-reviewer"],
        min_length=1,
        max_length=64,
    )
    knowledge_base_id: str = Field(
        default="taskforge-evaluation", min_length=1, max_length=128
    )
    version: str = Field(default="1", min_length=1, max_length=64)
    version_order: int = Field(default=1, ge=0)

    @field_validator(
        "tenant_id", "knowledge_base_id", "version", mode="before"
    )
    @classmethod
    def strings_are_clean(cls, value: object, info: Any) -> str:
        candidate = str(value).strip()
        if not candidate:
            raise ValueError(f"{info.field_name} must not be blank")
        return candidate

    @field_validator("request_principals", "indexed_acl_principals", mode="before")
    @classmethod
    def principals_are_clean(cls, value: object, info: Any) -> list[str]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ValueError(f"{info.field_name} must be a list")
        cleaned = sorted({str(item).strip() for item in value if str(item).strip()})
        if not cleaned:
            raise ValueError(f"{info.field_name} must not be empty")
        return cleaned

    @model_validator(mode="after")
    def requested_identity_can_read_index(self) -> ExperimentFilterConfig:
        if _DENIED_PRINCIPAL in self.request_principals:
            raise ValueError("request principals contain the reserved denied probe")
        if not set(self.request_principals).intersection(self.indexed_acl_principals):
            raise ValueError("request principals cannot read the evaluation corpus")
        return self


class RAGExperimentConfig(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset: ExperimentDatasetConfig = Field(default_factory=ExperimentDatasetConfig)
    retrieval: ExperimentRetrievalConfig = Field(
        default_factory=ExperimentRetrievalConfig
    )
    filters: ExperimentFilterConfig = Field(default_factory=ExperimentFilterConfig)


@dataclass(frozen=True)
class RAGExperimentResult:
    output_dir: Path
    predictions_path: Path
    metrics_path: Path
    manifest_path: Path
    metrics: Mapping[str, Any]
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class _PreparedDataset:
    dataset: RAGEvalDataset
    cases: tuple[RAGEvalCase, ...]
    provenance: Mapping[str, Any]
    pdf_artifacts: tuple[Mapping[str, Any], ...]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json(row) + b"\n" for row in rows)


def load_experiment_config(path: str | Path) -> RAGExperimentConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"RAG experiment config does not exist: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"RAG experiment config is not valid JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError("RAG experiment config must be a JSON object")
    return RAGExperimentConfig.model_validate(payload)


def _repository_file(
    repository: Path,
    relative_path: str,
    *,
    missing_message: str,
) -> Path:
    candidate = (repository / relative_path).resolve()
    try:
        candidate.relative_to(repository)
    except ValueError as exc:
        raise ValueError("experiment source path escapes the repository") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"{missing_message}: {candidate}")
    return candidate


def _normalized_dataset_hash(
    dataset: RAGEvalDataset, cases: Sequence[RAGEvalCase]
) -> str:
    payload = {
        "dataset": dataset.dataset,
        "license": dataset.license,
        "attribution_url": dataset.attribution_url,
        "documents": [
            item.model_dump(mode="json")
            for item in sorted(dataset.documents, key=lambda value: value.document_id)
        ],
        "cases": [item.model_dump(mode="json") for item in cases],
    }
    return _sha256_bytes(_canonical_json(payload))


def _prepare_synthetic_dataset(
    config: RAGExperimentConfig,
    repository: Path,
    staging: Path,
) -> _PreparedDataset:
    suite_path = _repository_file(
        repository,
        config.dataset.synthetic_suite_path,
        missing_message="synthetic PDF suite is missing",
    )
    pdf_root = staging / "source_pdfs"
    generation = generate_synthetic_pdfs(suite_path, pdf_root)
    dataset = load_generated_page_dataset(suite_path, generation)
    if not dataset.documents or not dataset.cases:
        raise ValueError("synthetic PDF adapter produced an empty evaluation dataset")
    pdf_artifacts = _pdf_artifact_records(generation, pdf_root)
    aggregate = _sha256_bytes(_canonical_json(pdf_artifacts))
    provenance = {
        "name": dataset.dataset,
        "adapter": "taskforge_synthetic_pdf_real_pypdf",
        "suite_path": config.dataset.synthetic_suite_path,
        "suite_sha256": sha256_file(suite_path),
        "normalized_sha256": _normalized_dataset_hash(dataset, dataset.cases),
        "license": dataset.license,
        "attribution_url": dataset.attribution_url,
        "corpus_documents": len(dataset.documents),
        "selected_cases": len(dataset.cases),
        "pdf_sha256": aggregate,
    }
    return _PreparedDataset(
        dataset=dataset,
        cases=tuple(dataset.cases),
        provenance=provenance,
        pdf_artifacts=tuple(pdf_artifacts),
    )


def _pdf_artifact_records(
    generation: SyntheticGenerationManifest, pdf_root: Path
) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for generated in sorted(generation.documents, key=lambda item: item.document_id):
        path = Path(generated.path).resolve(strict=True)
        try:
            relative = path.relative_to(pdf_root.resolve())
        except ValueError as exc:
            raise ValueError("generated PDF escaped the staging PDF directory") from exc
        actual_hash = sha256_file(path)
        if actual_hash != generated.sha256:
            raise ValueError("generated PDF checksum changed after parsing")
        records.append(
            {
                "document_id": generated.document_id,
                "path": (Path("source_pdfs") / relative).as_posix(),
                "sha256": actual_hash,
                "size_bytes": path.stat().st_size,
                "pages": generated.pages,
            }
        )
    return records


def _prepare_tatqa_dataset(
    config: RAGExperimentConfig,
    repository: Path,
) -> _PreparedDataset:
    input_path = _repository_file(
        repository,
        config.dataset.tatqa_input_path,
        missing_message=(
            "TAT-QA external cache is missing; fetch the pinned dataset before "
            "running the locked experiment"
        ),
    )
    split_path = _repository_file(
        repository,
        config.dataset.tatqa_locked_split_path,
        missing_message="TAT-QA locked split manifest is missing",
    )
    source_hash = sha256_file(input_path)
    dataset = load_tatqa_dataset(
        input_path,
        table_cleaning=config.dataset.tatqa_table_cleaning,
    )
    if not dataset.documents:
        raise ValueError("TAT-QA adapter produced an empty corpus")
    split = load_locked_split(split_path)
    if split.dataset != dataset.dataset:
        raise ValueError("TAT-QA locked split belongs to another dataset")
    selected = select_locked_cases(
        dataset.cases,
        split,
        dataset_sha256=source_hash,
    )
    if not selected:
        raise ValueError("TAT-QA locked split selected no cases")
    provenance = {
        "name": dataset.dataset,
        "adapter": "tatqa_locked",
        "input_path": config.dataset.tatqa_input_path,
        "input_sha256": source_hash,
        "input_size_bytes": input_path.stat().st_size,
        "locked_split_path": config.dataset.tatqa_locked_split_path,
        "locked_split_id": split.split_id,
        "locked_split_sha256": sha256_file(split_path),
        "normalized_sha256": _normalized_dataset_hash(dataset, selected),
        "license": dataset.license,
        "attribution_url": dataset.attribution_url,
        "corpus_documents": len(dataset.documents),
        "available_cases": len(dataset.cases),
        "selected_cases": len(selected),
        "table_cleaning": {
            "enabled": config.dataset.tatqa_table_cleaning,
            "contract": "coordinate_preserving_search_representation_v1",
        },
        "pdf_sha256": None,
    }
    return _PreparedDataset(
        dataset=dataset,
        cases=tuple(selected),
        provenance=provenance,
        pdf_artifacts=(),
    )


def _prepare_multihop_rag_dataset(
    config: RAGExperimentConfig,
    repository: Path,
) -> _PreparedDataset:
    queries_path = _repository_file(
        repository,
        config.dataset.multihop_rag_queries_path,
        missing_message=(
            "MultiHop-RAG query cache is missing; fetch the pinned dataset before "
            "running the locked experiment"
        ),
    )
    corpus_path = _repository_file(
        repository,
        config.dataset.multihop_rag_corpus_path,
        missing_message="MultiHop-RAG corpus cache is missing",
    )
    split_path = _repository_file(
        repository,
        config.dataset.multihop_rag_locked_split_path,
        missing_message="MultiHop-RAG locked split manifest is missing",
    )
    source_hash = sha256_file(queries_path)
    dataset = load_multihop_rag_dataset(queries_path, corpus_path)
    if not dataset.documents:
        raise ValueError("MultiHop-RAG adapter produced an empty corpus")
    split = load_locked_split(split_path)
    if split.dataset != dataset.dataset:
        raise ValueError("MultiHop-RAG locked split belongs to another dataset")
    selected = select_locked_cases(
        dataset.cases,
        split,
        dataset_sha256=source_hash,
    )
    if not selected:
        raise ValueError("MultiHop-RAG locked split selected no cases")
    provenance = {
        "name": dataset.dataset,
        "adapter": "multihop_rag_locked",
        "queries_path": config.dataset.multihop_rag_queries_path,
        "queries_sha256": source_hash,
        "queries_size_bytes": queries_path.stat().st_size,
        "corpus_path": config.dataset.multihop_rag_corpus_path,
        "corpus_sha256": sha256_file(corpus_path),
        "corpus_size_bytes": corpus_path.stat().st_size,
        "locked_split_path": config.dataset.multihop_rag_locked_split_path,
        "locked_split_id": split.split_id,
        "locked_split_sha256": sha256_file(split_path),
        "normalized_sha256": _normalized_dataset_hash(dataset, selected),
        "license": dataset.license,
        "attribution_url": dataset.attribution_url,
        "corpus_documents": len(dataset.documents),
        "available_cases": len(dataset.cases),
        "selected_cases": len(selected),
        "pdf_sha256": None,
    }
    return _PreparedDataset(
        dataset=dataset,
        cases=tuple(selected),
        provenance=provenance,
        pdf_artifacts=(),
    )


def _prepare_qasper_dataset(
    config: RAGExperimentConfig,
    repository: Path,
) -> _PreparedDataset:
    input_path = _repository_file(
        repository,
        config.dataset.qasper_input_path,
        missing_message=(
            "QASPER external cache is missing; fetch the pinned dataset before "
            "running the locked experiment"
        ),
    )
    split_path = _repository_file(
        repository,
        config.dataset.qasper_locked_split_path,
        missing_message="QASPER locked split manifest is missing",
    )
    source_hash = sha256_file(input_path)
    dataset = load_qasper_dataset(input_path)
    if not dataset.documents:
        raise ValueError("QASPER adapter produced an empty corpus")
    split = load_locked_split(split_path)
    if split.dataset != dataset.dataset:
        raise ValueError("QASPER locked split belongs to another dataset")
    selected = select_locked_cases(
        dataset.cases,
        split,
        dataset_sha256=source_hash,
    )
    if not selected:
        raise ValueError("QASPER locked split selected no cases")
    provenance = {
        "name": dataset.dataset,
        "adapter": "qasper_locked",
        "input_path": config.dataset.qasper_input_path,
        "input_sha256": source_hash,
        "input_size_bytes": input_path.stat().st_size,
        "locked_split_path": config.dataset.qasper_locked_split_path,
        "locked_split_id": split.split_id,
        "locked_split_sha256": sha256_file(split_path),
        "normalized_sha256": _normalized_dataset_hash(dataset, selected),
        "license": dataset.license,
        "attribution_url": dataset.attribution_url,
        "corpus_documents": len(dataset.documents),
        "available_cases": len(dataset.cases),
        "selected_cases": len(selected),
        "pdf_sha256": None,
    }
    return _PreparedDataset(
        dataset=dataset,
        cases=tuple(selected),
        provenance=provenance,
        pdf_artifacts=(),
    )


def _prepare_dataset(
    config: RAGExperimentConfig,
    repository: Path,
    staging: Path,
) -> _PreparedDataset:
    if config.dataset.kind == "synthetic_pdf":
        return _prepare_synthetic_dataset(config, repository, staging)
    if config.dataset.kind == "tatqa_locked":
        return _prepare_tatqa_dataset(config, repository)
    if config.dataset.kind == "multihop_rag_locked":
        return _prepare_multihop_rag_dataset(config, repository)
    if config.dataset.kind == "qasper_locked":
        return _prepare_qasper_dataset(config, repository)
    raise ValueError(f"unsupported dataset kind: {config.dataset.kind}")


def _validate_evidence(dataset: RAGEvalDataset, cases: Sequence[RAGEvalCase]) -> None:
    document_ids = {document.document_id for document in dataset.documents}
    missing = sorted(
        {
            evidence_id
            for case in cases
            for evidence_id in case.relevant_ids
            if evidence_id not in document_ids
        }
    )
    if missing:
        raise ValueError(f"evaluation case references missing evidence: {missing[0]}")


def _hybrid_chunks(
    dataset: RAGEvalDataset,
    cases: Sequence[RAGEvalCase],
    config: RAGExperimentConfig,
) -> list[HybridChunk]:
    filters = config.filters
    # QASPER supplies the paper for each question.  In that mode, indexing
    # unrelated papers is both wasteful and a source of accidental leakage;
    # retain the full normalized dataset for provenance/evidence validation,
    # but build the retrieval index only from papers represented in the locked
    # cases.  Global discovery remains available through the explicit mode.
    qasper_parent_scope: set[str] | None = None
    if (
        dataset.dataset == "QASPER"
        and config.dataset.qasper_context_mode == "provided_document_context"
    ):
        qasper_parent_scope = {
            str(case.metadata.get("parent_document_id"))
            for case in cases
            if isinstance(case.metadata.get("parent_document_id"), str)
        }
    chunks: list[HybridChunk] = []
    for document in sorted(dataset.documents, key=lambda value: value.document_id):
        if (
            qasper_parent_scope is not None
            and str(document.metadata.get("parent_document_id"))
            not in qasper_parent_scope
        ):
            continue
        texts = _chunk_document_text(
            document.text,
            document.metadata,
            config.retrieval,
        )
        for index, text in enumerate(texts):
            chunk_id = (
                f"{document.document_id}{_CHUNK_SEP}{index}"
                if len(texts) > 1
                else document.document_id
            )
            chunks.append(
                HybridChunk(
                    chunk_id=chunk_id,
                    tenant_id=filters.tenant_id,
                    text=text,
                    source_uri=document.source_uri,
                    document_id=document.document_id,
                    knowledge_base_id=filters.knowledge_base_id,
                    version=filters.version,
                    version_order=filters.version_order,
                    acl_principals=frozenset(filters.indexed_acl_principals),
                    metadata={
                        "evaluation": True,
                        "chunk_index": index,
                        "chunk_count": len(texts),
                        "parent_document_id": document.document_id,
                        "table_complete": (
                            document.metadata.get("kind") == "table"
                            and len(texts) == 1
                            and not config.retrieval.table_aware_chunking
                        ),
                        **document.metadata,
                    },
                )
            )
    # These high-overlap records are intentionally unreadable.  Their absence
    # in every ranking is an executable assertion that scope is applied before
    # candidate selection, rather than merely documented in configuration.
    probe_text = " ".join(case.query for case in cases)
    if len(probe_text) > 500_000:
        probe_text = probe_text[:500_000]
    chunks.extend(
        [
            HybridChunk(
                chunk_id=f"{_PROBE_PREFIX}:acl",
                tenant_id=filters.tenant_id,
                text=probe_text,
                source_uri="taskforge://filter-probe/acl",
                document_id=f"{_PROBE_PREFIX}:acl",
                knowledge_base_id=filters.knowledge_base_id,
                version=filters.version,
                version_order=filters.version_order,
                acl_principals=frozenset({_DENIED_PRINCIPAL}),
                metadata={"filter_probe": "acl"},
            ),
            HybridChunk(
                chunk_id=f"{_PROBE_PREFIX}:tenant",
                tenant_id=f"{filters.tenant_id}::denied",
                text=probe_text,
                source_uri="taskforge://filter-probe/tenant",
                document_id=f"{_PROBE_PREFIX}:tenant",
                knowledge_base_id=filters.knowledge_base_id,
                version=filters.version,
                version_order=filters.version_order,
                acl_principals=frozenset(filters.request_principals),
                metadata={"filter_probe": "tenant"},
            ),
        ]
    )
    return chunks


def _table_search_rows(
    document: EvalCorpusDocument,
) -> tuple[list[list[object]], list[tuple[int, ...]]]:
    """Return searchable rows and their original zero-based row lineage."""

    raw_rows = document.metadata.get("table_rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        return [], []
    rows_value = document.metadata.get("table_rows_cleaned")
    cleaning = document.metadata.get("table_cleaning")
    if not isinstance(rows_value, list) or not isinstance(cleaning, Mapping):
        rows = [
            list(row) if isinstance(row, (list, tuple)) else [row]
            for row in raw_rows
        ]
        return rows, [(index,) for index in range(len(rows))]
    raw_lineage = cleaning.get("row_source_indices")
    if not isinstance(raw_lineage, list) or len(raw_lineage) != len(rows_value):
        raise ValueError("cleaned TAT-QA rows require one lineage entry per row")
    rows = [
        list(row) if isinstance(row, (list, tuple)) else [row]
        for row in rows_value
    ]
    lineage: list[tuple[int, ...]] = []
    for indices in raw_lineage:
        if (
            not isinstance(indices, list)
            or not indices
            or any(
                not isinstance(index, int) or index < 0 or index >= len(raw_rows)
                for index in indices
            )
        ):
            raise ValueError("cleaned TAT-QA row lineage is outside the raw table")
        lineage.append(tuple(indices))
    return rows, lineage


def _table_representation_chunks(
    base_chunks: Sequence[HybridChunk],
    dataset: RAGEvalDataset,
    config: RAGExperimentConfig,
    *,
    representation: Literal["schema", "row", "cell", "section"],
) -> list[HybridChunk]:
    """Build an isolated schema/row/cell branch while retaining paragraphs."""

    filters = config.filters
    # TAT-QA frequently puts the query's semantic anchor in the prose that
    # introduces a table (for example, the Black-Scholes method or the names
    # of revenue segments), while the table itself contains only row labels
    # and numbers.  Keep a bounded, table_uid-linked context synopsis on the
    # table representation so candidate generation can route to that table.
    # This is corpus metadata, not evaluation evidence: it is built before a
    # query is seen and is never selected from relevant_ids or answers.
    context_by_table_uid: dict[str, list[tuple[int, str]]] = {}
    for document in dataset.documents:
        if document.metadata.get("kind") != "paragraph":
            continue
        table_uid = document.metadata.get("table_uid")
        if not isinstance(table_uid, str) or not table_uid.strip():
            continue
        text = " ".join(str(document.text).split())
        if not text:
            continue
        raw_order = document.metadata.get("order")
        try:
            order = int(raw_order)
        except (TypeError, ValueError):
            order = 1_000_000
        context_by_table_uid.setdefault(table_uid, []).append((order, text))
    table_context: dict[str, str] = {}
    for table_uid, paragraphs in context_by_table_uid.items():
        pieces: list[str] = []
        total = 0
        for _, paragraph in sorted(paragraphs, key=lambda item: (item[0], item[1])):
            # The first 280 characters retain headings and the usual
            # sentence-level table description while bounding index growth.
            piece = paragraph[:280]
            if not piece:
                continue
            remaining = 1_200 - total
            if remaining <= 0:
                break
            piece = piece[:remaining]
            pieces.append(piece)
            total += len(piece)
        if pieces:
            table_context[table_uid] = "Table context: " + " ".join(pieces)

    branch = [
        chunk
        for chunk in base_chunks
        if chunk.metadata.get("kind") != "table"
        or chunk.chunk_id.startswith(_PROBE_PREFIX)
    ]
    for document in sorted(dataset.documents, key=lambda value: value.document_id):
        if document.metadata.get("kind") != "table":
            continue
        raw_rows = document.metadata.get("table_rows")
        if not isinstance(raw_rows, list) or not raw_rows:
            continue
        rows, row_lineage = _table_search_rows(document)
        width = max((len(row) for row in rows), default=0)
        if width == 0:
            continue
        header = [
            str(value)
            for value in (
                document.metadata.get("table_cleaning", {}).get("headers", [])
                if document.metadata.get("table_cleaning_enabled") is True
                else []
            )
        ]
        cleaning_audit = (
            document.metadata.get("table_cleaning", {}).get("audit", {})
            if document.metadata.get("table_cleaning_enabled") is True
            else {}
        )
        header_depth = (
            int(cleaning_audit.get("header_depth", 1))
            if isinstance(cleaning_audit, Mapping)
            else 1
        )
        if len(header) != width or header_depth < 1 or header_depth > len(rows):
            header = [
                str(rows[0][index]).strip()
                if index < len(rows[0])
                else f"column_{index + 1}"
                for index in range(width)
            ]
            header_depth = 1
        parent_id = str(
            document.metadata.get("parent_document_id", document.document_id)
        )

        entries: list[tuple[str, str, int | None, int | None]] = []
        if representation == "schema":
            row_labels = [
                str(row[0]).strip()
                for row in rows[header_depth:]
                if row and str(row[0]).strip()
            ]
            schema = "Table schema: columns=" + " | ".join(header)
            if row_labels:
                schema += " | row_labels=" + " | ".join(row_labels)
            entries.append(("schema", schema, None, None))
            header_text = " | ".join(header)
            # A long all-row schema is useful for broad matching but suffers
            # BM25 length normalization on exact section/label questions.
            # Add compact label records with the same table document id so a
            # precise table heading (for example, "Deferred tax liabilities")
            # can win its representation branch without changing evidence ids.
            for row_index, row in enumerate(
                rows[header_depth:], start=header_depth
            ):
                label = str(row[0]).strip() if row else ""
                if len(label) >= 3:
                    entries.append(
                        (
                            "schema",
                            f"Table label: {label} | columns={header_text}",
                            row_index,
                            None,
                        )
                    )
        elif representation == "row":
            for row_index, row in enumerate(
                rows[header_depth:], start=header_depth
            ):
                fields = [
                    f"{header[column_index]}={str(value).strip()}"
                    for column_index, value in enumerate(row)
                    if str(value).strip()
                ]
                if fields:
                    row_text = (
                        "Table row: "
                        + " | ".join(fields)
                        + " | columns="
                        + " | ".join(header)
                    )
                    entries.append(("row", row_text, row_index, None))
        elif representation == "cell":
            for row_index, row in enumerate(
                rows[header_depth:], start=header_depth
            ):
                row_label = str(row[0]).strip() if row else ""
                for column_index, value in enumerate(row):
                    value_text = str(value).strip()
                    if not value_text:
                        continue
                    normalised = re.sub(r"[$€£¥,%\s,]", "", value_text)
                    cell_text = (
                        f"Table cell: row={row_label} | "
                        f"column={header[column_index]} | value={value_text} | "
                        f"value_normalized={normalised}"
                    )
                    entries.append(("cell", cell_text, row_index, column_index))
        else:
            # Keep the header attached to bounded contiguous row windows.  A
            # section is intentionally a single searchable evidence unit, so
            # metric, year and value terms survive table serialization even
            # when a query refers to more than one row.
            header_text = " | ".join(header)
            window_size = 8
            body_rows = rows[header_depth:]
            for window_start in range(0, len(body_rows), window_size):
                window = body_rows[window_start : window_start + window_size]
                rendered_rows = []
                for row in window:
                    fields = [
                        str(value).strip()
                        for value in row[:width]
                        if str(value).strip()
                    ]
                    if fields:
                        rendered_rows.append(" | ".join(fields))
                if rendered_rows:
                    section_text = (
                        "Table section: columns="
                        + header_text
                        + " | rows=\n"
                        + "\n".join(rendered_rows)
                    )
                    entries.append(
                        (
                            "section",
                            section_text,
                            window_start + header_depth,
                            None,
                        )
                    )

        for entry_index, (kind, text, row_index, column_index) in enumerate(entries):
            metadata = {
                "evaluation": True,
                "kind": "table",
                "representation": kind,
                "parent_document_id": parent_id,
                "table_uid": document.metadata.get("table_uid"),
                "table_rows": raw_rows,
                "table_complete": False,
            }
            if document.metadata.get("table_cleaning_enabled") is True:
                metadata["table_cleaning_enabled"] = True
                metadata["table_rows_cleaned"] = document.metadata.get(
                    "table_rows_cleaned"
                )
                metadata["table_cleaning"] = document.metadata.get("table_cleaning")
            context = table_context.get(str(document.metadata.get("table_uid", "")))
            if context:
                metadata["table_context"] = context
            if row_index is not None:
                source_indices = row_lineage[row_index]
                metadata["table_row_index"] = source_indices[0]
                metadata["table_row_source_indices"] = list(source_indices)
            if column_index is not None:
                metadata["table_column_index"] = column_index
            branch.append(
                HybridChunk(
                    chunk_id=(
                        f"{document.document_id}::repr::{representation}::"
                        f"{entry_index}"
                    ),
                    tenant_id=filters.tenant_id,
                    text=text,
                    source_uri=document.source_uri,
                    document_id=document.document_id,
                    knowledge_base_id=filters.knowledge_base_id,
                    version=filters.version,
                    version_order=filters.version_order,
                    acl_principals=frozenset(filters.indexed_acl_principals),
                    metadata=metadata,
                )
            )
    return branch


_MISSING_TABLE_VALUES = frozenset({"", "-", "--", "—", "–", "n/a", "na"})
_TABLE_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_TABLE_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _structured_header_depth(rows: Sequence[Sequence[object]]) -> int:
    """Infer a conservative one/two-row table header from corpus structure."""

    if len(rows) < 2:
        return 1
    first = [str(value).strip() for value in rows[0]]
    second = [str(value).strip() for value in rows[1]]
    first_label = first[0] if first else ""
    second_label = second[0] if second else ""
    second_fields = [value for value in second[1:] if value]
    if first_label or second_label or not second_fields:
        return 1
    header_signal = re.compile(
        r"(?:\b(?:19|20)\d{2}\b|\byears?\b|\bmonths?\b|\bquarters?\b|"
        r"\bup to\b|\bmore than\b|\btotal\b|%|january|february|march|april|"
        r"may|june|july|august|september|october|november|december)",
        re.IGNORECASE,
    )
    first_super_header = any(
        re.search(
            r"\b(years? ended|as of|payments? due|december 31|months? ended)\b",
            value,
            re.IGNORECASE,
        )
        for value in first[1:]
        if value
    )
    signalled = sum(bool(header_signal.search(value)) for value in second_fields)
    return 2 if first_super_header or signalled >= max(1, len(second_fields) // 2) else 1


def _structured_headers(
    rows: Sequence[Sequence[object]], header_depth: int, width: int
) -> list[str]:
    header_rows = [
        [str(row[index]).strip() if index < len(row) else "" for index in range(width)]
        for row in rows[:header_depth]
    ]
    super_headers = [
        value
        for value in header_rows[0][1:]
        if value and not _TABLE_YEAR.fullmatch(value)
    ]
    super_header = " | ".join(dict.fromkeys(super_headers))
    output = ["row_label"]
    for column_index in range(1, width):
        pieces: list[str] = []
        if header_depth > 1 and super_header:
            pieces.append(super_header)
        for header_row in header_rows:
            value = header_row[column_index]
            if value and value not in pieces:
                pieces.append(value)
        output.append(" | ".join(pieces) or f"column_{column_index + 1}")
    return output


def _structured_numeric_value(value: str) -> tuple[str | None, float | None, str]:
    cleaned = value.strip()
    if cleaned.casefold() in _MISSING_TABLE_VALUES:
        return None, None, "missing"
    match = _TABLE_NUMBER.search(cleaned)
    if match is None:
        return None, None, "non_numeric"
    normalised = match.group(0).replace(",", "")
    try:
        numeric = float(normalised)
    except ValueError:
        return None, None, "non_numeric"
    if cleaned.startswith("(") and cleaned.endswith(")"):
        numeric = -abs(numeric)
    if numeric < 0:
        sign = "negative"
    elif numeric > 0:
        sign = "positive"
    else:
        sign = "zero"
    return format(numeric, ".15g"), numeric, sign


def _structured_scale(value: str) -> str | None:
    match = re.search(r"\b(thousands?|millions?|billions?)\b", value, re.I)
    if match is None:
        return None
    return {
        "thousands": "thousand",
        "millions": "million",
        "billions": "billion",
    }.get(match.group(1).casefold(), match.group(1).casefold())


def _structured_unit(value: str, header: str) -> str:
    combined = f"{header} {value}".casefold()
    if "%" in value or "percent" in combined or "% change" in combined:
        return "percent"
    if any(symbol in value for symbol in "$€£¥"):
        return "currency"
    if "share" in combined:
        return "shares"
    return "number"


def _structured_table_fact_chunks(
    base_chunks: Sequence[HybridChunk],
    dataset: RAGEvalDataset,
) -> list[HybridChunk]:
    """Materialize query-independent table facts with explicit lineage.

    Every searchable row retains its merged header, original cell coordinates,
    table/document identity, parent context, and linked paragraph IDs.  The
    representation is candidate-generation-only; evaluation IDs remain the
    original table document IDs.
    """

    canonical_tables = {
        chunk.document_id: chunk
        for chunk in base_chunks
        if chunk.metadata.get("kind") == "table"
        and not chunk.chunk_id.startswith(_PROBE_PREFIX)
    }
    paragraphs_by_table: dict[str, list[tuple[int, str, str]]] = {}
    for document in dataset.documents:
        if document.metadata.get("kind") != "paragraph":
            continue
        table_uid = document.metadata.get("table_uid")
        if not isinstance(table_uid, str) or not table_uid.strip():
            continue
        try:
            order = int(document.metadata.get("order", 1_000_000))
        except (TypeError, ValueError):
            order = 1_000_000
        paragraphs_by_table.setdefault(table_uid, []).append(
            (order, document.document_id, " ".join(document.text.split()))
        )

    output: list[HybridChunk] = []
    for document in sorted(dataset.documents, key=lambda value: value.document_id):
        if document.metadata.get("kind") != "table":
            continue
        canonical = canonical_tables.get(document.document_id)
        raw_rows = document.metadata.get("table_rows")
        if canonical is None or not isinstance(raw_rows, list) or not raw_rows:
            continue
        rows, row_lineage = _table_search_rows(document)
        width = max((len(row) for row in rows), default=0)
        if width < 2:
            continue
        cleaning = document.metadata.get("table_cleaning")
        cleaned_headers = (
            cleaning.get("headers") if isinstance(cleaning, Mapping) else None
        )
        if (
            document.metadata.get("table_cleaning_enabled") is True
            and isinstance(cleaned_headers, list)
            and len(cleaned_headers) == width
        ):
            raw_audit = cleaning.get("audit")
            header_depth = (
                int(raw_audit.get("header_depth", 1))
                if isinstance(raw_audit, Mapping)
                else 1
            )
            if header_depth < 1 or header_depth > len(rows):
                raise ValueError("cleaned TAT-QA header depth is outside the table")
            headers = [str(value) for value in cleaned_headers]
        else:
            header_depth = _structured_header_depth(rows)
            headers = _structured_headers(rows, header_depth, width)
        header_row_indices = sorted(
            {
                source_index
                for clean_index in range(header_depth)
                for source_index in row_lineage[clean_index]
            }
        )
        table_uid = str(document.metadata.get("table_uid", ""))
        linked_paragraphs = sorted(
            paragraphs_by_table.get(table_uid, []),
            key=lambda item: (item[0], item[1]),
        )
        paragraph_ids = [item[1] for item in linked_paragraphs]
        context = " ".join(item[2] for item in linked_paragraphs)[:1_600]
        scale = _structured_scale(
            " ".join(
                [
                    context,
                    *(
                        str(value)
                        for row in rows[:header_depth]
                        for value in row
                    ),
                ]
            )
        )
        parent_id = str(
            document.metadata.get("parent_document_id", document.document_id)
        )
        row_labels: list[str] = []
        current_group = ""
        row_records: list[
            tuple[int, str, str, list[dict[str, Any]], tuple[int, ...]]
        ] = []
        for clean_row_index, row in enumerate(
            rows[header_depth:], start=header_depth
        ):
            source_row_indices = row_lineage[clean_row_index]
            row_index = source_row_indices[0]
            values = [
                str(row[index]).strip() if index < len(row) else ""
                for index in range(width)
            ]
            label = values[0]
            populated_values = [
                value
                for value in values[1:]
                if value.casefold() not in _MISSING_TABLE_VALUES
            ]
            if label and not populated_values:
                current_group = label.rstrip(":").strip()
                continue
            if not label and populated_values:
                label = "total"
            if not label and not populated_values:
                continue
            row_labels.append(label)
            cells: list[dict[str, Any]] = []
            rendered: list[str] = []
            for column_index in range(1, width):
                raw_value = values[column_index]
                if raw_value.casefold() in _MISSING_TABLE_VALUES:
                    normalised, numeric, sign = None, None, "missing"
                else:
                    normalised, numeric, sign = _structured_numeric_value(raw_value)
                header = headers[column_index]
                years = _TABLE_YEAR.findall(header)
                unit = _structured_unit(raw_value, header)
                cell = {
                    "column_index": column_index,
                    "source_row_indices": list(source_row_indices),
                    "header": header,
                    "raw": raw_value,
                    "normalized": normalised,
                    "numeric": numeric,
                    "years": years,
                    "unit": unit,
                    "scale": scale,
                    "sign": sign,
                }
                cells.append(cell)
                rendered.append(
                    f"column={header} value={raw_value or 'missing'} "
                    f"normalized={normalised or 'missing'} years={','.join(years) or 'none'} "
                    f"unit={unit} scale={scale or 'none'} sign={sign}"
                )
            row_records.append(
                (row_index, label, current_group, cells, source_row_indices)
            )
            row_text = (
                f"Structured table row: row={label} | group={current_group or 'none'} | "
                + " | ".join(rendered)
            )
            if context:
                row_text += f" | parent_context={context}"
            metadata = {
                **canonical.metadata,
                "representation": "structured_row",
                "parent_document_id": parent_id,
                "table_uid": table_uid,
                "table_rows": raw_rows,
                "table_complete": False,
                "structured_header_depth": header_depth,
                "structured_headers": headers,
                "structured_row_index": row_index,
                "structured_row_source_indices": list(source_row_indices),
                "structured_row_label": label,
                "structured_group_label": current_group or None,
                "structured_values": cells,
                "structured_scale": scale,
                "lineage": {
                    "parent_document_id": parent_id,
                    "table_document_id": document.document_id,
                    "table_uid": table_uid,
                    "header_row_indices": header_row_indices,
                    "row_index": row_index,
                    "row_source_indices": list(source_row_indices),
                    "parent_paragraph_ids": paragraph_ids,
                },
            }
            output.append(
                canonical.model_copy(
                    update={
                        "chunk_id": (
                            f"{document.document_id}::repr::structured::row::{row_index}"
                        ),
                        "text": row_text,
                        "metadata": metadata,
                        "previous_chunk_id": None,
                        "next_chunk_id": None,
                    }
                )
            )

        schema_aliases: list[str] = []
        meaningful_labels = [
            value
            for value in row_labels
            if value.casefold() not in {"total", "subtotal"}
        ]
        if len(meaningful_labels) >= 2:
            schema_aliases.extend(["components", "categories", "breakdown"])
            if "revenue" in tokenise(" ".join([context, *row_labels])):
                schema_aliases.append("segments")
        schema_text = (
            "Structured table schema: headers="
            + " | ".join(headers)
            + " | row_labels="
            + " | ".join(row_labels)
            + " | structure_aliases="
            + " ".join(schema_aliases)
        )
        if context:
            schema_text += f" | parent_context={context}"
        schema_metadata = {
            **canonical.metadata,
            "representation": "structured_schema",
            "parent_document_id": parent_id,
            "table_uid": table_uid,
            "table_rows": raw_rows,
            "table_complete": False,
            "structured_header_depth": header_depth,
            "structured_headers": headers,
            "structured_row_labels": row_labels,
            "structured_scale": scale,
            "structured_aliases": schema_aliases,
            "lineage": {
                "parent_document_id": parent_id,
                "table_document_id": document.document_id,
                "table_uid": table_uid,
                "header_row_indices": header_row_indices,
                "row_indices": sorted(
                    {
                        source_index
                        for record in row_records
                        for source_index in record[4]
                    }
                ),
                "parent_paragraph_ids": paragraph_ids,
            },
        }
        output.append(
            canonical.model_copy(
                update={
                    "chunk_id": f"{document.document_id}::repr::structured::schema",
                    "text": schema_text,
                    "metadata": schema_metadata,
                    "previous_chunk_id": None,
                    "next_chunk_id": None,
                }
            )
        )
    return output


def _parent_hybrid_chunks(children: Sequence[HybridChunk]) -> list[HybridChunk]:
    """Build routing-only parent records from already-filterable children."""

    grouped: dict[str, list[HybridChunk]] = {}
    for child in children:
        if child.chunk_id.startswith(_PROBE_PREFIX):
            continue
        raw_parent = child.metadata.get("parent_document_id")
        parent_id = (
            raw_parent.strip()
            if isinstance(raw_parent, str) and raw_parent.strip()
            else child.document_id
        )
        grouped.setdefault(parent_id, []).append(child)

    parents: list[HybridChunk] = []
    for parent_id, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda value: value.chunk_id)
        common_acl = set(ordered[0].acl_principals)
        for child in ordered[1:]:
            common_acl.intersection_update(child.acl_principals)
        if not common_acl:
            # A parent containing differently protected children is not safe to
            # aggregate into one routing record. Its children remain searchable
            # through the flat stages.
            continue
        first = ordered[0]
        parents.append(
            HybridChunk(
                chunk_id=f"parent::{parent_id}",
                tenant_id=first.tenant_id,
                text="\n\n".join(child.text for child in ordered),
                source_uri=f"taskforge://parent/{parent_id}",
                document_id=parent_id,
                knowledge_base_id=first.knowledge_base_id,
                version=first.version,
                version_order=first.version_order,
                acl_principals=frozenset(common_acl),
                metadata={
                    "evaluation": True,
                    "kind": "parent",
                    "parent_document_id": parent_id,
                    "child_count": len(ordered),
                },
            )
        )
    return parents


_QASPER_HIERARCHICAL_STAGES = frozenset(
    {
        "bm25_qasper_hierarchical",
        "qdrant_qasper_dense",
        "qdrant_qasper_dense_rerank",
        "bm25_dense_qasper_candidate_union",
        "bm25_dense_qasper_section_parent",
        "bm25_dense_qasper_section_parent_rrf",
    }
)


def _qasper_contextual_chunks(
    chunks: Sequence[HybridChunk],
) -> list[HybridChunk]:
    """Build a search-only title/section representation over raw evidence."""

    output: list[HybridChunk] = []
    for chunk in chunks:
        metadata = dict(chunk.metadata)
        if chunk.chunk_id.startswith(_PROBE_PREFIX):
            output.append(chunk)
            continue
        if metadata.get("source") != "qasper":
            output.append(chunk)
            continue
        title = str(metadata.get("paper_title", "")).strip()
        section = str(metadata.get("section_title", "")).strip()
        subsection = str(metadata.get("subsection_title", "")).strip()
        prefix = [part for part in (title, section, subsection) if part]
        contextual_text = "\n".join(prefix + [chunk.text])
        metadata["search_representation"] = "paper_title_section_body"
        metadata["display_text"] = chunk.text
        output.append(chunk.model_copy(update={"text": contextual_text, "metadata": metadata}))
    return output


def _qasper_section_parent_chunks(
    raw_chunks: Sequence[HybridChunk],
) -> list[HybridChunk]:
    """Build section-level routing parents while retaining paper ACL scope."""

    grouped: dict[str, list[HybridChunk]] = {}
    for chunk in raw_chunks:
        if chunk.chunk_id.startswith(_PROBE_PREFIX):
            continue
        section_id = chunk.metadata.get("section_id")
        if isinstance(section_id, str) and section_id.strip():
            grouped.setdefault(section_id.strip(), []).append(chunk)
    parents: list[HybridChunk] = []
    for section_id, children in sorted(grouped.items()):
        ordered = sorted(children, key=lambda item: item.chunk_id)
        first = ordered[0]
        title = str(first.metadata.get("paper_title", "")).strip()
        section = str(first.metadata.get("section_title", "")).strip()
        header = "\n".join(part for part in (title, section) if part)
        common_acl = set(first.acl_principals)
        for child in ordered[1:]:
            common_acl.intersection_update(child.acl_principals)
        if not common_acl:
            continue
        parents.append(
            HybridChunk(
                chunk_id=f"qasper-section-parent::{section_id}",
                tenant_id=first.tenant_id,
                text=(header + "\n" if header else "")
                + "\n\n".join(child.text for child in ordered),
                source_uri=f"taskforge://qasper-section/{section_id}",
                document_id=section_id,
                knowledge_base_id=first.knowledge_base_id,
                version=first.version,
                version_order=first.version_order,
                acl_principals=frozenset(common_acl),
                metadata={
                    "evaluation": True,
                    "kind": "section_parent",
                    "node_type": "section",
                    "section_id": section_id,
                    "parent_document_id": first.metadata.get("parent_document_id"),
                    "paper_id": first.metadata.get("paper_id"),
                    "paper_title": title,
                    "section_title": section,
                    "child_count": len(ordered),
                },
            )
        )
    return parents


def _compact_tatqa_parent_hybrid_chunks(
    children: Sequence[HybridChunk],
) -> list[HybridChunk]:
    """Build short, query-independent TAT-QA context routing records.

    Full parent concatenation dilutes row labels and years in long reports.
    This view retains the table schema, row labels, values and bounded paragraph
    introductions.  It is routing-only and carries the same ACL intersection as
    the full parent record; returned evidence still comes from child chunks.
    """

    grouped: dict[str, list[HybridChunk]] = {}
    for child in children:
        if child.chunk_id.startswith(_PROBE_PREFIX):
            continue
        raw_parent = child.metadata.get("parent_document_id")
        parent_id = (
            raw_parent.strip()
            if isinstance(raw_parent, str) and raw_parent.strip()
            else child.document_id
        )
        grouped.setdefault(parent_id, []).append(child)

    parents: list[HybridChunk] = []
    for parent_id, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda value: value.chunk_id)
        table = next(
            (
                item
                for item in ordered
                if item.metadata.get("kind") == "table"
                and isinstance(item.metadata.get("table_rows"), list)
            ),
            None,
        )
        if table is None:
            continue
        common_acl = set(ordered[0].acl_principals)
        for child in ordered[1:]:
            common_acl.intersection_update(child.acl_principals)
        if not common_acl:
            continue
        raw_rows = table.metadata["table_rows"]
        rows = [
            list(row) if isinstance(row, (list, tuple)) else [row]
            for row in raw_rows
        ]
        width = max((len(row) for row in rows), default=0)
        if width == 0:
            continue
        header_depth = _structured_header_depth(rows)
        headers = _structured_headers(rows, header_depth, width)
        row_labels = [
            " ".join(str(row[0]).split())
            for row in rows[header_depth:]
            if row and str(row[0]).strip()
        ]
        table_values = [
            " ".join(str(value).split())
            for row in rows[header_depth:]
            for value in row[1:]
            if str(value).strip()
        ]
        introductions: list[str] = []
        def paragraph_order(item: HybridChunk) -> tuple[int, str]:
            try:
                order = int(item.metadata.get("order", 1_000_000))
            except (TypeError, ValueError):
                order = 1_000_000
            return order, item.chunk_id

        paragraphs = sorted(
            (
                item
                for item in ordered
                if item.metadata.get("kind") == "paragraph"
            ),
            key=paragraph_order,
        )
        for paragraph in paragraphs:
            text = " ".join(paragraph.text.split())
            if not text:
                continue
            punctuation = [
                position + 1
                for marker in (". ", "; ")
                if 80 <= (position := text.find(marker)) <= 319
            ]
            limit = min(punctuation) if punctuation else min(len(text), 320)
            introductions.append(text[:limit])
        compact_text = "\n".join(
            (
                "Table headers: " + " | ".join(headers),
                "Table row labels: " + " | ".join(row_labels),
                "Context: " + " ".join(introductions),
                "Table values: " + " ".join(table_values),
            )
        )
        parents.append(
            HybridChunk(
                chunk_id=f"compact-parent::{parent_id}",
                tenant_id=table.tenant_id,
                text=compact_text,
                source_uri=f"taskforge://compact-parent/{parent_id}",
                document_id=parent_id,
                knowledge_base_id=table.knowledge_base_id,
                version=table.version,
                version_order=table.version_order,
                acl_principals=frozenset(common_acl),
                metadata={
                    "evaluation": True,
                    "kind": "parent",
                    "representation": "tatqa_compact_parent",
                    "parent_document_id": parent_id,
                    "child_count": len(ordered),
                    "table_uid": table.metadata.get("table_uid"),
                    "header_depth": header_depth,
                },
            )
        )
    return parents


def _search_request(
    query: str,
    config: RAGExperimentConfig,
    *,
    rerank: bool,
    parent_document_ids: frozenset[str] | None = None,
) -> HybridSearchRequest:
    filters = config.filters
    # Always retain the full candidate pool. Final Recall@n is computed by
    # slicing this stable ranking, while Recall@candidate_k exposes rerank headroom.
    retrieval_k = config.retrieval.candidate_k
    return HybridSearchRequest(
        query=query,
        tenant_id=filters.tenant_id,
        acl_principals=frozenset(filters.request_principals),
        versions=frozenset({filters.version}),
        version_orders=frozenset({filters.version_order}),
        knowledge_base_ids=frozenset({filters.knowledge_base_id}),
        parent_document_ids=parent_document_ids,
        top_k=retrieval_k,
        candidate_k=config.retrieval.candidate_k,
        max_chunks_per_document=config.retrieval.max_chunks_per_document,
        rerank=rerank,
        neighbor_window=0,
        max_expanded_hits=retrieval_k,
    )


def _percentile_nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("latency sample must not be empty")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _latency_summary(values: Sequence[float]) -> Mapping[str, Any]:
    if not values:
        raise ValueError("latency sample must not be empty")
    return {
        "unit": "milliseconds",
        "method": "nearest_rank",
        "count": len(values),
        "p50": round(_percentile_nearest_rank(values, 0.50), 6),
        "p95": round(_percentile_nearest_rank(values, 0.95), 6),
        "mean": round(sum(values) / len(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _expected_filter(
    config: RAGExperimentConfig,
    *,
    parent_document_ids: frozenset[str] | None = None,
) -> Mapping[str, Any]:
    request = _search_request(
        "filter contract",
        config,
        rerank=False,
        parent_document_ids=parent_document_ids,
    )
    return AppliedRetrievalFilters.from_request(request).model_dump(mode="json")


def _case_parent_scope(
    case: RAGEvalCase,
    config: RAGExperimentConfig,
) -> frozenset[str] | None:
    provided_scope = (
        config.dataset.kind == "tatqa_locked"
        and config.dataset.tatqa_context_mode == "provided_hybrid_context"
    ) or (
        config.dataset.kind == "qasper_locked"
        and config.dataset.qasper_context_mode == "provided_document_context"
    )
    if not provided_scope:
        return None
    raw_parent = case.metadata.get("parent_document_id")
    if not isinstance(raw_parent, str) or not raw_parent.strip():
        raise ValueError(
            f"case {case.case_id!r} lacks its provided parent document ID"
        )
    return frozenset({raw_parent.strip()})


class _TableQueryRouter:
    """Route explicit aggregate/count questions without using evaluation labels."""

    _COUNT_QUERY = re.compile(
        r"\b(how many|number of|count of|how often)\b",
        flags=re.IGNORECASE,
    )

    def __init__(self, generic: BM25Index, table_aware: BM25Index) -> None:
        self.generic = generic
        self.table_aware = table_aware

    def search(self, request: HybridSearchRequest) -> Any:
        backend = (
            self.table_aware
            if self._COUNT_QUERY.search(request.query)
            else self.generic
        )
        return backend.search(request)


class _TableMultiRepresentationRouter:
    """Use table-aware max fusion only for explicit count-style queries."""

    _COUNT_QUERY = _TableQueryRouter._COUNT_QUERY

    def __init__(self, generic: BM25Index, table_fused: Any) -> None:
        self.generic = generic
        self.table_fused = table_fused

    def search(self, request: HybridSearchRequest) -> Any:
        if self._COUNT_QUERY.search(request.query):
            return self.table_fused.search(request)
        response = self.generic.search(request)
        # Keep one backend label for the ablation matrix while preserving the
        # branch's original retrieval source in every hit.
        return response.model_copy(update={"backend": "multi_representation_rrf"})


def _cross_document_subqueries(query: str) -> list[str]:
    """Expand long multi-source questions using only visible query text."""

    output = [query]
    for phrase in re.findall(r"(?<!\w)'([^']{3,80})'(?!\w)", query):
        cleaned = " ".join(phrase.split())
        if cleaned and cleaned not in output:
            output.append(cleaned)
    clauses = re.split(
        r",\s+(?:while|and|but|whereas)\s+|\bwhile\b|\bwhereas\b",
        query,
        flags=re.IGNORECASE,
    )
    for clause in clauses:
        cleaned = " ".join(clause.strip(" ,?.").split())
        if len(cleaned) >= 20 and cleaned not in output:
            output.append(cleaned)
    return output[:8]


def _tatqa_subqueries(query: str) -> list[str]:
    """Generate deterministic finance/table-focused lexical subqueries."""

    output = [query]
    tokens = tokenise(query)
    stopwords = {
        "what",
        "which",
        "how",
        "many",
        "much",
        "was",
        "were",
        "is",
        "are",
        "the",
        "a",
        "an",
        "in",
        "on",
        "of",
        "for",
        "from",
        "to",
        "by",
        "and",
        "or",
        "did",
        "does",
        "do",
        "year",
        "years",
        "respectively",
        "recorded",
        "calculated",
        "listed",
        "show",
    }
    content = [token for token in tokens if token not in stopwords]
    if content:
        output.append(" ".join(content))
    years = re.findall(r"\b(?:19|20)\d{2}\b", query)
    if years:
        non_year = [token for token in content if token not in years]
        if non_year:
            output.append(" ".join([*non_year, *years]))
    if len(content) >= 3:
        output.append(" ".join(content[: min(5, len(content))]))
    return list(dict.fromkeys(output))[:6]


def _tatqa_query_plan_subqueries(query: str) -> list[str]:
    """Expand a TAT-QA query with deterministic operation/numeric terms.

    This is an opt-in candidate-generation stage.  It does not inspect the
    answer or gold evidence, and it leaves the default ablation unchanged.
    """
    plan = build_tatqa_query_plan_from_text(query)
    output = [query]
    metric_terms = list(plan["metric_terms"])
    years = [str(value) for value in plan["years"]]
    operation = str(plan["operation"])
    if metric_terms:
        output.append(" ".join([operation, *metric_terms]))
        if years:
            output.append(" ".join([*metric_terms, *years]))
    if plan["thresholds"]:
        output.append(" ".join([operation, *plan["thresholds"], *years]))
    if plan["scale"]:
        output.append(" ".join([operation, str(plan["scale"]), *metric_terms]))
    return list(dict.fromkeys(value.strip() for value in output if value.strip()))[:8]


def _tatqa_query_plan_query(query: str) -> str:
    """Render one compact deterministic QueryPlan query for BM25."""
    plan = build_tatqa_query_plan_from_text(query)
    parts = [query, str(plan["operation"])]
    parts.extend(str(value) for value in plan["metric_terms"][:8])
    parts.extend(str(value) for value in plan["years"])
    if plan["scale"]:
        parts.append(str(plan["scale"]))
    parts.extend(str(value) for value in plan["thresholds"][:3])
    return " ".join(dict.fromkeys(part for part in parts if part))


def _tatqa_query_plan_compact_query(query: str) -> str:
    """Render a low-noise QueryPlan query for the latency-sensitive branch."""

    plan = build_tatqa_query_plan_from_text(query)
    stopwords = _NumericTableScanIndex._STOPWORDS
    parts = [
        token
        for token in tokenise(query)
        if token not in stopwords
    ]
    parts.append(str(plan["operation"]))
    parts.extend(
        token
        for token in tokenise(" ".join(str(value) for value in plan["metric_terms"]))
        if token not in stopwords
    )
    parts.extend(str(value) for value in plan["years"])
    parts.extend(str(value) for value in plan["thresholds"][:3])
    if plan["scale"]:
        parts.append(str(plan["scale"]))
    return " ".join(dict.fromkeys(part for part in parts if part))


def _tatqa_requires_numeric_branch(query: str) -> bool:
    plan = build_tatqa_query_plan_from_text(query)
    return bool(
        plan["years"]
        or plan["thresholds"]
        or plan["operation"] in {"count", "arithmetic", "comparison"}
    )


def _tatqa_should_route_structured(query: str) -> bool:
    """Route only queries with an observable table/structured signal.

    A year or a word such as ``increase`` is not enough to replace a
    paragraph retriever: TAT-QA text questions often mention dates or numeric
    concepts while their gold evidence is prose.  Count/comparison queries
    with explicit numeric facts and row/segment/period language are safe
    structured candidates; generic lookup questions remain on the default
    profile.
    """

    plan = build_tatqa_query_plan_from_text(query)
    if plan["operation"] == "count":
        return True
    explicit_structure = bool(
        re.search(
            r"\b(table|row|rows|column|columns|cell|cells|segment|segments|"
            r"period|periods|under|provided|highlighted)\b",
            query,
            re.IGNORECASE,
        )
    )
    if explicit_structure:
        return not bool(
            re.search(r"\bwhat does the table show(?: us)?\b", query, re.I)
        )
    return plan["operation"] in {"arithmetic", "comparison"} and bool(
        plan["years"] or plan["thresholds"]
    )


def _tatqa_should_route_table_profile_lookup(query: str) -> bool:
    """Add generic table-lookup signals without routing narrative questions.

    Some table questions ask for a set of components, values, amounts, or
    periods without saying ``table``/``row`` and without triggering the
    numeric QueryPlan.  These terms are corpus/query signals, not dataset
    labels.  Narrative prompts such as ``what caused`` remain on the normal
    branch.
    """

    if _tatqa_should_route_structured(query):
        return True
    return bool(
        re.search(
            r"\b(component|components|respective|value|values|amount|amounts|"
            r"item|items|making up|which years?|in which years?|for the years?|"
            r"year ended|periods?)\b",
            query,
            re.IGNORECASE,
        )
    ) and not bool(
        re.search(
            r"\b(why|caused|recognised|recognized|impairment|loss|comprise|"
            r"included|disclosed|program|guidance|determined|what does|where|"
            r"when)\b",
            query,
            re.IGNORECASE,
        )
    )


def _tatqa_should_route_structured_candidate(query: str) -> bool:
    """Enable structured tail candidates without broadening the ranked head."""

    if re.search(r"\bwhat does the table show(?: us)?\b", query, re.I):
        return False
    if _tatqa_should_route_table_profile_lookup(query):
        return True
    operation = str(build_tatqa_query_plan_from_text(query)["operation"])
    if operation in {"count", "arithmetic", "comparison"}:
        return True
    return bool(
        re.search(
            r"\b(total|within|due|segments?|components?|respective|respectively|"
            r"weighted-average|highest|lowest)\b",
            query,
            re.IGNORECASE,
        )
    )


class _QueryRewriteIndex:
    def __init__(
        self,
        backend: Any,
        rewriter: Callable[[str], str],
        *,
        max_chunks_per_document: int | None = None,
    ) -> None:
        self.backend = backend
        self.rewriter = rewriter
        self.max_chunks_per_document = max_chunks_per_document

    def search(self, request: HybridSearchRequest) -> Any:
        updates: dict[str, Any] = {"query": self.rewriter(request.query)}
        if self.max_chunks_per_document is not None:
            updates["max_chunks_per_document"] = self.max_chunks_per_document
        return self.backend.search(
            request.model_copy(update=updates)
        )


class _ParentDiversePassageIndex:
    """Keep the best authorized passage from each distinct parent context.

    Concatenating a whole financial context into one BM25 record dilutes the
    discriminative row or paragraph terms, while expanding every routed parent
    consumes the fixed candidate budget with siblings.  This adapter searches
    original evidence units with a low-noise query and retains at most one hit
    per parent.  Gold labels are never consulted; lineage closure can add the
    few complementary siblings that are actually needed later in the pipeline.
    """

    def __init__(
        self,
        child_backend: Any,
        *,
        query_rewriter: Callable[[str], str],
        probe_k: int = 100,
    ) -> None:
        if not callable(getattr(child_backend, "search", None)):
            raise TypeError("child_backend must implement search(HybridSearchRequest)")
        if not callable(query_rewriter):
            raise TypeError("query_rewriter must be callable")
        if not 1 <= int(probe_k) <= 500:
            raise ValueError("probe_k must be between 1 and 500")
        self.child_backend = child_backend
        self.query_rewriter = query_rewriter
        self.probe_k = int(probe_k)

    def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        probe_request = request.model_copy(
            update={
                "query": self.query_rewriter(request.query),
                "top_k": self.probe_k,
                "candidate_k": self.probe_k,
                "max_chunks_per_document": 1,
                "rerank": False,
                "neighbor_window": 0,
                "max_expanded_hits": self.probe_k,
            }
        )
        response = self.child_backend.search(probe_request)
        if response.filters_applied_before_ranking != AppliedRetrievalFilters.from_request(
            probe_request
        ):
            raise RuntimeError("passage parent probe changed trusted filters")

        hits: list[HybridSearchHit] = []
        seen_parents: set[str] = set()
        for child_hit in response.hits:
            raw_parent = child_hit.chunk.metadata.get("parent_document_id")
            parent_id = (
                raw_parent.strip()
                if isinstance(raw_parent, str) and raw_parent.strip()
                else child_hit.chunk.document_id
            )
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
            hits.append(
                child_hit.model_copy(
                    update={
                        "rank": len(hits) + 1,
                        "retrieval_sources": [
                            *child_hit.retrieval_sources,
                            "parent_child_retrieval",
                        ],
                    }
                )
            )
            if len(hits) >= request.top_k:
                break
        return HybridSearchResponse(
            backend="multi_representation_rrf",
            collection_name=response.collection_name,
            query=request.query,
            filters_applied_before_ranking=AppliedRetrievalFilters.from_request(request),
            seed_count=len(hits),
            expanded_neighbor_count=0,
            raw_candidate_counts={
                "passage_probe": len(response.hits),
                "distinct_parents": len(seen_parents),
            },
            hits=hits,
        )


class _NumericTableScanIndex:
    """Deterministically rank authorized table representations by query facts.

    This is a candidate-generation branch, not an answer calculator.  It
    scans the already materialized table rows/cells after the request scope is
    applied, scores each table document using visible query terms, years and
    numeric markers, and emits at most ``candidate_k`` table documents.  The
    full scan is intentional: a BM25 prefix can never recover a table that
    fell outside its own candidate budget.
    """

    _NUMBER = re.compile(r"(?<!\w)-?\$?\d[\d,]*(?:\.\d+)?%?(?!\w)")
    _YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
    _ROW = re.compile(r"\brow=([^|]+)", re.IGNORECASE)
    _COLUMN = re.compile(r"\bcolumn=([^|]+)", re.IGNORECASE)
    _COLUMNS = re.compile(r"\bcolumns=([^|]+)", re.IGNORECASE)
    _TEMPORAL_ROWS_MARKER = "__temporal_rows__"
    _SPARSE_YEAR_MARKER = "__sparse_year_values__"
    _SEGMENT_ROWS_MARKER = "__segment_rows__"
    _STOPWORDS = frozenset(
        {
            "a",
            "an",
            "and",
            "are",
            "as",
            "amount",
            "average",
            "be",
            "between",
            "by",
            "below",
            "change",
            "component",
            "components",
            "did",
            "does",
            "count",
            "decrease",
            "exceed",
            "for",
            "from",
            "had",
            "how",
            "in",
            "is",
            "item",
            "items",
            "listed",
            "many",
            "but",
            "not",
            "number",
            "of",
            "on",
            "or",
            "period",
            "periods",
            "percentage",
            "ratio",
            "reported",
            "reporting",
            "segment",
            "segments",
            "shown",
            "table",
            "the",
            "to",
            "under",
            "was",
            "were",
            "value",
            "values",
            "provided",
            "what",
            "which",
            "increase",
            "year",
            "years",
        }
    )

    def __init__(self, chunks: Sequence[HybridChunk]) -> None:
        self._chunks = tuple(
            chunk for chunk in chunks if chunk.metadata.get("kind") == "table"
        )
        self._features = {
            chunk.chunk_id: self._extract_features(chunk)
            for chunk in self._chunks
        }

    @classmethod
    def _normalise_number(cls, value: str) -> str:
        return value.replace("$", "").replace(",", "").strip()

    @staticmethod
    def _normalise_term(value: str) -> str:
        if value.endswith("ies") and len(value) > 4:
            return value[:-3] + "y"
        if value.endswith("s") and len(value) > 3:
            return value[:-1]
        return value

    @classmethod
    def _terms(cls, value: str) -> set[str]:
        return {cls._normalise_term(term) for term in tokenise(value)}

    @classmethod
    def _extract_features(
        cls,
        chunk: HybridChunk,
    ) -> tuple[set[str], set[str], set[str], set[str], set[str], float]:
        # Keep linked prose as a separate candidate-generation field.  It
        # helps the numeric scan route a table whose explanation carries the
        # query anchor, without lengthening the BM25 representation branch or
        # changing the final evidence text returned to the answerer.
        context = str(chunk.metadata.get("table_context", ""))
        feature_text = f"{chunk.text} {context}".strip()
        terms = cls._terms(feature_text)
        years = set(cls._YEAR.findall(feature_text))
        numbers = {
            cls._normalise_number(value)
            for value in cls._NUMBER.findall(feature_text)
        }
        if "rows=" in chunk.text:
            row_text = chunk.text.split("rows=", 1)[1]
        else:
            row_text = " ".join(cls._ROW.findall(chunk.text))
        # Section representations serialize their header as ``columns=``;
        # cell representations use one ``column=`` field.  Treat both as
        # column evidence so a year-pair query can prefer the right table
        # before the candidate budget is spent on sibling tables.
        column_text = " ".join(
            [*cls._COLUMN.findall(chunk.text), *cls._COLUMNS.findall(chunk.text)]
        )
        row_terms = cls._terms(row_text)
        column_years = set(cls._YEAR.findall(column_text))
        markers: set[str] = set()
        raw_rows = chunk.metadata.get("table_rows")
        if isinstance(raw_rows, list) and raw_rows:
            rows = [
                list(row) if isinstance(row, (list, tuple)) else [row]
                for row in raw_rows
            ]
            header = rows[0]
            year_columns = {
                index
                for index, value in enumerate(header)
                if cls._YEAR.fullmatch(str(value).strip())
            }
            body = rows[1:]
            first_column = [str(row[0]).strip() for row in body if row]
            temporal_rows = sum(
                bool(cls._YEAR.search(value)) or value.casefold() == "thereafter"
                for value in first_column
            )
            if temporal_rows >= 3:
                markers.add(cls._TEMPORAL_ROWS_MARKER)
            if len(year_columns) >= 2:
                sparse_pairs = 0
                ordered_year_columns = sorted(year_columns)
                for row in body:
                    # Some SEC tables serialize a second header row (for
                    # example, ``Number | Number``) before the actual body.
                    # It must not manufacture a false sparse-value signal.
                    if not row or not str(row[0]).strip():
                        continue
                    values = [
                        str(row[index]).strip() if index < len(row) else ""
                        for index in ordered_year_columns
                    ]
                    for current, previous in zip(values, values[1:], strict=False):
                        current_present = current.casefold() not in {
                            "",
                            "-",
                            "--",
                            "—",
                            "–",
                            "每",
                            "n/a",
                            "na",
                        }
                        previous_missing = previous.casefold() in {
                            "",
                            "-",
                            "--",
                            "—",
                            "–",
                            "每",
                            "n/a",
                            "na",
                        }
                        if current_present and previous_missing:
                            sparse_pairs += 1
                if sparse_pairs:
                    markers.add(cls._SPARSE_YEAR_MARKER)
            row_label_terms = cls._terms(" ".join(first_column))
            if len(first_column) >= 2 and row_label_terms.intersection(
                {"revenue", "expense", "sale", "service", "business", "geographic"}
            ):
                markers.add(cls._SEGMENT_ROWS_MARKER)
        terms.update(markers)
        representation_bonus = {
            "cell": 0.04,
            "row": 0.03,
            "schema": 0.02,
            "section": 0.04,
        }.get(str(chunk.metadata.get("representation")), 0.0)
        return terms, years, numbers, row_terms, column_years, representation_bonus

    @classmethod
    def _facts(cls, query: str) -> tuple[set[str], set[str], set[str]]:
        plan = build_tatqa_query_plan_from_text(query)
        query_terms = {
            cls._normalise_term(term)
            for term in tokenise(query)
            if term not in cls._STOPWORDS
            and not cls._YEAR.fullmatch(term)
            and not cls._NUMBER.fullmatch(term)
        }
        plan_terms = {
            cls._normalise_term(term)
            for term in tokenise(" ".join(str(value) for value in plan["metric_terms"]))
            if term not in cls._STOPWORDS
            and not cls._YEAR.fullmatch(term)
            and not cls._NUMBER.fullmatch(term)
        }
        metric_terms = query_terms | plan_terms
        years = set(cls._YEAR.findall(query))
        years.update(str(value) for value in plan["years"])
        if re.search(r"\bprovided\b.*\bnot\b|\bnot\b.*\bprovided\b", query, re.I):
            if len(years) >= 2:
                metric_terms.add(cls._SPARSE_YEAR_MARKER)
        if re.search(r"\bperiods?\b", query, re.I):
            metric_terms.add(cls._TEMPORAL_ROWS_MARKER)
        if re.search(r"\bsegments?\b", query, re.I):
            metric_terms.add(cls._SEGMENT_ROWS_MARKER)
        numbers = {
            cls._normalise_number(value)
            for value in cls._NUMBER.findall(query)
            if cls._normalise_number(value)
            and cls._normalise_number(value) not in years
        }
        return metric_terms, years, numbers

    @classmethod
    def _chunk_score(
        cls,
        chunk: HybridChunk,
        query_terms: set[str],
        years: set[str],
        numbers: set[str],
    ) -> float:
        terms, text_years, text_numbers, row_terms, column_years, representation_bonus = (
            cls._extract_features(chunk)
        )
        return cls._chunk_score_features(
            (terms, text_years, text_numbers, row_terms, column_years, representation_bonus),
            query_terms,
            years,
            numbers,
        )

    @staticmethod
    def _chunk_score_features(
        features: tuple[set[str], set[str], set[str], set[str], set[str], float],
        query_terms: set[str],
        years: set[str],
        numbers: set[str],
    ) -> float:
        terms, text_years, text_numbers, row_terms, column_years, representation_bonus = (
            features
        )
        metric_overlap = len(query_terms.intersection(terms))
        metric_coverage = metric_overlap / len(query_terms) if query_terms else 0.0
        year_coverage = len(years.intersection(text_years)) / len(years) if years else 0.0
        number_coverage = (
            len(numbers.intersection(text_numbers)) / len(numbers) if numbers else 0.0
        )
        row_coverage = (
            len(query_terms.intersection(row_terms)) / len(query_terms)
            if query_terms
            else 0.0
        )
        column_coverage = (
            len(years.intersection(column_years)) / len(years)
            if years
            else 0.0
        )
        # Metric/row overlap dominates; numeric facts are tie-breakers.  This
        # keeps broad table questions from being promoted solely by a year.
        return (
            0.52 * metric_coverage
            + 0.18 * row_coverage
            + 0.14 * year_coverage
            + 0.10 * number_coverage
            + 0.04 * column_coverage
            + representation_bonus
        )

    def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        query_terms, years, numbers = self._facts(request.query)
        grouped: dict[str, list[HybridChunk]] = {}
        for chunk in self._chunks:
            if not _matches_scope(chunk, request):
                continue
            if (
                request.allowed_chunk_ids is not None
                and chunk.chunk_id not in request.allowed_chunk_ids
            ):
                continue
            grouped.setdefault(chunk.document_id, []).append(chunk)

        ranked: list[tuple[float, str, HybridChunk]] = []
        for document_id, chunks in grouped.items():
            document_terms: set[str] = set()
            document_years: set[str] = set()
            document_numbers: set[str] = set()
            for chunk in chunks:
                features = self._features[chunk.chunk_id]
                document_terms.update(features[0])
                document_years.update(features[1])
                document_numbers.update(features[2])
            document_metric_coverage = (
                len(query_terms.intersection(document_terms)) / len(query_terms)
                if query_terms
                else 0.0
            )
            document_year_coverage = (
                len(years.intersection(document_years)) / len(years)
                if years
                else 0.0
            )
            document_number_coverage = (
                len(numbers.intersection(document_numbers)) / len(numbers)
                if numbers
                else 0.0
            )
            best = max(
                (
                    (
                        self._chunk_score_features(
                            self._features[chunk.chunk_id],
                            query_terms,
                            years,
                            numbers,
                        ),
                        chunk,
                    )
                    for chunk in chunks
                ),
                key=lambda item: (item[0], item[1].chunk_id),
            )
            score = (
                0.45 * document_metric_coverage
                + 0.20 * document_year_coverage
                + 0.12 * document_number_coverage
                + 0.23 * best[0]
            )
            if score > 0.0:
                ranked.append((score, document_id, best[1]))
        ranked.sort(key=lambda item: (-item[0], item[1], item[2].chunk_id))
        hits = [
            HybridSearchHit(
                chunk=chunk,
                rank=index,
                score=score,
                base_score=score,
                retrieval_sources=["tatqa_numeric_scan"],
            )
            for index, (score, _, chunk) in enumerate(
                ranked[: request.candidate_k], start=1
            )
        ]
        return HybridSearchResponse(
            backend="multi_representation_rrf",
            collection_name=None,
            query=request.query,
            filters_applied_before_ranking=AppliedRetrievalFilters.from_request(request),
            seed_count=len(hits),
            expanded_neighbor_count=0,
            raw_candidate_counts={"numeric_scan": len(hits)},
            hits=hits,
        )


class _NumericTableContextIndex:
    """Add bounded same-context paragraph coverage to numeric table hits."""

    def __init__(
        self,
        seed_backend: _NumericTableScanIndex,
        chunks: Sequence[HybridChunk],
        *,
        max_siblings: int = 2,
    ) -> None:
        if not 1 <= int(max_siblings) <= 10:
            raise ValueError("max_siblings must be between 1 and 10")
        self.seed_backend = seed_backend
        self.max_siblings = int(max_siblings)
        self._siblings: dict[str, list[HybridChunk]] = {}
        for chunk in chunks:
            if chunk.metadata.get("kind") == "table":
                continue
            raw_parent = chunk.metadata.get("parent_document_id")
            parent_id = (
                raw_parent.strip()
                if isinstance(raw_parent, str) and raw_parent.strip()
                else chunk.document_id
            )
            self._siblings.setdefault(parent_id, []).append(chunk)
        for siblings in self._siblings.values():
            siblings.sort(
                key=lambda chunk: (
                    int(chunk.metadata["order"])
                    if str(chunk.metadata.get("order", "")).isdigit()
                    else 1_000_000,
                    chunk.chunk_id,
                )
            )

    def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        seed_response = self.seed_backend.search(request)
        seed_limit = max(1, request.candidate_k - self.max_siblings * 5)
        output = list(seed_response.hits[:seed_limit])
        seen = {hit.chunk.chunk_id for hit in output}
        parent_count: dict[str, int] = {}
        for seed in output:
            raw_parent = seed.chunk.metadata.get("parent_document_id")
            parent_id = (
                raw_parent.strip()
                if isinstance(raw_parent, str) and raw_parent.strip()
                else seed.chunk.document_id
            )
            for sibling in self._siblings.get(parent_id, ()):
                if parent_count.get(parent_id, 0) >= self.max_siblings:
                    break
                if sibling.chunk_id in seen or not _matches_scope(sibling, request):
                    continue
                if (
                    request.allowed_chunk_ids is not None
                    and sibling.chunk_id not in request.allowed_chunk_ids
                ):
                    continue
                output.append(
                    HybridSearchHit(
                        chunk=sibling,
                        rank=1,
                        score=0.0,
                        base_score=0.0,
                        retrieval_sources=["context_sibling_coverage"],
                    )
                )
                seen.add(sibling.chunk_id)
                parent_count[parent_id] = parent_count.get(parent_id, 0) + 1
                if len(output) >= request.candidate_k:
                    break
            if len(output) >= request.candidate_k:
                break
        output = [hit.model_copy(update={"rank": index}) for index, hit in enumerate(output[: request.candidate_k], start=1)]
        return HybridSearchResponse(
            backend="multi_representation_rrf",
            collection_name=None,
            query=request.query,
            filters_applied_before_ranking=AppliedRetrievalFilters.from_request(request),
            seed_count=sum(hit.neighbor_of_chunk_id is None for hit in output),
            expanded_neighbor_count=0,
            raw_candidate_counts={
                "numeric_context": len(output),
                "numeric_scan": len(seed_response.hits),
            },
            hits=output,
        )


class _EvidenceContextClosureIndex:
    """Append bounded same-parent evidence for already-ranked TAT-QA hits.

    This is a retrieval-time context expansion, not a gold-evidence shortcut:
    the seed ranking is produced by the wrapped backend, and only authorized
    siblings already present in the indexed corpus can be appended.  The
    output remains capped by ``candidate_k`` and discloses the closure count.
    """

    def __init__(
        self,
        backend: Any,
        chunks: Sequence[HybridChunk],
        *,
        seed_k: int = 10,
        max_siblings_per_seed: int = 4,
    ) -> None:
        if not callable(getattr(backend, "search", None)):
            raise TypeError("backend must implement search(request)")
        if not 1 <= int(seed_k) <= 100:
            raise ValueError("seed_k must be between 1 and 100")
        if not 1 <= int(max_siblings_per_seed) <= 20:
            raise ValueError("max_siblings_per_seed must be between 1 and 20")
        self.backend = backend
        self.seed_k = int(seed_k)
        self.max_siblings_per_seed = int(max_siblings_per_seed)
        self._siblings: dict[str, list[HybridChunk]] = {}
        for chunk in chunks:
            raw_parent = chunk.metadata.get("parent_document_id")
            parent_id = (
                raw_parent.strip()
                if isinstance(raw_parent, str) and raw_parent.strip()
                else chunk.document_id
            )
            self._siblings.setdefault(parent_id, []).append(chunk)
        for siblings in self._siblings.values():
            siblings.sort(
                key=lambda chunk: (
                    0 if chunk.metadata.get("kind") == "table" else 1,
                    int(chunk.metadata["order"])
                    if str(chunk.metadata.get("order", "")).isdigit()
                    else 1_000_000,
                    chunk.chunk_id,
                )
            )

    @staticmethod
    def _parent_id(chunk: HybridChunk) -> str:
        raw_parent = chunk.metadata.get("parent_document_id")
        return (
            raw_parent.strip()
            if isinstance(raw_parent, str) and raw_parent.strip()
            else chunk.document_id
        )

    def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        response = self.backend.search(request)
        closure_budget = min(
            request.candidate_k,
            self.seed_k * self.max_siblings_per_seed,
        )
        seed_limit = max(1, request.candidate_k - closure_budget)
        output = list(response.hits[:seed_limit])
        seen = {hit.chunk.chunk_id for hit in output}
        added = 0
        for seed in output[: self.seed_k]:
            parent_id = self._parent_id(seed.chunk)
            added_for_seed = 0
            for sibling in self._siblings.get(parent_id, ()):
                if added_for_seed >= self.max_siblings_per_seed:
                    break
                if sibling.chunk_id in seen or not _matches_scope(sibling, request):
                    continue
                if (
                    request.allowed_chunk_ids is not None
                    and sibling.chunk_id not in request.allowed_chunk_ids
                ):
                    continue
                output.append(
                    HybridSearchHit(
                        chunk=sibling,
                        rank=1,
                        score=0.0,
                        base_score=0.0,
                        retrieval_sources=["same_parent_evidence_closure"],
                    )
                )
                seen.add(sibling.chunk_id)
                added += 1
                added_for_seed += 1
                if added >= closure_budget or len(output) >= request.candidate_k:
                    break
            if added >= closure_budget or len(output) >= request.candidate_k:
                break
        return response.model_copy(
            update={
                "hits": [
                    hit.model_copy(update={"rank": index})
                    for index, hit in enumerate(
                        output[: request.candidate_k], start=1
                    )
                ],
                "expanded_neighbor_count": added,
                "raw_candidate_counts": {
                    **response.raw_candidate_counts,
                    "same_parent_evidence_closure": added,
                },
            }
        )


class _QueryAwareLineageClosureIndex:
    """Reserve candidate-tail slots for query-relevant same-parent evidence.

    The wrapped retriever keeps the complete ranked head.  This adapter only
    changes the Candidate@K tail: it follows corpus-derived parent links from
    the top seed documents, scores authorized sibling table/paragraph units
    against the query, and appends a bounded number of complementary units.
    Gold evidence IDs, answers, and dataset names are never consulted.
    """

    _STOPWORDS = frozenset(
        {
            "a",
            "an",
            "and",
            "are",
            "as",
            "be",
            "by",
            "did",
            "do",
            "does",
            "for",
            "from",
            "had",
            "has",
            "have",
            "how",
            "in",
            "is",
            "of",
            "on",
            "or",
            "the",
            "to",
            "was",
            "were",
            "what",
            "when",
            "which",
            "with",
        }
    )

    def __init__(
        self,
        backend: Any,
        chunks: Sequence[HybridChunk],
        *,
        preserve_head_k: int = 10,
        seed_k: int = 20,
        closure_slots: int = 12,
        max_siblings_per_parent: int = 2,
    ) -> None:
        if not callable(getattr(backend, "search", None)):
            raise TypeError("backend must implement search(request)")
        if not 1 <= int(preserve_head_k) <= 100:
            raise ValueError("preserve_head_k must be between 1 and 100")
        if not 1 <= int(seed_k) <= 100:
            raise ValueError("seed_k must be between 1 and 100")
        if not 1 <= int(closure_slots) <= 100:
            raise ValueError("closure_slots must be between 1 and 100")
        if not 1 <= int(max_siblings_per_parent) <= 10:
            raise ValueError("max_siblings_per_parent must be between 1 and 10")
        self.backend = backend
        self.preserve_head_k = int(preserve_head_k)
        self.seed_k = int(seed_k)
        self.closure_slots = int(closure_slots)
        self.max_siblings_per_parent = int(max_siblings_per_parent)
        grouped: dict[str, dict[str, list[HybridChunk]]] = {}
        for chunk in chunks:
            if chunk.chunk_id.startswith(_PROBE_PREFIX):
                continue
            parent_id = self._parent_id(chunk)
            grouped.setdefault(parent_id, {}).setdefault(
                chunk.document_id, []
            ).append(chunk)
        self._siblings = {
            parent_id: {
                document_id: tuple(sorted(items, key=lambda item: item.chunk_id))
                for document_id, items in sorted(documents.items())
            }
            for parent_id, documents in sorted(grouped.items())
        }
        self._features = {
            chunk.chunk_id: (
                _NumericTableScanIndex._terms(chunk.text),
                set(_NumericTableScanIndex._YEAR.findall(chunk.text)),
                {
                    _NumericTableScanIndex._normalise_number(value)
                    for value in _NumericTableScanIndex._NUMBER.findall(chunk.text)
                },
                str(chunk.metadata.get("kind", "")),
            )
            for documents in self._siblings.values()
            for sibling_chunks in documents.values()
            for chunk in sibling_chunks
        }

    @staticmethod
    def _parent_id(chunk: HybridChunk) -> str:
        raw_parent = chunk.metadata.get("parent_document_id")
        return (
            raw_parent.strip()
            if isinstance(raw_parent, str) and raw_parent.strip()
            else chunk.document_id
        )

    @classmethod
    def _query_facts(cls, query: str) -> tuple[set[str], set[str], set[str]]:
        plan = build_tatqa_query_plan_from_text(query)
        terms = {
            _NumericTableScanIndex._normalise_term(term)
            for term in tokenise(query)
            if term not in cls._STOPWORDS
            and not _NumericTableScanIndex._YEAR.fullmatch(term)
            and not _NumericTableScanIndex._NUMBER.fullmatch(term)
        }
        terms.update(
            _NumericTableScanIndex._normalise_term(term)
            for term in tokenise(
                " ".join(str(value) for value in plan["metric_terms"])
            )
            if term not in cls._STOPWORDS
        )
        years = set(_NumericTableScanIndex._YEAR.findall(query))
        years.update(str(value) for value in plan["years"])
        numbers = {
            _NumericTableScanIndex._normalise_number(value)
            for value in _NumericTableScanIndex._NUMBER.findall(query)
            if _NumericTableScanIndex._normalise_number(value) not in years
        }
        return terms, years, numbers

    @classmethod
    def _feature_score(
        cls,
        features: tuple[set[str], set[str], set[str], str],
        *,
        query_terms: set[str],
        years: set[str],
        numbers: set[str],
        seed_kind: str,
        seed_rank: int,
    ) -> float:
        terms, text_years, text_numbers, sibling_kind = features
        term_coverage = (
            len(query_terms.intersection(terms)) / len(query_terms)
            if query_terms
            else 0.0
        )
        year_coverage = (
            len(years.intersection(text_years)) / len(years)
            if years
            else 0.0
        )
        number_coverage = (
            len(numbers.intersection(text_numbers)) / len(numbers)
            if numbers
            else 0.0
        )
        complement_bonus = (
            1.0
            if {seed_kind, sibling_kind} == {"table", "paragraph"}
            else 0.0
        )
        seed_prior = 1.0 / max(1, seed_rank)
        return (
            0.60 * term_coverage
            + 0.14 * year_coverage
            + 0.10 * number_coverage
            + 0.10 * complement_bonus
            + 0.06 * seed_prior
        )

    def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        response = self.backend.search(request)
        if request.candidate_k <= self.preserve_head_k or not response.hits:
            return response
        query_terms, years, numbers = self._query_facts(request.query)
        protected_document_ids = {
            hit.chunk.document_id
            for hit in response.hits[: self.preserve_head_k]
        }
        seed_parents: dict[str, tuple[int, HybridSearchHit]] = {}
        for seed_rank, seed in enumerate(response.hits[: self.seed_k], start=1):
            parent_id = self._parent_id(seed.chunk)
            seed_parents.setdefault(parent_id, (seed_rank, seed))

        candidates: list[tuple[float, int, str, str, HybridChunk, HybridSearchHit]] = []
        for parent_id, (seed_rank, seed) in seed_parents.items():
            seed_kind = str(seed.chunk.metadata.get("kind", ""))
            for document_id, sibling_chunks in self._siblings.get(
                parent_id, {}
            ).items():
                if (
                    document_id == seed.chunk.document_id
                    or document_id in protected_document_ids
                ):
                    continue
                scored_chunks = [
                    (
                        self._feature_score(
                            self._features[sibling.chunk_id],
                            query_terms=query_terms,
                            years=years,
                            numbers=numbers,
                            seed_kind=seed_kind,
                            seed_rank=seed_rank,
                        ),
                        sibling,
                    )
                    for sibling in sibling_chunks
                    if _matches_scope(sibling, request)
                    and (
                        request.allowed_chunk_ids is None
                        or sibling.chunk_id in request.allowed_chunk_ids
                    )
                ]
                if not scored_chunks:
                    continue
                score, sibling = max(
                    scored_chunks,
                    key=lambda item: (item[0], item[1].chunk_id),
                )
                if score <= 0.0:
                    continue
                candidates.append(
                    (score, seed_rank, parent_id, document_id, sibling, seed)
                )
        candidates.sort(
            key=lambda item: (-item[0], item[1], item[2], item[3], item[4].chunk_id)
        )
        selected: list[HybridSearchHit] = []
        selected_documents: set[str] = set()
        parent_counts: Counter[str] = Counter()
        selection_limit = min(self.closure_slots, request.candidate_k)

        def add_candidate(
            candidate: tuple[
                float, int, str, str, HybridChunk, HybridSearchHit
            ],
        ) -> bool:
            score, _, parent_id, document_id, sibling, seed = candidate
            if (
                len(selected) >= selection_limit
                or document_id in selected_documents
                or parent_counts[parent_id] >= self.max_siblings_per_parent
            ):
                return False
            selected_documents.add(document_id)
            parent_counts[parent_id] += 1
            selected.append(
                HybridSearchHit(
                    chunk=sibling,
                    rank=1,
                    score=score,
                    base_score=score,
                    retrieval_sources=["same_parent_evidence_closure"],
                    neighbor_of_chunk_id=seed.chunk.chunk_id,
                    neighbor_distance=1,
                )
            )
            return True

        candidates_by_parent: dict[
            str,
            list[tuple[float, int, str, str, HybridChunk, HybridSearchHit]],
        ] = {}
        for candidate in candidates:
            candidates_by_parent.setdefault(candidate[2], []).append(candidate)
        # Lineage is the primary signal: first give each highest-ranked seed
        # parent one query-relevant complementary unit.  Only then spend any
        # remaining tail budget on a second sibling from the strongest parents.
        for parent_id, _ in sorted(
            seed_parents.items(), key=lambda item: item[1][0]
        ):
            if len(selected) >= selection_limit:
                break
            parent_candidates = candidates_by_parent.get(parent_id, [])
            if parent_candidates:
                add_candidate(parent_candidates[0])
        for candidate in candidates:
            if len(selected) >= selection_limit:
                break
            add_candidate(candidate)
        if not selected:
            return response

        base_budget = max(
            self.preserve_head_k,
            request.candidate_k - len(selected),
        )
        base_hits: list[HybridSearchHit] = []
        for hit in response.hits:
            if hit.chunk.document_id in selected_documents:
                continue
            base_hits.append(hit)
            if len(base_hits) >= base_budget:
                break
        output = [*base_hits, *selected][: request.candidate_k]
        return response.model_copy(
            update={
                "hits": [
                    hit.model_copy(update={"rank": rank})
                    for rank, hit in enumerate(output, start=1)
                ],
                "expanded_neighbor_count": (
                    response.expanded_neighbor_count + len(selected)
                ),
                "raw_candidate_counts": {
                    **response.raw_candidate_counts,
                    "query_aware_lineage_closure": len(selected),
                },
            }
        )


class _StructuredTableFactIndex:
    """Rank table documents from explicit row/header/value facts."""

    _MODES = frozenset({"count", "arithmetic", "multi_span"})
    _SOURCE_BY_MODE = {
        "count": "tatqa_structured_count",
        "arithmetic": "tatqa_structured_arithmetic",
        "multi_span": "tatqa_structured_multi_span",
    }
    _STOPWORDS = _NumericTableScanIndex._STOPWORDS.union(
        {
            "company",
            "respective",
            "respectively",
            "total",
            "within",
        }
    )

    def __init__(
        self,
        chunks: Sequence[HybridChunk],
        *,
        mode: Literal["count", "arithmetic", "multi_span"],
    ) -> None:
        if mode not in self._MODES:
            raise ValueError("structured table mode is invalid")
        self.mode = mode
        self.source = self._SOURCE_BY_MODE[mode]
        self._chunks = tuple(
            chunk
            for chunk in chunks
            if chunk.metadata.get("representation")
            in {"structured_row", "structured_schema"}
        )
        grouped: dict[str, list[HybridChunk]] = {}
        for chunk in self._chunks:
            grouped.setdefault(chunk.document_id, []).append(chunk)
        self._documents = {
            document_id: tuple(sorted(items, key=lambda item: item.chunk_id))
            for document_id, items in sorted(grouped.items())
        }
        self._features = {
            chunk.chunk_id: self._extract_features(chunk) for chunk in self._chunks
        }

    @classmethod
    def _extract_features(cls, chunk: HybridChunk) -> dict[str, Any]:
        metadata = chunk.metadata
        terms = _NumericTableScanIndex._terms(chunk.text)
        row_terms = _NumericTableScanIndex._terms(
            " ".join(
                str(value)
                for value in (
                    metadata.get("structured_row_label", ""),
                    metadata.get("structured_group_label", ""),
                    *(
                        metadata.get("structured_row_labels", [])
                        if isinstance(metadata.get("structured_row_labels"), list)
                        else []
                    ),
                )
                if value
            )
        )
        header_terms = _NumericTableScanIndex._terms(
            " ".join(
                str(value)
                for value in metadata.get("structured_headers", [])
            )
        )
        years: set[str] = set()
        numbers: set[str] = set()
        units: set[str] = set()
        signs: set[str] = set()
        numeric_count = 0
        raw_values = metadata.get("structured_values", [])
        if isinstance(raw_values, list):
            for value in raw_values:
                if not isinstance(value, Mapping):
                    continue
                years.update(str(item) for item in value.get("years", []) if item)
                normalised = value.get("normalized")
                if normalised is not None:
                    numbers.add(str(normalised))
                    numeric_count += 1
                unit = value.get("unit")
                sign = value.get("sign")
                if unit:
                    units.add(str(unit))
                if sign:
                    signs.add(str(sign))
        years.update(_TABLE_YEAR.findall(" ".join(metadata.get("structured_headers", []))))
        aliases = {
            str(value)
            for value in metadata.get("structured_aliases", [])
            if value
        }
        terms.update(aliases)
        return {
            "terms": terms,
            "row_terms": row_terms,
            "header_terms": header_terms,
            "years": years,
            "numbers": numbers,
            "units": units,
            "signs": signs,
            "numeric_count": numeric_count,
            "scale": str(metadata.get("structured_scale") or ""),
            "representation": str(metadata.get("representation", "")),
            "row_count": len(metadata.get("structured_row_labels", []))
            if isinstance(metadata.get("structured_row_labels"), list)
            else 1,
        }

    @classmethod
    def _query_features(cls, query: str) -> dict[str, Any]:
        plan = build_tatqa_query_plan_from_text(query)
        years = set(_TABLE_YEAR.findall(query))
        years.update(str(value) for value in plan["years"])
        terms = {
            _NumericTableScanIndex._normalise_term(term)
            for term in tokenise(query)
            if term not in cls._STOPWORDS
            and not _TABLE_YEAR.fullmatch(term)
            and not _NumericTableScanIndex._NUMBER.fullmatch(term)
        }
        terms.update(
            _NumericTableScanIndex._normalise_term(term)
            for term in tokenise(
                " ".join(str(value) for value in plan["metric_terms"])
            )
            if term not in cls._STOPWORDS
            and not _TABLE_YEAR.fullmatch(term)
        )
        numbers = {
            _NumericTableScanIndex._normalise_number(value)
            for value in _NumericTableScanIndex._NUMBER.findall(query)
            if _NumericTableScanIndex._normalise_number(value) not in years
        }
        scale = _structured_scale(query) or str(plan["scale"] or "")
        return {
            "terms": terms,
            "years": years,
            "numbers": numbers,
            "scale": scale,
            "comparator": str(plan["comparator"] or ""),
        }

    def _score(self, features: Mapping[str, Any], query: Mapping[str, Any]) -> float:
        query_terms = set(query["terms"])
        years = set(query["years"])
        numbers = set(query["numbers"])
        term_coverage = (
            len(query_terms.intersection(features["terms"])) / len(query_terms)
            if query_terms
            else 0.0
        )
        row_coverage = (
            len(query_terms.intersection(features["row_terms"])) / len(query_terms)
            if query_terms
            else 0.0
        )
        header_coverage = (
            len(query_terms.intersection(features["header_terms"])) / len(query_terms)
            if query_terms
            else 0.0
        )
        year_coverage = (
            len(years.intersection(features["years"])) / len(years)
            if years
            else 0.0
        )
        number_coverage = (
            len(numbers.intersection(features["numbers"])) / len(numbers)
            if numbers
            else 0.0
        )
        multi_value = min(1.0, float(features["numeric_count"]) / 2.0)
        scale_match = 1.0 if query["scale"] and query["scale"] == features["scale"] else 0.0
        schema_bonus = 1.0 if features["representation"] == "structured_schema" else 0.0
        if self.mode == "count":
            return (
                0.42 * row_coverage
                + 0.20 * term_coverage
                + 0.13 * year_coverage
                + 0.08 * number_coverage
                + 0.10 * multi_value
                + 0.07 * scale_match
            )
        if self.mode == "arithmetic":
            return (
                0.34 * row_coverage
                + 0.24 * term_coverage
                + 0.18 * year_coverage
                + 0.08 * header_coverage
                + 0.06 * number_coverage
                + 0.07 * multi_value
                + 0.03 * scale_match
            )
        plurality = min(1.0, float(features["row_count"]) / 2.0)
        return (
            0.40 * term_coverage
            + 0.22 * row_coverage
            + 0.10 * header_coverage
            + 0.08 * year_coverage
            + 0.10 * plurality
            + 0.10 * schema_bonus
        )

    def chunk_ids_for_documents(self, document_ids: set[str]) -> frozenset[str]:
        return frozenset(
            chunk.chunk_id
            for document_id in document_ids
            for chunk in self._documents.get(document_id, ())
        )

    def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        query = self._query_features(request.query)
        ranked: list[tuple[float, str, HybridChunk]] = []
        for document_id, chunks in self._documents.items():
            scored = [
                (self._score(self._features[chunk.chunk_id], query), chunk)
                for chunk in chunks
                if _matches_scope(chunk, request)
                and (
                    request.allowed_chunk_ids is None
                    or chunk.chunk_id in request.allowed_chunk_ids
                )
            ]
            if not scored:
                continue
            score, best = max(scored, key=lambda item: (item[0], item[1].chunk_id))
            if score > 0.0:
                ranked.append((score, document_id, best))
        ranked.sort(key=lambda item: (-item[0], item[1], item[2].chunk_id))
        hits = [
            HybridSearchHit(
                chunk=chunk,
                rank=rank,
                score=score,
                base_score=score,
                retrieval_sources=[self.source],
            )
            for rank, (score, _, chunk) in enumerate(
                ranked[: request.candidate_k], start=1
            )
        ]
        return HybridSearchResponse(
            backend="multi_representation_rrf",
            collection_name=None,
            query=request.query,
            filters_applied_before_ranking=AppliedRetrievalFilters.from_request(request),
            seed_count=len(hits),
            expanded_neighbor_count=0,
            raw_candidate_counts={self.source: len(hits)},
            hits=hits,
        )


class _QueryTypedStructuredTableIndex:
    """Route visible query types to isolated structured candidate branches."""

    _ARITHMETIC_LOOKUP = re.compile(
        r"\b(total|within|due|average|difference|change|ratio|percentage)\b",
        re.IGNORECASE,
    )
    _MULTI_SPAN = re.compile(
        r"\b(respective|respectively|segments?|components?|categories|breakdown|both|each|which)\b",
        re.IGNORECASE,
    )

    def __init__(self, chunks: Sequence[HybridChunk]) -> None:
        self._branches = {
            mode: _StructuredTableFactIndex(chunks, mode=mode)
            for mode in ("count", "arithmetic", "multi_span")
        }

    @classmethod
    def branch_name(cls, query: str) -> str:
        operation = str(build_tatqa_query_plan_from_text(query)["operation"])
        if operation == "count":
            return "count"
        if operation == "arithmetic":
            return "arithmetic"
        if cls._ARITHMETIC_LOOKUP.search(query):
            return "arithmetic"
        if operation == "comparison" or cls._MULTI_SPAN.search(query):
            return "multi_span"
        return "arithmetic" if _TABLE_YEAR.search(query) else "multi_span"

    def chunk_ids_for_documents(self, document_ids: set[str]) -> frozenset[str]:
        return frozenset(
            chunk_id
            for branch in self._branches.values()
            for chunk_id in branch.chunk_ids_for_documents(document_ids)
        )

    def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        return self._branches[self.branch_name(request.query)].search(request)


class _StructuredLineageCandidateIndex:
    """Add query-typed table facts and their lineage only to the candidate tail."""

    _INTRO = re.compile(
        r"\b(following table|following summarizes?|table presents|summary of|"
        r"consist of|consists of|"
        r"in thousands|in millions|in billions|reconciliation of)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        backend: Any,
        structured_backend: _QueryTypedStructuredTableIndex,
        chunks: Sequence[HybridChunk],
        *,
        preserve_head_k: int = 10,
        candidate_slots: int = 10,
        seed_k: int = 10,
        max_siblings_per_parent: int = 2,
    ) -> None:
        if not callable(getattr(backend, "search", None)):
            raise TypeError("backend must implement search(request)")
        if not 1 <= preserve_head_k <= 100:
            raise ValueError("preserve_head_k must be between 1 and 100")
        if not 1 <= candidate_slots <= 100:
            raise ValueError("candidate_slots must be between 1 and 100")
        if not 1 <= seed_k <= 100:
            raise ValueError("seed_k must be between 1 and 100")
        if not 1 <= max_siblings_per_parent <= 10:
            raise ValueError("max_siblings_per_parent must be between 1 and 10")
        self.backend = backend
        self.structured_backend = structured_backend
        self.preserve_head_k = int(preserve_head_k)
        self.candidate_slots = int(candidate_slots)
        self.seed_k = int(seed_k)
        self.max_siblings_per_parent = int(max_siblings_per_parent)
        self._base_chunks = tuple(
            chunk for chunk in chunks if not chunk.chunk_id.startswith(_PROBE_PREFIX)
        )
        self._canonical_tables = {
            chunk.document_id: chunk
            for chunk in self._base_chunks
            if chunk.metadata.get("kind") == "table"
        }
        grouped: dict[str, list[HybridChunk]] = {}
        for chunk in self._base_chunks:
            if chunk.metadata.get("kind") == "table":
                continue
            parent_id = _QueryAwareLineageClosureIndex._parent_id(chunk)
            grouped.setdefault(parent_id, []).append(chunk)
        self._siblings = {
            parent_id: tuple(sorted(items, key=lambda item: item.chunk_id))
            for parent_id, items in sorted(grouped.items())
        }

    @classmethod
    def _paragraph_score(
        cls, query: str, paragraph: HybridChunk, table: HybridChunk
    ) -> float:
        query_terms, years, numbers = _QueryAwareLineageClosureIndex._query_facts(query)
        paragraph_terms = _NumericTableScanIndex._terms(paragraph.text)
        raw_rows = table.metadata.get("table_rows", [])
        row_labels = " ".join(
            str(row[0])
            for row in raw_rows
            if isinstance(row, (list, tuple)) and row
        ) if isinstance(raw_rows, list) else ""
        row_terms = _NumericTableScanIndex._terms(row_labels)
        term_coverage = (
            len(query_terms.intersection(paragraph_terms)) / len(query_terms)
            if query_terms
            else 0.0
        )
        row_link = min(1.0, len(row_terms.intersection(paragraph_terms)) / 2.0)
        paragraph_years = set(_TABLE_YEAR.findall(paragraph.text))
        year_coverage = (
            len(years.intersection(paragraph_years)) / len(years)
            if years
            else 0.0
        )
        paragraph_numbers = {
            _NumericTableScanIndex._normalise_number(value)
            for value in _NumericTableScanIndex._NUMBER.findall(paragraph.text)
        }
        number_coverage = (
            len(numbers.intersection(paragraph_numbers)) / len(numbers)
            if numbers
            else 0.0
        )
        intro = 1.0 if cls._INTRO.search(paragraph.text) else 0.0
        entity_list = bool(
            re.search(r"\b(segments?|components?|categories|breakdown)\b", query, re.I)
        )
        if not entity_list:
            return (
                0.36 * term_coverage
                + 0.12 * row_link
                + 0.10 * year_coverage
                + 0.04 * number_coverage
                + 0.38 * intro
            )
        return (
            0.48 * term_coverage
            + 0.24 * row_link
            + 0.12 * year_coverage
            + 0.06 * number_coverage
            + 0.10 * intro
        )

    def _structured_request(
        self, request: HybridSearchRequest
    ) -> HybridSearchRequest | None:
        if request.allowed_chunk_ids is None:
            return request
        allowed_documents = {
            chunk.document_id
            for chunk in self._base_chunks
            if chunk.chunk_id in request.allowed_chunk_ids
        }
        structured_ids = self.structured_backend.chunk_ids_for_documents(
            allowed_documents
        )
        if not structured_ids:
            return None
        return request.model_copy(update={"allowed_chunk_ids": structured_ids})

    def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        response = self.backend.search(request)
        if request.candidate_k <= self.preserve_head_k or not response.hits:
            return response
        structured_request = self._structured_request(request)
        if structured_request is None:
            return response
        structured = self.structured_backend.search(structured_request)
        if not structured.hits:
            return response
        existing_document_ids = {hit.chunk.document_id for hit in response.hits}
        base_positions = {
            hit.chunk.document_id: rank
            for rank, hit in enumerate(response.hits, start=1)
        }
        seed_candidates: list[
            tuple[
                HybridSearchHit,
                HybridChunk,
                list[tuple[float, HybridChunk]],
                int,
            ]
        ] = []
        # Candidate injection is deliberately high-confidence: a structured
        # fact may complete lineage for a table already present in the stable
        # Top-10, but it cannot introduce a new table and displace ten tail
        # documents merely because that table was a weak structured match.
        for seed in structured.hits[: min(self.seed_k, 8)]:
            table = self._canonical_tables.get(seed.chunk.document_id)
            if table is None or not _matches_scope(table, request):
                continue
            base_rank = base_positions.get(table.document_id)
            if base_rank is None or base_rank > self.preserve_head_k:
                continue
            if (
                request.allowed_chunk_ids is not None
                and table.chunk_id not in request.allowed_chunk_ids
            ):
                continue
            parent_id = _QueryAwareLineageClosureIndex._parent_id(table)
            siblings = [
                (self._paragraph_score(request.query, sibling, table), sibling)
                for sibling in self._siblings.get(parent_id, ())
                if _matches_scope(sibling, request)
                and (
                    request.allowed_chunk_ids is None
                    or sibling.chunk_id in request.allowed_chunk_ids
                )
            ]
            siblings = [
                item
                for item in siblings
                if item[0] >= 0.22
                and item[1].document_id not in existing_document_ids
            ]
            siblings.sort(key=lambda item: (-item[0], item[1].chunk_id))
            if siblings:
                seed_candidates.append((seed, table, siblings, base_rank))

        selected: list[HybridSearchHit] = []
        selected_documents: set[str] = set()

        def add_sibling(
            seed: HybridSearchHit,
            table: HybridChunk,
            item: tuple[float, HybridChunk],
        ) -> None:
            score, sibling = item
            if (
                len(selected) >= self.candidate_slots
                or sibling.document_id in existing_document_ids
                or sibling.document_id in selected_documents
            ):
                return
            selected_documents.add(sibling.document_id)
            selected.append(
                HybridSearchHit(
                    chunk=sibling,
                    rank=1,
                    score=score,
                    base_score=score,
                    retrieval_sources=[
                        *seed.retrieval_sources,
                        "structured_lineage_candidate",
                    ],
                    neighbor_of_chunk_id=table.chunk_id,
                    neighbor_distance=1,
                )
            )

        selected_parent_ids: set[str] = set()
        mutation_limit = min(
            self.candidate_slots,
            self.max_siblings_per_parent,
        )
        seed_candidates.sort(
            key=lambda item: (
                -item[2][0][0],
                -item[0].score,
                item[3],
                item[1].document_id,
            )
        )
        # Spend at most one slot per high-confidence table first.  This lets
        # two independent table/text lineage gaps be repaired without a single
        # ambiguous table consuming the entire candidate-tail mutation budget.
        for seed, table, siblings, _ in seed_candidates:
            if len(selected) >= mutation_limit:
                break
            add_sibling(seed, table, siblings[0])
            selected_parent_ids.add(
                _QueryAwareLineageClosureIndex._parent_id(table)
            )
        if not selected:
            return response.model_copy(
                update={
                    "raw_candidate_counts": {
                        **response.raw_candidate_counts,
                        **structured.raw_candidate_counts,
                        "structured_lineage_candidate": 0,
                    }
                }
            )
        removal_indexes: list[int] = []
        # Prefer replacing low-ranked generic lineage expansions from other
        # parents.  The original fused candidates and the selected table's
        # own lineage remain intact whenever enough such slots exist.
        for index in range(len(response.hits) - 1, self.preserve_head_k - 1, -1):
            hit = response.hits[index]
            parent_id = _QueryAwareLineageClosureIndex._parent_id(hit.chunk)
            if (
                "same_parent_evidence_closure" in hit.retrieval_sources
                and parent_id not in selected_parent_ids
            ):
                removal_indexes.append(index)
                if len(removal_indexes) >= len(selected):
                    break
        if len(removal_indexes) < len(selected):
            for index in range(
                len(response.hits) - 1, self.preserve_head_k - 1, -1
            ):
                if index in removal_indexes:
                    continue
                removal_indexes.append(index)
                if len(removal_indexes) >= len(selected):
                    break
        removal_set = set(removal_indexes)
        base_hits = [
            hit
            for index, hit in enumerate(response.hits)
            if index not in removal_set
            and hit.chunk.document_id not in selected_documents
        ]
        output = [*base_hits, *selected][: request.candidate_k]
        return response.model_copy(
            update={
                "hits": [
                    hit.model_copy(update={"rank": rank})
                    for rank, hit in enumerate(output, start=1)
                ],
                "expanded_neighbor_count": (
                    response.expanded_neighbor_count + len(selected)
                ),
                "raw_candidate_counts": {
                    **response.raw_candidate_counts,
                    **structured.raw_candidate_counts,
                    "structured_lineage_candidate": len(selected),
                },
            }
        )


class _StructuredLineagePairRerankIndex:
    """Promote one audited parent pair for explicit temporal/count queries.

    The wrapper only reorders candidates already returned by the structured
    lineage stage.  A tail candidate must have been produced by the existing
    ACL-filtered lineage closure and share a parent with a stable Top-K hit.
    It cannot add evidence, change Candidate@K, or route narrative queries.
    """

    _QUERY = re.compile(
        r"(?:\bhow many\b.*\b(?:years?|quarters?|periods?)\b|"
        r"\b(?:in|for) which (?:years?|periods?)\b|"
        r"\bhow many\b.*\bas at\b|"
        r"\bhow many\b.*\bincluded within\b)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        backend: Any,
        *,
        preserve_head_k: int = 10,
        rerank_slots: int = 1,
        min_score: float = 0.24,
    ) -> None:
        if not callable(getattr(backend, "search", None)):
            raise TypeError("backend must implement search(request)")
        if not 2 <= preserve_head_k <= 100:
            raise ValueError("preserve_head_k must be between 2 and 100")
        if not 1 <= rerank_slots < preserve_head_k:
            raise ValueError("rerank_slots must be smaller than preserve_head_k")
        if not math.isfinite(min_score) or not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")
        self.backend = backend
        self.preserve_head_k = int(preserve_head_k)
        self.rerank_slots = int(rerank_slots)
        self.min_score = float(min_score)

    def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        response = self.backend.search(request)
        if (
            not self._QUERY.search(request.query)
            or len(response.hits) <= self.preserve_head_k
        ):
            return response
        head = response.hits[: self.preserve_head_k]
        head_parent_ids = {
            _QueryAwareLineageClosureIndex._parent_id(hit.chunk) for hit in head
        }
        eligible = [
            hit
            for hit in response.hits[self.preserve_head_k :]
            if "same_parent_evidence_closure" in hit.retrieval_sources
            and hit.score >= self.min_score
            and _QueryAwareLineageClosureIndex._parent_id(hit.chunk)
            in head_parent_ids
        ]
        eligible.sort(key=lambda hit: (-hit.score, hit.rank, hit.chunk.chunk_id))
        promoted = eligible[: self.rerank_slots]
        if not promoted:
            return response.model_copy(
                update={
                    "raw_candidate_counts": {
                        **response.raw_candidate_counts,
                        "structured_lineage_pair_rerank": 0,
                    }
                }
            )
        promoted_ids = {hit.chunk.chunk_id for hit in promoted}
        marked = [
            hit.model_copy(
                update={
                    "retrieval_sources": [
                        *hit.retrieval_sources,
                        "structured_lineage_pair_rerank",
                    ]
                }
            )
            for hit in promoted
        ]
        retained = [
            hit for hit in response.hits if hit.chunk.chunk_id not in promoted_ids
        ]
        insertion = self.preserve_head_k - len(marked)
        reordered = [*retained[:insertion], *marked, *retained[insertion:]]
        return response.model_copy(
            update={
                "hits": [
                    hit.model_copy(update={"rank": rank})
                    for rank, hit in enumerate(reordered, start=1)
                ],
                "raw_candidate_counts": {
                    **response.raw_candidate_counts,
                    "structured_lineage_pair_rerank": len(marked),
                },
            }
        )


class _ConditionalIndex:
    def __init__(
        self,
        default_backend: Any,
        conditional_backend: Any,
        predicate: Callable[[str], bool],
        *,
        force_rerank_for_conditional: bool = False,
    ) -> None:
        self.default_backend = default_backend
        self.conditional_backend = conditional_backend
        self.predicate = predicate
        self.force_rerank_for_conditional = force_rerank_for_conditional

    def search(self, request: HybridSearchRequest) -> Any:
        conditional = self.predicate(request.query)
        backend = (
            self.conditional_backend
            if conditional
            else self.default_backend
        )
        if self.force_rerank_for_conditional:
            request = request.model_copy(update={"rerank": conditional})
        response = backend.search(request)
        if conditional:
            return response
        return response.model_copy(
            update={
                "backend": "multi_representation_rrf",
                "raw_candidate_counts": {
                    "query_plan_default": len(response.hits),
                },
            }
        )


class _ProfileConditionalIndex:
    """Route a stage by query/corpus profile without using dataset names.

    The wrapper keeps one stable stage backend label for the ablation manifest,
    while the hit-level retrieval sources and raw branch counts disclose which
    backend handled each case.  This makes an isolated profile change auditable
    and prevents a cross-document optimization from silently rewriting
    general-text results.
    """

    def __init__(
        self,
        default_backend: Any,
        profile_backend: Any,
        corpus: Any,
        profile_name: str,
    ) -> None:
        self.default_backend = default_backend
        self.profile_backend = profile_backend
        self.corpus = corpus
        self.profile_name = profile_name

    def search(self, request: HybridSearchRequest) -> Any:
        selected = select_retrieval_profile(request.query, self.corpus)
        backend = (
            self.profile_backend
            if selected == self.profile_name
            else self.default_backend
        )
        response = backend.search(request)
        return response.model_copy(update={"backend": "profile_routed"})


def _build_reranker(config: ExperimentRetrievalConfig) -> tuple[Any, Mapping[str, Any]]:
    if config.domain_reranker_path:
        path = Path(config.domain_reranker_path)
        if not path.is_file():
            raise FileNotFoundError(f"domain reranker artifact is missing: {path}")
        try:
            reranker = TATQADomainReranker.load(path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"domain reranker artifact is invalid: {path}") from exc
        return reranker, {
            "kind": "tatqa_domain_linear",
            "model": reranker.model_id,
            "artifact_path": str(path),
            "learned": True,
            "production": False,
        }
    if config.learned_reranker:
        reranker = FastEmbedCrossEncoderReranker(
            config.reranker_model,
            batch_size=config.reranker_batch_size,
        )
        return reranker, {
            "kind": "fastembed_cross_encoder",
            "model": config.reranker_model,
            "learned": True,
            "production": True,
        }
    return LexicalOverlapFallbackReranker(), {
        "kind": "lexical_overlap_fallback",
        "model": None,
        "learned": False,
        "production": False,
    }


def _run_stages(
    prepared: _PreparedDataset,
    config: RAGExperimentConfig,
    *,
    timer_ns: Callable[[], int],
    repository_root: Path | None = None,
    response_observer: Callable[
        [str, RAGEvalCase, HybridSearchResponse | None, float], None
    ]
    | None = None,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    _validate_evidence(prepared.dataset, prepared.cases)
    chunks = _hybrid_chunks(prepared.dataset, prepared.cases, config)
    profile_corpus = corpus_metadata(prepared.dataset.documents)
    chunk_count = sum(
        1
        for chunk in chunks
        if not chunk.chunk_id.startswith(_PROBE_PREFIX)
    )
    lexical = BM25Index(
        chunks,
        k1=config.retrieval.bm25_k1,
        b=config.retrieval.bm25_b,
        field_weights=config.retrieval.bm25_field_weights,
    )
    stages = list(config.retrieval.stages)
    if config.retrieval.graph_fusion:
        stages.append("graph_fused")
    if config.retrieval.graph_feature_rerank:
        if "graph_feature_rerank" not in stages:
            stages.append("graph_feature_rerank")
    qasper_stages = _QASPER_HIERARCHICAL_STAGES.intersection(stages)
    if qasper_stages and config.dataset.kind != "qasper_locked":
        raise ValueError(
            "QASPER hierarchical stages are only valid for the QASPER dataset"
        )
    child_vector_stages = {
        "bm25_dense_table_row_cell_rrf",
        "bm25_dense_rrf_coverage",
        "bm25_dense_max_coverage",
        "bm25_dense_tatqa_query_rrf",
        "bm25_dense_tatqa_query_context_rrf",
        "bm25_dense_tatqa_query_context_query_weighted_rrf",
        "bm25_dense_tatqa_query_context_dense_weighted_rrf",
        "bm25_dense_tatqa_query_table_candidate_rrf",
        "bm25_dense_tatqa_query_plan_parent_scan_rrf",
        _TATQA_DENSE_CANDIDATE_UNION_STAGE,
        "bm25_dense_tatqa_query_context_rerank",
        "bm25_dense_tatqa_dual_query_rrf",
        "bm25_dense_tatqa_dual_query_context_rrf",
        "bm25_dense_tatqa_table_rrf",
        "bm25_dense_tatqa_table_context_rrf",
        "bm25_dense_tatqa_query_feature_rerank",
        "qdrant_dense",
        "bm25_dense_rrf",
        "bm25_dense_table_profile_rrf",
        "bm25_dense_rrf_rerank",
        "qdrant_rrf",
        "qdrant_rrf_rerank",
    }
    qasper_vector_stages = {
        "qdrant_qasper_dense",
        "qdrant_qasper_dense_rerank",
        "bm25_dense_qasper_candidate_union",
        "bm25_dense_qasper_section_parent",
        "bm25_dense_qasper_section_parent_rrf",
    }
    parent_vector_stages = {
        "bm25_dense_parent_child",
        "bm25_dense_parent_child_rrf",
    }
    needs_child_vector = any(stage in child_vector_stages for stage in stages)
    needs_qasper_vector = bool(qasper_vector_stages.intersection(stages))
    needs_parent_vector = any(stage in parent_vector_stages for stage in stages)
    needs_vector_index = needs_child_vector or needs_parent_vector or needs_qasper_vector
    rerank_stages = {
        "lexical_bm25_rerank",
        "bm25_dense_rrf_rerank",
        "qdrant_rrf_rerank",
        "bm25_dense_tatqa_query_feature_rerank",
        "bm25_dense_tatqa_query_context_rerank",
        "bm25_tatqa_query_plan_parent_scan_closure_table_profile_cross_encoder_rerank",
        "qdrant_qasper_dense_rerank",
    }
    if config.retrieval.semantic_embedding:
        mode_label = "semantic_dense"
        production_dense = True
        dense_embedding_desc: dict[str, Any] = {
            "kind": "fastembed_bge",
            "model": config.retrieval.semantic_model,
            "semantic": True,
            "production": True,
        }
    else:
        mode_label = (
            "learned_sparse"
            if config.retrieval.learned_sparse
            else EXPERIMENT_MODE
        )
        production_dense = False
        dense_embedding_desc = {
            "kind": "deterministic_hash",
            "dimension": config.retrieval.hash_dimension,
            "semantic": False,
            "production": False,
        }
    reranker_desc: Mapping[str, Any] = {
        "kind": "none",
        "model": None,
        "learned": False,
        "production": False,
    }
    parent_stage_names = {
        "bm25_parent_child",
        "bm25_dense_parent_child",
        "bm25_dense_parent_child_rrf",
        "bm25_tatqa_query_plan_parent_scan_rrf",
        "bm25_dense_tatqa_query_plan_parent_scan_rrf",
        "bm25_tatqa_query_plan_parent_scan_feature_rerank",
        "bm25_tatqa_query_plan_parent_scan_closure_rrf",
        "bm25_tatqa_query_plan_parent_scan_closure_all_profile_rrf",
        "bm25_tatqa_query_plan_parent_scan_closure_table_profile_rrf",
        "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_rrf",
        "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_structured_rrf",
        "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_pair_rerank_rrf",
        _TATQA_DENSE_CANDIDATE_UNION_STAGE,
        _TATQA_COMPACT_PARENT_PAIR_STAGE,
        _TATQA_PASSAGE_PARENT_PAIR_STAGE,
        "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_splade_rrf",
        "bm25_tatqa_query_plan_parent_scan_closure_table_profile_feature_rerank",
        "bm25_tatqa_query_plan_parent_scan_closure_table_profile_feature_rerank_blend05",
        "bm25_tatqa_query_plan_parent_scan_closure_table_profile_domain_rerank",
        "bm25_tatqa_query_plan_parent_scan_closure_table_profile_cross_encoder_rerank",
        "bm25_tatqa_query_plan_parent_scan_closure_feature_rerank",
    }
    parent_chunks = (
        _parent_hybrid_chunks(chunks)
        if any(stage in parent_stage_names for stage in stages)
        else []
    )
    qasper_contextual_chunks = (
        _qasper_contextual_chunks(chunks) if qasper_stages else []
    )
    qasper_lexical: SearchRepresentationIndex | None = None
    qasper_dense: SearchRepresentationIndex | None = None
    qasper_section_parents = (
        _qasper_section_parent_chunks(chunks) if qasper_stages else []
    )
    qasper_section_lexical = (
        BM25Index(
            qasper_section_parents,
            k1=config.retrieval.bm25_k1,
            b=config.retrieval.bm25_b,
            field_weights=config.retrieval.bm25_field_weights,
        )
        if qasper_section_parents
        else None
    )
    if qasper_stages:
        qasper_lexical = SearchRepresentationIndex(
            BM25Index(
                qasper_contextual_chunks,
                k1=config.retrieval.bm25_k1,
                b=config.retrieval.bm25_b,
                field_weights=config.retrieval.bm25_field_weights,
            ),
            chunks,
            backend_label="qasper_hierarchical_bm25",
        )
    parent_lexical = (
        BM25Index(
            parent_chunks,
            k1=config.retrieval.bm25_k1,
            b=config.retrieval.bm25_b,
            field_weights=config.retrieval.bm25_field_weights,
        )
        if parent_chunks
        else None
    )
    compact_parent_lexical: BM25Index | None = None
    if _TATQA_COMPACT_PARENT_PAIR_STAGE in stages:
        compact_parent_chunks = _compact_tatqa_parent_hybrid_chunks(chunks)
        if not compact_parent_chunks:
            raise RuntimeError("compact TAT-QA parent stage has no safe parent records")
        compact_parent_lexical = BM25Index(
            compact_parent_chunks,
            k1=config.retrieval.bm25_k1,
            b=config.retrieval.bm25_b,
            field_weights=config.retrieval.bm25_field_weights,
        )
    indexes: dict[str, tuple[Any, bool]] = {"lexical_bm25": (lexical, False)}
    if qasper_lexical is not None:
        indexes["bm25_qasper_hierarchical"] = (qasper_lexical, False)
    structured_lineage_pair_backend: Any | None = None
    if "bm25_parent_child" in stages:
        if parent_lexical is None:
            raise RuntimeError("parent-child stage has no safe parent records")
        indexes["bm25_parent_child"] = (
            ParentChildIndex(
                parent_lexical,
                lexical,
                chunks,
                parent_top_k=config.retrieval.parent_top_k,
                include_parent_siblings=config.retrieval.parent_sibling_coverage,
            ),
            False,
        )
    if "bm25_multi_query_rrf" in stages:
        indexes["bm25_multi_query_rrf"] = (
            MultiQueryRRFIndex(
                lexical,
                _cross_document_subqueries,
                rrf_k=config.retrieval.rrf_k,
            ),
            False,
        )
    if "bm25_tatqa_query_rrf" in stages:
        indexes["bm25_tatqa_query_rrf"] = (
            MultiQueryRRFIndex(
                lexical,
                _tatqa_subqueries,
                rrf_k=config.retrieval.rrf_k,
            ),
            False,
        )
    if "bm25_tatqa_query_plan_rrf" in stages:
        plan_query = _QueryRewriteIndex(lexical, _tatqa_query_plan_query)
        numeric_config = config.model_copy(
            update={
                "retrieval": config.retrieval.model_copy(
                    update={"chunking": True, "table_aware_chunking": False}
                )
            }
        )
        numeric_chunks = _table_representation_chunks(
            chunks,
            prepared.dataset,
            numeric_config,
            representation="cell",
        )
        numeric_index = BM25Index(
            numeric_chunks,
            k1=config.retrieval.bm25_k1,
            b=config.retrieval.bm25_b,
            field_weights=config.retrieval.bm25_field_weights,
        )
        numeric_query = _QueryRewriteIndex(
            numeric_index,
            _tatqa_query_plan_query,
        )
        table_plan_rrf = RepresentationRRFIndex(
            [
                ("query_plan", plan_query, chunks),
                ("numeric_cell", numeric_query, numeric_chunks),
            ],
            rrf_k=config.retrieval.rrf_k,
            fusion="rrf",
            candidate_strategy="coverage",
            branch_weights={
                "query_plan": 1.0,
                "numeric_cell": config.retrieval.tatqa_numeric_cell_weight,
            },
        )
        indexes["bm25_tatqa_query_plan_rrf"] = (
            _ConditionalIndex(
                lexical,
                table_plan_rrf,
                lambda query: (
                    select_retrieval_profile(query, profile_corpus) == "table_numeric"
                    and _tatqa_requires_numeric_branch(query)
                ),
            ),
            False,
        )
    if {
        "bm25_tatqa_query_plan_scan_rrf",
        "bm25_tatqa_query_plan_scan_context_rrf",
        "bm25_tatqa_query_plan_parent_scan_rrf",
        "bm25_tatqa_query_plan_context_scan_rrf",
        "bm25_dense_tatqa_query_plan_parent_scan_rrf",
        "bm25_tatqa_query_plan_parent_scan_feature_rerank",
        "bm25_tatqa_query_plan_parent_scan_closure_rrf",
        "bm25_tatqa_query_plan_parent_scan_closure_all_profile_rrf",
        "bm25_tatqa_query_plan_parent_scan_closure_table_profile_rrf",
        "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_rrf",
        "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_structured_rrf",
        "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_pair_rerank_rrf",
        _TATQA_DENSE_CANDIDATE_UNION_STAGE,
        _TATQA_COMPACT_PARENT_PAIR_STAGE,
        _TATQA_PASSAGE_PARENT_PAIR_STAGE,
        "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_splade_rrf",
        "bm25_tatqa_query_plan_parent_scan_closure_table_profile_feature_rerank",
        "bm25_tatqa_query_plan_parent_scan_closure_table_profile_feature_rerank_blend05",
        "bm25_tatqa_query_plan_parent_scan_closure_table_profile_domain_rerank",
        "bm25_tatqa_query_plan_parent_scan_closure_table_profile_cross_encoder_rerank",
        "bm25_tatqa_query_plan_parent_scan_closure_feature_rerank",
    }.intersection(stages):
        plan_query = _QueryRewriteIndex(lexical, _tatqa_query_plan_query)
        numeric_config = config.model_copy(
            update={
                "retrieval": config.retrieval.model_copy(
                    update={"chunking": True, "table_aware_chunking": False}
                )
            }
        )
        numeric_chunks = _table_representation_chunks(
            chunks,
            prepared.dataset,
            numeric_config,
            representation="section",
        )
        numeric_index = BM25Index(
            numeric_chunks,
            k1=config.retrieval.bm25_k1,
            b=config.retrieval.bm25_b,
            field_weights=config.retrieval.bm25_field_weights,
        )
        numeric_query = _QueryRewriteIndex(
            numeric_index,
            _tatqa_query_plan_query,
        )
        numeric_scan = _NumericTableScanIndex(numeric_chunks)
        learned_sparse_index: FastEmbedSparseIndex | None = None
        learned_sparse_stage = (
            "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_splade_rrf"
        )
        if learned_sparse_stage in stages:
            if not config.retrieval.learned_sparse:
                raise ValueError(
                    "table-profile SPLADE stage requires --learned-sparse"
                )
            learned_sparse_index = FastEmbedSparseIndex(
                numeric_chunks,
                model_name=config.retrieval.sparse_model,
            )
        compact_plan_query = _QueryRewriteIndex(
            lexical,
            _tatqa_query_plan_compact_query,
        )
        def build_plan_scan_index(
            *,
            include_context: bool = False,
            include_parent: bool = False,
            with_context_expansion: bool = False,
            with_reranker: bool = False,
            include_numeric_cell: bool = True,
            include_query_plan: bool = True,
            compact_query: bool = False,
            include_learned_sparse: bool = False,
            rerank_blend_weight: float = 0.2,
            reranker_override: Any | None = None,
            parent_backend_override: Any | None = None,
            diverse_passage_backend: Any | None = None,
        ) -> RepresentationRRFIndex:
            branches: list[tuple[str, Any, Sequence[HybridChunk]]] = []
            if include_query_plan:
                branches.append(
                    (
                        "query_plan",
                        compact_plan_query if compact_query else plan_query,
                        chunks,
                    )
                )
            if include_numeric_cell:
                branches.append(("numeric_cell", numeric_query, numeric_chunks))
            branches.append(("numeric_scan", numeric_scan, numeric_chunks))
            reserves: dict[str, int] = {"numeric_scan": 25}
            if include_query_plan:
                reserves["query_plan"] = 20
            if include_numeric_cell:
                reserves["numeric_cell"] = 5
            weights: dict[str, float] = {
                "numeric_scan": config.retrieval.tatqa_numeric_scan_weight,
            }
            if include_query_plan:
                weights["query_plan"] = 1.0
            if include_numeric_cell:
                weights["numeric_cell"] = config.retrieval.tatqa_numeric_cell_weight
            if include_context:
                numeric_context = _NumericTableContextIndex(
                    numeric_scan,
                    numeric_chunks,
                    max_siblings=2,
                )
                branches.append(("numeric_context", numeric_context, numeric_chunks))
                reserves = {"numeric_scan": 20, "numeric_context": 10}
                if include_query_plan:
                    reserves["query_plan"] = 15
                if include_numeric_cell:
                    reserves["numeric_cell"] = 5
                weights["numeric_context"] = 0.5
            if include_parent:
                if parent_lexical is None and parent_backend_override is None:
                    raise RuntimeError("parent scan stage has no safe parent records")
                parent_backend: Any = parent_backend_override or parent_lexical
                if (
                    parent_backend_override is None
                    and config.retrieval.tatqa_parent_query_expansion
                ):
                    parent_backend = MultiQueryRRFIndex(
                        parent_lexical,
                        _tatqa_subqueries,
                        rrf_k=config.retrieval.rrf_k,
                    )
                parent_child_scan = ParentChildIndex(
                    parent_backend,
                    lexical,
                    chunks,
                    parent_top_k=config.retrieval.parent_top_k,
                    include_parent_siblings=config.retrieval.parent_sibling_coverage,
                )
                branches.append(("parent_child", parent_child_scan, chunks))
                reserves = {"numeric_scan": 10, "parent_child": 20}
                if include_query_plan:
                    reserves["query_plan"] = 15
                if include_numeric_cell:
                    reserves["numeric_cell"] = 5
                weights["parent_child"] = 0.5
                if diverse_passage_backend is not None:
                    branches.append(
                        ("parent_diverse_passage", diverse_passage_backend, chunks)
                    )
                    reserves["parent_child"] = 10
                    reserves["parent_diverse_passage"] = 10
                    weights["parent_diverse_passage"] = 0.5
            if include_learned_sparse:
                if learned_sparse_index is None:
                    raise RuntimeError("learned sparse index was not initialized")
                branches.append(
                    ("learned_sparse", learned_sparse_index, numeric_chunks)
                )
                reserves["learned_sparse"] = 5
                weights["learned_sparse"] = config.retrieval.tatqa_sparse_weight
            return RepresentationRRFIndex(
                branches,
                rrf_k=config.retrieval.rrf_k,
                fusion="rrf",
                candidate_strategy="coverage",
                context_sibling_coverage=with_context_expansion,
                context_seed_k=config.retrieval.context_seed_k,
                context_sibling_limit=5 if with_context_expansion else None,
                coverage_branch_reserves=reserves,
                branch_weights=weights,
                reranker=(
                    reranker_override
                    if reranker_override is not None
                    else TATQAFeatureReranker(blend_weight=rerank_blend_weight)
                    if with_reranker
                    else None
                ),
                rerank_top_k=config.retrieval.rerank_top_k,
            )
        def conditional_predicate(query: str) -> bool:
            return (
                select_retrieval_profile(query, profile_corpus) == "table_numeric"
                and _tatqa_requires_numeric_branch(query)
            )

        def table_profile_predicate(query: str) -> bool:
            return (
                select_retrieval_profile(query, profile_corpus) == "table_numeric"
                and _tatqa_should_route_structured(query)
            )
        if "bm25_tatqa_query_plan_scan_rrf" in stages:
            indexes["bm25_tatqa_query_plan_scan_rrf"] = (
                _ConditionalIndex(
                    lexical,
                    build_plan_scan_index(),
                    conditional_predicate,
                ),
                False,
            )
        if "bm25_tatqa_query_plan_context_scan_rrf" in stages:
            indexes["bm25_tatqa_query_plan_context_scan_rrf"] = (
                _ConditionalIndex(
                    lexical,
                    build_plan_scan_index(include_context=True),
                    conditional_predicate,
                ),
                False,
            )
        if "bm25_tatqa_query_plan_scan_context_rrf" in stages:
            indexes["bm25_tatqa_query_plan_scan_context_rrf"] = (
                _ConditionalIndex(
                    lexical,
                    build_plan_scan_index(with_context_expansion=True),
                    conditional_predicate,
                ),
                False,
            )
        if "bm25_tatqa_query_plan_parent_scan_rrf" in stages:
            indexes["bm25_tatqa_query_plan_parent_scan_rrf"] = (
                _ConditionalIndex(
                    lexical,
                    build_plan_scan_index(include_parent=True),
                    conditional_predicate,
                ),
                False,
            )
        if "bm25_tatqa_query_plan_parent_scan_feature_rerank" in stages:
            indexes["bm25_tatqa_query_plan_parent_scan_feature_rerank"] = (
                _ConditionalIndex(
                    lexical,
                    build_plan_scan_index(include_parent=True, with_reranker=True),
                    conditional_predicate,
                    force_rerank_for_conditional=True,
                ),
                True,
            )
        if "bm25_tatqa_query_plan_parent_scan_closure_rrf" in stages:
            indexes["bm25_tatqa_query_plan_parent_scan_closure_rrf"] = (
                _ConditionalIndex(
                    lexical,
                    build_plan_scan_index(
                        include_parent=True,
                        with_context_expansion=True,
                        include_numeric_cell=False,
                        compact_query=True,
                    ),
                    table_profile_predicate,
                ),
                False,
            )
        if "bm25_tatqa_query_plan_parent_scan_closure_all_profile_rrf" in stages:
            indexes["bm25_tatqa_query_plan_parent_scan_closure_all_profile_rrf"] = (
                _ConditionalIndex(
                    lexical,
                    build_plan_scan_index(
                        include_parent=True,
                        with_context_expansion=True,
                        include_numeric_cell=False,
                        compact_query=True,
                    ),
                    lambda query: select_retrieval_profile(query, profile_corpus)
                    == "table_numeric",
                ),
                False,
            )
        if "bm25_tatqa_query_plan_parent_scan_closure_table_profile_rrf" in stages:
            indexes["bm25_tatqa_query_plan_parent_scan_closure_table_profile_rrf"] = (
                _ConditionalIndex(
                    lexical,
                    build_plan_scan_index(
                        include_parent=True,
                        with_context_expansion=True,
                        include_numeric_cell=False,
                        compact_query=True,
                    ),
                    lambda query: select_retrieval_profile(query, profile_corpus)
                    == "table_numeric"
                    and _tatqa_should_route_table_profile_lookup(query),
                ),
                False,
            )
        if "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_rrf" in stages:
            lineage_backend = _QueryAwareLineageClosureIndex(
                build_plan_scan_index(
                    include_parent=True,
                    with_context_expansion=True,
                    include_numeric_cell=False,
                    compact_query=True,
                ),
                chunks,
                preserve_head_k=max(config.retrieval.top_k),
                seed_k=config.retrieval.tatqa_lineage_seed_k,
                closure_slots=config.retrieval.tatqa_lineage_closure_slots,
                max_siblings_per_parent=(
                    config.retrieval.tatqa_lineage_max_siblings_per_parent
                ),
            )
            indexes[
                "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_rrf"
            ] = (
                _ConditionalIndex(
                    lexical,
                    lineage_backend,
                    lambda query: select_retrieval_profile(query, profile_corpus)
                    == "table_numeric"
                    and _tatqa_should_route_table_profile_lookup(query),
                ),
                False,
            )
        structured_stage = (
            "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_structured_rrf"
        )
        lineage_pair_stage = (
            "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_pair_rerank_rrf"
        )
        if {
            structured_stage,
            lineage_pair_stage,
            _TATQA_DENSE_CANDIDATE_UNION_STAGE,
            _TATQA_COMPACT_PARENT_PAIR_STAGE,
            _TATQA_PASSAGE_PARENT_PAIR_STAGE,
        }.intersection(stages):
            structured_chunks = _structured_table_fact_chunks(
                chunks,
                prepared.dataset,
            )
            structured_index = _QueryTypedStructuredTableIndex(structured_chunks)
            structured_lineage_raw = _QueryAwareLineageClosureIndex(
                build_plan_scan_index(
                    include_parent=True,
                    with_context_expansion=True,
                    include_numeric_cell=False,
                    compact_query=True,
                ),
                chunks,
                preserve_head_k=max(config.retrieval.top_k),
                seed_k=config.retrieval.tatqa_lineage_seed_k,
                closure_slots=config.retrieval.tatqa_lineage_closure_slots,
                max_siblings_per_parent=(
                    config.retrieval.tatqa_lineage_max_siblings_per_parent
                ),
            )
            structured_lineage_base = _ConditionalIndex(
                lexical,
                structured_lineage_raw,
                lambda query: select_retrieval_profile(query, profile_corpus)
                == "table_numeric"
                and _tatqa_should_route_table_profile_lookup(query),
            )
            structured_lineage_backend = _StructuredLineageCandidateIndex(
                structured_lineage_base,
                structured_index,
                chunks,
                preserve_head_k=max(config.retrieval.top_k),
                candidate_slots=config.retrieval.tatqa_structured_candidate_slots,
                seed_k=min(10, config.retrieval.tatqa_lineage_seed_k),
                max_siblings_per_parent=(
                    config.retrieval.tatqa_lineage_max_siblings_per_parent
                ),
            )
            def structured_predicate(query: str) -> bool:
                return (
                    select_retrieval_profile(query, profile_corpus)
                    == "table_numeric"
                    and _tatqa_should_route_structured_candidate(query)
                )
            if structured_stage in stages:
                indexes[structured_stage] = (
                    _ConditionalIndex(
                        lexical,
                        structured_lineage_backend,
                        structured_predicate,
                    ),
                    False,
                )
            if {
                lineage_pair_stage,
                _TATQA_DENSE_CANDIDATE_UNION_STAGE,
            }.intersection(stages):
                structured_lineage_pair_backend = _ConditionalIndex(
                    lexical,
                    _StructuredLineagePairRerankIndex(
                        structured_lineage_backend,
                        preserve_head_k=max(config.retrieval.top_k),
                        rerank_slots=(
                            config.retrieval.tatqa_lineage_pair_rerank_slots
                        ),
                        min_score=config.retrieval.tatqa_lineage_pair_min_score,
                    ),
                    structured_predicate,
                )
            if lineage_pair_stage in stages:
                if structured_lineage_pair_backend is None:  # pragma: no cover
                    raise RuntimeError("structured lineage pair backend was not built")
                indexes[lineage_pair_stage] = (
                    structured_lineage_pair_backend,
                    False,
                )
            if _TATQA_COMPACT_PARENT_PAIR_STAGE in stages:
                if compact_parent_lexical is None:
                    raise RuntimeError("compact TAT-QA parent index was not initialized")
                compact_parent_backend = _QueryRewriteIndex(
                    compact_parent_lexical,
                    _tatqa_query_plan_compact_query,
                )
                compact_lineage_raw = _QueryAwareLineageClosureIndex(
                    build_plan_scan_index(
                        include_parent=True,
                        with_context_expansion=True,
                        include_numeric_cell=False,
                        compact_query=True,
                        parent_backend_override=compact_parent_backend,
                    ),
                    chunks,
                    preserve_head_k=max(config.retrieval.top_k),
                    seed_k=config.retrieval.tatqa_lineage_seed_k,
                    closure_slots=config.retrieval.tatqa_lineage_closure_slots,
                    max_siblings_per_parent=(
                        config.retrieval.tatqa_lineage_max_siblings_per_parent
                    ),
                )
                compact_lineage_base = _ConditionalIndex(
                    lexical,
                    compact_lineage_raw,
                    lambda query: select_retrieval_profile(query, profile_corpus)
                    == "table_numeric"
                    and _tatqa_should_route_table_profile_lookup(query),
                )
                compact_lineage_backend = _StructuredLineageCandidateIndex(
                    compact_lineage_base,
                    structured_index,
                    chunks,
                    preserve_head_k=max(config.retrieval.top_k),
                    candidate_slots=config.retrieval.tatqa_structured_candidate_slots,
                    seed_k=min(10, config.retrieval.tatqa_lineage_seed_k),
                    max_siblings_per_parent=(
                        config.retrieval.tatqa_lineage_max_siblings_per_parent
                    ),
                )
                indexes[_TATQA_COMPACT_PARENT_PAIR_STAGE] = (
                    _ConditionalIndex(
                        lexical,
                        _StructuredLineagePairRerankIndex(
                            compact_lineage_backend,
                            preserve_head_k=max(config.retrieval.top_k),
                            rerank_slots=(
                                config.retrieval.tatqa_lineage_pair_rerank_slots
                            ),
                            min_score=config.retrieval.tatqa_lineage_pair_min_score,
                        ),
                        structured_predicate,
                    ),
                    False,
                )
            if _TATQA_PASSAGE_PARENT_PAIR_STAGE in stages:
                diverse_passage_backend = _ParentDiversePassageIndex(
                    lexical,
                    query_rewriter=_tatqa_query_plan_compact_query,
                    probe_k=100,
                )
                passage_lineage_raw = _QueryAwareLineageClosureIndex(
                    build_plan_scan_index(
                        include_parent=True,
                        with_context_expansion=True,
                        include_numeric_cell=False,
                        compact_query=True,
                        diverse_passage_backend=diverse_passage_backend,
                    ),
                    chunks,
                    preserve_head_k=max(config.retrieval.top_k),
                    seed_k=config.retrieval.tatqa_lineage_seed_k,
                    closure_slots=config.retrieval.tatqa_lineage_closure_slots,
                    max_siblings_per_parent=(
                        config.retrieval.tatqa_lineage_max_siblings_per_parent
                    ),
                )
                passage_lineage_base = _ConditionalIndex(
                    lexical,
                    passage_lineage_raw,
                    lambda query: select_retrieval_profile(query, profile_corpus)
                    == "table_numeric"
                    and _tatqa_should_route_table_profile_lookup(query),
                )
                passage_lineage_backend = _StructuredLineageCandidateIndex(
                    passage_lineage_base,
                    structured_index,
                    chunks,
                    preserve_head_k=max(config.retrieval.top_k),
                    candidate_slots=config.retrieval.tatqa_structured_candidate_slots,
                    seed_k=min(10, config.retrieval.tatqa_lineage_seed_k),
                    max_siblings_per_parent=(
                        config.retrieval.tatqa_lineage_max_siblings_per_parent
                    ),
                )
                indexes[_TATQA_PASSAGE_PARENT_PAIR_STAGE] = (
                    _ConditionalIndex(
                        lexical,
                        _StructuredLineagePairRerankIndex(
                            passage_lineage_backend,
                            preserve_head_k=max(config.retrieval.top_k),
                            rerank_slots=(
                                config.retrieval.tatqa_lineage_pair_rerank_slots
                            ),
                            min_score=config.retrieval.tatqa_lineage_pair_min_score,
                        ),
                        structured_predicate,
                    ),
                    False,
                )
        if learned_sparse_stage in stages:
            sparse_lineage_backend = _QueryAwareLineageClosureIndex(
                build_plan_scan_index(
                    include_parent=True,
                    with_context_expansion=True,
                    include_numeric_cell=False,
                    compact_query=True,
                    include_learned_sparse=True,
                ),
                chunks,
                preserve_head_k=max(config.retrieval.top_k),
                seed_k=config.retrieval.tatqa_lineage_seed_k,
                closure_slots=config.retrieval.tatqa_lineage_closure_slots,
                max_siblings_per_parent=(
                    config.retrieval.tatqa_lineage_max_siblings_per_parent
                ),
            )
            indexes[learned_sparse_stage] = (
                _ConditionalIndex(
                    lexical,
                    sparse_lineage_backend,
                    lambda query: select_retrieval_profile(query, profile_corpus)
                    == "table_numeric"
                    and _tatqa_should_route_table_profile_lookup(query),
                ),
                False,
            )
        if "bm25_tatqa_query_plan_parent_scan_closure_table_profile_feature_rerank" in stages:
            indexes[
                "bm25_tatqa_query_plan_parent_scan_closure_table_profile_feature_rerank"
            ] = (
                _ConditionalIndex(
                    lexical,
                    build_plan_scan_index(
                        include_parent=True,
                        with_context_expansion=True,
                        with_reranker=True,
                        include_numeric_cell=False,
                        compact_query=True,
                    ),
                    lambda query: select_retrieval_profile(query, profile_corpus)
                    == "table_numeric"
                    and _tatqa_should_route_table_profile_lookup(query),
                    force_rerank_for_conditional=True,
                ),
                True,
            )
        if (
            "bm25_tatqa_query_plan_parent_scan_closure_table_profile_feature_rerank_blend05"
            in stages
        ):
            indexes[
                "bm25_tatqa_query_plan_parent_scan_closure_table_profile_feature_rerank_blend05"
            ] = (
                _ConditionalIndex(
                    lexical,
                    build_plan_scan_index(
                        include_parent=True,
                        with_context_expansion=True,
                        with_reranker=True,
                        include_numeric_cell=False,
                        compact_query=True,
                        rerank_blend_weight=0.05,
                    ),
                    lambda query: select_retrieval_profile(query, profile_corpus)
                    == "table_numeric"
                    and _tatqa_should_route_table_profile_lookup(query),
                    force_rerank_for_conditional=True,
                ),
                True,
            )
        if "bm25_tatqa_query_plan_parent_scan_closure_table_profile_domain_rerank" in stages:
            if not config.retrieval.domain_reranker_path:
                raise ValueError(
                    "table-profile domain rerank stage requires --domain-reranker"
                )
            domain_reranker, _ = _build_reranker(config.retrieval)
            indexes[
                "bm25_tatqa_query_plan_parent_scan_closure_table_profile_domain_rerank"
            ] = (
                _ConditionalIndex(
                    lexical,
                    build_plan_scan_index(
                        include_parent=True,
                        with_context_expansion=True,
                        with_reranker=True,
                        include_numeric_cell=False,
                        compact_query=True,
                        reranker_override=domain_reranker,
                    ),
                    lambda query: select_retrieval_profile(query, profile_corpus)
                    == "table_numeric"
                    and _tatqa_should_route_table_profile_lookup(query),
                    force_rerank_for_conditional=True,
                ),
                True,
            )
        if "bm25_tatqa_query_plan_parent_scan_closure_table_profile_cross_encoder_rerank" in stages:
            if not config.retrieval.learned_reranker:
                raise ValueError(
                    "table-profile cross-encoder stage requires --learned-reranker"
                )
            if config.retrieval.domain_reranker_path:
                raise ValueError(
                    "table-profile cross-encoder stage cannot use --domain-reranker"
                )
            cross_encoder_reranker, _ = _build_reranker(config.retrieval)
            indexes[
                "bm25_tatqa_query_plan_parent_scan_closure_table_profile_cross_encoder_rerank"
            ] = (
                _ConditionalIndex(
                    lexical,
                    build_plan_scan_index(
                        include_parent=True,
                        with_context_expansion=True,
                        with_reranker=True,
                        include_numeric_cell=False,
                        compact_query=True,
                        reranker_override=cross_encoder_reranker,
                    ),
                    lambda query: select_retrieval_profile(query, profile_corpus)
                    == "table_numeric"
                    and _tatqa_should_route_table_profile_lookup(query),
                    force_rerank_for_conditional=True,
                ),
                True,
            )
        if "bm25_tatqa_query_plan_parent_scan_closure_feature_rerank" in stages:
            indexes["bm25_tatqa_query_plan_parent_scan_closure_feature_rerank"] = (
                _ConditionalIndex(
                    lexical,
                    build_plan_scan_index(
                        include_parent=True,
                        with_context_expansion=True,
                        with_reranker=True,
                        include_numeric_cell=False,
                        compact_query=True,
                    ),
                    table_profile_predicate,
                    force_rerank_for_conditional=True,
                ),
                True,
            )
    if "bm25_source_coverage_rrf" in stages:
        source_coverage = SourceCoverageRRFIndex(
            lexical,
            chunks,
            rrf_k=config.retrieval.rrf_k,
        )
        indexes["bm25_source_coverage_rrf"] = (
            _ProfileConditionalIndex(
                lexical,
                source_coverage,
                profile_corpus,
                "cross_document",
            ),
            False,
        )
    if "bm25_source_coverage_anchor_rrf" in stages:
        source_coverage_anchor = SourceCoverageRRFIndex(
            lexical,
            chunks,
            rrf_k=config.retrieval.rrf_k,
            lexical_anchor_k=3,
        )
        indexes["bm25_source_coverage_anchor_rrf"] = (
            _ProfileConditionalIndex(
                lexical,
                source_coverage_anchor,
                profile_corpus,
                "cross_document",
            ),
            False,
        )
    if "bm25_table_router" in stages:
        generic_retrieval = config.retrieval.model_copy(
            update={"chunking": True, "table_aware_chunking": False}
        )
        table_retrieval = config.retrieval.model_copy(
            update={"chunking": True, "table_aware_chunking": True}
        )
        generic_chunks = _hybrid_chunks(
            prepared.dataset,
            prepared.cases,
            config.model_copy(update={"retrieval": generic_retrieval}),
        )
        table_chunks = _hybrid_chunks(
            prepared.dataset,
            prepared.cases,
            config.model_copy(update={"retrieval": table_retrieval}),
        )
        generic_bm25 = BM25Index(
            generic_chunks,
            k1=config.retrieval.bm25_k1,
            b=config.retrieval.bm25_b,
            field_weights=config.retrieval.bm25_field_weights,
        )
        table_bm25 = BM25Index(
            table_chunks,
            k1=config.retrieval.bm25_k1,
            b=config.retrieval.bm25_b,
            field_weights=config.retrieval.bm25_field_weights,
        )
        indexes["bm25_table_router"] = (
            _TableQueryRouter(generic_bm25, table_bm25),
            False,
        )
    if any(
        stage
        in {
            "bm25_table_multi_rep_rrf",
            "bm25_table_multi_rep_max",
            "bm25_table_multi_rep_adaptive",
        }
        for stage in stages
    ):
        generic_retrieval = config.retrieval.model_copy(
            update={"chunking": True, "table_aware_chunking": False}
        )
        table_retrieval = config.retrieval.model_copy(
            update={"chunking": True, "table_aware_chunking": True}
        )
        generic_chunks = _hybrid_chunks(
            prepared.dataset,
            prepared.cases,
            config.model_copy(update={"retrieval": generic_retrieval}),
        )
        table_chunks = _hybrid_chunks(
            prepared.dataset,
            prepared.cases,
            config.model_copy(update={"retrieval": table_retrieval}),
        )
        generic_bm25 = BM25Index(
            generic_chunks,
            k1=config.retrieval.bm25_k1,
            b=config.retrieval.bm25_b,
            field_weights=config.retrieval.bm25_field_weights,
        )
        table_bm25 = BM25Index(
            table_chunks,
            k1=config.retrieval.bm25_k1,
            b=config.retrieval.bm25_b,
            field_weights=config.retrieval.bm25_field_weights,
        )
        for representation_stage, fusion in (
            ("bm25_table_multi_rep_rrf", "rrf"),
            ("bm25_table_multi_rep_max", "max"),
        ):
            if representation_stage not in stages:
                continue
            indexes[representation_stage] = (
                RepresentationRRFIndex(
                    [
                        ("generic", generic_bm25, generic_chunks),
                        ("table_aware", table_bm25, table_chunks),
                    ],
                    rrf_k=config.retrieval.rrf_k,
                    fusion=fusion,
                ),
                False,
            )
        if "bm25_table_multi_rep_adaptive" in stages:
            indexes["bm25_table_multi_rep_adaptive"] = (
                _TableMultiRepresentationRouter(
                    generic_bm25,
                    RepresentationRRFIndex(
                        [
                            ("generic", generic_bm25, generic_chunks),
                            ("table_aware", table_bm25, table_chunks),
                        ],
                        rrf_k=config.retrieval.rrf_k,
                        fusion="max",
                    ),
                ),
                False,
            )
    if "bm25_table_row_cell_rrf" in stages:
        generic_retrieval = config.retrieval.model_copy(
            update={"chunking": True, "table_aware_chunking": False}
        )
        generic_config = config.model_copy(update={"retrieval": generic_retrieval})
        full_chunks = _hybrid_chunks(
            prepared.dataset,
            prepared.cases,
            generic_config,
        )
        full_bm25 = BM25Index(
            full_chunks,
            k1=config.retrieval.bm25_k1,
            b=config.retrieval.bm25_b,
            field_weights=config.retrieval.bm25_field_weights,
        )
        branches: list[tuple[str, BM25Index, Sequence[HybridChunk]]] = [
            ("full", full_bm25, full_chunks)
        ]
        for representation in ("schema", "row", "cell"):
            branch_chunks = _table_representation_chunks(
                full_chunks,
                prepared.dataset,
                generic_config,
                representation=representation,
            )
            branch_index = BM25Index(
                branch_chunks,
                k1=config.retrieval.bm25_k1,
                b=config.retrieval.bm25_b,
                field_weights=config.retrieval.bm25_field_weights,
            )
            branches.append((representation, branch_index, branch_chunks))
        indexes["bm25_table_row_cell_rrf"] = (
            RepresentationRRFIndex(
                branches,
                rrf_k=config.retrieval.rrf_k,
                fusion="rrf",
            ),
            False,
        )
    if "lexical_bm25_rerank" in stages and not needs_vector_index:
        reranker, reranker_desc = _build_reranker(config.retrieval)
        lexical.reranker = reranker
        indexes["lexical_bm25_rerank"] = (lexical, True)
    if needs_vector_index:
        dense_embedder: DenseEmbedder
        if config.retrieval.semantic_embedding:
            embedding_repository = (
                repository_root.resolve()
                if repository_root is not None
                else Path(__file__).resolve().parents[2]
            )
            model_cache_id = hashlib.sha256(
                config.retrieval.semantic_model.encode("utf-8")
            ).hexdigest()[:16]
            cache_relative_path = (
                Path(".taskforge")
                / "eval-cache"
                / "embeddings"
                / f"{model_cache_id}.sqlite3"
            )
            dense_embedder = FastEmbedEmbedder(
                config.retrieval.semantic_model,
                cache_path=embedding_repository / cache_relative_path,
                batch_size=_SEMANTIC_EMBEDDING_BATCH_SIZE,
            )
            # Warm the locked query set once.  FastEmbed's query encoder is an
            # ONNX call; keeping those exact vectors in memory makes the
            # reported search latency a warm-query measurement rather than
            # charging one model start-up/IPC cost to every case.
            dense_embedder.warm_queries(case.query for case in prepared.cases)
            dense_embedding_desc["dimension"] = dense_embedder.dimension
            dense_embedding_desc["cache"] = {
                "kind": "sqlite_content_addressed",
                "schema": "embeddings_v1",
                "path": cache_relative_path.as_posix(),
                "model_key": model_cache_id,
                "embedding_kinds": ["document", "query"],
                "vector_format": "float32_little_endian",
                "batch_size": _SEMANTIC_EMBEDDING_BATCH_SIZE,
            }
        else:
            dense_embedder = DeterministicHashEmbedder(config.retrieval.hash_dimension)
        reranker, reranker_desc = _build_reranker(config.retrieval)
        if "lexical_bm25_rerank" in stages:
            lexical.reranker = reranker
        qdrant: QdrantHybridIndex | None = None
        composite: BM25DenseRRFIndex | None = None
        if needs_child_vector:
            qdrant = QdrantHybridIndex.in_memory(
                collection_name="taskforge-rag-ablation",
                embedder=dense_embedder,
                reranker=reranker,
                embedding_metadata_fields=config.retrieval.bm25_field_weights,
                upsert_batch_size=_QDRANT_UPSERT_BATCH_SIZE,
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
            indexes.update(
                {
                    "qdrant_dense": (QdrantDenseIndex(qdrant), False),
                    "bm25_dense_rrf": (composite, False),
                    "bm25_dense_table_profile_rrf": (
                        _ProfileConditionalIndex(
                            lexical,
                            composite,
                            profile_corpus,
                            "table_numeric",
                        ),
                        False,
                    ),
                    "bm25_dense_rrf_rerank": (composite, True),
                    "qdrant_rrf": (qdrant, False),
                    "qdrant_rrf_rerank": (qdrant, True),
                }
            )
            if "bm25_dense_rrf_coverage" in stages:
                indexes["bm25_dense_rrf_coverage"] = (
                    RepresentationRRFIndex(
                        [
                            ("bm25", lexical, chunks),
                            ("dense", QdrantDenseIndex(qdrant), chunks),
                        ],
                        rrf_k=config.retrieval.rrf_k,
                        fusion="rrf",
                        candidate_strategy="coverage",
                    ),
                    False,
                )
            if "bm25_dense_max_coverage" in stages:
                indexes["bm25_dense_max_coverage"] = (
                    RepresentationRRFIndex(
                        [
                            ("bm25", lexical, chunks),
                            ("dense", QdrantDenseIndex(qdrant), chunks),
                        ],
                        rrf_k=config.retrieval.rrf_k,
                        fusion="max",
                        candidate_strategy="coverage",
                    ),
                    False,
                )
            if "bm25_dense_table_row_cell_rrf" in stages:
                branches: list[tuple[str, Any, Sequence[HybridChunk]]] = [
                    ("full_bm25", lexical, chunks),
                    ("dense", QdrantDenseIndex(qdrant), chunks),
                ]
                generic_config = config.model_copy(
                    update={
                        "retrieval": config.retrieval.model_copy(
                            update={
                                "chunking": True,
                                "table_aware_chunking": False,
                            }
                        )
                    }
                )
                for representation in ("schema", "row", "cell"):
                    branch_chunks = _table_representation_chunks(
                        chunks,
                        prepared.dataset,
                        generic_config,
                        representation=representation,
                    )
                    branch_index = BM25Index(
                        branch_chunks,
                        k1=config.retrieval.bm25_k1,
                        b=config.retrieval.bm25_b,
                        field_weights=config.retrieval.bm25_field_weights,
                    )
                    branches.append((representation, branch_index, branch_chunks))
                indexes["bm25_dense_table_row_cell_rrf"] = (
                    RepresentationRRFIndex(
                        branches,
                        rrf_k=config.retrieval.rrf_k,
                        fusion="rrf",
                        candidate_strategy="coverage",
                    ),
                    False,
                )
            if "bm25_dense_tatqa_query_rrf" in stages:
                tatqa_query = MultiQueryRRFIndex(
                    lexical,
                    _tatqa_subqueries,
                    rrf_k=config.retrieval.rrf_k,
                )
                indexes["bm25_dense_tatqa_query_rrf"] = (
                    RepresentationRRFIndex(
                        [
                            ("tatqa_query", tatqa_query, chunks),
                            ("dense", QdrantDenseIndex(qdrant), chunks),
                        ],
                        rrf_k=config.retrieval.rrf_k,
                        fusion="rrf",
                        candidate_strategy="coverage",
                    ),
                    False,
                )
            if {
                "bm25_dense_tatqa_query_context_rrf",
                "bm25_dense_tatqa_query_context_query_weighted_rrf",
                "bm25_dense_tatqa_query_context_dense_weighted_rrf",
            }.intersection(stages):
                tatqa_query = MultiQueryRRFIndex(
                    lexical,
                    _tatqa_subqueries,
                    rrf_k=config.retrieval.rrf_k,
                )
                context_specs = (
                    ("bm25_dense_tatqa_query_context_rrf", None),
                    (
                        "bm25_dense_tatqa_query_context_query_weighted_rrf",
                        {"tatqa_query": 2.0, "dense": 1.0},
                    ),
                    (
                        "bm25_dense_tatqa_query_context_dense_weighted_rrf",
                        {"tatqa_query": 1.0, "dense": 2.0},
                    ),
                )
                for stage_name, branch_weights in context_specs:
                    if stage_name not in stages:
                        continue
                    indexes[stage_name] = (
                        RepresentationRRFIndex(
                            [
                                ("tatqa_query", tatqa_query, chunks),
                                ("dense", QdrantDenseIndex(qdrant), chunks),
                            ],
                            rrf_k=config.retrieval.rrf_k,
                            fusion="rrf",
                            candidate_strategy="coverage",
                            context_sibling_coverage=True,
                            branch_weights=branch_weights,
                        ),
                        False,
                    )
            if "bm25_dense_tatqa_query_context_rerank" in stages:
                tatqa_query = MultiQueryRRFIndex(
                    lexical,
                    _tatqa_subqueries,
                    rrf_k=config.retrieval.rrf_k,
                )
                indexes["bm25_dense_tatqa_query_context_rerank"] = (
                    RepresentationRRFIndex(
                        [
                            ("tatqa_query", tatqa_query, chunks),
                            ("dense", QdrantDenseIndex(qdrant), chunks),
                        ],
                        rrf_k=config.retrieval.rrf_k,
                        fusion="rrf",
                        candidate_strategy="coverage",
                        context_sibling_coverage=True,
                        reranker=reranker,
                        rerank_top_k=config.retrieval.rerank_top_k,
                    ),
                    True,
                )
            if "bm25_dense_tatqa_query_table_candidate_rrf" in stages:
                tatqa_query = MultiQueryRRFIndex(
                    lexical,
                    _tatqa_subqueries,
                    rrf_k=config.retrieval.rrf_k,
                )
                generic_config = config.model_copy(
                    update={
                        "retrieval": config.retrieval.model_copy(
                            update={"chunking": True, "table_aware_chunking": False}
                        )
                    }
                )
                table_branches: list[tuple[str, Any, Sequence[HybridChunk]]] = [
                    ("tatqa_query", tatqa_query, chunks),
                    ("dense", QdrantDenseIndex(qdrant), chunks),
                ]
                for representation in ("schema", "row", "cell"):
                    branch_chunks = _table_representation_chunks(
                        chunks,
                        prepared.dataset,
                        generic_config,
                        representation=representation,
                    )
                    branch_index = BM25Index(
                        branch_chunks,
                        k1=config.retrieval.bm25_k1,
                        b=config.retrieval.bm25_b,
                        field_weights=config.retrieval.bm25_field_weights,
                    )
                    table_branches.append((representation, branch_index, branch_chunks))
                indexes["bm25_dense_tatqa_query_table_candidate_rrf"] = (
                    RepresentationRRFIndex(
                        table_branches,
                        rrf_k=config.retrieval.rrf_k,
                        fusion="rrf",
                        candidate_strategy="coverage",
                        context_sibling_coverage=True,
                        context_seed_k=config.retrieval.context_seed_k,
                        coverage_branch_reserves={
                            "cell": 20,
                            "row": 20,
                            "schema": 20,
                        },
                        branch_weights={
                            "tatqa_query": 1.0,
                            "dense": 1.0,
                            "schema": 0.01,
                            "row": 0.01,
                            "cell": 0.01,
                        },
                    ),
                    False,
                )
            if "bm25_dense_tatqa_query_plan_parent_scan_rrf" in stages:
                if parent_lexical is None:
                    raise RuntimeError(
                        "semantic parent scan stage has no safe parent records"
                    )
                parent_child_scan = ParentChildIndex(
                    parent_lexical,
                    lexical,
                    chunks,
                    parent_top_k=config.retrieval.parent_top_k,
                    include_parent_siblings=config.retrieval.parent_sibling_coverage,
                )
                structured_dense = RepresentationRRFIndex(
                    [
                        ("query_plan", plan_query, chunks),
                        ("numeric_cell", numeric_query, numeric_chunks),
                        ("numeric_scan", numeric_scan, numeric_chunks),
                        ("parent_child", parent_child_scan, chunks),
                        ("dense", QdrantDenseIndex(qdrant), chunks),
                    ],
                    rrf_k=config.retrieval.rrf_k,
                    fusion="rrf",
                    candidate_strategy="coverage",
                    coverage_branch_reserves={
                        "query_plan": 10,
                        "numeric_cell": 5,
                        "numeric_scan": 10,
                        "parent_child": 10,
                        "dense": 15,
                    },
                    branch_weights={
                        "query_plan": 1.0,
                        "numeric_cell": config.retrieval.tatqa_numeric_cell_weight,
                        "numeric_scan": config.retrieval.tatqa_numeric_scan_weight,
                        "parent_child": 0.5,
                        "dense": config.retrieval.rrf_dense_weight,
                    },
                )
                indexes["bm25_dense_tatqa_query_plan_parent_scan_rrf"] = (
                    _ConditionalIndex(
                        lexical,
                        structured_dense,
                        conditional_predicate,
                    ),
                    False,
                )
            if _TATQA_DENSE_CANDIDATE_UNION_STAGE in stages:
                if structured_lineage_pair_backend is None:
                    raise RuntimeError(
                        "semantic candidate union has no structured lineage primary backend"
                    )
                indexes[_TATQA_DENSE_CANDIDATE_UNION_STAGE] = (
                    CandidateTailUnionIndex(
                        structured_lineage_pair_backend,
                        QdrantDenseIndex(qdrant),
                        preserve_head_k=max(config.retrieval.top_k),
                        candidate_slots=_SEMANTIC_DENSE_CANDIDATE_SLOTS,
                    ),
                    False,
                )
            if "bm25_dense_tatqa_dual_query_rrf" in stages:
                # Run the same deterministic finance-focused query family
                # against both lexical and dense branches.  This is a
                # candidate-generation ablation: it deliberately preserves
                # the full Top-50 union before the final RRF head is chosen.
                tatqa_query = MultiQueryRRFIndex(
                    lexical,
                    _tatqa_subqueries,
                    rrf_k=config.retrieval.rrf_k,
                )
                tatqa_dense_query = MultiQueryRRFIndex(
                    QdrantDenseIndex(qdrant),
                    _tatqa_subqueries,
                    rrf_k=config.retrieval.rrf_k,
                )
                indexes["bm25_dense_tatqa_dual_query_rrf"] = (
                    RepresentationRRFIndex(
                        [
                            ("tatqa_query", tatqa_query, chunks),
                            ("tatqa_dense_query", tatqa_dense_query, chunks),
                        ],
                        rrf_k=config.retrieval.rrf_k,
                        fusion="rrf",
                        candidate_strategy="coverage",
                    ),
                    False,
                )
            if "bm25_dense_tatqa_dual_query_context_rrf" in stages:
                tatqa_query = MultiQueryRRFIndex(
                    lexical,
                    _tatqa_subqueries,
                    rrf_k=config.retrieval.rrf_k,
                )
                tatqa_dense_query = MultiQueryRRFIndex(
                    QdrantDenseIndex(qdrant),
                    _tatqa_subqueries,
                    rrf_k=config.retrieval.rrf_k,
                )
                indexes["bm25_dense_tatqa_dual_query_context_rrf"] = (
                    RepresentationRRFIndex(
                        [
                            ("tatqa_query", tatqa_query, chunks),
                            ("tatqa_dense_query", tatqa_dense_query, chunks),
                        ],
                        rrf_k=config.retrieval.rrf_k,
                        fusion="rrf",
                        candidate_strategy="coverage",
                        context_sibling_coverage=True,
                    ),
                    False,
                )
            if "bm25_dense_tatqa_query_feature_rerank" in stages:
                tatqa_query = MultiQueryRRFIndex(
                    lexical,
                    _tatqa_subqueries,
                    rrf_k=config.retrieval.rrf_k,
                )
                indexes["bm25_dense_tatqa_query_feature_rerank"] = (
                    RepresentationRRFIndex(
                        [
                            ("tatqa_query", tatqa_query, chunks),
                            ("dense", QdrantDenseIndex(qdrant), chunks),
                        ],
                        rrf_k=config.retrieval.rrf_k,
                        fusion="rrf",
                        candidate_strategy="coverage",
                        reranker=TATQAFeatureReranker(),
                    ),
                    True,
                )
            if {
                "bm25_dense_tatqa_table_rrf",
                "bm25_dense_tatqa_table_context_rrf",
            }.intersection(stages):
                tatqa_query = MultiQueryRRFIndex(
                    lexical,
                    _tatqa_subqueries,
                    rrf_k=config.retrieval.rrf_k,
                )
                generic_config = config.model_copy(
                    update={
                        "retrieval": config.retrieval.model_copy(
                            update={
                                "chunking": True,
                                "table_aware_chunking": False,
                            }
                        )
                    }
                )
                branches: list[tuple[str, Any, Sequence[HybridChunk]]] = [
                    ("tatqa_query", tatqa_query, chunks),
                    ("dense", QdrantDenseIndex(qdrant), chunks),
                ]
                for representation in ("row", "cell"):
                    branch_chunks = _table_representation_chunks(
                        chunks,
                        prepared.dataset,
                        generic_config,
                        representation=representation,
                    )
                    branch_index = BM25Index(
                        branch_chunks,
                        k1=config.retrieval.bm25_k1,
                        b=config.retrieval.bm25_b,
                        field_weights=config.retrieval.bm25_field_weights,
                    )
                    branches.append((representation, branch_index, branch_chunks))
                table_index = RepresentationRRFIndex(
                    branches,
                    rrf_k=config.retrieval.rrf_k,
                    fusion="rrf",
                    candidate_strategy="coverage",
                    branch_weights={
                        "tatqa_query": 2.0,
                        "dense": 1.0,
                        "row": 0.75,
                        "cell": 0.75,
                    },
                    context_sibling_coverage=(
                        "bm25_dense_tatqa_table_context_rrf" in stages
                    ),
                )
                if "bm25_dense_tatqa_table_rrf" in stages:
                    indexes["bm25_dense_tatqa_table_rrf"] = (table_index, False)
                if "bm25_dense_tatqa_table_context_rrf" in stages:
                    indexes["bm25_dense_tatqa_table_context_rrf"] = (table_index, False)
        if needs_qasper_vector:
            if qasper_lexical is None or qasper_section_lexical is None:
                raise RuntimeError("QASPER hierarchical stage has no safe parent records")
            qasper_dense = SearchRepresentationIndex(
                InMemoryDenseIndex(
                    qasper_contextual_chunks,
                    dense_embedder,
                    collection_name="taskforge-rag-qasper-hierarchical",
                    reranker=reranker,
                    rerank_top_k=config.retrieval.rerank_top_k,
                    adaptive_rerank_min_k=(
                        config.retrieval.adaptive_rerank_min_k
                        if config.retrieval.adaptive_rerank_enabled
                        else None
                    ),
                    adaptive_rerank_margin_threshold=(
                        config.retrieval.adaptive_rerank_margin_threshold
                    ),
                ),
                chunks,
                backend_label="qasper_hierarchical_in_memory_dense",
            )
            if "qdrant_qasper_dense" in stages:
                indexes["qdrant_qasper_dense"] = (qasper_dense, False)
            if "qdrant_qasper_dense_rerank" in stages:
                indexes["qdrant_qasper_dense_rerank"] = (qasper_dense, True)
            qasper_union = CandidateTailUnionIndex(
                qasper_lexical,
                qasper_dense,
                preserve_head_k=30,
                candidate_slots=20,
            )
            if "bm25_dense_qasper_candidate_union" in stages:
                indexes["bm25_dense_qasper_candidate_union"] = (
                    qasper_union,
                    False,
                )
            if "bm25_dense_qasper_section_parent" in stages:
                indexes["bm25_dense_qasper_section_parent"] = (
                    ParentChildIndex(
                        qasper_section_lexical,
                        qasper_union,
                        chunks,
                        parent_field="section_id",
                        parent_top_k=config.retrieval.parent_top_k,
                        include_parent_siblings=True,
                    ),
                    False,
                )
            if "bm25_dense_qasper_section_parent_rrf" in stages:
                qasper_rrf = RepresentationRRFIndex(
                    [
                        ("bm25", qasper_lexical, chunks),
                        ("dense", qasper_dense, chunks),
                    ],
                    rrf_k=config.retrieval.rrf_k,
                    fusion="rrf",
                    candidate_strategy="coverage",
                )
                indexes["bm25_dense_qasper_section_parent_rrf"] = (
                    ParentChildIndex(
                        qasper_section_lexical,
                        qasper_rrf,
                        chunks,
                        parent_field="section_id",
                        parent_top_k=config.retrieval.parent_top_k,
                        include_parent_siblings=True,
                    ),
                    False,
                )
        if needs_parent_vector:
            if parent_lexical is None:
                raise RuntimeError("parent semantic stage has no safe parent records")
            parent_qdrant = QdrantHybridIndex.in_memory(
                collection_name="taskforge-rag-parent-ablation",
                embedder=dense_embedder,
                reranker=reranker,
                embedding_metadata_fields=config.retrieval.bm25_field_weights,
            )
            parent_qdrant.upsert(parent_chunks)
            parent_composite = BM25DenseRRFIndex(
                parent_lexical,
                parent_qdrant,
                reranker=reranker,
                rrf_k=config.retrieval.rrf_k,
                bm25_weight=config.retrieval.rrf_bm25_weight,
                dense_weight=config.retrieval.rrf_dense_weight,
            )
            indexes["bm25_dense_parent_child"] = (
                ParentChildIndex(
                    parent_composite,
                    lexical,
                    chunks,
                    parent_top_k=config.retrieval.parent_top_k,
                    include_parent_siblings=config.retrieval.parent_sibling_coverage,
                ),
                False,
            )
            if composite is None:
                child_backend: Any = lexical
            else:
                child_backend = composite
            indexes["bm25_dense_parent_child_rrf"] = (
                ParentChildIndex(
                    parent_composite,
                    child_backend,
                    chunks,
                    parent_top_k=config.retrieval.parent_top_k,
                    include_parent_siblings=config.retrieval.parent_sibling_coverage,
                ),
                False,
            )
        if "lexical_bm25_rerank" in stages:
            indexes["lexical_bm25_rerank"] = (lexical, True)
    base_expected_filter = _expected_filter(config)
    contextual_scope = (
        config.dataset.kind == "tatqa_locked"
        and config.dataset.tatqa_context_mode == "provided_hybrid_context"
    ) or (
        config.dataset.kind == "qasper_locked"
        and config.dataset.qasper_context_mode == "provided_document_context"
    )
    stage_filter_contract: Mapping[str, Any] = (
        {
            **base_expected_filter,
            "parent_document_ids": ["$case.metadata.parent_document_id"],
        }
        if contextual_scope
        else base_expected_filter
    )
    graph = (
        LocalDocumentGraph(
            chunk
            for chunk in chunks
            if not chunk.chunk_id.startswith(_PROBE_PREFIX)
        )
        if config.retrieval.graph_fusion
        else None
    )
    evidence_graph = (
        LocalEvidenceGraph(
            chunk
            for chunk in chunks
            if not chunk.chunk_id.startswith(_PROBE_PREFIX)
        )
        if config.retrieval.graph_feature_rerank or "graph_feature_rerank" in stages
        else None
    )
    learned_graph_reranker: LearnedGraphReranker | None = None
    learned_graph_model_sha256: str | None = None
    if config.retrieval.graph_learned_reranker_path is not None:
        if repository_root is None:
            raise ValueError("learned graph reranker requires repository_root")
        model_path = (
            repository_root / config.retrieval.graph_learned_reranker_path
        ).resolve()
        if repository_root.resolve() not in model_path.parents:
            raise ValueError("learned graph reranker path escaped repository root")
        learned_graph_reranker = LearnedGraphReranker.load(model_path)
        learned_graph_model_sha256 = sha256_model(model_path)
    all_rows: list[Mapping[str, Any]] = []
    stage_metrics: dict[str, Any] = {}
    if "graph_feature_rerank" in stages:
        base_stage = config.retrieval.graph_rerank_base_stage
        if base_stage not in stages:
            raise ValueError(
                "graph_feature_rerank requires graph_rerank_base_stage in stages: "
                + base_stage
            )
        if base_stage not in indexes:
            raise ValueError(
                "graph_rerank_base_stage is not available for this dataset: "
                + base_stage
            )
    case_ids = [case.case_id for case in prepared.cases]
    response_cache: dict[tuple[str, str], HybridSearchResponse] = {}
    selected_profiles = {
        case.case_id: profile_metadata(
            select_retrieval_profile(case.query, profile_corpus),
            profile_corpus,
            query_features(case.query),
        )
        for case in prepared.cases
    }
    for stage in stages:
        predictions: list[RetrievalPrediction] = []
        durations_ms: list[float] = []
        rows: list[Mapping[str, Any]] = []
        retrieved_texts_by_case: dict[str, list[str]] = {}
        observed_backend: str | None = None
        rerank = stage in rerank_stages
        graph_rerank_base_stage = config.retrieval.graph_rerank_base_stage
        effective_stage = (
            graph_rerank_base_stage
            if stage == "graph_feature_rerank"
            else stage
        )
        index = indexes[effective_stage][0] if stage != "graph_fused" else None
        if stage == "graph_feature_rerank":
            rerank = effective_stage in rerank_stages
        for case in prepared.cases:
            parent_scope = _case_parent_scope(case, config)
            request = _search_request(
                case.query,
                config,
                rerank=rerank,
                parent_document_ids=parent_scope,
            )
            expected_filter = AppliedRetrievalFilters.from_request(
                request
            ).model_dump(mode="json")
            started = timer_ns()
            graph_result = None
            graph_query_plan: EvidenceQueryPlan | None = None
            graph_expanded_chunk_ids: tuple[str, ...] = ()
            graph_base_candidate_ids: frozenset[str] = frozenset()
            if stage == "graph_fused":
                if parent_scope is not None:
                    raise ValueError(
                        "graph_fused does not implement provided-context filtering"
                    )
                retrieved_ids, backend_label = _graph_fused_search(
                    graph, lexical, config, case.query
                )
                response = None
            else:
                if stage == "lexical_bm25" and config.retrieval.query_expansion:
                    expanded = _expand_with_prf(lexical, request)
                    if expanded is not None:
                        request = request.model_copy(update={"query": expanded})
                if stage == "graph_feature_rerank":
                    response = response_cache.get(
                        (effective_stage, case.case_id)
                    )
                    if response is None:
                        response = index.search(request)
                else:
                    response = index.search(request)
                    response_cache[(stage, case.case_id)] = response
                if stage == "graph_feature_rerank":
                    if evidence_graph is None:
                        raise RuntimeError("graph feature reranker is not initialized")
                    graph_base_candidate_ids = frozenset(
                        hit.chunk.chunk_id for hit in response.hits
                    )
                    if (
                        config.retrieval.graph_candidate_expansion
                        and config.retrieval.graph_expansion_slots > 0
                    ):
                        expanded = evidence_graph.expand_candidates(
                            response.hits,
                            max_add=config.retrieval.graph_expansion_slots,
                            hops=config.retrieval.graph_expansion_hops,
                            allowed_parent_document_ids=parent_scope,
                        )
                        if expanded:
                            graph_expanded_chunk_ids = tuple(
                                hit.chunk.chunk_id for hit in expanded
                            )
                            response = response.model_copy(
                                update={
                                    "hits": [*response.hits, *expanded],
                                    "expanded_neighbor_count": (
                                        response.expanded_neighbor_count + len(expanded)
                                    ),
                                }
                            )
                    if response.hits:
                        graph_query_plan = EvidenceQueryPlan.from_text(
                            case.query,
                            max_hops=config.retrieval.graph_expansion_hops,
                        )
                        graph_result = evidence_graph.rerank(
                            case.query,
                            response.hits,
                            query_plan=graph_query_plan,
                            seed_k=min(
                                config.retrieval.graph_rerank_seed_k,
                                len(response.hits),
                            ),
                            graph_weight=config.retrieval.graph_rerank_graph_weight,
                            entity_weight=config.retrieval.graph_rerank_entity_weight,
                            section_weight=config.retrieval.graph_rerank_section_weight,
                            adjacency_weight=config.retrieval.graph_rerank_adjacency_weight,
                            ppr_weight=config.retrieval.graph_rerank_ppr_weight,
                            allowed_parent_document_ids=parent_scope,
                        )
                        if learned_graph_reranker is not None:
                            graph_result = learned_graph_reranker.rerank(graph_result)
                        response = response.model_copy(
                            update={"hits": list(graph_result.hits)}
                        )
                    backend_label = (
                        "local_graph_learned_rerank"
                        if learned_graph_reranker is not None
                        else "local_graph_feature_rerank"
                    )
                else:
                    backend_label = response.backend
            ended = timer_ns()
            if ended < started:
                raise RuntimeError("experiment timer moved backwards")
            duration_ms = (ended - started) / 1_000_000.0
            durations_ms.append(duration_ms)
            if response is None:
                actual_filter = expected_filter
            else:
                actual_filter = response.filters_applied_before_ranking.model_dump(mode="json")
                if actual_filter != expected_filter:
                    raise RuntimeError("retrieval backend changed the trusted filter request")
            if observed_backend is None:
                observed_backend = backend_label
            elif observed_backend != backend_label:
                raise RuntimeError("a retrieval stage reported inconsistent backends")
            if response is not None:
                retrieved_ids = _deduped_document_ids(
                    response.hits, max_documents=config.retrieval.candidate_k
                )
                retrieved_parent_ids = _deduped_parent_ids(
                    response.hits, max_documents=config.retrieval.candidate_k
                )
                (
                    retrieved_row_ids,
                    retrieved_cell_ids,
                    retrieved_complete_table_ids,
                    retrieved_table_units_by_hit,
                ) = _retrieved_table_units(response.hits)
                retrieved_texts_by_case[case.case_id] = [
                    hit.chunk.text for hit in response.hits
                ]
            else:
                document_map = {
                    document.document_id: document
                    for document in prepared.dataset.documents
                }
                retrieved_parent_ids = []
                retrieved_row_ids = []
                retrieved_cell_ids = []
                retrieved_complete_table_ids = []
                retrieved_table_units_by_hit = []
                retrieved_texts_by_case[case.case_id] = []
                for document_id in retrieved_ids:
                    document = document_map.get(document_id)
                    if document is None:
                        continue
                    parent_id = str(
                        document.metadata.get("parent_document_id", document_id)
                    )
                    if parent_id not in retrieved_parent_ids:
                        retrieved_parent_ids.append(parent_id)
                    if document.metadata.get("table_complete") is True:
                        retrieved_complete_table_ids.append(document_id)
                    retrieved_texts_by_case[case.case_id].append(document.text)
            if any(value.startswith(_PROBE_PREFIX) for value in retrieved_ids):
                raise RuntimeError("an inaccessible filter probe entered a ranking")
            if response_observer is not None:
                response_observer(stage, case, response, duration_ms)
            predictions.append(
                RetrievalPrediction(
                    case_id=case.case_id,
                    retrieved_ids=retrieved_ids,
                    retrieved_parent_ids=retrieved_parent_ids,
                    retrieved_row_ids=retrieved_row_ids,
                    retrieved_cell_ids=retrieved_cell_ids,
                    retrieved_complete_table_ids=retrieved_complete_table_ids,
                    retrieved_table_units_by_hit=retrieved_table_units_by_hit,
                )
            )
            rows.append(
                {
                    "stage": stage,
                    "case_id": case.case_id,
                    "category": case.category,
                    "retrieval_profile": selected_profiles[case.case_id],
                    "query": case.query,
                    "search_query": request.query,
                    "latency_ms": duration_ms,
                    "relevant_ids": case.relevant_ids,
                    "retrieved_ids": retrieved_ids,
                    "retrieved_parent_ids": retrieved_parent_ids,
                    "retrieved_row_ids": retrieved_row_ids,
                    "retrieved_cell_ids": retrieved_cell_ids,
                    "retrieved_complete_table_ids": retrieved_complete_table_ids,
                    "retrieved_table_units_by_hit": retrieved_table_units_by_hit,
                    "raw_candidate_counts": (
                        dict(response.raw_candidate_counts)
                        if response is not None and response.raw_candidate_counts
                        else {
                            str(backend_label): len(retrieved_ids)
                            if response is None
                            else len(response.hits)
                        }
                    ),
                    "adaptive_rerank": (
                        response.adaptive_rerank.model_dump(mode="json")
                        if response is not None
                        and response.adaptive_rerank is not None
                        else None
                    ),
                    "scores": (
                        [hit.score for hit in response.hits]
                        if response is not None
                        else []
                    ),
                    "base_scores": (
                        [hit.base_score for hit in response.hits]
                        if response is not None
                        else []
                    ),
                    "reranker_scores": (
                        [hit.reranker_score for hit in response.hits]
                        if response is not None
                        else []
                    ),
                    "retrieval_sources": (
                        [hit.retrieval_sources for hit in response.hits]
                        if response is not None
                        else []
                    ),
                    "graph": (
                        {
                            "seed_chunk_ids": list(graph_result.seed_chunk_ids),
                            "node_count": graph_result.node_count,
                            "edge_count": graph_result.edge_count,
                            "candidate_set_preserved": {
                                hit.chunk.chunk_id for hit in graph_result.hits
                            }
                            == {hit.chunk.chunk_id for hit in response.hits}
                            if response is not None
                            else False,
                            "base_candidate_set_preserved": (
                                graph_base_candidate_ids
                                == {hit.chunk.chunk_id for hit in graph_result.hits}
                                if graph_result is not None
                                else False
                            ),
                            "expanded_chunk_ids": list(graph_expanded_chunk_ids),
                            "query_plan": (
                                {
                                    "raw_query": graph_query_plan.raw_query,
                                    "terms": sorted(graph_query_plan.terms),
                                    "entities": sorted(graph_query_plan.entities),
                                    "section_hints": sorted(graph_query_plan.section_hints),
                                    "max_hops": graph_query_plan.max_hops,
                                }
                                if graph_query_plan is not None
                                else None
                            ),
                            "learned_reranker": (
                                {
                                    "enabled": learned_graph_reranker is not None,
                                    "path": config.retrieval.graph_learned_reranker_path,
                                    "sha256": learned_graph_model_sha256,
                                }
                                if stage == "graph_feature_rerank"
                                else None
                            ),
                            "features": {
                                chunk_id: asdict(features)
                                for chunk_id, features in graph_result.features.items()
                            },
                        }
                        if graph_result is not None
                        else None
                    ),
                    "backend": backend_label,
                    "filter_request": actual_filter,
                    "filters_applied_before_ranking": True,
                    "experiment_mode": mode_label,
                    "production_semantic_dense": production_dense,
                }
            )
        report = evaluate_retrieval(
            prepared.cases,
            predictions,
            ks=sorted(
                set([*config.retrieval.top_k, config.retrieval.candidate_k])
            ),
        )
        hierarchical = evaluate_hierarchical_retrieval(
            prepared.cases,
            predictions,
            prepared.dataset.documents,
            retrieved_texts_by_case=retrieved_texts_by_case,
            ks=sorted(
                set(
                    [
                        *config.retrieval.top_k,
                        5,
                        20,
                        config.retrieval.candidate_k,
                    ]
                )
            ),
        )
        if [row["case_id"] for row in rows] != case_ids:
            raise RuntimeError("ablation stages did not use identical case ordering")
        raw_candidate_values: dict[str, list[int]] = {}
        for row in rows:
            for branch, count in row["raw_candidate_counts"].items():
                raw_candidate_values.setdefault(branch, []).append(int(count))
        adaptive_rows = [
            row for row in rows if row.get("adaptive_rerank") is not None
        ]
        adaptive_summary: Mapping[str, Any] = {
            "enabled": False,
            "cases": 0,
        }
        if adaptive_rows:
            escalated_rows = [
                row
                for row in adaptive_rows
                if bool(row["adaptive_rerank"]["escalated"])
            ]
            non_escalated_rows = [
                row
                for row in adaptive_rows
                if not bool(row["adaptive_rerank"]["escalated"])
            ]
            adaptive_summary = {
                "enabled": True,
                "cases": len(adaptive_rows),
                "min_k": config.retrieval.adaptive_rerank_min_k,
                "max_k": config.retrieval.rerank_top_k,
                "margin_threshold": (
                    config.retrieval.adaptive_rerank_margin_threshold
                ),
                "escalated_cases": len(escalated_rows),
                "escalation_rate": len(escalated_rows) / len(adaptive_rows),
                "mean_applied_k": sum(
                    int(row["adaptive_rerank"]["applied_k"])
                    for row in adaptive_rows
                )
                / len(adaptive_rows),
                "non_escalated_latency": _latency_summary(
                    [float(row["latency_ms"]) for row in non_escalated_rows]
                )
                if non_escalated_rows
                else None,
                "escalated_latency": _latency_summary(
                    [float(row["latency_ms"]) for row in escalated_rows]
                )
                if escalated_rows
                else None,
            }
        all_rows.extend(rows)
        if needs_vector_index and isinstance(reranker, FastEmbedCrossEncoderReranker):
            reranker_desc = {
                **reranker_desc,
                **reranker.telemetry(),
            }
        stage_metrics[stage] = {
            "backend": observed_backend,
            "filter_request": stage_filter_contract,
            "filters_applied_before_ranking": True,
            "experiment_mode": mode_label,
            "embedding": (
                {
                    "kind": "fastembed_splade_sparse",
                    "model": config.retrieval.sparse_model,
                    "semantic": True,
                    "sparse": True,
                    "production": True,
                }
                if stage
                == "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_splade_rrf"
                else
                {
                    "kind": "none",
                    "semantic": False,
                    "production": False,
                }
                if stage
                in {
                    "lexical_bm25",
                    "lexical_bm25_rerank",
                    "bm25_table_router",
                    "bm25_table_multi_rep_rrf",
                    "bm25_table_multi_rep_max",
                    "bm25_table_multi_rep_adaptive",
                    "bm25_table_row_cell_rrf",
                    "bm25_multi_query_rrf",
                    "bm25_tatqa_query_rrf",
                    "bm25_tatqa_query_plan_rrf",
                    "bm25_tatqa_query_plan_scan_rrf",
                    "bm25_tatqa_query_plan_scan_context_rrf",
                    "bm25_tatqa_query_plan_parent_scan_rrf",
                    "bm25_tatqa_query_plan_context_scan_rrf",
                    "bm25_tatqa_query_plan_parent_scan_feature_rerank",
                    "bm25_tatqa_query_plan_parent_scan_closure_rrf",
                    "bm25_tatqa_query_plan_parent_scan_closure_all_profile_rrf",
                    "bm25_tatqa_query_plan_parent_scan_closure_table_profile_rrf",
                    "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_rrf",
                    "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_structured_rrf",
                    "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_pair_rerank_rrf",
                    _TATQA_COMPACT_PARENT_PAIR_STAGE,
                    _TATQA_PASSAGE_PARENT_PAIR_STAGE,
                    "bm25_tatqa_query_plan_parent_scan_closure_table_profile_feature_rerank",
                    "bm25_tatqa_query_plan_parent_scan_closure_table_profile_feature_rerank_blend05",
                    "bm25_tatqa_query_plan_parent_scan_closure_table_profile_domain_rerank",
                    "bm25_tatqa_query_plan_parent_scan_closure_table_profile_cross_encoder_rerank",
                    "bm25_tatqa_query_plan_parent_scan_closure_feature_rerank",
                    "bm25_source_coverage_rrf",
                    "bm25_source_coverage_anchor_rrf",
                    "bm25_parent_child",
                    "bm25_qasper_hierarchical",
                    "graph_fused",
                }
                else dense_embedding_desc
            ),
            "reranker": (
                {
                    "kind": (
                        "learned_graph_pairwise_linear"
                        if learned_graph_reranker is not None
                        else "local_evidence_graph_features"
                    ),
                    "model": config.retrieval.graph_learned_reranker_path,
                    "model_sha256": learned_graph_model_sha256,
                    "learned": learned_graph_reranker is not None,
                    "production": False,
                    "base_stage": config.retrieval.graph_rerank_base_stage,
                }
                if stage == "graph_feature_rerank"
                else {
                    "kind": "tatqa_domain_linear",
                    "model": "taskforge-tatqa-linear-v1",
                    "learned": True,
                    "production": False,
                }
                if stage
                == "bm25_tatqa_query_plan_parent_scan_closure_table_profile_domain_rerank"
                else {
                    "kind": "fastembed_cross_encoder",
                    "model": config.retrieval.reranker_model,
                    "learned": True,
                    "production": True,
                }
                if stage
                == "bm25_tatqa_query_plan_parent_scan_closure_table_profile_cross_encoder_rerank"
                else {
                    "kind": "structured_lineage_pair_rule",
                    "model": None,
                    "learned": False,
                    "production": False,
                }
                if stage
                in {
                    "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_pair_rerank_rrf",
                    _TATQA_DENSE_CANDIDATE_UNION_STAGE,
                    _TATQA_COMPACT_PARENT_PAIR_STAGE,
                    _TATQA_PASSAGE_PARENT_PAIR_STAGE,
                }
                else {
                    "kind": "tatqa_feature_reranker",
                    "model": None,
                    "learned": False,
                    "production": False,
                }
                if stage
                in {
                    "bm25_dense_tatqa_query_feature_rerank",
                    "bm25_tatqa_query_plan_parent_scan_feature_rerank",
                    "bm25_tatqa_query_plan_parent_scan_closure_feature_rerank",
                    "bm25_tatqa_query_plan_parent_scan_closure_table_profile_feature_rerank",
                    "bm25_tatqa_query_plan_parent_scan_closure_table_profile_feature_rerank_blend05",
                }
                else reranker_desc
                if stage in rerank_stages
                else {
                    "kind": "none",
                    "model": None,
                    "learned": False,
                    "production": False,
                }
            ),
            "query_router": (
                "candidate_preserving_local_evidence_graph_rerank"
                if stage == "graph_feature_rerank"
                else
                "parent_context_then_child_retrieval"
                if stage
                in {
                    "bm25_parent_child",
                    "bm25_dense_parent_child",
                    "bm25_dense_parent_child_rrf",
                }
                else "explicit_count_phrase_to_table_aware_bm25"
                if stage == "bm25_table_router"
                else "generic_and_table_aware_rrf"
                if stage == "bm25_table_multi_rep_rrf"
                else "generic_and_table_aware_max_rank"
                if stage == "bm25_table_multi_rep_max"
                else "count_query_table_max_else_generic_bm25"
                if stage == "bm25_table_multi_rep_adaptive"
                else "full_schema_row_cell_paragraph_rrf"
                if stage == "bm25_table_row_cell_rrf"
                else "dense_full_schema_row_cell_paragraph_rrf"
                if stage == "bm25_dense_table_row_cell_rrf"
                else "dense_bm25_rrf_candidate_coverage"
                if stage == "bm25_dense_rrf_coverage"
                else "table_numeric_profile_dense_rrf"
                if stage == "bm25_dense_table_profile_rrf"
                else "dense_bm25_max_candidate_coverage"
                if stage == "bm25_dense_max_coverage"
                else "dense_tatqa_query_rrf_candidate_coverage"
                if stage == "bm25_dense_tatqa_query_rrf"
                else "tatqa_query_plan_numeric_rrf"
                if stage == "bm25_tatqa_query_plan_rrf"
                else "tatqa_query_plan_numeric_scan_rrf"
                if stage == "bm25_tatqa_query_plan_scan_rrf"
                else "tatqa_query_plan_numeric_scan_context_rrf"
                if stage == "bm25_tatqa_query_plan_scan_context_rrf"
                else "tatqa_query_plan_parent_numeric_scan_rrf"
                if stage == "bm25_tatqa_query_plan_parent_scan_rrf"
                else "tatqa_query_plan_parent_numeric_scan_feature_rerank"
                if stage == "bm25_tatqa_query_plan_parent_scan_feature_rerank"
                else "tatqa_query_plan_compact_parent_scan_evidence_closure_rrf"
                if stage == "bm25_tatqa_query_plan_parent_scan_closure_rrf"
                else "table_numeric_profile_compact_parent_scan_evidence_closure_rrf"
                if stage == "bm25_tatqa_query_plan_parent_scan_closure_all_profile_rrf"
                else "table_lookup_profile_compact_parent_scan_evidence_closure_rrf"
                if stage == "bm25_tatqa_query_plan_parent_scan_closure_table_profile_rrf"
                else "table_lookup_profile_query_aware_lineage_closure_rrf"
                if stage == "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_rrf"
                else "table_lookup_profile_query_typed_structured_lineage_candidates"
                if stage
                == "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_structured_rrf"
                else "table_lookup_profile_structured_lineage_pair_rerank"
                if stage
                == "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_pair_rerank_rrf"
                else "structured_lineage_pair_head_with_semantic_dense_candidate_tail"
                if stage == _TATQA_DENSE_CANDIDATE_UNION_STAGE
                else "table_lookup_profile_compact_parent_structured_lineage_pair_rerank"
                if stage == _TATQA_COMPACT_PARENT_PAIR_STAGE
                else "table_lookup_profile_passage_parent_structured_lineage_pair_rerank"
                if stage == _TATQA_PASSAGE_PARENT_PAIR_STAGE
                else "table_lookup_profile_lineage_splade_rrf"
                if stage == "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_splade_rrf"
                else "table_lookup_profile_compact_parent_scan_feature_rerank"
                if stage == "bm25_tatqa_query_plan_parent_scan_closure_table_profile_feature_rerank"
                else "table_lookup_profile_compact_parent_scan_feature_rerank_blend05"
                if stage == "bm25_tatqa_query_plan_parent_scan_closure_table_profile_feature_rerank_blend05"
                else "table_lookup_profile_compact_parent_scan_domain_rerank"
                if stage == "bm25_tatqa_query_plan_parent_scan_closure_table_profile_domain_rerank"
                else "table_lookup_profile_compact_parent_scan_cross_encoder_rerank"
                if stage == "bm25_tatqa_query_plan_parent_scan_closure_table_profile_cross_encoder_rerank"
                else "tatqa_query_plan_compact_parent_scan_feature_rerank"
                if stage == "bm25_tatqa_query_plan_parent_scan_closure_feature_rerank"
                else "dense_tatqa_query_plan_parent_numeric_scan_rrf"
                if stage == "bm25_dense_tatqa_query_plan_parent_scan_rrf"
                else "tatqa_query_plan_context_scan_rrf"
                if stage == "bm25_tatqa_query_plan_context_scan_rrf"
                else "dense_tatqa_query_rrf_context_candidate_coverage"
                if stage == "bm25_dense_tatqa_query_context_rrf"
                else "dense_tatqa_query_context_query_weighted_rrf"
                if stage == "bm25_dense_tatqa_query_context_query_weighted_rrf"
                else "dense_tatqa_query_context_dense_weighted_rrf"
                if stage == "bm25_dense_tatqa_query_context_dense_weighted_rrf"
                else "dense_tatqa_query_table_tail_candidate_coverage"
                if stage == "bm25_dense_tatqa_query_table_candidate_rrf"
                else "dense_tatqa_query_context_learned_rerank"
                if stage == "bm25_dense_tatqa_query_context_rerank"
                else "dense_and_lexical_tatqa_query_rrf_candidate_coverage"
                if stage == "bm25_dense_tatqa_dual_query_rrf"
                else "dense_and_lexical_tatqa_query_context_rrf_candidate_coverage"
                if stage == "bm25_dense_tatqa_dual_query_context_rrf"
                else "dense_tatqa_query_row_cell_weighted_rrf"
                if stage == "bm25_dense_tatqa_table_rrf"
                else "dense_tatqa_query_row_cell_context_weighted_rrf"
                if stage == "bm25_dense_tatqa_table_context_rrf"
                else "dense_tatqa_query_feature_rerank"
                if stage == "bm25_dense_tatqa_query_feature_rerank"
                else (
                    "tatqa_content_year_query_rrf"
                    if stage == "bm25_tatqa_query_rrf"
                    else "quoted_phrase_and_clause_multi_query_rrf"
                    if stage == "bm25_multi_query_rrf"
                    else (
                        "profile_routed_cross_document_source_coverage"
                        if stage == "bm25_source_coverage_rrf"
                        else "profile_routed_cross_document_anchor_source_coverage"
                        if stage == "bm25_source_coverage_anchor_rrf"
                        else "qasper_paper_title_section_contextual_bm25"
                        if stage == "bm25_qasper_hierarchical"
                        else "qasper_paper_title_section_dense"
                        if stage == "qdrant_qasper_dense"
                        else "qasper_paper_title_section_dense_cross_encoder_rerank"
                        if stage == "qdrant_qasper_dense_rerank"
                        else "qasper_bm25_dense_candidate_tail_union"
                        if stage == "bm25_dense_qasper_candidate_union"
                        else "qasper_section_parent_bm25_dense_candidate_union"
                        if stage == "bm25_dense_qasper_section_parent"
                        else "qasper_section_parent_bm25_dense_rrf"
                        if stage == "bm25_dense_qasper_section_parent_rrf"
                        else "none"
                    )
                )
            ),
            "latency": _latency_summary(durations_ms),
            "adaptive_rerank": adaptive_summary,
            "raw_candidate_counts": {
                branch: {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                }
                for branch, values in sorted(raw_candidate_values.items())
            },
            "graph": (
                {
                    "enabled": True,
                    "candidate_set_preserved": all(
                        bool(row["graph"]["candidate_set_preserved"])
                        for row in rows
                        if row["graph"] is not None
                    ),
                    "node_count": max(
                        (int(row["graph"]["node_count"]) for row in rows if row["graph"] is not None),
                        default=0,
                    ),
                    "edge_count": max(
                        (int(row["graph"]["edge_count"]) for row in rows if row["graph"] is not None),
                        default=0,
                    ),
                    "seed_k": config.retrieval.graph_rerank_seed_k,
                    "weights": {
                        "graph": config.retrieval.graph_rerank_graph_weight,
                        "entity": config.retrieval.graph_rerank_entity_weight,
                        "section": config.retrieval.graph_rerank_section_weight,
                        "adjacency": config.retrieval.graph_rerank_adjacency_weight,
                    },
                }
                if stage == "graph_feature_rerank"
                else None
            ),
            "candidate_recall": report.summary.recall_at_k[
                str(config.retrieval.candidate_k)
            ],
            "hierarchical": hierarchical,
            "retrieval": report.model_dump(mode="json"),
        }
    expected_row_count = len(prepared.cases) * len(stages)
    if len(all_rows) != expected_row_count:
        raise RuntimeError("experiment produced an incomplete ablation matrix")
    metrics = {
        "schema_version": "1.0",
        "experiment_mode": mode_label,
        "production_semantic_dense": production_dense,
        "embedding_cache": dense_embedding_desc.get("cache"),
        "retrieval_scope": {
            "mode": (
                config.dataset.qasper_context_mode
                if config.dataset.kind == "qasper_locked"
                else config.dataset.tatqa_context_mode
                if config.dataset.kind == "tatqa_locked"
                else "global_discovery"
            ),
            "source": (
                "case.metadata.parent_document_id"
                if contextual_scope
                else "global_knowledge_base"
            ),
            "derived_from_relevance_labels": False,
        },
        "case_ids": case_ids,
        "top_k": config.retrieval.top_k,
        "candidate_k": config.retrieval.candidate_k,
        "max_chunks_per_document": config.retrieval.max_chunks_per_document,
        "parent_routing": {
            "parent_top_k": config.retrieval.parent_top_k,
            "sibling_coverage": config.retrieval.parent_sibling_coverage,
            "query_expansion": config.retrieval.tatqa_parent_query_expansion,
        },
        "rerank_top_k": config.retrieval.rerank_top_k,
        "adaptive_rerank": {
            "enabled": config.retrieval.adaptive_rerank_enabled,
            "min_k": config.retrieval.adaptive_rerank_min_k,
            "max_k": config.retrieval.rerank_top_k,
            "margin": "top1_minus_top2_cross_encoder_score",
            "margin_threshold": config.retrieval.adaptive_rerank_margin_threshold,
            "selection_split": (
                "qasper-train-tuning-200-v1"
                if config.retrieval.adaptive_rerank_enabled
                else None
            ),
        },
        "context_seed_k": config.retrieval.context_seed_k,
        "same_case_ids_and_top_k": True,
        "chunking": {
            "enabled": config.retrieval.chunking,
            "table_aware": config.retrieval.table_aware_chunking,
            "max_chars": config.retrieval.chunk_max_chars,
            "overlap_chars": config.retrieval.chunk_overlap_chars,
        },
        "chunk_count": chunk_count,
        "query_expansion": config.retrieval.query_expansion,
        "bm25_field_weights": dict(config.retrieval.bm25_field_weights),
        "rrf": {
            "k": config.retrieval.rrf_k,
            "bm25_weight": config.retrieval.rrf_bm25_weight,
            "dense_weight": config.retrieval.rrf_dense_weight,
        },
            "graph_fusion": config.retrieval.graph_fusion,
            "graph_max_neighbors": config.retrieval.graph_max_neighbors,
            "graph_feature_rerank": "graph_feature_rerank" in stages,
            "graph_rerank_base_stage": config.retrieval.graph_rerank_base_stage,
            "graph_rerank_seed_k": config.retrieval.graph_rerank_seed_k,
            "graph_rerank_weights": {
                "graph": config.retrieval.graph_rerank_graph_weight,
                "entity": config.retrieval.graph_rerank_entity_weight,
                "section": config.retrieval.graph_rerank_section_weight,
                "adjacency": config.retrieval.graph_rerank_adjacency_weight,
                "ppr": config.retrieval.graph_rerank_ppr_weight,
            },
            "graph_candidate_expansion": {
                "enabled": config.retrieval.graph_candidate_expansion,
                "hops": config.retrieval.graph_expansion_hops,
                "slots": config.retrieval.graph_expansion_slots,
            },
            "graph_learned_reranker": {
                "enabled": learned_graph_reranker is not None,
                "path": config.retrieval.graph_learned_reranker_path,
                "sha256": learned_graph_model_sha256,
            },
            "candidate_expansion": {
                "enabled": config.retrieval.graph_candidate_expansion,
                "hops": config.retrieval.graph_expansion_hops,
                "slots": config.retrieval.graph_expansion_slots,
            },
            "retrieval_profiles": {
            "selection": "query_features+corpus_metadata",
            "corpus": {
                "document_count": profile_corpus.document_count,
                "table_count": profile_corpus.table_count,
                "page_count": profile_corpus.page_count,
                "source_count": profile_corpus.source_count,
                "has_page_coordinates": profile_corpus.has_page_coordinates,
                "has_table_structure": profile_corpus.has_table_structure,
            },
            "case_counts": dict(
                Counter(
                    str(value["name"])
                    for value in selected_profiles.values()
                )
            ),
        },
        "stages": stage_metrics,
    }
    return all_rows, metrics


def _source_hashes() -> Mapping[str, str]:
    root = Path(__file__).resolve().parent
    names = (
        "rag_experiment.py",
        "rag_evaluation.py",
        "rag_answer_eval.py",
        "rag_baseline.py",
        "hybrid_retrieval.py",
        "evidence_graph.py",
        "graph_reranker.py",
        "synthetic_pdf_eval.py",
    )
    hashes = {
        f"taskforge.{Path(name).stem}": sha256_file(root / name)
        for name in names
    }
    runner = root.parents[1] / "scripts" / "run_rag_experiment.py"
    if not runner.is_file():
        raise FileNotFoundError(f"RAG experiment CLI source is missing: {runner}")
    hashes["scripts.run_rag_experiment"] = sha256_file(runner)
    return hashes


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def run_rag_experiment(
    *,
    output_dir: str | Path,
    config: RAGExperimentConfig,
    repository_root: str | Path | None = None,
    config_source_path: str | Path | None = None,
    created_at: datetime | None = None,
    timer_ns: Callable[[], int] = time.perf_counter_ns,
) -> RAGExperimentResult:
    """Run all M1 stages and atomically publish evidence artifacts.

    The caller supplies no model provider or network endpoint.  Synthetic PDF
    generation, pypdf parsing, BM25, and qdrant-client local mode all execute
    in-process.  Existing output directories are never overwritten.
    """

    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"RAG experiment output already exists: {target}")
    repository = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    if not repository.is_dir():
        raise FileNotFoundError(f"repository root does not exist: {repository}")
    timestamp = created_at or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("created_at must be timezone-aware")
    timestamp = timestamp.astimezone(UTC)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        prepared = _prepare_dataset(config, repository, staging)
        rows, metrics = _run_stages(
            prepared,
            config,
            timer_ns=timer_ns,
            repository_root=repository,
        )
        predictions_payload = _jsonl_bytes(rows)
        metrics_payload = _canonical_json(metrics) + b"\n"
        effective_config = config.model_dump(mode="json")
        effective_config_hash = _sha256_bytes(_canonical_json(effective_config))
        config_source: Mapping[str, Any] | None = None
        if config_source_path is not None:
            source = Path(config_source_path).resolve(strict=True)
            config_source = {
                "path": source.name,
                "sha256": sha256_file(source),
                "size_bytes": source.stat().st_size,
            }
        source_hashes = _source_hashes()
        code_hash = _sha256_bytes(_canonical_json(source_hashes))
        pdf_hash = _sha256_bytes(_canonical_json(list(prepared.pdf_artifacts)))
        run_id = _sha256_bytes(
            "\0".join(
                (
                    prepared.provenance["normalized_sha256"],
                    effective_config_hash,
                    pdf_hash,
                    code_hash,
                )
            ).encode("ascii")
        )[:20]
        category_counts = Counter(case.category for case in prepared.cases)
        stage_manifest = {
            stage: {
                "backend": metrics["stages"][stage]["backend"],
                "filters_applied_before_ranking": True,
                "filter_request": metrics["stages"][stage]["filter_request"],
                "experiment_mode": metrics["experiment_mode"],
                "production_semantic_dense": metrics["production_semantic_dense"],
                "embedding": metrics["stages"][stage]["embedding"],
                "reranker": metrics["stages"][stage]["reranker"],
            }
            for stage in config.retrieval.stages
        }
        manifest: dict[str, Any] = {
            "schema_version": "1.0",
            "run_id": run_id,
            "created_at": timestamp.isoformat().replace("+00:00", "Z"),
            "experiment_mode": metrics["experiment_mode"],
            "production_semantic_dense": metrics["production_semantic_dense"],
            "limitations": [
                *(
                    ["development sweep is incomplete and cannot be promoted"]
                    if config.retrieval.development_sweep
                    else []
                ),
                *(
                    []
                    if metrics["production_semantic_dense"]
                    else [
                        "deterministic hash vectors are nonsemantic and not production dense embeddings"
                    ]
                ),
                *(
                    []
                    if config.retrieval.learned_reranker
                    else [
                        "the offline reranker is lexical overlap, not a learned cross-encoder"
                    ]
                ),
            ],
            "dataset": dict(prepared.provenance),
            "config": {
                "effective": effective_config,
                "sha256": effective_config_hash,
                "source": config_source,
            },
            "sample": {
                "case_ids": [case.case_id for case in prepared.cases],
                "selected_cases": len(prepared.cases),
                "category_counts": dict(sorted(category_counts.items())),
            },
            "ablation": {
                "stages": stage_manifest,
                "top_k": config.retrieval.top_k,
                "same_case_ids_and_top_k": True,
                "qdrant_location": ":memory:",
                "embedding_cache": metrics["embedding_cache"],
                "qdrant_upsert_batch_size": _QDRANT_UPSERT_BATCH_SIZE,
                "inaccessible_filter_probes": 2,
                "chunking": metrics["chunking"],
                "chunk_count": metrics["chunk_count"],
                "query_expansion": metrics["query_expansion"],
                "bm25_field_weights": metrics["bm25_field_weights"],
                "rrf": metrics["rrf"],
                "graph_fusion": metrics["graph_fusion"],
                "graph_max_neighbors": metrics["graph_max_neighbors"],
                "graph_feature_rerank": metrics["graph_feature_rerank"],
                "graph_rerank_base_stage": metrics["graph_rerank_base_stage"],
                "graph_rerank_seed_k": metrics["graph_rerank_seed_k"],
                "graph_rerank_weights": metrics["graph_rerank_weights"],
                "graph_candidate_expansion": metrics["graph_candidate_expansion"],
                "graph_learned_reranker": metrics["graph_learned_reranker"],
                "adaptive_rerank": metrics["adaptive_rerank"],
                "retrieval_profiles": metrics["retrieval_profiles"],
                "retrieval_scope": metrics["retrieval_scope"],
            },
            "code": {
                "package": "taskforge-agent",
                "package_version": __version__,
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "qdrant_client_version": _package_version("qdrant-client"),
                "source_sha256": source_hashes,
                "sha256": code_hash,
            },
            "pdf_artifacts": list(prepared.pdf_artifacts),
            "pdf_artifacts_sha256": pdf_hash,
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
            "artifact_hash_scope": (
                "predictions.jsonl, metrics.json, and generated source PDFs; "
                "manifest.json is the hash index and is intentionally not self-hashed"
            ),
        }
        manifest_payload = _canonical_json(manifest) + b"\n"
        _write_new(staging / "predictions.jsonl", predictions_payload)
        _write_new(staging / "metrics.json", metrics_payload)
        _write_new(staging / "manifest.json", manifest_payload)
        if target.exists():
            raise FileExistsError(f"RAG experiment output already exists: {target}")
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return RAGExperimentResult(
        output_dir=target,
        predictions_path=target / "predictions.jsonl",
        metrics_path=target / "metrics.json",
        manifest_path=target / "manifest.json",
        metrics=metrics,
        manifest=manifest,
    )


__all__ = [
    "EXPERIMENT_MODE",
    "REQUIRED_STAGES",
    "DatasetKind",
    "ExperimentDatasetConfig",
    "ExperimentFilterConfig",
    "ExperimentRetrievalConfig",
    "RAGExperimentConfig",
    "RAGExperimentResult",
    "StageName",
    "load_experiment_config",
    "run_rag_experiment",
]

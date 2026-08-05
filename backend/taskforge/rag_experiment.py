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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from . import __version__
from .domain import StrictModel
from .hybrid_retrieval import (
    AppliedRetrievalFilters,
    BM25Index,
    DenseEmbedder,
    DeterministicHashEmbedder,
    FastEmbedEmbedder,
    HybridChunk,
    HybridSearchRequest,
    LexicalOverlapFallbackReranker,
    QdrantHybridIndex,
)
from .knowledge import tokenise
from .local_graph import LocalDocumentGraph
from .rag_baseline import (
    load_locked_split,
    select_locked_cases,
    sha256_file,
)
from .rag_evaluation import (
    RAGEvalCase,
    RAGEvalDataset,
    RetrievalPrediction,
    evaluate_retrieval,
    load_multihop_rag_dataset,
    load_tatqa_dataset,
)
from .synthetic_pdf_eval import (
    SyntheticGenerationManifest,
    generate_synthetic_pdfs,
    load_generated_page_dataset,
)

StageName = Literal[
    "lexical_bm25",
    "qdrant_rrf",
    "qdrant_rrf_rerank",
    "graph_fused",
]
DatasetKind = Literal["synthetic_pdf", "tatqa_locked", "multihop_rag_locked"]
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
        document_id = _document_id_from_chunk_id(hit.chunk.chunk_id)
        if document_id not in seen:
            seen.add(document_id)
            result.append(document_id)
            if max_documents is not None and len(result) >= max_documents:
                break
    return result


class ExperimentDatasetConfig(StrictModel):
    kind: DatasetKind = "synthetic_pdf"
    synthetic_suite_path: str = "eval/synthetic_pdf_suite.json"
    tatqa_input_path: str = ".taskforge/eval-cache/tatqa_dataset_dev.json"
    tatqa_locked_split_path: str = "eval/splits/tatqa-dev-m0-100-v1.json"
    multihop_rag_queries_path: str = ".taskforge/eval-cache/MultiHopRAG.json"
    multihop_rag_corpus_path: str = ".taskforge/eval-cache/corpus.json"
    multihop_rag_locked_split_path: str = (
        "eval/splits/multihop-rag-dev-m0-100-v1.json"
    )

    @field_validator(
        "synthetic_suite_path",
        "tatqa_input_path",
        "tatqa_locked_split_path",
        "multihop_rag_queries_path",
        "multihop_rag_corpus_path",
        "multihop_rag_locked_split_path",
        mode="before",
    )
    @classmethod
    def paths_are_repository_relative(cls, value: object, info: Any) -> str:
        return _safe_repository_path(value, info.field_name)


class ExperimentRetrievalConfig(StrictModel):
    stages: list[StageName] = Field(default_factory=lambda: list(REQUIRED_STAGES))
    top_k: list[int] = Field(default_factory=lambda: [1, 5, 10])
    candidate_k: int = Field(default=25, ge=1, le=500)
    bm25_k1: float = Field(default=1.5, gt=0.0, le=10.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)
    hash_dimension: int = Field(default=64, ge=8, le=65_536)
    semantic_embedding: bool = False
    semantic_model: str = Field(default="BAAI/bge-small-en-v1.5", min_length=1)
    chunking: bool = False
    chunk_max_chars: int = Field(default=1500, ge=200, le=20_000)
    chunk_overlap_chars: int = Field(default=150, ge=0, le=10_000)
    query_expansion: bool = False
    bm25_field_weights: dict[str, float] = Field(default_factory=dict)
    graph_fusion: bool = False
    graph_max_neighbors: int = Field(default=12, ge=1, le=100)

    @model_validator(mode="after")
    def chunk_overlap_is_smaller_than_budget(self) -> ExperimentRetrievalConfig:
        if self.chunking and self.chunk_overlap_chars >= self.chunk_max_chars:
            raise ValueError("chunk_overlap_chars must be smaller than chunk_max_chars")
        for field, weight in self.bm25_field_weights.items():
            if not isinstance(weight, (int, float)) or weight <= 0:
                raise ValueError("bm25_field_weights must be finite positive numbers")
        return self

    @model_validator(mode="after")
    def stages_and_budgets_are_comparable(self) -> ExperimentRetrievalConfig:
        if len(self.stages) != len(set(self.stages)):
            raise ValueError("retrieval stages must not contain duplicates")
        missing = [stage for stage in REQUIRED_STAGES if stage not in self.stages]
        if missing:
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
    dataset = load_tatqa_dataset(input_path)
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
    chunks: list[HybridChunk] = []
    for document in sorted(dataset.documents, key=lambda value: value.document_id):
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


def _search_request(
    query: str,
    config: RAGExperimentConfig,
    *,
    rerank: bool,
) -> HybridSearchRequest:
    filters = config.filters
    final_k = max(config.retrieval.top_k)
    # With chunking, retrieve more chunks than the final document count so a
    # single document cannot crowd the top-k; grouping happens after ranking.
    retrieval_k = (
        config.retrieval.candidate_k if config.retrieval.chunking else final_k
    )
    return HybridSearchRequest(
        query=query,
        tenant_id=filters.tenant_id,
        acl_principals=frozenset(filters.request_principals),
        versions=frozenset({filters.version}),
        version_orders=frozenset({filters.version_order}),
        knowledge_base_ids=frozenset({filters.knowledge_base_id}),
        top_k=retrieval_k,
        candidate_k=config.retrieval.candidate_k,
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


def _expected_filter(config: RAGExperimentConfig) -> Mapping[str, Any]:
    request = _search_request("filter contract", config, rerank=False)
    return AppliedRetrievalFilters.from_request(request).model_dump(mode="json")


def _run_stages(
    prepared: _PreparedDataset,
    config: RAGExperimentConfig,
    *,
    timer_ns: Callable[[], int],
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    _validate_evidence(prepared.dataset, prepared.cases)
    chunks = _hybrid_chunks(prepared.dataset, prepared.cases, config)
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
    if config.retrieval.semantic_embedding:
        mode_label = "semantic_dense"
        production_dense = True
        dense_embedder: DenseEmbedder = FastEmbedEmbedder(
            config.retrieval.semantic_model
        )
        dense_embedding_desc: dict[str, Any] = {
            "kind": "fastembed_bge",
            "model": config.retrieval.semantic_model,
            "dimension": dense_embedder.dimension,
            "semantic": True,
            "production": True,
        }
    else:
        mode_label = EXPERIMENT_MODE
        production_dense = False
        dense_embedder = DeterministicHashEmbedder(config.retrieval.hash_dimension)
        dense_embedding_desc = {
            "kind": "deterministic_hash",
            "dimension": config.retrieval.hash_dimension,
            "semantic": False,
            "production": False,
        }
    qdrant = QdrantHybridIndex.in_memory(
        collection_name="taskforge-rag-ablation",
        embedder=dense_embedder,
        reranker=LexicalOverlapFallbackReranker(),
    )
    qdrant.upsert(chunks)
    indexes = {
        "lexical_bm25": (lexical, False),
        "qdrant_rrf": (qdrant, False),
        "qdrant_rrf_rerank": (qdrant, True),
    }
    expected_filter = _expected_filter(config)
    stages = list(config.retrieval.stages)
    if config.retrieval.graph_fusion:
        stages.append("graph_fused")
    graph = (
        LocalDocumentGraph(
            chunk
            for chunk in chunks
            if not chunk.chunk_id.startswith(_PROBE_PREFIX)
        )
        if config.retrieval.graph_fusion
        else None
    )
    all_rows: list[Mapping[str, Any]] = []
    stage_metrics: dict[str, Any] = {}
    case_ids = [case.case_id for case in prepared.cases]
    for stage in stages:
        predictions: list[RetrievalPrediction] = []
        durations_ms: list[float] = []
        rows: list[Mapping[str, Any]] = []
        observed_backend: str | None = None
        rerank = stage == "qdrant_rrf_rerank"
        index = indexes[stage][0] if stage != "graph_fused" else None
        for case in prepared.cases:
            request = _search_request(case.query, config, rerank=rerank)
            started = timer_ns()
            if stage == "graph_fused":
                retrieved_ids, backend_label = _graph_fused_search(
                    graph, lexical, config, case.query
                )
                response = None
            else:
                if stage == "lexical_bm25" and config.retrieval.query_expansion:
                    expanded = _expand_with_prf(lexical, request)
                    if expanded is not None:
                        request = request.model_copy(update={"query": expanded})
                response = index.search(request)
                backend_label = response.backend
            ended = timer_ns()
            if ended < started:
                raise RuntimeError("experiment timer moved backwards")
            durations_ms.append((ended - started) / 1_000_000.0)
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
                    response.hits, max_documents=max(config.retrieval.top_k)
                )
            if any(value.startswith(_PROBE_PREFIX) for value in retrieved_ids):
                raise RuntimeError("an inaccessible filter probe entered a ranking")
            predictions.append(
                RetrievalPrediction(case_id=case.case_id, retrieved_ids=retrieved_ids)
            )
            rows.append(
                {
                    "stage": stage,
                    "case_id": case.case_id,
                    "category": case.category,
                    "query": case.query,
                    "search_query": request.query,
                    "relevant_ids": case.relevant_ids,
                    "retrieved_ids": retrieved_ids,
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
            ks=config.retrieval.top_k,
        )
        if [row["case_id"] for row in rows] != case_ids:
            raise RuntimeError("ablation stages did not use identical case ordering")
        all_rows.extend(rows)
        stage_metrics[stage] = {
            "backend": observed_backend,
            "filter_request": expected_filter,
            "filters_applied_before_ranking": True,
            "experiment_mode": mode_label,
            "embedding": (
                {
                    "kind": "none",
                    "semantic": False,
                    "production": False,
                }
                if stage in {"lexical_bm25", "graph_fused"}
                else dense_embedding_desc
            ),
            "reranker": (
                "lexical_overlap_fallback"
                if stage == "qdrant_rrf_rerank"
                else "none"
            ),
            "latency": _latency_summary(durations_ms),
            "retrieval": report.model_dump(mode="json"),
        }
    expected_row_count = len(prepared.cases) * len(stages)
    if len(all_rows) != expected_row_count:
        raise RuntimeError("experiment produced an incomplete ablation matrix")
    metrics = {
        "schema_version": "1.0",
        "experiment_mode": mode_label,
        "production_semantic_dense": production_dense,
        "case_ids": case_ids,
        "top_k": config.retrieval.top_k,
        "same_case_ids_and_top_k": True,
        "chunking": {
            "enabled": config.retrieval.chunking,
            "max_chars": config.retrieval.chunk_max_chars,
            "overlap_chars": config.retrieval.chunk_overlap_chars,
        },
        "chunk_count": chunk_count,
        "query_expansion": config.retrieval.query_expansion,
        "bm25_field_weights": dict(config.retrieval.bm25_field_weights),
        "graph_fusion": config.retrieval.graph_fusion,
        "graph_max_neighbors": config.retrieval.graph_max_neighbors,
        "stages": stage_metrics,
    }
    return all_rows, metrics


def _source_hashes() -> Mapping[str, str]:
    root = Path(__file__).resolve().parent
    names = (
        "rag_experiment.py",
        "rag_evaluation.py",
        "rag_baseline.py",
        "hybrid_retrieval.py",
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
        rows, metrics = _run_stages(prepared, config, timer_ns=timer_ns)
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
                    []
                    if metrics["production_semantic_dense"]
                    else [
                        "deterministic hash vectors are nonsemantic and not production dense embeddings"
                    ]
                ),
                "the offline reranker is lexical overlap, not a learned cross-encoder",
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
                "inaccessible_filter_probes": 2,
                "chunking": metrics["chunking"],
                "chunk_count": metrics["chunk_count"],
                "query_expansion": metrics["query_expansion"],
                "bm25_field_weights": metrics["bm25_field_weights"],
                "graph_fusion": metrics["graph_fusion"],
                "graph_max_neighbors": metrics["graph_max_neighbors"],
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

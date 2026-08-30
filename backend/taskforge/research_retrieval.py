"""Evidence-driven retrieval for the paper-research Agent.

This module is deliberately an adapter around the existing knowledge
contracts.  It does not replace the runtime router.  A research query first
uses one shared lexical/dense candidate pool and only then runs deterministic
evidence-gap operators.  Every returned item is still resolved through the
authoritative knowledge store before it can be read or cited.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import Field, field_validator, model_validator

from .domain import StrictModel
from .evidence_graph import LocalEvidenceGraph
from .hybrid_knowledge import knowledge_to_hybrid_chunk
from .hybrid_retrieval import (
    BM25Index,
    HybridSearchHit,
    HybridSearchRequest,
    InMemoryDenseIndex,
    Reranker,
)
from .knowledge import (
    AccessContext,
    KnowledgeChunk,
    KnowledgeHit,
    lexical_match,
    tokenise,
)
from .rag_experiment_profile import (
    RAGExperimentProfile,
    resolve_rag_experiment_profile,
)
from .research_protocol import RetrievalTrace, RetrievalTraceHit
from .research_supervised_ranker import SupervisedResearchRanker, row_features


@runtime_checkable
class ResearchKnowledgeStore(Protocol):
    def visible_chunks(
        self,
        principal: AccessContext,
        *,
        now: Any = None,
        source_uris: Iterable[str] | None = None,
        knowledge_base_ids: Iterable[str] | None = None,
        latest_only: bool = True,
    ) -> tuple[KnowledgeChunk, ...]: ...

    def get(
        self,
        chunk_id: str,
        principal: AccessContext,
        *,
        now: Any = None,
    ) -> KnowledgeChunk | None: ...


class ResearchQuery(StrictModel):
    query: str = Field(min_length=1, max_length=4_000)
    query_variants: tuple[str, ...] = Field(default=(), max_length=2)
    # Keep the live paper-search head aligned with the eight-card Agent budget.
    top_k: int = Field(default=8, ge=1, le=50)
    candidate_k: int = Field(default=50, ge=10, le=100)
    mode: str = Field(default="standard", pattern="^(standard|rigorous)$")
    source_uris: tuple[str, ...] = Field(default=())
    knowledge_base_ids: tuple[str, ...] = Field(default=())
    latest_only: bool = True

    @field_validator("query", mode="before")
    @classmethod
    def clean_query(cls, value: object) -> str:
        cleaned = str(value).strip()
        if not cleaned or not tokenise(cleaned):
            raise ValueError("query must contain at least one searchable token")
        return cleaned

    @field_validator("query_variants", mode="before")
    @classmethod
    def normalize_query_variants(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ValueError("query_variants must be an array")
        normalized: list[str] = []
        for raw in value:
            cleaned = " ".join(str(raw).split())
            if not cleaned or len(cleaned) > 4_000:
                raise ValueError("query variants must contain 1 to 4000 characters")
            if cleaned.casefold() not in {item.casefold() for item in normalized}:
                normalized.append(cleaned)
        return tuple(normalized)

    @field_validator("source_uris", "knowledge_base_ids", mode="before")
    @classmethod
    def clean_filters(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            value = (value,)
        cleaned = tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
        if len(cleaned) > 128:
            raise ValueError("research filters exceed the configured limit")
        return cleaned

    @model_validator(mode="after")
    def budgets_are_consistent(self) -> ResearchQuery:
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        return self

    @property
    def all_queries(self) -> tuple[str, ...]:
        values: list[str] = []
        for value in (self.query, *self.query_variants):
            if value.casefold() not in {item.casefold() for item in values}:
                values.append(value)
        return tuple(values[:3])


class EvidenceRequirement(StrictModel):
    subquestion: str = Field(min_length=1, max_length=4_000)
    required_entities: tuple[str, ...] = Field(default=())
    evidence_types: tuple[str, ...] = Field(default=("claim",))
    numeric_constraints: tuple[str, ...] = Field(default=())
    minimum_sources: int = Field(default=1, ge=1, le=20)
    needs_comparison: bool = False
    needs_conflict_check: bool = False


class EvidenceGap(StrictModel):
    operator: str = Field(
        pattern=(
            "^(parent_section|structured_table|source_coverage|layout_neighbor|"
            "entity_keyword|list_paragraph|experiment_section|"
            "per_source_comparison|visual_evidence)$"
        )
    )
    reason: str = Field(min_length=1, max_length=500)
    query: str = Field(min_length=1, max_length=4_000)
    requirement_index: int = Field(default=0, ge=0)


class CoverageReport(StrictModel):
    covered_requirement_indices: tuple[int, ...] = Field(default=())
    gaps: tuple[EvidenceGap, ...] = Field(default=())
    covered_entities: tuple[str, ...] = Field(default=())
    source_count: int = Field(default=0, ge=0)
    citation_ready_count: int = Field(default=0, ge=0)
    unresolved_visual_count: int = Field(default=0, ge=0)

    @property
    def sufficient(self) -> bool:
        return not self.gaps


class ResearchEvidence(StrictModel):
    evidence_id: str = Field(min_length=1, max_length=1_024)
    chunk_id: str = Field(min_length=1, max_length=512)
    title: str | None = Field(default=None, max_length=1_000)
    source: str = Field(min_length=1, max_length=2_048)
    section: str | None = Field(default=None, max_length=1_000)
    page: str | None = Field(default=None, max_length=100)
    version: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=64_000)
    parent_chunk_id: str | None = Field(default=None, max_length=512)
    evidence_type: str = Field(default="paragraph", min_length=1, max_length=128)
    visual_artifact_ids: tuple[str, ...] = Field(default=(), max_length=32)
    visual_pending: bool = False
    text_start: int = Field(default=0, ge=0)
    text_end: int | None = Field(default=None, ge=1)
    presentation_strategy: str = Field(default="full_child", min_length=1, max_length=128)
    score: float = Field(ge=0.0)
    retrieval_sources: tuple[str, ...] = Field(default=())


class ResearchSearchResult(StrictModel):
    query: str
    evidence: tuple[ResearchEvidence, ...]
    candidate_count: int = Field(ge=0)
    retrieval_rounds: int = Field(ge=1, le=2)
    activated_operators: tuple[str, ...] = Field(default=())
    coverage: CoverageReport
    elapsed_ms: float = Field(ge=0.0)
    trace: RetrievalTrace | None = None
    # ``multilingual`` is selected only when the host has configured the
    # multilingual model pair and either the query or corpus contains CJK
    # text. ``multilingual_fallback`` is explicit telemetry for a Chinese
    # request made before that optional pair is installed; it never pretends
    # that the English model is multilingual.
    retrieval_route: Literal["english", "multilingual", "multilingual_fallback"] = "english"


class CitationVerification(StrictModel):
    verified: bool
    verification_type: str = "lexical_support"
    claim: str
    evidence_ids: tuple[str, ...]
    resolved_evidence_ids: tuple[str, ...] = Field(default=())
    missing_evidence_ids: tuple[str, ...] = Field(default=())
    token_coverage: float = Field(ge=0.0, le=1.0)
    caveat: str = (
        "This is an evidence-identity and lexical-support check; semantic entailment "
        "still requires Evaluator/Critic review."
    )


@dataclass(frozen=True, slots=True)
class _Candidate:
    hit: KnowledgeHit
    sources: tuple[str, ...]
    diagnostics: Mapping[str, object] = field(default_factory=dict)


_NUMERIC_RE = re.compile(r"\b(?:19|20)\d{2}\b|[-+]?\d+(?:\.\d+)?%?")
_TABLE_WORDS = frozenset(
    {"table", "row", "rows", "column", "columns", "metric", "percentage", "percent", "rate", "value"}
)
_LAYOUT_WORDS = frozenset(
    {"page", "pages", "figure", "fig", "footnote", "caption", "appendix", "layout", "table"}
)
_COMPARISON_WORDS = frozenset(
    {"compare", "comparison", "versus", "vs", "difference", "both", "across", "contrast"}
)
_EXPERIMENT_WORDS = frozenset(
    {
        "experiment",
        "experimental",
        "setup",
        "implementation",
        "training",
        "hyperparameter",
        "baseline",
        "ablation",
    }
)
_LIST_WORDS = frozenset(
    {"list", "enumerate", "which", "what are", "components", "steps", "datasets"}
)
_VISUAL_WORDS = frozenset(
    {"figure", "fig", "chart", "diagram", "plot", "axis", "legend", "image"}
)
_REFERENTIAL_QUERY_WORDS = frozenset(
    {"it", "its", "they", "their", "them", "this", "that", "these", "those"}
)
_QUERY_FUNCTION_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "did",
        "do",
        "does",
        "for",
        "how",
        "in",
        "is",
        "of",
        "the",
        "to",
        "was",
        "were",
        "what",
        "which",
        "with",
    }
)
_CONTEXT_TOKEN_RE = re.compile(
    r"[A-Za-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]|[^\s]",
    re.UNICODE,
)
_ENTITY_RE = re.compile(r"\b(?:[A-Z]{2,}[A-Za-z0-9.-]*|[A-Z][A-Za-z0-9.-]{2,})\b")
_ENTITY_STOP = frozenset(
    {
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "does",
        "do",
        "is",
        "are",
        "compare",
    }
)
_STRUCTURAL_NUMBER_PREFIX_RE = re.compile(
    r"(?:figure|fig\.?|table|page|section|equation|eq\.?)\s*$",
    re.IGNORECASE,
)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_PAPER_REFERENCE_TERMS = frozenset(
    {
        "reference",
        "references",
        "bibliography",
        "acknowledg",
        "参考文献",
        "致谢",
        "作者简介",
    }
)
_PAPER_DATA_TERMS = frozenset(
    {"dataset", "datasets", "data", "corpus", "benchmark", "数据集", "数据", "语料"}
)
_PAPER_METHOD_TERMS = frozenset(
    {
        "method",
        "methods",
        "approach",
        "algorithm",
        "model",
        "architecture",
        "方法",
        "算法",
        "模型",
    }
)
_PAPER_RESULT_TERMS = frozenset(
    {
        "result",
        "results",
        "experiment",
        "experiments",
        "evaluation",
        "performance",
        "accuracy",
        "结果",
        "实验",
        "评估",
        "性能",
        "准确率",
    }
)
_BIBLIOGRAPHY_MARKER_RE = re.compile(r"\[\s*\d{1,3}\s*\]")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_FRONT_MATTER_TERMS = (
    "corresponding author",
    "copyright for this paper",
    "creative commons",
    "orcid",
    "作者简介",
    "通讯作者",
    "基金项目",
)
_REFERENCE_QUERY_TERMS = (
    "reference",
    "references",
    "bibliography",
    "citation list",
    "参考文献",
    "文献目录",
)
_AUTHOR_QUERY_TERMS = (
    "author",
    "authors",
    "affiliation",
    "email",
    "orcid",
    "作者",
    "单位",
    "邮箱",
)


def _contains_cjk(value: str, *, minimum: int = 1) -> bool:
    """Detect Chinese/Japanese/Korean ideographs without language packages."""

    return len(_CJK_RE.findall(str(value))) >= minimum


def _query_entities(query: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            value
            for value in _ENTITY_RE.findall(query)
            if value.casefold() not in _ENTITY_STOP
        )
    )


def _numeric_constraints(query: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in _NUMERIC_RE.finditer(query):
        prefix = query[max(0, match.start() - 24) : match.start()]
        if _STRUCTURAL_NUMBER_PREFIX_RE.search(prefix):
            continue
        values.append(match.group(0))
    return tuple(dict.fromkeys(values))


def _metadata_text(chunk: KnowledgeChunk) -> str:
    values: list[str] = []
    for key in (
        "title",
        "source",
        "heading",
        "section",
        "sections",
        "kind",
        "block_types",
        "table_rows",
        "page",
        "pages",
    ):
        value = chunk.metadata.get(key)
        if value is not None:
            values.append(str(value))
    return " ".join(values)


def _evidence_id(chunk: KnowledgeChunk) -> str:
    raw = chunk.metadata.get("evidence_id")
    return str(raw).strip() if isinstance(raw, str) and raw.strip() else chunk.chunk_id


def _source_label(chunk: KnowledgeChunk) -> str:
    return str(chunk.metadata.get("source") or chunk.source_uri)


def _page_label(chunk: KnowledgeChunk) -> str | None:
    raw = chunk.metadata.get("page", chunk.metadata.get("pages"))
    if raw is None:
        return None
    if isinstance(raw, (list, tuple, set, frozenset)):
        return ",".join(str(value) for value in raw)
    return str(raw)


def _section_label(chunk: KnowledgeChunk) -> str | None:
    raw = chunk.metadata.get("section", chunk.metadata.get("heading"))
    return str(raw) if raw is not None else None


def _evidence_type(chunk: KnowledgeChunk) -> str:
    block_types = {
        str(value)
        for value in chunk.metadata.get("block_types", ())
    }
    if block_types & {"image", "chart"}:
        return "figure"
    if "table" in block_types or chunk.metadata.get("kind") == "table":
        return "table"
    if "equation" in block_types:
        return "equation"
    return "paragraph"


def _visual_artifact_ids(chunk: KnowledgeChunk) -> tuple[str, ...]:
    values = chunk.metadata.get("visual_artifact_ids", ())
    if isinstance(values, (str, bytes)):
        return (str(values),) if str(values).strip() else ()
    if not isinstance(values, Sequence):
        return ()
    return tuple(str(value) for value in values if str(value).strip())[:32]


def _visual_pending(chunk: KnowledgeChunk) -> bool:
    return bool(chunk.metadata.get("visual_pending")) or (
        bool(_visual_artifact_ids(chunk))
        and not bool(chunk.metadata.get("visual_text_ready", True))
    )


def _rrf_merge(
    left: Sequence[KnowledgeHit],
    right: Sequence[KnowledgeHit],
    *,
    rrf_k: int = 60,
) -> list[_Candidate]:
    by_id: dict[str, _Candidate] = {}
    scores: dict[str, float] = {}
    source_map: dict[str, list[str]] = {}
    for label, hits in (("bm25", left), ("dense", right)):
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit.chunk.chunk_id
            if chunk_id not in by_id:
                by_id[chunk_id] = _Candidate(hit=hit, sources=())
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            source_map.setdefault(chunk_id, []).append(label)
    output = [
        _Candidate(
            hit=replace(
                item.hit,
                score=scores[item.hit.chunk.chunk_id],
                retrieval_profile="research_unified",
                retrieval_backend="bm25_dense_rrf",
            ),
            sources=tuple(dict.fromkeys(source_map[item.hit.chunk.chunk_id])),
        )
        for item in by_id.values()
    ]
    output.sort(key=lambda item: (-item.hit.score, item.hit.chunk.chunk_id))
    return output


class _Operator(Protocol):
    name: str

    def search(
        self,
        service: ResearchRetrievalService,
        query: str,
        principal: AccessContext,
        corpus: Sequence[KnowledgeChunk],
        candidate_k: int,
    ) -> list[KnowledgeHit]: ...


class ParentSectionOperator:
    name = "parent_section"

    def search(self, service: ResearchRetrievalService, query: str, principal: AccessContext, corpus: Sequence[KnowledgeChunk], candidate_k: int) -> list[KnowledgeHit]:
        return service._lexical_search(query, principal, corpus, candidate_k, field_weights={"heading": 2.0, "section": 2.0})


class StructuredTableOperator:
    name = "structured_table"

    def search(self, service: ResearchRetrievalService, query: str, principal: AccessContext, corpus: Sequence[KnowledgeChunk], candidate_k: int) -> list[KnowledgeHit]:
        return service._lexical_search(query, principal, corpus, candidate_k, field_weights={"retrieval_structure": 2.5, "table_rows": 2.0, "kind": 1.5}, require_table=True)


class SourceCoverageOperator:
    name = "source_coverage"

    def search(self, service: ResearchRetrievalService, query: str, principal: AccessContext, corpus: Sequence[KnowledgeChunk], candidate_k: int) -> list[KnowledgeHit]:
        hits = service._lexical_search(query, principal, corpus, candidate_k)
        by_source: dict[str, KnowledgeHit] = {}
        for hit in hits:
            by_source.setdefault(_source_label(hit.chunk).casefold(), hit)
        return list(by_source.values())


class LayoutNeighborOperator:
    name = "layout_neighbor"

    def search(self, service: ResearchRetrievalService, query: str, principal: AccessContext, corpus: Sequence[KnowledgeChunk], candidate_k: int) -> list[KnowledgeHit]:
        return service._lexical_search(query, principal, corpus, candidate_k, field_weights={"retrieval_layout": 2.5, "page": 2.0, "heading": 1.5})


class EntityKeywordOperator:
    name = "entity_keyword"

    def search(self, service: ResearchRetrievalService, query: str, principal: AccessContext, corpus: Sequence[KnowledgeChunk], candidate_k: int) -> list[KnowledgeHit]:
        return service._lexical_search(
            query,
            principal,
            corpus,
            candidate_k,
            field_weights={"title": 2.5, "heading": 2.0, "section": 2.0},
        )


class ListParagraphOperator:
    name = "list_paragraph"

    def search(self, service: ResearchRetrievalService, query: str, principal: AccessContext, corpus: Sequence[KnowledgeChunk], candidate_k: int) -> list[KnowledgeHit]:
        return service._lexical_search(
            query,
            principal,
            corpus,
            candidate_k,
            field_weights={"kind": 2.5, "block_types": 2.0, "heading": 1.5},
            allowed_types=frozenset({"list", "paragraph", "mixed"}),
        )


class ExperimentSectionOperator:
    name = "experiment_section"

    def search(self, service: ResearchRetrievalService, query: str, principal: AccessContext, corpus: Sequence[KnowledgeChunk], candidate_k: int) -> list[KnowledgeHit]:
        return service._lexical_search(
            query,
            principal,
            corpus,
            candidate_k,
            field_weights={"heading": 3.0, "section": 3.0, "title": 1.5},
            metadata_terms=_EXPERIMENT_WORDS,
        )


class PerSourceComparisonOperator:
    name = "per_source_comparison"

    def search(self, service: ResearchRetrievalService, query: str, principal: AccessContext, corpus: Sequence[KnowledgeChunk], candidate_k: int) -> list[KnowledgeHit]:
        output: list[KnowledgeHit] = []
        by_source: dict[str, list[KnowledgeChunk]] = {}
        for chunk in corpus:
            by_source.setdefault(_source_label(chunk), []).append(chunk)
        per_source_k = max(1, min(candidate_k, math.ceil(candidate_k / max(1, len(by_source)))))
        for source_chunks in by_source.values():
            output.extend(
                service._lexical_search(
                    query,
                    principal,
                    source_chunks,
                    per_source_k,
                    field_weights={"heading": 2.0, "section": 2.0},
                )
            )
        return output


class VisualEvidenceOperator:
    name = "visual_evidence"

    def search(self, service: ResearchRetrievalService, query: str, principal: AccessContext, corpus: Sequence[KnowledgeChunk], candidate_k: int) -> list[KnowledgeHit]:
        return service._lexical_search(
            query,
            principal,
            corpus,
            candidate_k,
            field_weights={"retrieval_structure": 3.0, "retrieval_layout": 2.5, "heading": 1.5},
            allowed_types=frozenset({"image", "chart", "table", "mixed"}),
        )


class ResearchRetrievalService:
    """Unified research retrieval with bounded evidence-gap completion."""

    # Search is a candidate-discovery operation.  Returning the whole chunk
    # here makes every subsequent Agent turn pay for text that may never be
    # read or cited.  Keep the authoritative chunk in the store and expose a
    # short extract; ``paper_read`` remains the explicit full-text operation.
    # Child chunks are the citation unit. Returning only their first 500
    # characters hid semantically reranked evidence from the Agent whenever
    # the support appeared later in the Child. A 3k bound normally presents
    # the complete 300-600 token Child while still bounding atomic tables.
    SEARCH_SNIPPET_CHARS = 3_000
    READ_TEXT_CHARS = 64_000

    def __init__(
        self,
        knowledge_store: ResearchKnowledgeStore,
        *,
        dense_embedder: Any | None = None,
        reranker: Reranker | None = None,
        multilingual_dense_embedder: Any | None = None,
        multilingual_reranker: Reranker | None = None,
        feature_ranker: SupervisedResearchRanker | None = None,
        graph_enabled: bool = True,
        structure_fusion_enabled: bool = False,
        structure_section_weight: float = 0.5,
        structure_query_coverage_weight: float = 0.1,
        preserve_head_k: int = 0,
        reranker_context_window: int = 0,
        contextual_child_rerank_enabled: bool = False,
        contextual_child_neighbor_tokens: int = 120,
        contextual_child_max_tokens: int = 500,
        lexical_fusion_weight: float = 0.0,
        parent_aware_rerank_enabled: bool = True,
        parent_referential_guard_enabled: bool = True,
        parent_aware_candidate_k: int = 20,
        parent_context_max_tokens: int = 800,
        parent_include_document_title: bool = True,
        parent_include_heading_path: bool = True,
        parent_include_neighbor_chunks: bool = True,
        parent_child_score_weight: float = 0.55,
        parent_context_score_weight: float = 0.35,
        parent_retrieval_score_weight: float = 0.10,
        lineage_diversity_enabled: bool = True,
        lineage_preferred_children_per_parent: int = 2,
        lineage_parent_penalty: float = 0.08,
        lineage_overlap_weight: float = 0.05,
        intent_section_fusion_enabled: bool = False,
        intent_section_fusion_weight: float = 0.1,
        intent_query_overlap_weight: float = 0.05,
        intent_rank_fusion_weight: float = 0.45,
        operator_budget_standard: int = 1,
        operator_budget_rigorous: int = 2,
        dual_route_enabled: bool = False,
        dual_route_flat_candidate_k: int = 30,
        dual_route_child_candidate_k: int = 20,
        dual_route_flat_head_k: int = 2,
        dual_route_rerank_candidate_k: int = 10,
        dual_route_tail_rerank_candidate_k: int = 0,
        dual_route_min_confidence: float = 0.35,
        index_cache_size: int = 64,
        experiment_profile: RAGExperimentProfile | None = None,
    ) -> None:
        if not isinstance(knowledge_store, ResearchKnowledgeStore):
            raise TypeError("knowledge_store must expose visible_chunks and get")
        if not 0 <= operator_budget_standard <= 4 or not 0 <= operator_budget_rigorous <= 4:
            raise ValueError("operator budgets must be between 0 and 4")
        if not 0 <= index_cache_size <= 256:
            raise ValueError("index_cache_size must be between 0 and 256")
        if not 16 <= contextual_child_neighbor_tokens <= 240:
            raise ValueError("contextual Child neighbour budget must be between 16 and 240")
        if not 256 <= contextual_child_max_tokens <= 1_024:
            raise ValueError("contextual Child maximum must be between 256 and 1024")
        if contextual_child_neighbor_tokens * 2 >= contextual_child_max_tokens:
            raise ValueError("contextual Child neighbours must leave room for the target")
        if not 1 <= parent_aware_candidate_k <= 100:
            raise ValueError("parent-aware candidate count must be between 1 and 100")
        if not 1 <= dual_route_flat_candidate_k <= 100:
            raise ValueError("dual-route Flat candidate count must be between 1 and 100")
        if not 1 <= dual_route_child_candidate_k <= 100:
            raise ValueError("dual-route Child candidate count must be between 1 and 100")
        if not 0 <= dual_route_flat_head_k <= 10:
            raise ValueError("dual-route Flat fallback head must be between 0 and 10")
        if not 1 <= dual_route_rerank_candidate_k <= 100:
            raise ValueError("dual-route rerank candidate count must be between 1 and 100")
        if not 0 <= dual_route_tail_rerank_candidate_k <= 100:
            raise ValueError("dual-route tail rerank candidate count must be between 0 and 100")
        if not 0.0 <= dual_route_min_confidence <= 1.0:
            raise ValueError("dual-route minimum confidence must be between 0 and 1")
        if not 128 <= parent_context_max_tokens <= 3_000:
            raise ValueError("Parent context token budget must be between 128 and 3000")
        parent_weights = (
            float(parent_child_score_weight),
            float(parent_context_score_weight),
            float(parent_retrieval_score_weight),
        )
        if any(value < 0 for value in parent_weights) or sum(parent_weights) <= 0:
            raise ValueError("Parent-aware rerank weights must be non-negative with a positive sum")
        if not 1 <= lineage_preferred_children_per_parent <= 10:
            raise ValueError("preferred Children per Parent must be between 1 and 10")
        if lineage_parent_penalty < 0 or lineage_overlap_weight < 0:
            raise ValueError("lineage diversity penalties must be non-negative")
        self.knowledge_store = knowledge_store
        self.dense_embedder = dense_embedder
        self.reranker = reranker
        self.multilingual_dense_embedder = multilingual_dense_embedder
        self.multilingual_reranker = multilingual_reranker
        self.feature_ranker = feature_ranker
        self.experiment_profile = experiment_profile or resolve_rag_experiment_profile(
            "current"
        )
        self.graph_enabled = bool(graph_enabled)
        self.structure_fusion_enabled = bool(structure_fusion_enabled)
        self.structure_section_weight = float(structure_section_weight)
        self.structure_query_coverage_weight = float(structure_query_coverage_weight)
        self.preserve_head_k = int(preserve_head_k)
        self.reranker_context_window = int(reranker_context_window)
        self.contextual_child_rerank_enabled = bool(
            contextual_child_rerank_enabled
            and self.experiment_profile.name == "optimized"
        )
        self.contextual_child_neighbor_tokens = int(
            contextual_child_neighbor_tokens
        )
        self.contextual_child_max_tokens = int(contextual_child_max_tokens)
        self.lexical_fusion_weight = float(lexical_fusion_weight)
        self.parent_aware_rerank_enabled = bool(
            parent_aware_rerank_enabled
            and self.experiment_profile.parent_aware_rerank_enabled
        )
        self.parent_referential_guard_enabled = bool(
            parent_referential_guard_enabled
        )
        self.parent_aware_candidate_k = int(parent_aware_candidate_k)
        self.parent_context_max_tokens = int(parent_context_max_tokens)
        self.parent_include_document_title = bool(parent_include_document_title)
        self.parent_include_heading_path = bool(parent_include_heading_path)
        self.parent_include_neighbor_chunks = bool(parent_include_neighbor_chunks)
        weight_total = sum(parent_weights)
        self.parent_child_score_weight = parent_weights[0] / weight_total
        self.parent_context_score_weight = parent_weights[1] / weight_total
        self.parent_retrieval_score_weight = parent_weights[2] / weight_total
        self.lineage_diversity_enabled = bool(
            lineage_diversity_enabled
            and self.experiment_profile.lineage_diversity_enabled
        )
        self.lineage_preferred_children_per_parent = int(
            lineage_preferred_children_per_parent
        )
        self.lineage_parent_penalty = float(lineage_parent_penalty)
        self.lineage_overlap_weight = float(lineage_overlap_weight)
        self.intent_section_fusion_enabled = bool(intent_section_fusion_enabled)
        self.intent_section_fusion_weight = float(intent_section_fusion_weight)
        self.intent_query_overlap_weight = float(intent_query_overlap_weight)
        self.intent_rank_fusion_weight = float(intent_rank_fusion_weight)
        self.operator_budget_standard = int(operator_budget_standard)
        self.operator_budget_rigorous = int(operator_budget_rigorous)
        self.dual_route_enabled = bool(dual_route_enabled)
        self.dual_route_flat_candidate_k = int(dual_route_flat_candidate_k)
        self.dual_route_child_candidate_k = int(dual_route_child_candidate_k)
        self.dual_route_flat_head_k = int(dual_route_flat_head_k)
        self.dual_route_rerank_candidate_k = int(dual_route_rerank_candidate_k)
        self.dual_route_tail_rerank_candidate_k = int(
            dual_route_tail_rerank_candidate_k
        )
        self.dual_route_min_confidence = float(dual_route_min_confidence)
        self.index_cache_size = int(index_cache_size)
        self._index_cache: OrderedDict[
            tuple[tuple[str, str, str, int], ...],
            tuple[BM25Index, InMemoryDenseIndex | None],
        ] = OrderedDict()
        self._index_cache_lock = threading.RLock()
        # The cache follows the current async/thread execution context. A
        # paper_search and its subsequent paper_read can reuse validated Parent
        # chunks, while unrelated requests receive a separate mapping.
        self._parent_context_cache: ContextVar[dict[str, KnowledgeChunk] | None] = (
            ContextVar(
                f"taskforge_parent_context_{id(self)}",
                default=None,
            )
        )
        self.operators: dict[str, _Operator] = {
            item.name: item
            for item in (
                ParentSectionOperator(),
                StructuredTableOperator(),
                SourceCoverageOperator(),
                LayoutNeighborOperator(),
                EntityKeywordOperator(),
                ListParagraphOperator(),
                ExperimentSectionOperator(),
                PerSourceComparisonOperator(),
                VisualEvidenceOperator(),
            )
        }

    def _visible(self, request: ResearchQuery, principal: AccessContext) -> tuple[KnowledgeChunk, ...]:
        return tuple(
            chunk
            for chunk in self.knowledge_store.visible_chunks(
                principal,
                source_uris=request.source_uris or None,
                knowledge_base_ids=request.knowledge_base_ids or None,
                latest_only=request.latest_only,
            )
            if chunk.metadata.get("retrieval_role") != "parent"
            and self.experiment_profile.matches(chunk.metadata)
        )

    @staticmethod
    def _hybrid_lane(chunk: KnowledgeChunk) -> str | None:
        lane = str(chunk.metadata.get("hybrid_route") or "").strip().casefold()
        return lane if lane in {"flat_primary", "child_aux"} else None

    def _dual_route_rankings(
        self,
        request: ResearchQuery,
        principal: AccessContext,
        corpus: Sequence[KnowledgeChunk],
        *,
        dense_embedder: Any | None,
    ) -> tuple[list[_Candidate], bool]:
        """Merge a Flat recall lane with an auxiliary Child lane.

        The legacy path remains untouched unless an explicitly hybrid-indexed
        document is present and ``dual_route_enabled`` is set. Each lane is
        searched independently so the larger Child index cannot consume the
        Flat candidate budget before the two ranked lists are fused.
        """

        flat = tuple(
            chunk for chunk in corpus if self._hybrid_lane(chunk) == "flat_primary"
        )
        child = tuple(
            chunk for chunk in corpus if self._hybrid_lane(chunk) == "child_aux"
        )
        if not self.dual_route_enabled or not flat or not child:
            return [], False

        flat_request = request.model_copy(
            update={
                "candidate_k": min(
                    request.candidate_k,
                    self.dual_route_flat_candidate_k,
                )
            }
        )
        child_request = request.model_copy(
            update={
                "candidate_k": min(
                    request.candidate_k,
                    self.dual_route_child_candidate_k,
                )
            }
        )
        per_query: list[list[_Candidate]] = []
        for query in request.all_queries:
            lane_rankings: list[list[_Candidate]] = []
            for lane, lane_corpus, lane_request in (
                ("flat", flat, flat_request),
                ("child", child, child_request),
            ):
                ranked = self._unified_search(
                    query,
                    lane_request,
                    principal,
                    lane_corpus,
                    dense_embedder=dense_embedder,
                )
                lane_rankings.append(
                    [
                        _Candidate(
                            item.hit,
                            tuple(
                                dict.fromkeys(
                                    (*item.sources, f"dual_route_{lane}")
                                )
                            ),
                            item.diagnostics,
                        )
                        for item in ranked
                    ]
                )
            per_query.append(
                self._query_rrf_merge(
                    lane_rankings,
                    original_query=request.query,
                    candidate_k=request.candidate_k,
                )
            )
        return (
            self._query_rrf_merge(
                per_query,
                original_query=request.query,
                candidate_k=request.candidate_k,
            ),
            True,
        )

    @staticmethod
    def _hybrid_chunk(chunk: KnowledgeChunk):
        hybrid = knowledge_to_hybrid_chunk(chunk)
        metadata = dict(hybrid.metadata)
        metadata.setdefault("source", _source_label(chunk))
        metadata.setdefault("retrieval_layout", _metadata_text(chunk))
        metadata.setdefault("retrieval_structure", _metadata_text(chunk))
        return hybrid.model_copy(update={"metadata": metadata})

    def _search_text(self, chunk: KnowledgeChunk) -> str:
        """Return index/rerank text while preserving authoritative citations.

        New Parent-Child chunks persist a deterministic title/heading enriched
        projection in metadata. Older indexes and non-hierarchical chunks
        transparently fall back to their original body text.
        """

        retrieval_text = chunk.metadata.get("retrieval_text")
        if (
            self.experiment_profile.retrieval_text_enabled
            and isinstance(retrieval_text, str)
            and retrieval_text.strip()
        ):
            return retrieval_text.strip()
        return chunk.text

    @staticmethod
    def _bounded_rerank_fragment(
        text: str,
        limit: int,
        *,
        tail: bool = False,
        query: str | None = None,
    ) -> str:
        """Return an exact-text token window for non-authoritative reranking."""

        cleaned = str(text).strip()
        if not cleaned or limit <= 0:
            return ""
        spans = list(_CONTEXT_TOKEN_RE.finditer(cleaned))
        if len(spans) <= limit:
            return cleaned
        if tail:
            return cleaned[spans[-limit].start() :].strip()
        query_terms = {
            value
            for value in tokenise(query or "")
            if value not in _QUERY_FUNCTION_WORDS
        }
        matching = [
            index
            for index, span in enumerate(spans)
            if span.group(0).casefold() in query_terms
        ]
        if matching:
            centre = matching[len(matching) // 2]
            start_index = min(
                max(0, centre - limit // 2),
                len(spans) - limit,
            )
        else:
            start_index = 0
        end_index = start_index + limit - 1
        return cleaned[
            spans[start_index].start() : spans[end_index].end()
        ].strip()

    @staticmethod
    def _same_parent_neighbor(
        target: KnowledgeChunk,
        neighbor: KnowledgeChunk | None,
    ) -> bool:
        if neighbor is None:
            return False
        target_parent = target.metadata.get("parent_chunk_id")
        return bool(
            target.metadata.get("retrieval_role") == "child"
            and neighbor.metadata.get("retrieval_role") == "child"
            and isinstance(target_parent, str)
            and target_parent
            and neighbor.metadata.get("parent_chunk_id") == target_parent
            and neighbor.document_id == target.document_id
        )

    def _contextual_child_rerank_text(
        self,
        query: str,
        chunk: KnowledgeChunk,
        by_id: Mapping[str, KnowledgeChunk],
    ) -> str:
        """Build one bounded local window while keeping Child text authoritative."""

        if chunk.metadata.get("retrieval_role") != "child":
            return chunk.text
        previous_id = chunk.metadata.get("previous_chunk_id")
        next_id = chunk.metadata.get("next_chunk_id")
        previous = by_id.get(previous_id) if isinstance(previous_id, str) else None
        following = by_id.get(next_id) if isinstance(next_id, str) else None
        if not self._same_parent_neighbor(chunk, previous):
            previous = None
        if not self._same_parent_neighbor(chunk, following):
            following = None

        heading_path = chunk.metadata.get("heading_path")
        if isinstance(heading_path, Sequence) and not isinstance(
            heading_path, (str, bytes)
        ):
            heading = " > ".join(
                str(value).strip()
                for value in heading_path
                if str(value).strip()
            )
        else:
            heading = str(chunk.metadata.get("heading") or "").strip()
        heading = self._bounded_rerank_fragment(heading, 32)

        marker_budget = 12
        target_budget = min(
            max(1, int(self.contextual_child_max_tokens * 0.70)),
            max(1, self.contextual_child_max_tokens - marker_budget),
        )
        target = self._bounded_rerank_fragment(
            chunk.text,
            target_budget,
            query=query,
        )
        used = marker_budget + len(_CONTEXT_TOKEN_RE.findall(heading)) + len(
            _CONTEXT_TOKEN_RE.findall(target)
        )
        remaining = max(0, self.contextual_child_max_tokens - used)
        neighbor_count = int(previous is not None) + int(following is not None)
        per_neighbor = (
            min(self.contextual_child_neighbor_tokens, remaining // neighbor_count)
            if neighbor_count
            else 0
        )
        previous_text = (
            self._bounded_rerank_fragment(previous.text, per_neighbor, tail=True)
            if previous is not None
            else ""
        )
        next_text = (
            self._bounded_rerank_fragment(following.text, per_neighbor)
            if following is not None
            else ""
        )
        pieces: list[str] = []
        if heading:
            pieces.append(f"[Section]\n{heading}")
        if previous_text:
            pieces.append(f"[Previous Context]\n{previous_text}")
        pieces.append(f"[Target Child]\n{target}")
        if next_text:
            pieces.append(f"[Next Context]\n{next_text}")
        return "\n\n".join(pieces)

    def _search_chunk(self, chunk: KnowledgeChunk):
        hybrid = self._hybrid_chunk(chunk)
        return hybrid.model_copy(update={"text": self._search_text(chunk)})

    def _corpus_cache_key(
        self,
        corpus: Sequence[KnowledgeChunk],
    ) -> tuple[tuple[str, str, str, int], ...]:
        return tuple(
            (
                chunk.chunk_id,
                chunk.version,
                hashlib.sha256(
                    self._search_text(chunk).encode("utf-8")
                ).hexdigest(),
                len(self._search_text(chunk)),
            )
            for chunk in corpus
        )

    @staticmethod
    def _embedder_cache_name(embedder: Any | None) -> str:
        if embedder is None:
            return "none"
        configured = getattr(embedder, "model_name", None)
        if configured:
            return str(configured)
        return f"{type(embedder).__module__}.{type(embedder).__qualname__}"

    def _indexes(
        self,
        corpus: Sequence[KnowledgeChunk],
        embedder: Any | None = None,
    ) -> tuple[BM25Index, InMemoryDenseIndex | None]:
        # Dense vectors from two language models are not interchangeable. The
        # model identity is therefore part of the cache key, preventing an
        # English index from being reused for a Chinese/cross-lingual query.
        key = (
            ("__dense_model__", self._embedder_cache_name(embedder), "", 0),
            *self._corpus_cache_key(corpus),
        )
        if self.index_cache_size:
            with self._index_cache_lock:
                cached = self._index_cache.get(key)
                if cached is not None:
                    self._index_cache.move_to_end(key)
                    return cached
        indexed = [self._search_chunk(chunk) for chunk in corpus]
        built = (
            BM25Index(indexed),
            (
                InMemoryDenseIndex(
                    indexed,
                    embedder,
                    collection_name=getattr(
                        embedder,
                        "index_name",
                        "knowledge-dense-v1",
                    ),
                )
                if embedder is not None
                else None
            ),
        )
        if self.index_cache_size:
            with self._index_cache_lock:
                self._index_cache[key] = built
                self._index_cache.move_to_end(key)
                while len(self._index_cache) > self.index_cache_size:
                    self._index_cache.popitem(last=False)
        return built

    def _lexical_search(
        self,
        query: str,
        principal: AccessContext,
        corpus: Sequence[KnowledgeChunk],
        candidate_k: int,
        *,
        field_weights: Mapping[str, float] | None = None,
        require_table: bool = False,
        allowed_types: frozenset[str] | None = None,
        metadata_terms: frozenset[str] | None = None,
    ) -> list[KnowledgeHit]:
        selected = [
            chunk
            for chunk in corpus
            if (
                not require_table
                or str(chunk.metadata.get("kind", "")).casefold() == "table"
                or bool(chunk.metadata.get("table_rows"))
                or "table" in {str(item).casefold() for item in chunk.metadata.get("block_types", ())}
            )
            and (
                allowed_types is None
                or str(chunk.metadata.get("kind", "")).casefold() in allowed_types
                or bool(
                    allowed_types
                    & {
                        str(item).casefold()
                        for item in chunk.metadata.get("block_types", ())
                    }
                )
            )
            and (
                metadata_terms is None
                or bool(
                    metadata_terms
                    & set(
                        re.findall(
                            r"[a-z][a-z-]+",
                            _metadata_text(chunk).casefold(),
                        )
                    )
                )
            )
        ]
        if not selected:
            return []
        indexed = [self._search_chunk(chunk) for chunk in selected]
        backend = BM25Index(indexed, field_weights=field_weights)
        request = HybridSearchRequest(
            query=query,
            tenant_id=principal.tenant_id,
            acl_principals=principal.acl_tokens,
            allowed_chunk_ids=frozenset(chunk.chunk_id for chunk in selected),
            top_k=min(candidate_k, 100),
            candidate_k=min(candidate_k, 500),
            max_expanded_hits=min(candidate_k, 1_000),
        )
        response = backend.search(request)
        by_id = {chunk.chunk_id: chunk for chunk in selected}
        return [
            self._to_hit(result, by_id[result.chunk.chunk_id], "research_operator", query)
            for result in response.hits
        ]

    def _to_hit(
        self,
        result: Any,
        chunk: KnowledgeChunk,
        backend: str,
        query: str = "",
    ) -> KnowledgeHit:
        search_text = self._search_text(chunk)
        match = lexical_match(query or search_text, search_text)
        return KnowledgeHit(
            chunk=chunk,
            score=max(0.0, float(result.score)),
            lexical_score=match.score,
            semantic_score=max(0.0, float(result.base_score or 0.0)),
            matched_terms=match.matched_terms,
            retrieval_profile="research_unified",
            retrieval_backend=backend,
        )

    def _unified_search(
        self,
        query: str,
        request: ResearchQuery,
        principal: AccessContext,
        corpus: Sequence[KnowledgeChunk],
        *,
        dense_embedder: Any | None = None,
    ) -> list[_Candidate]:
        bm25_index, dense_index = self._indexes(corpus, dense_embedder)
        allowed = frozenset(chunk.chunk_id for chunk in corpus)
        search_request = HybridSearchRequest(
            query=query,
            tenant_id=principal.tenant_id,
            acl_principals=principal.acl_tokens,
            allowed_chunk_ids=allowed,
            top_k=request.candidate_k,
            candidate_k=request.candidate_k,
            max_expanded_hits=request.candidate_k,
        )
        bm25_response = bm25_index.search(search_request)
        by_id = {chunk.chunk_id: chunk for chunk in corpus}
        bm25_hits = [
            self._to_hit(result, by_id[result.chunk.chunk_id], "research_bm25", query)
            for result in bm25_response.hits
        ]
        dense_hits: list[KnowledgeHit] = []
        if dense_index is not None:
            dense_response = dense_index.search(search_request)
            dense_hits = [
                self._to_hit(result, by_id[result.chunk.chunk_id], "research_dense", query)
                for result in dense_response.hits
            ]
        return _rrf_merge(bm25_hits, dense_hits)

    def _query_rrf_merge(
        self,
        rankings: Sequence[Sequence[_Candidate]],
        *,
        original_query: str,
        candidate_k: int,
        rrf_k: int = 60,
    ) -> list[_Candidate]:
        values: dict[str, _Candidate] = {}
        scores: dict[str, float] = {}
        sources: dict[str, list[str]] = {}
        for query_index, ranking in enumerate(rankings):
            label = (
                "query_original"
                if query_index == 0
                else f"query_variant_{query_index}"
            )
            for rank, item in enumerate(ranking, start=1):
                chunk_id = item.hit.chunk.chunk_id
                values.setdefault(chunk_id, item)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
                sources.setdefault(chunk_id, []).extend((*item.sources, label))
        output: list[_Candidate] = []
        for chunk_id, item in values.items():
            match = lexical_match(
                original_query,
                self._search_text(item.hit.chunk),
            )
            output.append(
                _Candidate(
                    hit=replace(
                        item.hit,
                        score=scores[chunk_id],
                        lexical_score=match.score,
                        matched_terms=match.matched_terms,
                        retrieval_backend="multi_query_rrf",
                    ),
                    sources=tuple(dict.fromkeys(sources[chunk_id])),
                )
            )
        output.sort(key=lambda item: (-item.hit.score, item.hit.chunk.chunk_id))
        return output[:candidate_k]

    @staticmethod
    def _requirements(query: str) -> tuple[EvidenceRequirement, ...]:
        lowered = query.casefold()
        words = set(re.findall(r"[a-z][a-z-]+", lowered))
        numbers = _numeric_constraints(query)
        comparison = bool(words & _COMPARISON_WORDS)
        minimum_sources = 2 if comparison else 1
        evidence_types = ["claim"]
        if numbers or words & _TABLE_WORDS:
            evidence_types.append("numeric_or_table")
        return (
            EvidenceRequirement(
                subquestion=query,
                required_entities=_query_entities(query),
                evidence_types=tuple(evidence_types),
                numeric_constraints=numbers,
                minimum_sources=minimum_sources,
                needs_comparison=comparison,
                needs_conflict_check=comparison,
            ),
        )

    def _coverage(self, requirements: Sequence[EvidenceRequirement], candidates: Sequence[_Candidate], query: str) -> CoverageReport:
        texts = [f"{item.hit.chunk.text} {_metadata_text(item.hit.chunk)}" for item in candidates]
        joined = " ".join(texts).casefold()
        sources = {_source_label(item.hit.chunk).casefold() for item in candidates}
        gaps: list[EvidenceGap] = []
        covered: list[int] = []
        features = set(re.findall(r"[a-z][a-z-]+", query.casefold()))
        phrases = {value for value in _LIST_WORDS if " " in value and value in query.casefold()}
        visual_needed_by_query = bool(features & _VISUAL_WORDS)
        visual_chunks = [
            item
            for item in candidates
            if (
                str(item.hit.chunk.metadata.get("kind", "")).casefold()
                in {"image", "chart"}
                or bool(
                    {"image", "chart"}
                    & {
                        str(value).casefold()
                        for value in item.hit.chunk.metadata.get("block_types", ())
                    }
                )
            )
        ]
        unresolved_visual_count = sum(
            bool(item.hit.chunk.metadata.get("visual_pending"))
            or (
                bool(_visual_artifact_ids(item.hit.chunk))
                and not bool(item.hit.chunk.metadata.get("visual_text_ready", True))
            )
            for item in visual_chunks
        )
        for index, requirement in enumerate(requirements):
            numeric_missing = any(value.casefold() not in joined for value in requirement.numeric_constraints)
            entity_missing = any(
                value.casefold() not in joined
                for value in requirement.required_entities
            )
            source_missing = len(sources) < requirement.minimum_sources
            table_needed = "numeric_or_table" in requirement.evidence_types and (
                not any(str(item.hit.chunk.metadata.get("kind", "")).casefold() == "table" for item in candidates)
                or numeric_missing
            )
            layout_needed = bool(features & _LAYOUT_WORDS) and not any(_page_label(item.hit.chunk) for item in candidates)
            list_needed = bool(features & _LIST_WORDS or phrases) and not any(
                str(item.hit.chunk.metadata.get("kind", "")).casefold() == "list"
                or "list" in {
                    str(value).casefold()
                    for value in item.hit.chunk.metadata.get("block_types", ())
                }
                for item in candidates
            )
            experiment_needed = bool(features & _EXPERIMENT_WORDS) and not any(
                bool(
                    _EXPERIMENT_WORDS
                    & set(
                        re.findall(
                            r"[a-z][a-z-]+",
                            _metadata_text(item.hit.chunk).casefold(),
                        )
                    )
                )
                for item in candidates
            )
            visual_missing = visual_needed_by_query and not visual_chunks
            if source_missing and requirement.needs_comparison:
                gaps.append(EvidenceGap(operator="per_source_comparison", reason="comparison lacks per-paper evidence", query=query, requirement_index=index))
            elif source_missing:
                gaps.append(EvidenceGap(operator="source_coverage", reason="not enough distinct paper sources", query=query, requirement_index=index))
            elif entity_missing:
                gaps.append(EvidenceGap(operator="entity_keyword", reason="one or more named entities are missing", query=query, requirement_index=index))
            elif table_needed:
                gaps.append(EvidenceGap(operator="structured_table", reason="numeric or table evidence is incomplete", query=query, requirement_index=index))
            elif list_needed:
                gaps.append(EvidenceGap(operator="list_paragraph", reason="list or enumerated evidence is incomplete", query=query, requirement_index=index))
            elif experiment_needed:
                gaps.append(EvidenceGap(operator="experiment_section", reason="experimental setup section evidence is incomplete", query=query, requirement_index=index))
            elif visual_missing or unresolved_visual_count:
                gaps.append(EvidenceGap(operator="visual_evidence", reason="visual evidence is missing or unparsed", query=query, requirement_index=index))
            elif layout_needed:
                gaps.append(EvidenceGap(operator="layout_neighbor", reason="page/layout evidence is incomplete", query=query, requirement_index=index))
            else:
                covered.append(index)
        if not gaps and not candidates:
            gaps.append(EvidenceGap(operator="parent_section", reason="no citation-ready evidence was retrieved", query=query))
        return CoverageReport(
            covered_requirement_indices=tuple(covered),
            gaps=tuple(gaps),
            covered_entities=(),
            source_count=len(sources),
            citation_ready_count=sum(1 for item in candidates if item.hit.chunk.text.strip()),
            unresolved_visual_count=unresolved_visual_count,
        )

    @staticmethod
    def _merge_candidates(base: Sequence[_Candidate], extras: Sequence[KnowledgeHit]) -> list[_Candidate]:
        scores: dict[str, float] = {}
        values: dict[str, _Candidate] = {item.hit.chunk.chunk_id: item for item in base}
        source_map: dict[str, list[str]] = {item.hit.chunk.chunk_id: list(item.sources) for item in base}
        for rank, hit in enumerate(extras, start=1):
            chunk_id = hit.chunk.chunk_id
            values.setdefault(chunk_id, _Candidate(hit=hit, sources=()))
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (60 + rank)
            source_map.setdefault(chunk_id, []).append("gap_operator")
        for rank, item in enumerate(base, start=1):
            scores[item.hit.chunk.chunk_id] = scores.get(item.hit.chunk.chunk_id, 0.0) + 1.0 / (60 + rank)
        output: list[_Candidate] = []
        for chunk_id, item in values.items():
            hit = replace(item.hit, score=scores.get(chunk_id, item.hit.score))
            output.append(_Candidate(hit=hit, sources=tuple(dict.fromkeys(source_map[chunk_id]))))
        output.sort(key=lambda item: (-item.hit.score, item.hit.chunk.chunk_id))
        return output

    def _rerank_once(
        self,
        query: str,
        candidates: Sequence[_Candidate],
        corpus: Sequence[KnowledgeChunk],
        top_k: int,
        *,
        reranker: Reranker | None = None,
        disable_reranker: bool = False,
    ) -> list[_Candidate]:
        limited = list(candidates)
        preserved_head = limited[: self.preserve_head_k]
        active_reranker = (
            None
            if disable_reranker
            else self.reranker
            if reranker is None
            else reranker
        )
        if active_reranker is not None and limited:
            by_id = {item.chunk_id: item for item in corpus}
            def _reranker_text(candidate: _Candidate) -> str:
                chunk = candidate.hit.chunk
                if self.contextual_child_rerank_enabled:
                    return self._contextual_child_rerank_text(
                        query,
                        chunk,
                        by_id,
                    )
                if not self.reranker_context_window:
                    return self._search_text(chunk)
                pieces = [self._search_text(chunk)]
                current = chunk
                for _ in range(self.reranker_context_window):
                    neighbor_id = current.metadata.get("next_chunk_id")
                    if not neighbor_id or neighbor_id not in by_id:
                        break
                    current = by_id[neighbor_id]
                    pieces.append(self._search_text(current))
                return "\n".join(pieces)
            scores = list(
                active_reranker.score(
                    query,
                    [_reranker_text(item) for item in limited],
                )
            )
            if len(scores) != len(limited) or not all(math.isfinite(float(value)) for value in scores):
                raise ValueError("research reranker returned invalid scores")
            raw_scores = [float(score) for score in scores]
            if self.lexical_fusion_weight:
                lexical = [float(item.hit.lexical_score or 0.0) for item in limited]
                low, high = min(lexical), max(lexical)
                lexical_norm = [
                    (value - low) / (high - low) if high > low else 0.0
                    for value in lexical
                ]
                score_low, score_high = min(raw_scores), max(raw_scores)
                score_norm = [
                    (value - score_low) / (score_high - score_low)
                    if score_high > score_low else 0.0
                    for value in raw_scores
                ]
                raw_scores = [
                    value + self.lexical_fusion_weight * lexical_value
                    for value, lexical_value in zip(score_norm, lexical_norm, strict=True)
                ]
            limited = [
                _Candidate(
                    replace(item.hit, score=score),
                    tuple(
                        dict.fromkeys(
                            (
                                *item.sources,
                                *(
                                    ("contextual_child_rerank",)
                                    if self.contextual_child_rerank_enabled
                                    else ()
                                ),
                            )
                        )
                    ),
                )
                for item, score in zip(limited, raw_scores, strict=True)
            ]
            if self.structure_fusion_enabled:
                def _normalise(values: Sequence[float]) -> list[float]:
                    low, high = min(values), max(values)
                    return [
                        (float(value) - low) / (high - low) if high > low else 0.0
                        for value in values
                    ]

                query_tokens = set(re.findall(r"[a-z0-9]+", query.casefold()))
                section_terms = {
                    "dataset", "data", "corpus", "benchmark", "evaluation",
                    "baseline", "model", "algorithm", "approach", "method",
                    "architecture", "component", "framework", "engine",
                    "result", "performance", "score", "accuracy", "experiment",
                    "evaluate", "evaluated", "test", "testing", "annotation",
                    "annotated", "crowd", "label",
                }
                query_intent_terms = query_tokens & section_terms
                coverage: list[float] = []
                section_match: list[float] = []
                for item in limited:
                    text_tokens = set(re.findall(r"[a-z0-9]+", item.hit.chunk.text.casefold()))
                    coverage.append(len(query_tokens & text_tokens) / max(1, len(query_tokens)))
                    metadata = item.hit.chunk.metadata
                    section = " ".join(
                        str(metadata.get(key, ""))
                        for key in ("section", "section_title", "subsection_title")
                    )
                    # PDF ingestion may not preserve section metadata, but
                    # the deterministic renderer keeps heading breadcrumbs in
                    # the first text fields (``Experiments ::: Setup ::: ...``).
                    # Recover those breadcrumbs for structure fusion without
                    # changing the evidence text or identity.
                    heading = " ::: ".join(item.hit.chunk.text.split(":::")[:3])
                    section = f"{section} {heading}".casefold()
                    section_match.append(
                        len(query_intent_terms & set(re.findall(r"[a-z0-9]+", section)))
                        / max(1, len(query_intent_terms))
                    )
                fused = [
                    base
                    + self.structure_query_coverage_weight * value
                    + self.structure_section_weight * section_value
                    for base, value, section_value in zip(
                        _normalise(raw_scores),
                        _normalise(coverage),
                        _normalise(section_match),
                        strict=True,
                    )
                ]
                limited = [
                    _Candidate(replace(item.hit, score=float(score)), item.sources)
                    for item, score in zip(limited, fused, strict=True)
                ]
            limited.sort(key=lambda item: (-item.hit.score, item.hit.chunk.chunk_id))
        if self.feature_ranker is not None and limited:
            ids = [item.hit.chunk.chunk_id for item in limited]
            feature_row = {
                "query": query,
                "retrieved_ids": ids,
                "base_scores": [
                    float(item.hit.semantic_score or item.hit.score)
                    for item in limited
                ],
                "reranker_scores": [float(item.hit.score) for item in limited],
            }
            documents = {
                item.hit.chunk.chunk_id: {
                    "text": item.hit.chunk.text,
                    "metadata": dict(item.hit.chunk.metadata),
                }
                for item in limited
            }
            order = self.feature_ranker.rerank(row_features(feature_row, documents))
            limited = [
                _Candidate(
                    replace(limited[index].hit, retrieval_backend="supervised_feature_rerank"),
                    tuple(dict.fromkeys((*limited[index].sources, "supervised_feature_rerank"))),
                )
                for index in order
            ]
        if self.graph_enabled and limited:
            graph = LocalEvidenceGraph(self._hybrid_chunk(chunk) for chunk in corpus)
            hybrid_hits = [
                HybridSearchHit(
                    chunk=self._hybrid_chunk(item.hit.chunk),
                    rank=rank,
                    score=float(item.hit.score),
                    base_score=float(item.hit.score),
                    retrieval_sources=["bm25_dense_rrf"],
                )
                for rank, item in enumerate(limited, start=1)
            ]
            graph_result = graph.rerank(
                query,
                hybrid_hits,
                seed_k=min(10, len(hybrid_hits)),
            )
            by_id = {item.hit.chunk.chunk_id: item for item in limited}
            limited = [
                _Candidate(
                    replace(
                        by_id[item.chunk.chunk_id].hit,
                        score=float(item.score),
                        retrieval_backend="evidence_graph",
                    ),
                    tuple(dict.fromkeys((*by_id[item.chunk.chunk_id].sources, "graph_feature_rerank"))),
                )
                for item in graph_result.hits
            ]
        if self.preserve_head_k and limited:
            head_ids = {item.hit.chunk.chunk_id for item in preserved_head}
            reranked = [item for item in limited if item.hit.chunk.chunk_id not in head_ids]
            # Preserve the original high-recall head, then fill remaining
            # slots with the learned ranking. This is an explicit hard
            # fallback and is disabled by default.
            limited = [*preserved_head, *reranked]
        return limited[:top_k]

    @staticmethod
    def _normalise_scores(values: Sequence[float]) -> list[float]:
        if not values:
            return []
        low, high = min(values), max(values)
        if high <= low:
            return [0.0 for _ in values]
        return [(float(value) - low) / (high - low) for value in values]

    @staticmethod
    def _bounded_prefix(text: str, token_budget: int) -> str:
        cleaned = text.strip()
        if not cleaned or token_budget <= 0:
            return ""
        if len(tokenise(cleaned)) <= token_budget:
            return cleaned
        low, high = 1, len(cleaned)
        best = ""
        while low <= high:
            middle = (low + high) // 2
            candidate = cleaned[:middle].rstrip()
            if len(tokenise(candidate)) <= token_budget:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        return best

    @classmethod
    def _bounded_suffix(cls, text: str, token_budget: int) -> str:
        cleaned = text.strip()
        if not cleaned or token_budget <= 0:
            return ""
        if len(tokenise(cleaned)) <= token_budget:
            return cleaned
        reversed_text = cleaned[::-1]
        # Binary-searching the reversed string preserves the exact requested
        # suffix while reusing the deterministic token budget implementation.
        return cls._bounded_prefix(reversed_text, token_budget)[::-1].lstrip()

    def _parent_for_child(
        self,
        child: KnowledgeChunk,
        principal: AccessContext,
        cache: dict[str, KnowledgeChunk],
    ) -> KnowledgeChunk | None:
        if child.metadata.get("retrieval_role") != "child":
            return None
        parent_id = child.metadata.get("parent_chunk_id")
        if not isinstance(parent_id, str) or not parent_id or parent_id == child.chunk_id:
            return None
        parent = cache.get(parent_id)
        if parent is None:
            parent = self.knowledge_store.get(parent_id, principal)
            if parent is None:
                return None
            cache[parent_id] = parent
        if (
            parent.metadata.get("retrieval_role") != "parent"
            or not self.experiment_profile.matches(parent.metadata)
            or parent.document_id != child.document_id
            or parent.version != child.version
            or parent.tenant_id != child.tenant_id
        ):
            cache.pop(parent_id, None)
            return None
        return parent

    def _parent_local_context(
        self,
        child: KnowledgeChunk,
        parent: KnowledgeChunk,
        corpus_by_id: Mapping[str, KnowledgeChunk],
    ) -> str:
        """Prefer sibling context and fall back to a Parent-centred window."""

        before = ""
        after = ""
        if self.parent_include_neighbor_chunks:
            previous_id = child.metadata.get("previous_chunk_id")
            next_id = child.metadata.get("next_chunk_id")
            previous = (
                corpus_by_id.get(previous_id)
                if isinstance(previous_id, str)
                else None
            )
            following = (
                corpus_by_id.get(next_id) if isinstance(next_id, str) else None
            )
            parent_id = parent.chunk_id
            if (
                previous is not None
                and previous.metadata.get("parent_chunk_id") == parent_id
            ):
                before = previous.text
            if (
                following is not None
                and following.metadata.get("parent_chunk_id") == parent_id
            ):
                after = following.text

        if not before and not after:
            position = parent.text.find(child.text)
            if position >= 0:
                before = parent.text[:position]
                after = parent.text[position + len(child.text) :]
            else:
                before = parent.text

        before_budget = self.parent_context_max_tokens // 2
        after_budget = self.parent_context_max_tokens - before_budget
        before = self._bounded_suffix(before, before_budget)
        after = self._bounded_prefix(after, after_budget)
        pieces: list[str] = []
        if before:
            pieces.append(f"Previous context:\n{before}")
        if after:
            pieces.append(f"Following context:\n{after}")
        return "\n\n".join(pieces)

    def _parent_rerank_text(
        self,
        child: KnowledgeChunk,
        parent: KnowledgeChunk | None,
        corpus_by_id: Mapping[str, KnowledgeChunk],
    ) -> tuple[str, bool, bool]:
        pieces: list[str] = []
        heading_used = False
        if self.parent_include_document_title:
            title = child.metadata.get("title")
            if isinstance(title, str) and title.strip():
                pieces.append(f"Document: {title.strip()}")
        if self.parent_include_heading_path:
            path = child.metadata.get("heading_path")
            if isinstance(path, Sequence) and not isinstance(path, (str, bytes)):
                heading = " > ".join(
                    str(value).strip() for value in path if str(value).strip()
                )
            else:
                heading = str(child.metadata.get("heading") or "").strip()
            if heading:
                pieces.append(f"Section: {heading}")
                heading_used = True
        parent_context = (
            self._parent_local_context(child, parent, corpus_by_id)
            if parent is not None
            else ""
        )
        if parent_context:
            pieces.append(parent_context)
        pieces.append(f"Child evidence:\n{child.text}")
        return "\n\n".join(pieces), bool(parent_context), heading_used

    def _parent_aware_rerank(
        self,
        query: str,
        candidates: Sequence[_Candidate],
        corpus: Sequence[KnowledgeChunk],
        principal: AccessContext,
        retrieval_scores: Mapping[str, float],
        *,
        reranker: Reranker | None,
        parent_cache: dict[str, KnowledgeChunk],
    ) -> list[_Candidate]:
        """Rerank the high-value Child head with validated Parent context."""

        values = list(candidates)
        if not self.parent_aware_rerank_enabled or reranker is None or not values:
            return values
        query_tokens = [value.casefold() for value in tokenise(query)]
        informative_tokens = [
            value
            for value in query_tokens
            if value not in _REFERENTIAL_QUERY_WORDS
            and value not in _QUERY_FUNCTION_WORDS
        ]
        if (
            self.parent_referential_guard_enabled
            and _REFERENTIAL_QUERY_WORDS.intersection(query_tokens)
            and len(informative_tokens) <= 3
            and not any(character.isdigit() for character in query)
        ):
            return [
                replace(
                    item,
                    sources=tuple(
                        dict.fromkeys(
                            (*item.sources, "parent_aware_referential_guard")
                        )
                    ),
                )
                for item in values
            ]
        head_size = min(self.parent_aware_candidate_k, len(values))
        head = values[:head_size]
        tail = values[head_size:]
        corpus_by_id = {chunk.chunk_id: chunk for chunk in corpus}
        documents: list[str] = []
        parent_used: list[bool] = []
        heading_used: list[bool] = []
        child_scores = [float(item.hit.score) for item in head]
        retrieval_values = [
            float(retrieval_scores.get(item.hit.chunk.chunk_id, 0.0))
            for item in head
        ]
        for item in head:
            parent = self._parent_for_child(item.hit.chunk, principal, parent_cache)
            document, used_parent, used_heading = self._parent_rerank_text(
                item.hit.chunk,
                parent,
                corpus_by_id,
            )
            documents.append(document)
            parent_used.append(used_parent)
            heading_used.append(used_heading)
        if not any(parent_used):
            return values

        context_scores = [float(value) for value in reranker.score(query, documents)]
        if len(context_scores) != len(head) or not all(
            math.isfinite(value) for value in context_scores
        ):
            raise ValueError("Parent-aware reranker returned invalid scores")
        child_norm = self._normalise_scores(child_scores)
        context_norm = self._normalise_scores(context_scores)
        retrieval_norm = self._normalise_scores(retrieval_values)
        fused = [
            self.parent_child_score_weight * child_value
            + self.parent_context_score_weight * context_value
            + self.parent_retrieval_score_weight * retrieval_value
            for child_value, context_value, retrieval_value in zip(
                child_norm,
                context_norm,
                retrieval_norm,
                strict=True,
            )
        ]
        updated: list[_Candidate] = []
        for rank_before, (item, context_score, final_score, used_parent, used_heading) in enumerate(
            zip(
                head,
                context_scores,
                fused,
                parent_used,
                heading_used,
                strict=True,
            ),
            start=1,
        ):
            diagnostics = dict(item.diagnostics)
            diagnostics.update(
                {
                    "child_score": float(item.hit.score),
                    "context_score": context_score,
                    "final_score": final_score,
                    "parent_context_used": used_parent,
                    "heading_path_used": used_heading,
                    "parent_rank_before": rank_before,
                }
            )
            updated.append(
                _Candidate(
                    hit=replace(
                        item.hit,
                        score=float(final_score),
                        retrieval_backend="parent_aware_rerank",
                    ),
                    sources=tuple(
                        dict.fromkeys((*item.sources, "parent_aware_rerank"))
                    ),
                    diagnostics=diagnostics,
                )
            )
        updated.sort(
            key=lambda item: (
                -item.hit.score,
                int(item.diagnostics.get("parent_rank_before", head_size + 1)),
                item.hit.chunk.chunk_id,
            )
        )
        ranked: list[_Candidate] = []
        for rank_after, item in enumerate(updated, start=1):
            diagnostics = dict(item.diagnostics)
            diagnostics["parent_rank_after"] = rank_after
            ranked.append(replace(item, diagnostics=diagnostics))
        return [*ranked, *tail]

    @staticmethod
    def _token_overlap(left: str, right: str) -> float:
        left_tokens = set(tokenise(left))
        right_tokens = set(tokenise(right))
        union = left_tokens | right_tokens
        return len(left_tokens & right_tokens) / len(union) if union else 0.0

    def _lineage_diversify(
        self,
        candidates: Sequence[_Candidate],
    ) -> list[_Candidate]:
        """Softly diversify siblings without imposing a destructive hard cap."""

        remaining = list(candidates)
        if (
            not self.lineage_diversity_enabled
            or len(remaining) < 2
            or not any(
                item.hit.chunk.metadata.get("retrieval_role") == "child"
                and item.hit.chunk.metadata.get("parent_chunk_id")
                != item.hit.chunk.chunk_id
                for item in remaining
            )
        ):
            return remaining
        selected: list[_Candidate] = []
        parent_counts: dict[str, int] = {}
        while remaining:
            scored: list[tuple[float, float, str, int]] = []
            for index, item in enumerate(remaining):
                parent_id = item.hit.chunk.metadata.get("parent_chunk_id")
                parent_key = (
                    parent_id
                    if isinstance(parent_id, str) and parent_id
                    else item.hit.chunk.chunk_id
                )
                count = parent_counts.get(parent_key, 0)
                excess = max(
                    0,
                    count - self.lineage_preferred_children_per_parent + 1,
                )
                overlap = max(
                    (
                        self._token_overlap(
                            item.hit.chunk.text,
                            chosen.hit.chunk.text,
                        )
                        for chosen in selected
                    ),
                    default=0.0,
                )
                adjusted = (
                    float(item.hit.score)
                    - self.lineage_parent_penalty * excess
                    - self.lineage_overlap_weight * overlap
                )
                scored.append(
                    (adjusted, float(item.hit.score), item.hit.chunk.chunk_id, index)
                )
            adjusted, _, _, chosen_index = min(
                scored,
                key=lambda value: (-value[0], -value[1], value[2]),
            )
            chosen = remaining.pop(chosen_index)
            parent_id = chosen.hit.chunk.metadata.get("parent_chunk_id")
            parent_key = (
                parent_id
                if isinstance(parent_id, str) and parent_id
                else chosen.hit.chunk.chunk_id
            )
            parent_counts[parent_key] = parent_counts.get(parent_key, 0) + 1
            diagnostics = dict(chosen.diagnostics)
            diagnostics["diversity_adjusted_score"] = adjusted
            selected.append(
                replace(
                    chosen,
                    sources=tuple(
                        dict.fromkeys((*chosen.sources, "lineage_diversity"))
                    ),
                    diagnostics=diagnostics,
                )
            )
        return selected

    @staticmethod
    def _low_information_table(chunk: KnowledgeChunk) -> bool:
        """Reject parser-produced table shells that contain no usable fact."""

        if _evidence_type(chunk) != "table":
            return False
        if "|" not in chunk.text:
            return False
        cells: list[str] = []
        for line in chunk.text.splitlines():
            values = [value.strip() for value in line.strip().strip("|").split("|")]
            cells.extend(
                value
                for value in values
                if value and not re.fullmatch(r":?-{3,}:?", value)
            )
        content = " ".join(cells).casefold()
        if not content:
            return True
        # A compact numeric result table remains useful even when it has only
        # a handful of cells. Bare row labels such as "relevant doc 1" do not.
        if re.search(
            r"\d+(?:\.\d+)?\s*(?:%|ms\b|s\b|sec\b|kg\b|gb\b|mb\b|kb\b)",
            content,
        ):
            return False
        words = re.findall(r"[a-z\u3400-\u9fff]+", content)
        placeholders = {
            "context",
            "doc",
            "document",
            "item",
            "prompt",
            "relevant",
            "row",
            "column",
            "value",
            "文档",
            "相关",
        }
        informative = [word for word in words if word not in placeholders]
        return len(cells) <= 8 and len(set(informative)) <= 2

    @staticmethod
    def _paper_noise_factor(query: str, chunk: KnowledgeChunk) -> tuple[float, str | None]:
        """Identify PDF boilerplate that is searchable but weak as evidence."""

        if ResearchRetrievalService._low_information_table(chunk):
            return 0.0, "empty_table_noise"
        query_lower = query.casefold()
        text = chunk.text.strip()
        lowered = text.casefold()
        section = " ".join(
            str(chunk.metadata.get(key) or "")
            for key in ("heading", "section", "section_title", "subsection_title")
        ).casefold()
        reference_requested = any(term in query_lower for term in _REFERENCE_QUERY_TERMS)
        if not reference_requested:
            searchable_head = f"{section} {lowered[:320]}"
            markers = list(_BIBLIOGRAPHY_MARKER_RE.finditer(text))
            bibliography_like = any(
                term in searchable_head for term in _PAPER_REFERENCE_TERMS
            ) or (
                len(markers) >= 3
                and markers[0].start() <= 180
            )
            if bibliography_like:
                return 0.08, "bibliography_noise"

        author_requested = any(term in query_lower for term in _AUTHOR_QUERY_TERMS)
        if not author_requested:
            prefix = lowered[:420]
            front_matter_signals = len(_EMAIL_RE.findall(text[:600])) + sum(
                term in prefix for term in _FRONT_MATTER_TERMS
            )
            if front_matter_signals >= 2:
                return 0.30, "front_matter_noise"
        return 1.0, None

    @classmethod
    def _paper_quality_rerank(
        cls,
        query: str,
        candidates: Sequence[_Candidate],
    ) -> list[_Candidate]:
        """Demote bibliography/contact blocks after semantic reranking."""

        updated: list[_Candidate] = []
        for item in candidates:
            factor, reason = cls._paper_noise_factor(query, item.hit.chunk)
            if reason is None:
                updated.append(item)
                continue
            diagnostics = dict(item.diagnostics)
            diagnostics["paper_noise_factor"] = factor
            diagnostics["paper_noise_reason"] = reason
            updated.append(
                _Candidate(
                    hit=replace(
                        item.hit,
                        score=max(0.0, float(item.hit.score) * factor),
                        retrieval_backend="paper_quality_rerank",
                    ),
                    sources=tuple(dict.fromkeys((*item.sources, reason))),
                    diagnostics=diagnostics,
                )
            )
        updated.sort(key=lambda item: (-item.hit.score, item.hit.chunk.chunk_id))
        return updated

    @staticmethod
    def _duplicate_shingles(text: str, size: int = 5) -> set[tuple[str, ...]]:
        tokens = [value.casefold() for value in tokenise(text)]
        if len(tokens) < size:
            return set()
        return {
            tuple(tokens[index : index + size])
            for index in range(len(tokens) - size + 1)
        }

    @classmethod
    def _near_duplicate_evidence(
        cls,
        left: KnowledgeChunk,
        right: KnowledgeChunk,
    ) -> bool:
        """Detect title/keyword-prefixed variants of the same paper passage."""

        if left.source_uri != right.source_uri:
            return False
        left_normalized = "".join(tokenise(left.text.casefold()))
        right_normalized = "".join(tokenise(right.text.casefold()))
        shorter, longer = sorted(
            (left_normalized, right_normalized),
            key=len,
        )
        if len(shorter) >= 160 and shorter in longer:
            return True
        left_shingles = cls._duplicate_shingles(left.text)
        right_shingles = cls._duplicate_shingles(right.text)
        if min(len(left_shingles), len(right_shingles)) < 20:
            return False
        intersection = len(left_shingles & right_shingles)
        containment = intersection / min(len(left_shingles), len(right_shingles))
        union = len(left_shingles | right_shingles)
        return containment >= 0.88 and intersection / max(1, union) >= 0.50

    @classmethod
    def _dedupe_final_evidence(
        cls,
        candidates: Sequence[_Candidate],
        top_k: int,
    ) -> list[_Candidate]:
        selected: list[_Candidate] = []
        for item in candidates:
            if "empty_table_noise" in item.sources:
                continue
            duplicate_index = next(
                (
                    index
                    for index, existing in enumerate(selected)
                    if cls._near_duplicate_evidence(
                        item.hit.chunk,
                        existing.hit.chunk,
                    )
                ),
                None,
            )
            if duplicate_index is not None:
                existing = selected[duplicate_index]
                item_length = len("".join(tokenise(item.hit.chunk.text)))
                existing_length = len("".join(tokenise(existing.hit.chunk.text)))
                # Prefer the tighter passage when it retains most of the
                # reranker score; this drops title/author/keyword-prefixed
                # wrappers around an otherwise identical abstract.
                if (
                    item_length < existing_length * 0.98
                    and item.hit.score >= existing.hit.score * 0.75
                ):
                    selected[duplicate_index] = replace(
                        item,
                        sources=tuple(
                            dict.fromkeys((*item.sources, "near_duplicate_dedupe"))
                        ),
                    )
                continue
            if len(selected) < top_k:
                selected.append(item)
        selected.sort(key=lambda item: (-item.hit.score, item.hit.chunk.chunk_id))
        return selected[:top_k]

    def _preserve_dual_route_flat_head(
        self,
        candidates: Sequence[_Candidate],
    ) -> list[_Candidate]:
        """Keep a small Flat head as a deterministic query-level fallback."""

        if not self.dual_route_enabled or self.dual_route_flat_head_k <= 0:
            return list(candidates)
        flat_head = [
            item
            for item in candidates
            if self._hybrid_lane(item.hit.chunk) == "flat_primary"
        ][: self.dual_route_flat_head_k]
        if not flat_head:
            return list(candidates)
        flat_ids = {item.hit.chunk.chunk_id for item in flat_head}
        return [
            *flat_head,
            *[
                item
                for item in candidates
                if item.hit.chunk.chunk_id not in flat_ids
            ],
        ]

    @staticmethod
    def _dual_route_structure_prior(
        query: str,
        candidates: Sequence[_Candidate],
    ) -> list[_Candidate]:
        """Apply conservative paper-section priors to the opt-in route.

        References and acknowledgements often repeat paper terminology and
        otherwise crowd the head. Intent boosts are deliberately small and
        only use explicit section metadata or the beginning of the chunk.
        """

        query_text = query.casefold()
        query_terms = set(tokenise(query_text))
        wants_data = bool(query_terms & _PAPER_DATA_TERMS) or any(
            term in query_text for term in ("数据集", "数据", "语料")
        )
        wants_method = bool(query_terms & _PAPER_METHOD_TERMS) or any(
            term in query_text for term in ("方法", "算法", "模型")
        )
        wants_result = bool(query_terms & _PAPER_RESULT_TERMS) or any(
            term in query_text for term in ("结果", "实验", "评估", "性能")
        )
        updated: list[_Candidate] = []
        for item in candidates:
            metadata = item.hit.chunk.metadata
            section = " ".join(
                str(metadata.get(key) or "")
                for key in ("heading", "section", "section_title", "subsection_title")
            ).casefold()
            text_prefix = item.hit.chunk.text[:240].casefold()
            searchable = f"{section} {text_prefix}"
            factor = 1.0
            if any(term in searchable for term in _PAPER_REFERENCE_TERMS):
                factor *= 0.35
            if wants_data and any(
                term in searchable for term in ("dataset", "data", "corpus", "数据集", "数据", "语料")
            ):
                factor *= 1.10
            if wants_method and any(
                term in searchable for term in ("method", "approach", "algorithm", "方法", "算法", "模型")
            ):
                factor *= 1.10
            if wants_result and any(
                term in searchable for term in ("result", "experiment", "evaluation", "performance", "结果", "实验", "评估", "性能")
            ):
                factor *= 1.10
            if factor == 1.0:
                updated.append(item)
                continue
            diagnostics = dict(item.diagnostics)
            diagnostics["paper_structure_factor"] = factor
            updated.append(
                _Candidate(
                    replace(item.hit, score=max(0.0, float(item.hit.score) * factor)),
                    item.sources,
                    diagnostics,
                )
            )
        updated.sort(key=lambda item: (-item.hit.score, item.hit.chunk.chunk_id))
        return updated

    @classmethod
    def _dual_route_score_fusion(
        cls,
        candidates: Sequence[_Candidate],
        retrieval_scores: Mapping[str, float],
    ) -> list[_Candidate]:
        """Keep the first-stage signal in the Dual rerank score.

        A cross-encoder is useful for ordering candidates, but it must not be
        allowed to erase a high-recall Flat result.  Scores are normalized per
        request because local and remote rerankers expose incompatible scales.
        The structure prior is already recorded by
        ``_dual_route_structure_prior`` and contributes a small bounded term.
        """

        if not candidates:
            return []
        retrieval_values = [
            float(retrieval_scores.get(item.hit.chunk.chunk_id, 0.0))
            for item in candidates
        ]
        rerank_values = [
            (
                float(retrieval_score)
                if item.diagnostics.get("dual_rerank_skipped")
                else float(item.hit.score)
            )
            for item, retrieval_score in zip(
                candidates,
                retrieval_values,
                strict=True,
            )
        ]
        rerank_norm = cls._normalise_scores(rerank_values)
        retrieval_norm = cls._normalise_scores(retrieval_values)
        fused: list[_Candidate] = []
        for item, rerank_score, retrieval_score in zip(
            candidates,
            rerank_norm,
            retrieval_norm,
            strict=True,
        ):
            factor = float(item.diagnostics.get("paper_structure_factor", 1.0))
            # Reference sections are suppressed at zero; ordinary paper
            # sections retain the full small structure contribution.
            structure_score = max(0.0, min(1.0, (factor - 0.35) / 0.65))
            score = (
                0.65 * rerank_score
                + 0.25 * retrieval_score
                + 0.10 * structure_score
            )
            diagnostics = dict(item.diagnostics)
            diagnostics.update(
                {
                    "dual_rerank_score_normalized": float(rerank_score),
                    "dual_retrieval_score_normalized": float(retrieval_score),
                    "dual_structure_score": float(structure_score),
                    "dual_fused_score": float(score),
                }
            )
            fused.append(
                _Candidate(
                    replace(
                        item.hit,
                        score=float(score),
                        retrieval_backend="dual_route_score_fusion",
                    ),
                    item.sources,
                    diagnostics,
                )
            )
        fused.sort(key=lambda item: (-item.hit.score, item.hit.chunk.chunk_id))
        return fused

    @staticmethod
    def _dual_route_needs_tail_rerank(
        candidates: Sequence[_Candidate],
        top_k: int,
    ) -> bool:
        """Detect an unstable visible head without language-specific labels.

        The normal Dual path reranks only its prefix.  A second pass is
        reserved for a weak/flat visible head: either the lowest visible fused
        score is small or the head has little separation between rank one and
        the last visible item.  These are deliberately conservative signals;
        they do not alter the legacy route or require CJK intent heuristics.
        """

        if len(candidates) <= top_k or top_k < 2:
            return False
        visible = candidates[:top_k]
        scores = [max(0.0, float(item.hit.score)) for item in visible]
        return min(scores) < 0.40 or (scores[0] - scores[-1]) < 0.50

    @staticmethod
    def _dual_route_structured(chunk: KnowledgeChunk) -> bool:
        kind = str(chunk.metadata.get("kind") or "").casefold()
        block_types = {
            str(value).casefold()
            for value in chunk.metadata.get("block_types", ())
        }
        return bool(
            {"table", "list", "image", "chart"} & ({kind} | block_types)
        )

    @classmethod
    def _dual_route_overlap(cls, left: KnowledgeChunk, right: KnowledgeChunk) -> float:
        """Estimate cross-lane overlap without changing citation text."""

        if left.source_uri != right.source_uri:
            return 0.0
        left_blocks = {
            str(value)
            for value in left.metadata.get("block_ids", ())
            if str(value).strip()
        }
        right_blocks = {
            str(value)
            for value in right.metadata.get("block_ids", ())
            if str(value).strip()
        }
        if left_blocks and right_blocks:
            return len(left_blocks & right_blocks) / max(
                1,
                min(len(left_blocks), len(right_blocks)),
            )
        left_tokens = set(tokenise(left.text))
        right_tokens = set(tokenise(right.text))
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / max(
            1,
            min(len(left_tokens), len(right_tokens)),
        )

    @classmethod
    def _dual_route_dedupe(
        cls,
        candidates: Sequence[_Candidate],
        *,
        max_children_per_parent: int = 2,
    ) -> list[_Candidate]:
        """Remove duplicate Flat/Child spans and cap sibling crowding."""

        selected: list[_Candidate] = []
        parent_counts: dict[tuple[str, str], int] = {}
        for item in candidates:
            chunk = item.hit.chunk
            lane = cls._hybrid_lane(chunk)
            if lane == "child_aux":
                parent_id = str(chunk.metadata.get("parent_chunk_id") or "")
                if parent_id and parent_id != chunk.chunk_id:
                    parent_key = (chunk.source_uri, parent_id)
                    if parent_counts.get(parent_key, 0) >= max_children_per_parent:
                        continue
            duplicate_index: int | None = None
            if lane in {"flat_primary", "child_aux"} and not cls._dual_route_structured(chunk):
                for index, existing in enumerate(selected):
                    other = existing.hit.chunk
                    other_lane = cls._hybrid_lane(other)
                    if other_lane == lane or other_lane not in {"flat_primary", "child_aux"}:
                        continue
                    if cls._dual_route_overlap(chunk, other) >= 0.70:
                        duplicate_index = index
                        break
            if duplicate_index is not None:
                existing = selected[duplicate_index]
                # Candidates arrive in descending fused score order.  Keep a
                # Child on ties so the citation points at the precise span.
                current_wins = (
                    item.hit.score > existing.hit.score
                    or (
                        item.hit.score == existing.hit.score
                        and lane == "child_aux"
                        and cls._hybrid_lane(existing.hit.chunk) != "child_aux"
                    )
                )
                if not current_wins:
                    continue
                selected[duplicate_index] = replace(
                    item,
                    sources=tuple(
                        dict.fromkeys(
                            (*item.sources, "dual_route_cross_granularity_dedupe")
                        )
                    ),
                )
                if lane == "child_aux":
                    parent_id = str(chunk.metadata.get("parent_chunk_id") or "")
                    if parent_id and parent_id != chunk.chunk_id:
                        parent_key = (chunk.source_uri, parent_id)
                        parent_counts[parent_key] = parent_counts.get(parent_key, 0) + 1
                continue
            if lane == "child_aux":
                parent_id = str(chunk.metadata.get("parent_chunk_id") or "")
                if parent_id and parent_id != chunk.chunk_id:
                    parent_counts[(chunk.source_uri, parent_id)] = (
                        parent_counts.get((chunk.source_uri, parent_id), 0) + 1
                    )
            selected.append(item)
        return selected

    def _intent_reorder(
        self,
        query: str,
        evidence: Sequence[_Candidate],
    ) -> list[_Candidate]:
        """Apply a conservative section-intent prior to the returned head.

        This runs after the learned ranker so it cannot change candidate
        discovery or deep-candidate membership.  It only reorders the already
        returned evidence and is therefore safe to disable for non-PDF data.
        """
        query_lower = query.casefold()
        groups: list[tuple[str, ...]] = []
        if any(word in query_lower for word in ("collect", "source", "sourced", "obtain", "acquire", "dataset", "datasets", "data")):
            groups.append(("data", "collection", "dataset", "corpus", "annotation"))
        if any(word in query_lower for word in ("methods", "approach", "algorithm", "variant", "baseline", "used in experiments")):
            groups.append(("method", "approach", "algorithm", "experiment", "experiments", "baseline"))
        if any(word in query_lower for word in ("result", "performance", "score", "accuracy", "compare", "comparison", "benchmark")):
            groups.append(("result", "evaluation", "experiment", "performance", "benchmark"))
        if not groups or len(evidence) < 2:
            return list(evidence)
        priors: list[float] = []
        query_tokens = set(re.findall(r"[a-z0-9]+", query_lower))
        overlaps: list[float] = []
        for item in evidence:
            text = item.hit.chunk.text.casefold()
            first = " ".join(text.split()[:10])
            heading = " ".join(text.split(":::")[:3]) if ":::" in text else first
            priors.append(
                sum(max(1.0 if term in heading else 0.5 if term in first else 0.0 for term in terms) for terms in groups)
            )
            overlaps.append(
                len(query_tokens & set(re.findall(r"[a-z0-9]+", text))) / max(1, len(query_tokens))
            )
        low, high = min(priors), max(priors)
        if high <= low:
            return list(evidence)
        span = high - low
        overlap_low, overlap_high = min(overlaps), max(overlaps)
        overlap_span = overlap_high - overlap_low
        adjusted = [
            _Candidate(
                replace(
                    item.hit,
                    score=(
                        1.0 / (index + 1)
                        + self.intent_section_fusion_weight * ((prior - low) / span)
                        + self.intent_query_overlap_weight * (
                            (overlap - overlap_low) / overlap_span if overlap_span > 0 else 0.0
                        )
                    ),
                    retrieval_backend="intent_section_fusion",
                ),
                item.sources,
                item.diagnostics,
            )
            for index, (item, prior, overlap) in enumerate(zip(evidence, priors, overlaps, strict=True))
        ]
        adjusted.sort(key=lambda item: (-item.hit.score, item.hit.chunk.chunk_id))
        return adjusted

    def _intent_rank_fuse(
        self,
        base: Sequence[_Candidate],
        intent: Sequence[_Candidate],
    ) -> list[_Candidate]:
        """Fuse learned and deterministic intent orders without new candidates."""
        if not base or not intent:
            return list(intent or base)
        alpha = self.intent_rank_fusion_weight
        base_rank = {item.hit.chunk.chunk_id: index for index, item in enumerate(base)}
        intent_rank = {item.hit.chunk.chunk_id: index for index, item in enumerate(intent)}
        values = {item.hit.chunk.chunk_id: item for item in (*base, *intent)}
        scored: list[tuple[float, str]] = []
        for chunk_id in values:
            learned = 1.0 / (base_rank.get(chunk_id, len(base)) + 1)
            intent_score = 1.0 / (intent_rank.get(chunk_id, len(intent)) + 1)
            scored.append(
                (
                    (1.0 - alpha) * learned + alpha * intent_score,
                    chunk_id,
                )
            )
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            _Candidate(
                replace(
                    values[chunk_id].hit,
                    score=float(score),
                    retrieval_backend="intent_rank_fusion",
                ),
                tuple(dict.fromkeys((*values[chunk_id].sources, "intent_rank_fusion"))),
                values[chunk_id].diagnostics,
            )
            for score, chunk_id in scored
        ]

    @classmethod
    def _query_excerpt(cls, query: str, text: str) -> tuple[str, int, int, str]:
        """Choose a bounded query-centred window instead of a fixed prefix."""

        limit = cls.SEARCH_SNIPPET_CHARS
        if len(text) <= limit:
            return text, 0, len(text), "full_child"
        lowered = text.casefold()
        starts = {0}
        for term in dict.fromkeys(tokenise(query)):
            position = lowered.find(term.casefold())
            while position >= 0:
                starts.add(max(0, min(len(text) - limit, position - limit // 2)))
                position = lowered.find(term.casefold(), position + max(1, len(term)))
        scored: list[tuple[float, int]] = []
        for start in starts:
            window = text[start : start + limit]
            match = lexical_match(query, window)
            prefix = window[:320].casefold()
            reference_markers = list(_BIBLIOGRAPHY_MARKER_RE.finditer(window))
            noise_penalty = min(4.0, 0.6 * len(_EMAIL_RE.findall(window[:600])))
            noise_penalty += 0.8 * sum(term in prefix for term in _FRONT_MATTER_TERMS)
            if len(reference_markers) >= 3 and reference_markers[0].start() <= 180:
                noise_penalty += 6.0
            scored.append(
                (
                    float(match.score) + len(match.matched_terms) - noise_penalty,
                    start,
                )
            )
        _, start = max(scored, key=lambda item: (item[0], -item[1]))
        if start:
            boundary = text.find(" ", start, min(len(text), start + 40))
            if boundary >= 0:
                start = boundary + 1
        end = min(len(text), start + limit)
        if end < len(text):
            boundary = text.rfind(" ", max(start + 1, end - 40), end)
            if boundary > start:
                end = boundary
        return text[start:end], start, end, "query_centered_lexical_window"

    @staticmethod
    def _trace_hits(
        candidates: Sequence[_Candidate],
        *,
        evidence: Sequence[ResearchEvidence] = (),
    ) -> list[RetrievalTraceHit]:
        spans = {
            item.chunk_id: (item.text_start, item.text_end)
            for item in evidence
        }
        return [
            RetrievalTraceHit(
                chunk_id=item.hit.chunk.chunk_id,
                rank=rank,
                score=float(item.hit.score),
                retrieval_sources=list(item.sources),
                text_start=spans.get(item.hit.chunk.chunk_id, (None, None))[0],
                text_end=spans.get(item.hit.chunk.chunk_id, (None, None))[1],
                child_score=(
                    float(item.diagnostics["child_score"])
                    if item.diagnostics.get("child_score") is not None
                    else None
                ),
                context_score=(
                    float(item.diagnostics["context_score"])
                    if item.diagnostics.get("context_score") is not None
                    else None
                ),
                final_score=(
                    float(item.diagnostics["final_score"])
                    if item.diagnostics.get("final_score") is not None
                    else None
                ),
                parent_context_used=bool(
                    item.diagnostics.get("parent_context_used", False)
                ),
                heading_path_used=bool(
                    item.diagnostics.get("heading_path_used", False)
                ),
                parent_rank_before=(
                    int(item.diagnostics["parent_rank_before"])
                    if item.diagnostics.get("parent_rank_before") is not None
                    else None
                ),
                parent_rank_after=(
                    int(item.diagnostics["parent_rank_after"])
                    if item.diagnostics.get("parent_rank_after") is not None
                    else None
                ),
            )
            for rank, item in enumerate(candidates, start=1)
        ]

    def _retrieval_route(
        self,
        query: str,
        corpus: Sequence[KnowledgeChunk],
    ) -> Literal["english", "multilingual", "multilingual_fallback"]:
        """Select a language-compatible dense/rerank profile.

        BM25 remains shared across profiles.  A small CJK threshold on corpus
        text avoids routing an English paper merely because a user pasted one
        Chinese title in metadata, while still routing Chinese papers and
        Chinese-question/English-paper cross-lingual queries.  The fallback
        label is deliberately explicit when the optional multilingual models
        are not configured, so evaluation cannot misreport the English model
        as a multilingual result.
        """

        cjk_query = _contains_cjk(query)
        cjk_corpus = any(_contains_cjk(chunk.text, minimum=8) for chunk in corpus)
        if not (cjk_query or cjk_corpus):
            return "english"
        if self.multilingual_dense_embedder is not None or self.multilingual_reranker is not None:
            return "multilingual"
        return "multilingual_fallback"

    def search(
        self,
        request: ResearchQuery | Mapping[str, Any] | str,
        principal: AccessContext,
    ) -> ResearchSearchResult:
        started = time.perf_counter()
        if not isinstance(request, ResearchQuery):
            request = (
                ResearchQuery.model_validate(dict(request))
                if isinstance(request, Mapping)
                else ResearchQuery(query=request)
            )
        parent_cache: dict[str, KnowledgeChunk] = {}
        self._parent_context_cache.set(parent_cache)
        corpus = self._visible(request, principal)
        if not corpus:
            empty = CoverageReport(gaps=(EvidenceGap(operator="parent_section", reason="authorized corpus is empty", query=request.query),))
            return ResearchSearchResult(
                query=request.query,
                evidence=(),
                candidate_count=0,
                retrieval_rounds=1,
                coverage=empty,
                elapsed_ms=0.0,
                retrieval_route=(
                    (
                        "multilingual"
                        if (
                            self.multilingual_dense_embedder is not None
                            or self.multilingual_reranker is not None
                        )
                        else "multilingual_fallback"
                    )
                    if _contains_cjk(request.query)
                    else "english"
                ),
            )
        retrieval_route = self._retrieval_route(request.query, corpus)
        dense_embedder = (
            (
                self.multilingual_dense_embedder or self.dense_embedder
            )
            if retrieval_route == "multilingual"
            else self.dense_embedder
        )
        reranker = (
            self.multilingual_reranker
            if retrieval_route == "multilingual"
            else self.reranker
        )
        requirements = self._requirements(request.query)
        base, dual_route_active = self._dual_route_rankings(
            request,
            principal,
            corpus,
            dense_embedder=dense_embedder,
        )
        # An opt-in Dual run must never score Chinese evidence with the
        # English cross-encoder.  If the multilingual pair is unavailable,
        # retain the deterministic Flat/RRF order and let the Flat fallback
        # policy decide; the legacy single-lane route is unchanged.
        if dual_route_active and retrieval_route != "multilingual":
            reranker = None
        if not dual_route_active:
            rankings = [
                self._unified_search(
                    query,
                    request,
                    principal,
                    corpus,
                    dense_embedder=dense_embedder,
                )
                for query in request.all_queries
            ]
            base = self._query_rrf_merge(
                rankings,
                original_query=request.query,
                candidate_k=request.candidate_k,
            )
        # Gap detection operates on the evidence head that would actually be
        # presented, not the whole Candidate@K tail. A relevant item at rank
        # 49 does not make a Top-10 answer context sufficient.
        coverage = self._coverage(
            requirements,
            base[: request.top_k],
            request.query,
        )
        activated: list[str] = []
        candidates = list(base)
        budget = self.operator_budget_rigorous if request.mode == "rigorous" else self.operator_budget_standard
        # The current gap operators intentionally use English-only intent and
        # entity regexes. Do not let them inject noisy second-round searches
        # into Chinese/cross-lingual measurements until their heuristics are
        # multilingual as well.
        if retrieval_route != "english":
            budget = 0
        for gap in coverage.gaps[:budget]:
            operator = self.operators[gap.operator]
            extras = operator.search(self, gap.query, principal, corpus, request.candidate_k)
            candidates = self._merge_candidates(candidates, extras)
            candidates = candidates[: request.candidate_k]
            activated.append(operator.name)
        retrieval_scores = {
            item.hit.chunk.chunk_id: float(item.hit.score) for item in candidates
        }
        rerank_failed = False
        reranked_head: list[_Candidate] = []
        # The multilingual cross-encoder is the expensive stage on CPU. The
        # Dual route keeps the complete Flat+Child candidate pool for recall
        # diagnostics, but only reranks the highest-scoring prefix. The
        # untouched tail participates in score fusion through its retrieval
        # score, so it can still win when the reranked head is weak.
        rerank_candidates = candidates
        rerank_tail: list[_Candidate] = []
        if dual_route_active and reranker is not None:
            rerank_limit = min(
                len(candidates),
                request.candidate_k,
                self.dual_route_rerank_candidate_k,
            )
            rerank_candidates = candidates[:rerank_limit]
            rerank_tail = [
                replace(
                    item,
                    diagnostics={
                        **dict(item.diagnostics),
                        "dual_rerank_skipped": True,
                    },
                    sources=tuple(
                        dict.fromkeys((*item.sources, "dual_route_rerank_tail"))
                    ),
                )
                for item in candidates[rerank_limit:]
            ]
        try:
            child_reranked = self._rerank_once(
                request.query,
                rerank_candidates,
                corpus,
                len(rerank_candidates),
                reranker=reranker,
                disable_reranker=(dual_route_active and reranker is None),
            )
            reranked_head = list(child_reranked)
            if rerank_tail:
                child_reranked = [*child_reranked, *rerank_tail]
        except Exception:
            if not dual_route_active:
                raise
            # A learned reranker is optional enrichment.  Preserve the
            # already-merged Flat/RRF candidates so the query can still be
            # served and the Flat coverage fallback can take over below.
            rerank_failed = True
            child_reranked = [
                replace(
                    item,
                    sources=tuple(
                        dict.fromkeys(
                            (*item.sources, "dual_route_reranker_fallback")
                        )
                    ),
                )
                for item in candidates
            ]
        if dual_route_active:
            child_reranked = self._dual_route_structure_prior(
                request.query,
                child_reranked,
            )
            child_reranked = self._dual_route_score_fusion(
                child_reranked,
                retrieval_scores,
            )
            child_reranked = self._dual_route_dedupe(child_reranked)
            # Most queries are served by the cheap Top-N pass.  Only an
            # unstable visible head gets a second pass over the next tail
            # candidates; a tail-reranker failure leaves the first pass intact.
            if (
                not rerank_failed
                and rerank_tail
                and reranked_head
                and self.dual_route_tail_rerank_candidate_k > 0
                and self._dual_route_needs_tail_rerank(
                    child_reranked,
                    request.top_k,
                )
            ):
                tail_limit = min(
                    len(rerank_tail),
                    self.dual_route_tail_rerank_candidate_k,
                )
                tail_for_rerank = [
                    replace(
                        item,
                        diagnostics={
                            key: value
                            for key, value in item.diagnostics.items()
                            if key != "dual_rerank_skipped"
                        },
                    )
                    for item in rerank_tail[:tail_limit]
                ]
                try:
                    tail_reranked = self._rerank_once(
                        request.query,
                        tail_for_rerank,
                        corpus,
                        len(tail_for_rerank),
                        reranker=reranker,
                    )
                except Exception:
                    child_reranked = [
                        replace(
                            item,
                            sources=tuple(
                                dict.fromkeys(
                                    (*item.sources, "dual_route_tail_reranker_fallback")
                                )
                            ),
                        )
                        for item in child_reranked
                    ]
                else:
                    tail_reranked = [
                        replace(
                            item,
                            sources=tuple(
                                dict.fromkeys(
                                    (*item.sources, "dual_route_tail_rerank")
                                )
                            ),
                        )
                        for item in tail_reranked
                    ]
                    child_reranked = [
                        *reranked_head,
                        *tail_reranked,
                        *rerank_tail[tail_limit:],
                    ]
                    child_reranked = self._dual_route_structure_prior(
                        request.query,
                        child_reranked,
                    )
                    child_reranked = self._dual_route_score_fusion(
                        child_reranked,
                        retrieval_scores,
                    )
                    child_reranked = self._dual_route_dedupe(child_reranked)
        try:
            if dual_route_active:
                # Parent is context-only for Dual.  Applying a second
                # cross-encoder pass to broad Parents was the main source of
                # rank regressions in the previous experiment.
                parent_reranked = [
                    replace(
                        item,
                        sources=tuple(
                            dict.fromkeys(
                                (*item.sources, "dual_route_parent_context_only")
                            )
                        ),
                    )
                    for item in child_reranked
                ]
            else:
                parent_reranked = self._parent_aware_rerank(
                    request.query,
                    child_reranked,
                    corpus,
                    principal,
                    retrieval_scores,
                    reranker=reranker,
                    parent_cache=parent_cache,
                )
        except Exception:
            # Parent context is an enhancement, never a condition for
            # returning evidence. Preserve the complete first-stage ranking.
            parent_reranked = [
                replace(
                    item,
                    sources=tuple(
                        dict.fromkeys((*item.sources, "parent_aware_fallback"))
                    ),
                )
                for item in child_reranked
            ]
        dual_route_flat_fallback = False
        if dual_route_active:
            # A dual route is allowed to add candidates, but it must not hide
            # a Flat answer that satisfies the same deterministic coverage
            # checks. This is the query-level rollback when the auxiliary
            # Child/Parent ranking produces a gap in its visible head.
            flat_fallback = [
                item
                for item in child_reranked
                if self._hybrid_lane(item.hit.chunk) == "flat_primary"
            ]
            dual_coverage = self._coverage(
                requirements,
                parent_reranked[: request.top_k],
                request.query,
            )
            flat_coverage = self._coverage(
                requirements,
                flat_fallback[: request.top_k],
                request.query,
            )
            visible_scores = [
                max(0.0, float(item.hit.score))
                for item in parent_reranked[: request.top_k]
            ]
            low_confidence = not visible_scores or max(visible_scores) < self.dual_route_min_confidence
            if not flat_coverage.gaps and (
                rerank_failed
                or dual_coverage.gaps
                or low_confidence
            ):
                dual_route_flat_fallback = True
                parent_reranked = [
                    replace(
                        item,
                        sources=tuple(
                            dict.fromkeys(
                                (
                                    *item.sources,
                                    "dual_route_flat_fallback",
                                    (
                                        "dual_route_low_confidence_fallback"
                                        if low_confidence
                                        else "dual_route_coverage_fallback"
                                    ),
                                )
                            )
                        ),
                    )
                    for item in flat_fallback
                ]
        final_pool = parent_reranked
        if self.intent_section_fusion_enabled:
            intent_final = self._intent_reorder(request.query, parent_reranked)
            final_pool = self._intent_rank_fuse(parent_reranked, intent_final)
        diversity_limit = min(
            len(final_pool),
            max(
                request.top_k,
                (
                    self.parent_aware_candidate_k
                    if self.parent_aware_rerank_enabled and reranker is not None
                    else request.top_k * 3
                ),
            ),
        )
        final_pool = [
            *self._lineage_diversify(final_pool[:diversity_limit]),
            *final_pool[diversity_limit:],
        ]
        if dual_route_flat_fallback:
            final_pool = self._preserve_dual_route_flat_head(final_pool)
        final_pool = self._paper_quality_rerank(request.query, final_pool)
        final = self._dedupe_final_evidence(final_pool, request.top_k)
        final_coverage = self._coverage(requirements, final, request.query)
        evidence_values: list[ResearchEvidence] = []
        for item in final:
            excerpt, text_start, text_end, strategy = self._query_excerpt(
                request.query,
                item.hit.chunk.text,
            )
            evidence_values.append(ResearchEvidence(
                evidence_id=_evidence_id(item.hit.chunk),
                chunk_id=item.hit.chunk.chunk_id,
                title=str(item.hit.chunk.metadata.get("title")) if item.hit.chunk.metadata.get("title") is not None else None,
                source=_source_label(item.hit.chunk),
                section=_section_label(item.hit.chunk),
                page=_page_label(item.hit.chunk),
                version=item.hit.chunk.version,
                text=excerpt,
                evidence_type=_evidence_type(item.hit.chunk),
                visual_artifact_ids=_visual_artifact_ids(item.hit.chunk),
                visual_pending=_visual_pending(item.hit.chunk),
                text_start=text_start,
                text_end=text_end,
                presentation_strategy=strategy,
                score=max(0.0, float(item.hit.score)),
                retrieval_sources=tuple(dict.fromkeys((*item.sources, item.hit.retrieval_backend or "research"))),
            ))
        evidence = tuple(evidence_values)
        trace = RetrievalTrace(
            query=request.query,
            query_variants=list(request.all_queries),
            candidate_hits=self._trace_hits(candidates),
            reranked_hits=self._trace_hits(parent_reranked),
            returned_hits=self._trace_hits(final, evidence=evidence),
        )
        return ResearchSearchResult(
            query=request.query,
            evidence=evidence,
            candidate_count=len(candidates),
            retrieval_rounds=1 + int(bool(activated)),
            activated_operators=tuple(dict.fromkeys(activated)),
            coverage=final_coverage,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            trace=trace,
            retrieval_route=retrieval_route,
        )

    def read_evidence(self, evidence_id: str, principal: AccessContext) -> ResearchEvidence:
        cleaned = str(evidence_id).strip()
        if not cleaned or len(cleaned) > 1_024:
            raise ValueError("evidence_id is invalid")
        visible = tuple(
            chunk
            for chunk in self.knowledge_store.visible_chunks(
                principal,
                latest_only=True,
            )
            if self.experiment_profile.matches(chunk.metadata)
        )
        for chunk in visible:
            if cleaned in {chunk.chunk_id, _evidence_id(chunk)}:
                readable = chunk
                parent_cache = self._parent_context_cache.get()
                if parent_cache is None:
                    parent_cache = {}
                    self._parent_context_cache.set(parent_cache)
                parent = self._parent_for_child(chunk, principal, parent_cache)
                if parent is not None:
                    readable = parent
                return ResearchEvidence(
                    evidence_id=_evidence_id(chunk),
                    chunk_id=chunk.chunk_id,
                    title=str(chunk.metadata.get("title")) if chunk.metadata.get("title") is not None else None,
                    source=_source_label(chunk),
                    section=_section_label(chunk),
                    page=_page_label(chunk),
                    version=chunk.version,
                    text=readable.text[: self.READ_TEXT_CHARS],
                    evidence_type=_evidence_type(chunk),
                    visual_artifact_ids=_visual_artifact_ids(chunk),
                    visual_pending=_visual_pending(chunk),
                    text_end=min(len(readable.text), self.READ_TEXT_CHARS),
                    parent_chunk_id=(
                        readable.chunk_id if readable.chunk_id != chunk.chunk_id else None
                    ),
                    presentation_strategy=(
                        "parent_context_for_child"
                        if readable.chunk_id != chunk.chunk_id
                        else "full_child"
                    ),
                    score=1.0,
                    retrieval_sources=("evidence_read",),
                )
        raise KeyError("evidence_id not found")

    def verify_citation(self, claim: str, evidence_ids: Sequence[str], principal: AccessContext) -> CitationVerification:
        cleaned_claim = str(claim).strip()
        ids = tuple(dict.fromkeys(str(item).strip() for item in evidence_ids if str(item).strip()))
        if not cleaned_claim or not ids or len(ids) > 10:
            raise ValueError("claim and one to ten evidence_ids are required")
        resolved: list[ResearchEvidence] = []
        support_texts: list[str] = []
        missing: list[str] = []
        for evidence_id in ids:
            try:
                item = self.read_evidence(evidence_id, principal)
                resolved.append(item)
                chunk = self.knowledge_store.get(item.chunk_id, principal)
                support_texts.append(chunk.text if chunk is not None else item.text)
            except KeyError:
                missing.append(evidence_id)
        claim_tokens = set(tokenise(cleaned_claim))
        support_tokens = set(tokenise(" ".join(support_texts)))
        coverage = len(claim_tokens & support_tokens) / len(claim_tokens) if claim_tokens else 0.0
        return CitationVerification(
            verified=not missing and coverage >= 0.55,
            claim=cleaned_claim,
            evidence_ids=ids,
            resolved_evidence_ids=tuple(item.evidence_id for item in resolved),
            missing_evidence_ids=tuple(missing),
            token_coverage=coverage,
        )


__all__ = [
    "CitationVerification",
    "CoverageReport",
    "EvidenceGap",
    "EvidenceRequirement",
    "LayoutNeighborOperator",
    "ParentSectionOperator",
    "ResearchEvidence",
    "ResearchQuery",
    "ResearchSearchResult",
    "ResearchRetrievalService",
    "SourceCoverageOperator",
    "StructuredTableOperator",
]

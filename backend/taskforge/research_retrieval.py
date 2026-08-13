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
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

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
    top_k: int = Field(default=10, ge=1, le=50)
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


class EvidenceRequirement(StrictModel):
    subquestion: str = Field(min_length=1, max_length=4_000)
    required_entities: tuple[str, ...] = Field(default=())
    evidence_types: tuple[str, ...] = Field(default=("claim",))
    numeric_constraints: tuple[str, ...] = Field(default=())
    minimum_sources: int = Field(default=1, ge=1, le=20)
    needs_comparison: bool = False
    needs_conflict_check: bool = False


class EvidenceGap(StrictModel):
    operator: str = Field(pattern="^(parent_section|structured_table|source_coverage|layout_neighbor)$")
    reason: str = Field(min_length=1, max_length=500)
    query: str = Field(min_length=1, max_length=4_000)
    requirement_index: int = Field(default=0, ge=0)


class CoverageReport(StrictModel):
    covered_requirement_indices: tuple[int, ...] = Field(default=())
    gaps: tuple[EvidenceGap, ...] = Field(default=())
    covered_entities: tuple[str, ...] = Field(default=())
    source_count: int = Field(default=0, ge=0)
    citation_ready_count: int = Field(default=0, ge=0)

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
    text: str = Field(min_length=1, max_length=8_000)
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


class ResearchRetrievalService:
    """Unified research retrieval with bounded evidence-gap completion."""

    # Search is a candidate-discovery operation.  Returning the whole chunk
    # here makes every subsequent Agent turn pay for text that may never be
    # read or cited.  Keep the authoritative chunk in the store and expose a
    # short extract; ``paper_read`` remains the explicit full-text operation.
    SEARCH_SNIPPET_CHARS = 500
    READ_TEXT_CHARS = 8_000

    def __init__(
        self,
        knowledge_store: ResearchKnowledgeStore,
        *,
        dense_embedder: Any | None = None,
        reranker: Reranker | None = None,
        feature_ranker: SupervisedResearchRanker | None = None,
        graph_enabled: bool = True,
        structure_fusion_enabled: bool = False,
        structure_section_weight: float = 0.5,
        structure_query_coverage_weight: float = 0.1,
        preserve_head_k: int = 0,
        reranker_context_window: int = 0,
        lexical_fusion_weight: float = 0.0,
        intent_section_fusion_enabled: bool = False,
        intent_section_fusion_weight: float = 0.1,
        intent_query_overlap_weight: float = 0.05,
        intent_rank_fusion_weight: float = 0.45,
        operator_budget_standard: int = 1,
        operator_budget_rigorous: int = 2,
        index_cache_size: int = 64,
    ) -> None:
        if not isinstance(knowledge_store, ResearchKnowledgeStore):
            raise TypeError("knowledge_store must expose visible_chunks and get")
        if not 0 <= operator_budget_standard <= 4 or not 0 <= operator_budget_rigorous <= 4:
            raise ValueError("operator budgets must be between 0 and 4")
        if not 0 <= index_cache_size <= 256:
            raise ValueError("index_cache_size must be between 0 and 256")
        self.knowledge_store = knowledge_store
        self.dense_embedder = dense_embedder
        self.reranker = reranker
        self.feature_ranker = feature_ranker
        self.graph_enabled = bool(graph_enabled)
        self.structure_fusion_enabled = bool(structure_fusion_enabled)
        self.structure_section_weight = float(structure_section_weight)
        self.structure_query_coverage_weight = float(structure_query_coverage_weight)
        self.preserve_head_k = int(preserve_head_k)
        self.reranker_context_window = int(reranker_context_window)
        self.lexical_fusion_weight = float(lexical_fusion_weight)
        self.intent_section_fusion_enabled = bool(intent_section_fusion_enabled)
        self.intent_section_fusion_weight = float(intent_section_fusion_weight)
        self.intent_query_overlap_weight = float(intent_query_overlap_weight)
        self.intent_rank_fusion_weight = float(intent_rank_fusion_weight)
        self.operator_budget_standard = int(operator_budget_standard)
        self.operator_budget_rigorous = int(operator_budget_rigorous)
        self.index_cache_size = int(index_cache_size)
        self._index_cache: OrderedDict[
            tuple[tuple[str, str, str, int], ...],
            tuple[BM25Index, InMemoryDenseIndex | None],
        ] = OrderedDict()
        self._index_cache_lock = threading.RLock()
        self.operators: dict[str, _Operator] = {
            item.name: item
            for item in (
                ParentSectionOperator(),
                StructuredTableOperator(),
                SourceCoverageOperator(),
                LayoutNeighborOperator(),
            )
        }

    def _visible(self, request: ResearchQuery, principal: AccessContext) -> tuple[KnowledgeChunk, ...]:
        return self.knowledge_store.visible_chunks(
            principal,
            source_uris=request.source_uris or None,
            knowledge_base_ids=request.knowledge_base_ids or None,
            latest_only=request.latest_only,
        )

    @staticmethod
    def _hybrid_chunk(chunk: KnowledgeChunk):
        hybrid = knowledge_to_hybrid_chunk(chunk)
        metadata = dict(hybrid.metadata)
        metadata.setdefault("source", _source_label(chunk))
        metadata.setdefault("retrieval_layout", _metadata_text(chunk))
        metadata.setdefault("retrieval_structure", _metadata_text(chunk))
        return hybrid.model_copy(update={"metadata": metadata})

    @staticmethod
    def _search_text(chunk: KnowledgeChunk) -> str:
        """Return authoritative body text for dense and learned reranking."""

        # PDF structure chunks already retain their heading in ``text``.  A
        # repeated title/heading prefix diluted short passages in paired
        # QASPER ablations, so metadata is weighted only by BM25 below.
        return chunk.text

    def _search_chunk(self, chunk: KnowledgeChunk):
        hybrid = self._hybrid_chunk(chunk)
        return hybrid.model_copy(update={"text": self._search_text(chunk)})

    @staticmethod
    def _corpus_cache_key(
        corpus: Sequence[KnowledgeChunk],
    ) -> tuple[tuple[str, str, str, int], ...]:
        return tuple(
            (
                chunk.chunk_id,
                chunk.version,
                str(
                    chunk.metadata.get("content_hash")
                    or hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
                ),
                len(chunk.text),
            )
            for chunk in corpus
        )

    def _indexes(
        self,
        corpus: Sequence[KnowledgeChunk],
    ) -> tuple[BM25Index, InMemoryDenseIndex | None]:
        key = self._corpus_cache_key(corpus)
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
                InMemoryDenseIndex(indexed, self.dense_embedder)
                if self.dense_embedder is not None
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
    ) -> list[KnowledgeHit]:
        selected = [
            chunk
            for chunk in corpus
            if not require_table
            or str(chunk.metadata.get("kind", "")).casefold() == "table"
            or bool(chunk.metadata.get("table_rows"))
            or "table" in {str(item).casefold() for item in chunk.metadata.get("block_types", ())}
        ]
        if not selected:
            return []
        indexed = [self._hybrid_chunk(chunk) for chunk in selected]
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

    @staticmethod
    def _to_hit(
        result: Any,
        chunk: KnowledgeChunk,
        backend: str,
        query: str = "",
    ) -> KnowledgeHit:
        match = lexical_match(query or chunk.text, chunk.text)
        return KnowledgeHit(
            chunk=chunk,
            score=max(0.0, float(result.score)),
            lexical_score=match.score,
            semantic_score=max(0.0, float(result.base_score or 0.0)),
            matched_terms=match.matched_terms,
            retrieval_profile="research_unified",
            retrieval_backend=backend,
        )

    def _unified_search(self, request: ResearchQuery, principal: AccessContext, corpus: Sequence[KnowledgeChunk]) -> list[_Candidate]:
        bm25_index, dense_index = self._indexes(corpus)
        allowed = frozenset(chunk.chunk_id for chunk in corpus)
        search_request = HybridSearchRequest(
            query=request.query,
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
            self._to_hit(result, by_id[result.chunk.chunk_id], "research_bm25", request.query)
            for result in bm25_response.hits
        ]
        dense_hits: list[KnowledgeHit] = []
        if dense_index is not None:
            dense_response = dense_index.search(search_request)
            dense_hits = [
                self._to_hit(result, by_id[result.chunk.chunk_id], "research_dense", request.query)
                for result in dense_response.hits
            ]
        return _rrf_merge(bm25_hits, dense_hits)

    @staticmethod
    def _requirements(query: str) -> tuple[EvidenceRequirement, ...]:
        lowered = query.casefold()
        words = set(re.findall(r"[a-z][a-z-]+", lowered))
        numbers = tuple(dict.fromkeys(_NUMERIC_RE.findall(query)))
        comparison = bool(words & _COMPARISON_WORDS)
        minimum_sources = 2 if comparison else 1
        evidence_types = ["claim"]
        if numbers or words & _TABLE_WORDS:
            evidence_types.append("numeric_or_table")
        return (
            EvidenceRequirement(
                subquestion=query,
                required_entities=(),
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
        for index, requirement in enumerate(requirements):
            numeric_missing = any(value.casefold() not in joined for value in requirement.numeric_constraints)
            source_missing = len(sources) < requirement.minimum_sources
            table_needed = "numeric_or_table" in requirement.evidence_types and (
                not any(str(item.hit.chunk.metadata.get("kind", "")).casefold() == "table" for item in candidates)
                or numeric_missing
            )
            layout_needed = bool(features & _LAYOUT_WORDS) and not any(_page_label(item.hit.chunk) for item in candidates)
            if source_missing:
                gaps.append(EvidenceGap(operator="source_coverage", reason="not enough distinct paper sources", query=query, requirement_index=index))
            elif table_needed:
                gaps.append(EvidenceGap(operator="structured_table", reason="numeric or table evidence is incomplete", query=query, requirement_index=index))
            elif layout_needed:
                gaps.append(EvidenceGap(operator="layout_neighbor", reason="page/layout evidence is incomplete", query=query, requirement_index=index))
            else:
                covered.append(index)
        if not gaps and candidates and len(candidates) < 1:
            gaps.append(EvidenceGap(operator="parent_section", reason="no citation-ready evidence was retrieved", query=query))
        return CoverageReport(
            covered_requirement_indices=tuple(covered),
            gaps=tuple(gaps),
            covered_entities=(),
            source_count=len(sources),
            citation_ready_count=sum(1 for item in candidates if item.hit.chunk.text.strip()),
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
    ) -> list[_Candidate]:
        limited = list(candidates)
        preserved_head = limited[: self.preserve_head_k]
        if self.reranker is not None and limited:
            by_id = {item.chunk_id: item for item in corpus}
            def _reranker_text(candidate: _Candidate) -> str:
                chunk = candidate.hit.chunk
                if not self.reranker_context_window:
                    return self._search_text(chunk)
                pieces = [self._search_text(chunk)]
                current = chunk
                for _ in range(self.reranker_context_window):
                    neighbor_id = current.next_chunk_id
                    if not neighbor_id or neighbor_id not in by_id:
                        break
                    current = by_id[neighbor_id]
                    pieces.append(self._search_text(current))
                return "\n".join(pieces)
            scores = list(
                self.reranker.score(
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
            limited = [_Candidate(replace(item.hit, score=score), item.sources) for item, score in zip(limited, raw_scores, strict=True)]
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
            )
            for score, chunk_id in scored
        ]

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
        corpus = self._visible(request, principal)
        if not corpus:
            empty = CoverageReport(gaps=(EvidenceGap(operator="parent_section", reason="authorized corpus is empty", query=request.query),))
            return ResearchSearchResult(query=request.query, evidence=(), candidate_count=0, retrieval_rounds=1, coverage=empty, elapsed_ms=0.0)
        requirements = self._requirements(request.query)
        base = self._unified_search(request, principal, corpus)
        coverage = self._coverage(requirements, base, request.query)
        activated: list[str] = []
        candidates = list(base)
        budget = self.operator_budget_rigorous if request.mode == "rigorous" else self.operator_budget_standard
        for gap in coverage.gaps[:budget]:
            operator = self.operators[gap.operator]
            extras = operator.search(self, gap.query, principal, corpus, request.candidate_k)
            candidates = self._merge_candidates(candidates, extras)
            activated.append(operator.name)
        learned_final = self._rerank_once(request.query, candidates, corpus, request.top_k)
        final = learned_final
        if self.intent_section_fusion_enabled:
            intent_final = self._intent_reorder(request.query, learned_final)
            final = self._intent_rank_fuse(learned_final, intent_final)[: request.top_k]
        final_coverage = self._coverage(requirements, final, request.query)
        evidence = tuple(
            ResearchEvidence(
                evidence_id=_evidence_id(item.hit.chunk),
                chunk_id=item.hit.chunk.chunk_id,
                title=str(item.hit.chunk.metadata.get("title")) if item.hit.chunk.metadata.get("title") is not None else None,
                source=_source_label(item.hit.chunk),
                section=_section_label(item.hit.chunk),
                page=_page_label(item.hit.chunk),
                version=item.hit.chunk.version,
                text=item.hit.chunk.text[: self.SEARCH_SNIPPET_CHARS],
                score=max(0.0, float(item.hit.score)),
                retrieval_sources=tuple(dict.fromkeys((*item.sources, item.hit.retrieval_backend or "research"))),
            )
            for item in final
        )
        return ResearchSearchResult(
            query=request.query,
            evidence=evidence,
            candidate_count=len(candidates),
            retrieval_rounds=1 + int(bool(activated)),
            activated_operators=tuple(dict.fromkeys(activated)),
            coverage=final_coverage,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    def read_evidence(self, evidence_id: str, principal: AccessContext) -> ResearchEvidence:
        cleaned = str(evidence_id).strip()
        if not cleaned or len(cleaned) > 1_024:
            raise ValueError("evidence_id is invalid")
        visible = self.knowledge_store.visible_chunks(principal, latest_only=True)
        for chunk in visible:
            if cleaned in {chunk.chunk_id, _evidence_id(chunk)}:
                return ResearchEvidence(
                    evidence_id=_evidence_id(chunk),
                    chunk_id=chunk.chunk_id,
                    title=str(chunk.metadata.get("title")) if chunk.metadata.get("title") is not None else None,
                    source=_source_label(chunk),
                    section=_section_label(chunk),
                    page=_page_label(chunk),
                    version=chunk.version,
                    text=chunk.text[: self.READ_TEXT_CHARS],
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
        missing: list[str] = []
        for evidence_id in ids:
            try:
                resolved.append(self.read_evidence(evidence_id, principal))
            except KeyError:
                missing.append(evidence_id)
        claim_tokens = set(tokenise(cleaned_claim))
        support_tokens = set(tokenise(" ".join(item.text for item in resolved)))
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

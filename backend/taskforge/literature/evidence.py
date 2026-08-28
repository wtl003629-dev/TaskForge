"""Intent-aware evidence retrieval that cannot execute without a ResearchScope."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from ..knowledge import AccessContext, tokenise
from ..rag_experiment_profile import (
    RAGExperimentProfile,
    resolve_rag_experiment_profile,
)
from ..research_protocol import (
    EvidenceCard,
    EvidenceIntent,
    EvidenceSearchRequest,
    RetrievalConfidence,
    ScopeEvidenceResult,
)
from ..research_retrieval import (
    CitationVerification,
    ResearchEvidence,
    ResearchQuery,
    ResearchRetrievalService,
    ResearchSearchResult,
)
from .evidence_query_expander import (
    EvidenceQueryExpander,
    EvidenceQueryExpansionError,
    protected_query_terms,
)
from .repository import LiteratureAccess, SQLiteLiteratureRepository

_INTENT_TERMS: tuple[tuple[EvidenceIntent, frozenset[str]], ...] = (
    ("numeric_table", frozenset({"table", "metric", "score", "percent", "recall", "accuracy", "数值", "表格", "指标"})),
    ("cross_paper_comparison", frozenset({"compare", "versus", "difference", "across", "对比", "比较", "差异"})),
    ("figure_or_layout", frozenset({"figure", "caption", "page", "appendix", "图", "页", "附录"})),
    ("experimental_setup", frozenset({"experiment", "baseline", "hyperparameter", "setup", "实验", "基线", "参数"})),
    ("method_definition", frozenset({"method", "algorithm", "architecture", "framework", "方法", "算法", "架构"})),
    ("claim_verification", frozenset({"verify", "support", "evidence", "claim", "验证", "证据", "论断"})),
    ("related_work", frozenset({"related", "prior", "literature", "相关工作", "已有研究"})),
)

def route_evidence_intent(query: str, declared: EvidenceIntent) -> EvidenceIntent:
    if declared != "general_fact":
        return declared
    lowered = query.casefold()
    tokens = set(re.findall(r"[\w-]+", lowered, re.UNICODE))
    for intent, terms in _INTENT_TERMS:
        if terms & tokens or any(term in lowered for term in terms if len(term) > 1):
            return intent
    return "general_fact"


class ScopeBoundEvidenceService:
    def __init__(
        self,
        repository: SQLiteLiteratureRepository,
        retrieval: ResearchRetrievalService,
        *,
        experiment_profile: RAGExperimentProfile | None = None,
        query_expander: EvidenceQueryExpander | None = None,
        query_expansion_mode: Literal["original", "keyword", "synonym", "full"] = "original",
    ) -> None:
        self.repository = repository
        self.retrieval = retrieval
        self.experiment_profile = experiment_profile or resolve_rag_experiment_profile(
            "current"
        )
        self.query_expander = query_expander
        self.query_expansion_mode = query_expansion_mode

    @staticmethod
    def _principal(access: LiteratureAccess) -> AccessContext:
        return AccessContext(tenant_id=access.tenant_id, user_id=access.user_id)

    def _cards(
        self,
        scope_id: str,
        scope_version: int,
        result: ResearchSearchResult,
    ) -> list[EvidenceCard]:
        cards: list[EvidenceCard] = []
        for item in result.evidence:
            paper_id = item.source.removeprefix("paper://") if item.source.startswith("paper://") else None
            cards.append(
                EvidenceCard(
                    evidence_id=item.evidence_id,
                    scope_id=scope_id,
                    scope_version=scope_version,
                    paper_id=paper_id,
                    chunk_id=item.chunk_id,
                    rag_profile=self.experiment_profile.name,
                    rag_ablation=self.experiment_profile.ablation,
                    source=item.source,
                    title=item.title,
                    section=item.section,
                    page=item.page,
                    evidence_type=item.evidence_type,
                    visual_artifact_ids=list(item.visual_artifact_ids),
                    visual_pending=item.visual_pending,
                    snippet=item.text[:3_000],
                    text_start=item.text_start,
                    text_end=item.text_end,
                    presentation_strategy=item.presentation_strategy,
                    score=item.score,
                    retrieval_sources=list(item.retrieval_sources),
                    verification_status="read",
                )
            )
        return cards

    @staticmethod
    def _merge_cards(
        first: Sequence[EvidenceCard],
        second: Sequence[EvidenceCard],
        top_k: int,
    ) -> list[EvidenceCard]:
        by_id: dict[str, EvidenceCard] = {card.evidence_id: card for card in first}
        for card in second:
            current = by_id.get(card.evidence_id)
            if current is None or card.score > current.score:
                by_id[card.evidence_id] = card
            elif current is not None:
                by_id[card.evidence_id] = current.model_copy(
                    update={
                        "retrieval_sources": list(
                            dict.fromkeys([*current.retrieval_sources, *card.retrieval_sources])
                        )
                    }
                )
        return sorted(by_id.values(), key=lambda card: (-card.score, card.evidence_id))[:top_k]

    @staticmethod
    def _confidence(
        query: str,
        cards: Sequence[EvidenceCard],
        *,
        selected_papers: Sequence[str],
        intent: EvidenceIntent,
    ) -> RetrievalConfidence:
        query_terms = set(tokenise(query))
        evidence_text = " ".join(card.snippet for card in cards)
        evidence_terms = set(tokenise(evidence_text))
        term_coverage = (
            len(query_terms & evidence_terms) / len(query_terms) if query_terms else 0.0
        )
        numeric = tuple(dict.fromkeys(re.findall(r"[-+]?\d+(?:\.\d+)?%?", query)))
        numeric_coverage = (
            sum(value in evidence_text for value in numeric) / len(numeric) if numeric else 1.0
        )
        paper_ids = {card.paper_id for card in cards if card.paper_id}
        scope_coverage = len(paper_ids) / max(1, len(selected_papers))
        top_score = cards[0].score if cards else 0.0
        second_score = cards[1].score if len(cards) > 1 else 0.0
        margin = top_score - second_score
        section_match = sum(bool(card.section) for card in cards) / len(cards) if cards else 0.0
        required_sources = 2 if intent == "cross_paper_comparison" else 1
        source_coverage = min(1.0, len(paper_ids) / required_sources)
        reasons: list[str] = []
        if not cards:
            reasons.append("no citation-ready evidence")
        if query_terms and term_coverage < 0.25:
            reasons.append("query term coverage is below 0.25")
        if numeric_coverage < 1.0:
            reasons.append("numeric constraints are not fully covered")
        if source_coverage < 1.0:
            reasons.append("not enough selected papers are represented")
        protected = protected_query_terms(query)
        entity_terms = tuple(
            term
            for term in protected
            if not re.fullmatch(r"[-+]?\d+(?:[.,]\d+)*(?:%|[a-z]+)?", term)
            and term
            not in {
                "not",
                "no",
                "without",
                "except",
                "exclude",
                "excluding",
                "never",
                "neither",
                "nor",
                "less",
                "more",
                "before",
                "after",
                "versus",
                "vs",
                "compare",
                "comparison",
                "between",
            }
        )
        entity_coverage = (
            sum(term in evidence_text.casefold() for term in entity_terms)
            / len(entity_terms)
            if entity_terms
            else 1.0
        )
        if entity_coverage < 1.0:
            reasons.append("named entity constraints are not fully covered")
        if intent == "figure_or_layout" and not any(card.page for card in cards):
            reasons.append("page/layout provenance is missing")
        if intent == "figure_or_layout" and not any(
            card.evidence_type == "figure" for card in cards
        ):
            reasons.append("figure evidence is not represented")
        unresolved_visual_count = sum(card.visual_pending for card in cards)
        if unresolved_visual_count:
            reasons.append("one or more retrieved visuals remain unparsed")
        return RetrievalConfidence(
            top_score=top_score,
            top1_top2_margin=margin,
            query_term_coverage=term_coverage,
            entity_coverage=entity_coverage,
            numeric_constraint_coverage=numeric_coverage,
            source_coverage=source_coverage,
            section_match=section_match,
            citation_ready_count=len(cards),
            scope_paper_coverage=scope_coverage,
            unresolved_visual_count=unresolved_visual_count,
            sufficient=not reasons,
            reasons=reasons,
        )

    async def search(
        self,
        access: LiteratureAccess,
        request: EvidenceSearchRequest,
    ) -> ScopeEvidenceResult:
        scope = self.repository.get_scope(
            access,
            request.scope_id,
            version=request.scope_version,
        )
        if scope.status != "ready":
            raise ValueError("research scope must be ready before evidence retrieval")
        intent = route_evidence_intent(request.query, request.intent)
        filters = {
            "source_uris": tuple(f"paper://{paper_id}" for paper_id in scope.selected_paper_ids),
            "knowledge_base_ids": (
                self.experiment_profile.knowledge_base_id(
                    f"research-scope:{scope.scope_id}:v{scope.scope_version}"
                ),
            ),
        }
        variants: tuple[str, ...] = ()
        expansion_error: str | None = None
        if self.query_expansion_mode != "original" and self.query_expander is not None:
            try:
                synonym, keyword = await self.query_expander.expand(
                    request.query,
                    intent,
                )
                variants = (
                    (keyword,)
                    if self.query_expansion_mode == "keyword"
                    else (synonym,)
                    if self.query_expansion_mode == "synonym"
                    else (synonym, keyword)
                )
            except EvidenceQueryExpansionError as exc:
                expansion_error = str(exc)
        first = self.retrieval.search(
            ResearchQuery(
                query=request.query,
                query_variants=variants,
                top_k=request.top_k,
                candidate_k=request.candidate_k,
                mode=request.mode,
                **filters,
            ),
            self._principal(access),
        )
        cards = self._cards(scope.scope_id, scope.scope_version, first)
        confidence = self._confidence(
            request.query,
            cards,
            selected_papers=scope.selected_paper_ids,
            intent=intent,
        )
        rewritten_query = variants[0] if variants else None
        rounds = first.retrieval_rounds
        operators = list(first.activated_operators)
        if cards:
            self.repository.save_evidence(access, cards)
        return ScopeEvidenceResult(
            scope_id=scope.scope_id,
            scope_version=scope.scope_version,
            query=request.query,
            query_variants=list((request.query, *variants)),
            query_expansion_error=expansion_error,
            routed_intent=intent,
            rewritten_query=rewritten_query,
            retrieval_rounds=rounds,
            activated_operators=list(dict.fromkeys(operators)),
            evidence=cards,
            confidence=confidence,
            retrieval_traces=[first.trace] if first.trace is not None else [],
            retrieval_route=first.retrieval_route,
        )

    async def aclose(self) -> None:
        close = getattr(self.query_expander, "aclose", None)
        if callable(close):
            await close()

    def list_evidence(
        self,
        access: LiteratureAccess,
        scope_id: str,
        *,
        scope_version: int | None = None,
        paper_id: str | None = None,
    ) -> list[EvidenceCard]:
        """List only evidence produced by the active RAG profile.

        Experimental cards can intentionally coexist with current cards in the
        literature database.  Keeping this filter in the scope-bound service
        prevents an API listing or later claim assembly from crossing that
        profile boundary.
        """

        return [
            card
            for card in self.repository.list_evidence(
                access,
                scope_id,
                version=scope_version,
                paper_id=paper_id,
            )
            if card.rag_profile == self.experiment_profile.name
            and card.rag_ablation == self.experiment_profile.ablation
        ]

    def read_evidence(
        self,
        access: LiteratureAccess,
        scope_id: str,
        evidence_id: str,
        *,
        scope_version: int | None = None,
    ) -> ResearchEvidence:
        scope = self.repository.get_scope(access, scope_id, version=scope_version)
        cards = {
            card.evidence_id: card
            for card in self.list_evidence(
                access,
                scope.scope_id,
                scope_version=scope.scope_version,
            )
        }
        stored = cards.get(evidence_id)
        if stored is None:
            raise KeyError("evidence_id is outside the research scope")
        resolved = self.retrieval.read_evidence(
            stored.chunk_id or evidence_id,
            self._principal(access),
        )
        return resolved.model_copy(update={"evidence_id": stored.evidence_id})

    def verify_citation(
        self,
        access: LiteratureAccess,
        scope_id: str,
        claim: str,
        evidence_ids: Sequence[str],
        *,
        scope_version: int | None = None,
    ) -> CitationVerification:
        scope = self.repository.get_scope(access, scope_id, version=scope_version)
        allowed = {
            card.evidence_id: card
            for card in self.list_evidence(
                access,
                scope.scope_id,
                scope_version=scope.scope_version,
            )
        }
        if set(evidence_ids) - set(allowed):
            raise KeyError("citation contains evidence outside the research scope")
        resolved_keys = [
            allowed[evidence_id].chunk_id or evidence_id
            for evidence_id in evidence_ids
        ]
        raw = self.retrieval.verify_citation(
            claim,
            resolved_keys,
            self._principal(access),
        )
        missing_keys = set(raw.missing_evidence_ids)
        resolved_ids = [
            evidence_id
            for evidence_id, key in zip(evidence_ids, resolved_keys, strict=True)
            if key not in missing_keys
        ]
        missing_ids = [
            evidence_id
            for evidence_id, key in zip(evidence_ids, resolved_keys, strict=True)
            if key in missing_keys
        ]
        return raw.model_copy(
            update={
                "evidence_ids": tuple(evidence_ids),
                "resolved_evidence_ids": tuple(resolved_ids),
                "missing_evidence_ids": tuple(missing_ids),
            }
        )


__all__ = ["ScopeBoundEvidenceService", "route_evidence_intent"]

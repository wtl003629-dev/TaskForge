"""Intent-aware evidence retrieval that cannot execute without a ResearchScope."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..knowledge import AccessContext, tokenise
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

_REWRITE_HINTS: dict[EvidenceIntent, str] = {
    "general_fact": "definition evidence conclusion",
    "method_definition": "method algorithm architecture implementation",
    "experimental_setup": "experimental setup dataset baseline hyperparameter",
    "numeric_table": "table metric result value percentage",
    "cross_paper_comparison": "comparison across papers similarities differences results",
    "figure_or_layout": "figure caption page appendix layout",
    "claim_verification": "supporting evidence result conclusion limitation",
    "related_work": "related work prior method comparison",
}


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
        rewrite_enabled: bool = True,
    ) -> None:
        self.repository = repository
        self.retrieval = retrieval
        self.rewrite_enabled = bool(rewrite_enabled)

    @staticmethod
    def _principal(access: LiteratureAccess) -> AccessContext:
        return AccessContext(tenant_id=access.tenant_id, user_id=access.user_id)

    @staticmethod
    def _cards(
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
                    source=item.source,
                    title=item.title,
                    section=item.section,
                    page=item.page,
                    evidence_type="table" if "structured_table" in item.retrieval_sources else "paragraph",
                    snippet=item.text[:500],
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
        if top_score < 0.005:
            reasons.append("top retrieval score is too low")
        if intent == "figure_or_layout" and not any(card.page for card in cards):
            reasons.append("page/layout provenance is missing")
        return RetrievalConfidence(
            top_score=top_score,
            top1_top2_margin=margin,
            query_term_coverage=term_coverage,
            entity_coverage=term_coverage,
            numeric_constraint_coverage=numeric_coverage,
            source_coverage=source_coverage,
            section_match=section_match,
            citation_ready_count=len(cards),
            scope_paper_coverage=scope_coverage,
            sufficient=not reasons,
            reasons=reasons,
        )

    def search(
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
                f"research-scope:{scope.scope_id}:v{scope.scope_version}",
            ),
        }
        first = self.retrieval.search(
            ResearchQuery(
                query=request.query,
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
        rewritten_query: str | None = None
        rounds = 1
        operators = list(first.activated_operators)
        if self.rewrite_enabled and not confidence.sufficient:
            rounds = 2
            hint = _REWRITE_HINTS[intent]
            rewritten_query = f"{request.query[: 3_999 - len(hint)]} {hint}"
            second = self.retrieval.search(
                ResearchQuery(
                    query=rewritten_query,
                    top_k=request.top_k,
                    candidate_k=request.candidate_k,
                    mode="rigorous",
                    **filters,
                ),
                self._principal(access),
            )
            cards = self._merge_cards(
                cards,
                self._cards(scope.scope_id, scope.scope_version, second),
                request.top_k,
            )
            operators.extend(second.activated_operators)
            confidence = self._confidence(
                request.query,
                cards,
                selected_papers=scope.selected_paper_ids,
                intent=intent,
            )
        if cards:
            self.repository.save_evidence(access, cards)
        return ScopeEvidenceResult(
            scope_id=scope.scope_id,
            scope_version=scope.scope_version,
            query=request.query,
            routed_intent=intent,
            rewritten_query=rewritten_query,
            retrieval_rounds=rounds,
            activated_operators=list(dict.fromkeys(operators)),
            evidence=cards,
            confidence=confidence,
        )

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
            for card in self.repository.list_evidence(
                access,
                scope.scope_id,
                version=scope.scope_version,
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
            for card in self.repository.list_evidence(
                access,
                scope.scope_id,
                version=scope.scope_version,
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

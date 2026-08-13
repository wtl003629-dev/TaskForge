"""Small, host-owned contracts for research-agent handoffs.

These models are intentionally projections, not a second conversation
history.  A role can publish IDs and bounded deltas to the shared blackboard;
the host keeps the original evidence and trajectory in durable storage.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from .domain import StrictModel, utc_now

LiteratureIntent = Literal[
    "topic",
    "method",
    "dataset",
    "author",
    "venue",
    "foundational",
    "recent",
    "citation",
]

EvidenceIntent = Literal[
    "general_fact",
    "method_definition",
    "experimental_setup",
    "numeric_table",
    "cross_paper_comparison",
    "figure_or_layout",
    "claim_verification",
    "related_work",
]

ShortResearchText = Annotated[str, Field(min_length=1, max_length=500)]
OutlineText = Annotated[str, Field(min_length=1, max_length=300)]


class LiteratureRequest(StrictModel):
    """One host-owned request to discover papers, not passages."""

    protocol: Literal["research.literature_request.v1"] = (
        "research.literature_request.v1"
    )
    request_id: str = Field(
        default_factory=lambda: f"literature-{uuid4()}",
        min_length=1,
        max_length=240,
    )
    query: str = Field(min_length=1, max_length=4_000)
    research_questions: list[str] = Field(default_factory=list, max_length=16)
    year_from: int | None = Field(default=None, ge=1000, le=3000)
    year_to: int | None = Field(default=None, ge=1000, le=3000)
    venues: list[str] = Field(default_factory=list, max_length=32)
    authors: list[str] = Field(default_factory=list, max_length=64)
    required_terms: list[str] = Field(default_factory=list, max_length=64)
    excluded_terms: list[str] = Field(default_factory=list, max_length=64)
    paper_types: list[str] = Field(default_factory=list, max_length=16)
    result_limit: int = Field(default=20, ge=1, le=100)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "query",
        "research_questions",
        "venues",
        "authors",
        "required_terms",
        "excluded_terms",
        "paper_types",
        mode="before",
    )
    @classmethod
    def clean_text_fields(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (list, tuple)):
            return list(
                dict.fromkeys(str(item).strip() for item in value if str(item).strip())
            )
        return value

    @model_validator(mode="after")
    def year_range_is_ordered(self) -> LiteratureRequest:
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("year_from must be less than or equal to year_to")
        overlap = {
            item.casefold() for item in self.required_terms
        } & {item.casefold() for item in self.excluded_terms}
        if overlap:
            raise ValueError("required_terms and excluded_terms must not overlap")
        return self


class SearchQuery(StrictModel):
    protocol: Literal["research.search_query.v1"] = "research.search_query.v1"
    query_id: str = Field(
        default_factory=lambda: f"query-{uuid4()}",
        min_length=1,
        max_length=240,
    )
    text: str = Field(min_length=1, max_length=4_000)
    intent: LiteratureIntent = "topic"
    priority: int = Field(default=1, ge=1, le=100)
    provider_filters: dict[str, object] = Field(default_factory=dict)

    @field_validator("text", mode="before")
    @classmethod
    def clean_text(cls, value: object) -> str:
        return str(value).strip()


class PaperCard(StrictModel):
    """Canonical paper-level result assembled from one or more providers."""

    protocol: Literal["research.paper_card.v1"] = "research.paper_card.v1"
    paper_id: str = Field(min_length=1, max_length=240)
    canonical_title: str = Field(min_length=1, max_length=2_000)
    authors: list[str] = Field(default_factory=list, max_length=256)
    abstract: str = Field(default="", max_length=50_000)
    short_description: str = Field(default="", max_length=500)
    year: int | None = Field(default=None, ge=1000, le=3000)
    venue: str | None = Field(default=None, max_length=1_000)
    doi: str | None = Field(default=None, max_length=512)
    arxiv_id: str | None = Field(default=None, max_length=128)
    semantic_scholar_id: str | None = Field(default=None, max_length=128)
    openalex_id: str | None = Field(default=None, max_length=256)
    source_urls: list[str] = Field(default_factory=list, max_length=32)
    pdf_url: str | None = Field(default=None, max_length=4_096)
    citation_count: int | None = Field(default=None, ge=0)
    references: list[str] = Field(default_factory=list, max_length=500)
    cited_by: list[str] = Field(default_factory=list, max_length=500)
    matched_queries: list[str] = Field(default_factory=list, max_length=32)
    provider_ranks: dict[str, int] = Field(default_factory=dict, max_length=32)
    matched_requirements: list[str] = Field(default_factory=list, max_length=64)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    relevance_reason: str = Field(default="", max_length=2_000)
    verification_status: Literal[
        "provider_verified",
        "cross_source_verified",
        "metadata_partial",
        "unverified",
    ] = "unverified"
    full_text_status: Literal[
        "not_requested",
        "available",
        "abstract_only",
        "ingested",
        "failed",
    ] = "not_requested"

    @field_validator("canonical_title", mode="before")
    @classmethod
    def clean_title(cls, value: object) -> str:
        return " ".join(str(value).split())

    @field_validator(
        "authors",
        "source_urls",
        "references",
        "cited_by",
        "matched_queries",
        "matched_requirements",
        mode="before",
    )
    @classmethod
    def unique_lists(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            return list(
                dict.fromkeys(str(item).strip() for item in value if str(item).strip())
            )
        return value


class ResearchScope(StrictModel):
    """Host-owned immutable-version boundary for evidence retrieval."""

    protocol: Literal["research.scope.v1"] = "research.scope.v1"
    scope_id: str = Field(
        default_factory=lambda: f"scope-{uuid4()}", min_length=1, max_length=240
    )
    tenant_id: str = Field(min_length=1, max_length=256)
    owner_user_id: str = Field(min_length=1, max_length=256)
    conversation_id: str = Field(min_length=1, max_length=240)
    request_id: str = Field(min_length=1, max_length=240)
    selected_paper_ids: list[str] = Field(min_length=1, max_length=128)
    selected_source_uris: list[str] = Field(default_factory=list, max_length=128)
    excluded_paper_ids: list[str] = Field(default_factory=list, max_length=256)
    user_intent: str = Field(min_length=1, max_length=4_000)
    allowed_expansion: bool = False
    scope_version: int = Field(default=1, ge=1)
    status: Literal[
        "draft",
        "confirmed",
        "ingesting",
        "ready",
        "expansion_requested",
        "closed",
    ] = "draft"
    created_at: datetime = Field(default_factory=utc_now)
    confirmed_at: datetime | None = None

    @field_validator(
        "selected_paper_ids",
        "selected_source_uris",
        "excluded_paper_ids",
        mode="before",
    )
    @classmethod
    def unique_scope_lists(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            return list(
                dict.fromkeys(str(item).strip() for item in value if str(item).strip())
            )
        return value

    @model_validator(mode="after")
    def selections_do_not_overlap(self) -> ResearchScope:
        if set(self.selected_paper_ids) & set(self.excluded_paper_ids):
            raise ValueError("selected and excluded papers must not overlap")
        if self.status != "draft" and self.confirmed_at is None:
            raise ValueError("a non-draft scope requires confirmed_at")
        return self


class EvidenceSearchRequest(StrictModel):
    protocol: Literal["research.evidence_search.v1"] = "research.evidence_search.v1"
    scope_id: str = Field(min_length=1, max_length=240)
    scope_version: int | None = Field(default=None, ge=1)
    query: str = Field(min_length=1, max_length=4_000)
    intent: EvidenceIntent = "general_fact"
    top_k: int = Field(default=10, ge=1, le=50)
    candidate_k: int = Field(default=50, ge=10, le=100)
    mode: Literal["standard", "rigorous"] = "standard"

    @model_validator(mode="after")
    def candidate_budget_covers_output(self) -> EvidenceSearchRequest:
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        return self


class ResearchPlan(StrictModel):
    protocol: Literal["research.plan.v1"] = "research.plan.v1"
    research_questions: list[ShortResearchText] = Field(default_factory=list, max_length=3)
    evidence_requirements: list[ShortResearchText] = Field(default_factory=list, max_length=4)
    output_outline: list[OutlineText] = Field(default_factory=list, max_length=5)


class EvidenceCard(StrictModel):
    protocol: Literal["research.evidence_card.v1"] = "research.evidence_card.v1"
    evidence_id: str = Field(min_length=1, max_length=1_024)
    scope_id: str | None = Field(default=None, max_length=240)
    scope_version: int | None = Field(default=None, ge=1)
    paper_id: str | None = Field(default=None, max_length=240)
    chunk_id: str | None = Field(default=None, max_length=512)
    source: str = Field(min_length=1, max_length=2_048)
    title: str | None = Field(default=None, max_length=500)
    section: str | None = Field(default=None, max_length=500)
    page: str | None = Field(default=None, max_length=240)
    evidence_type: str = Field(default="paragraph", min_length=1, max_length=128)
    snippet: str = Field(min_length=1, max_length=500)
    score: float = Field(default=0.0, ge=0.0)
    retrieval_sources: list[str] = Field(default_factory=list, max_length=16)
    supported_requirements: list[str] = Field(default_factory=list, max_length=32)
    verification_status: Literal["unread", "read", "verified", "unsupported"] = "unread"


class EvidenceLedger(StrictModel):
    protocol: Literal["research.evidence_ledger.v1"] = "research.evidence_ledger.v1"
    evidence_ids: list[str] = Field(default_factory=list, max_length=10)
    coverage_delta: list[ShortResearchText] = Field(default_factory=list, max_length=8)
    gaps: list[ShortResearchText] = Field(default_factory=list, max_length=8)
    receipt_ids: list[str] = Field(default_factory=list, max_length=4)


class ClaimRecord(StrictModel):
    protocol: Literal["research.claim.v1"] = "research.claim.v1"
    claim_id: str = Field(min_length=1, max_length=240)
    claim_text: str = Field(min_length=1, max_length=500)
    scope_id: str | None = Field(default=None, max_length=240)
    scope_version: int | None = Field(default=None, ge=1)
    paper_ids: list[str] = Field(default_factory=list, max_length=4)
    evidence_ids: list[str] = Field(default_factory=list, max_length=4)
    risk_level: Literal["low", "medium", "high"] = "medium"
    citation_status: Literal[
        "unverified", "verified", "unsupported", "scope_mismatch"
    ] = "unverified"
    verification_status: Literal["unverified", "verified", "needs_review"] = "unverified"


class LiteraturePlan(StrictModel):
    protocol: Literal["research.literature_plan.v1"] = "research.literature_plan.v1"
    request_id: str = Field(min_length=1, max_length=240)
    research_questions: list[str] = Field(default_factory=list, max_length=16)
    queries: list[SearchQuery] = Field(min_length=1, max_length=6)


class PaperCandidateLedger(StrictModel):
    protocol: Literal["research.paper_candidate_ledger.v1"] = (
        "research.paper_candidate_ledger.v1"
    )
    request_id: str = Field(min_length=1, max_length=240)
    paper_ids: list[str] = Field(default_factory=list, max_length=100)
    provider_failures: dict[str, str] = Field(default_factory=dict)
    total_candidates: int = Field(default=0, ge=0)


class RetrievalConfidence(StrictModel):
    protocol: Literal["research.retrieval_confidence.v1"] = (
        "research.retrieval_confidence.v1"
    )
    top_score: float = Field(default=0.0, ge=0.0)
    top1_top2_margin: float = Field(default=0.0)
    query_term_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    entity_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    numeric_constraint_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    source_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    section_match: float = Field(default=0.0, ge=0.0, le=1.0)
    citation_ready_count: int = Field(default=0, ge=0)
    scope_paper_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    sufficient: bool = False
    reasons: list[str] = Field(default_factory=list, max_length=16)


class ScopeEvidenceResult(StrictModel):
    protocol: Literal["research.scope_evidence_result.v1"] = (
        "research.scope_evidence_result.v1"
    )
    scope_id: str = Field(min_length=1, max_length=240)
    scope_version: int = Field(ge=1)
    query: str = Field(min_length=1, max_length=4_000)
    routed_intent: EvidenceIntent
    rewritten_query: str | None = Field(default=None, max_length=4_000)
    retrieval_rounds: int = Field(default=1, ge=1, le=2)
    activated_operators: list[str] = Field(default_factory=list, max_length=8)
    evidence: list[EvidenceCard] = Field(default_factory=list, max_length=50)
    confidence: RetrievalConfidence


class IngestionStatus(StrictModel):
    protocol: Literal["research.ingestion_status.v1"] = (
        "research.ingestion_status.v1"
    )
    job_id: str = Field(min_length=1, max_length=240)
    scope_id: str = Field(min_length=1, max_length=240)
    paper_id: str = Field(min_length=1, max_length=240)
    status: Literal[
        "queued", "uploaded", "fetching", "parsing", "indexed", "abstract_only", "failed"
    ]
    evidence_count: int = Field(default=0, ge=0)
    error: str | None = Field(default=None, max_length=2_000)
    updated_at: datetime = Field(default_factory=utc_now)


class ScopeExpansionRequest(StrictModel):
    protocol: Literal["research.scope_expansion_request.v1"] = (
        "research.scope_expansion_request.v1"
    )
    expansion_id: str = Field(
        default_factory=lambda: f"scope-expansion-{uuid4()}",
        min_length=1,
        max_length=240,
    )
    scope_id: str = Field(min_length=1, max_length=240)
    requested_by: Literal["evaluator", "critic"]
    reason: str = Field(min_length=1, max_length=2_000)
    proposed_paper_ids: list[str] = Field(default_factory=list, max_length=100)
    status: Literal["pending", "approved", "rejected"] = "pending"
    created_at: datetime = Field(default_factory=utc_now)
    decided_at: datetime | None = None


class DraftArtifact(StrictModel):
    protocol: Literal["research.draft.v1"] = "research.draft.v1"
    draft_id: str = Field(min_length=1, max_length=240)
    claim_ids: list[str] = Field(default_factory=list, max_length=6)
    section_count: int = Field(default=0, ge=0, le=128)


class ReviewPatch(StrictModel):
    protocol: Literal["research.review_patch.v1"] = "research.review_patch.v1"
    claim_id: str = Field(min_length=1, max_length=240)
    action: Literal["keep", "revise", "remove", "request_evidence"]
    replacement: str | None = Field(default=None, max_length=1_000)
    reason: str = Field(min_length=1, max_length=1_000)


class PlannerHandoff(StrictModel):
    protocol: Literal["research.planner_handoff.v1"] = "research.planner_handoff.v1"
    plan: ResearchPlan


class EvaluatorHandoff(StrictModel):
    protocol: Literal["research.evaluator_handoff.v1"] = "research.evaluator_handoff.v1"
    ledger: EvidenceLedger
    # The Host joins durable paper_search receipts into the downstream
    # blackboard. The model hands off IDs only and never serialises snippets.
    evidence_cards: list[EvidenceCard] = Field(default_factory=list, max_length=0)


class WriterHandoff(StrictModel):
    protocol: Literal["research.writer_handoff.v1"] = "research.writer_handoff.v1"
    draft: DraftArtifact
    claim_manifest: list[ClaimRecord] = Field(default_factory=list, max_length=6)


class CriticHandoff(StrictModel):
    protocol: Literal["research.critic_handoff.v1"] = "research.critic_handoff.v1"
    patches: list[ReviewPatch] = Field(default_factory=list, max_length=6)
    verdict: Literal["accept", "needs_revision", "more_evidence"]


ResearchRolePayload = Annotated[
    PlannerHandoff | EvaluatorHandoff | WriterHandoff | CriticHandoff,
    Field(discriminator="protocol"),
]


__all__ = [
    "ClaimRecord",
    "DraftArtifact",
    "EvidenceCard",
    "EvidenceLedger",
    "EvidenceIntent",
    "EvidenceSearchRequest",
    "IngestionStatus",
    "LiteratureIntent",
    "LiteraturePlan",
    "LiteratureRequest",
    "PaperCandidateLedger",
    "PaperCard",
    "PlannerHandoff",
    "ResearchPlan",
    "ResearchRolePayload",
    "ResearchScope",
    "RetrievalConfidence",
    "ReviewPatch",
    "EvaluatorHandoff",
    "WriterHandoff",
    "CriticHandoff",
    "ScopeEvidenceResult",
    "ScopeExpansionRequest",
    "SearchQuery",
]

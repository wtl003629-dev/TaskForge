from __future__ import annotations

from pathlib import Path

import pytest

from taskforge.knowledge import KnowledgeChunk
from taskforge.literature.evidence import (
    ScopeBoundEvidenceService,
    route_evidence_intent,
)
from taskforge.literature.repository import LiteratureAccess, SQLiteLiteratureRepository
from taskforge.persistent_context import SQLiteKnowledgeStore
from taskforge.research_protocol import (
    EvidenceSearchRequest,
    LiteratureRequest,
    PaperCard,
    ResearchScope,
)
from taskforge.research_retrieval import ResearchRetrievalService


def _ready_scope(
    tmp_path: Path,
) -> tuple[SQLiteLiteratureRepository, SQLiteKnowledgeStore, LiteratureAccess]:
    repository = SQLiteLiteratureRepository(tmp_path / "literature.sqlite3")
    knowledge = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite3")
    access = LiteratureAccess("tenant-a", "user-a", "conversation-a")
    repository.save_request(
        access,
        LiteratureRequest(request_id="request-1", query="retrieval evaluation"),
    )
    for paper_id in ("paper-selected", "paper-outside"):
        repository.upsert_paper(
            access,
            PaperCard(
                paper_id=paper_id,
                canonical_title=paper_id,
                abstract="abstract",
                verification_status="provider_verified",
                full_text_status="ingested",
            ),
        )
    repository.create_scope(
        access,
        ResearchScope(
            scope_id="scope-1",
            tenant_id=access.tenant_id,
            owner_user_id=access.user_id,
            conversation_id=access.conversation_id or "",
            request_id="request-1",
            selected_paper_ids=["paper-selected"],
            excluded_paper_ids=["paper-outside"],
            user_intent="Evaluate the selected paper.",
        ),
    )
    repository.transition_scope_status(access, "scope-1", "confirmed")
    repository.transition_scope_status(access, "scope-1", "ingesting")
    repository.transition_scope_status(access, "scope-1", "ready")
    knowledge.upsert_many(
        [
            KnowledgeChunk(
                chunk_id="chunk-selected",
                tenant_id=access.tenant_id,
                text="The dense retrieval method reports recall table value 95 percent.",
                source_uri="paper://paper-selected",
                document_id="research-paper:scope-1:paper-selected",
                acl=frozenset({"user:user-a"}),
                metadata={
                    "knowledge_base_id": "research-scope:scope-1:v1",
                    "scope_id": "scope-1",
                    "scope_version": 1,
                    "paper_id": "paper-selected",
                    "heading": "Results",
                    "pages": [4],
                    "kind": "table",
                    "table_rows": [["Recall", "95%"]],
                    "evidence_id": "evidence-selected",
                },
            ),
            KnowledgeChunk(
                chunk_id="chunk-outside",
                tenant_id=access.tenant_id,
                text="The forbidden secret target has an exact lexical match.",
                source_uri="paper://paper-outside",
                document_id="research-paper:scope-other:paper-outside",
                acl=frozenset({"user:user-a"}),
                metadata={
                    "knowledge_base_id": "research-scope:scope-other:v1",
                    "scope_id": "scope-other",
                    "paper_id": "paper-outside",
                    "evidence_id": "evidence-outside",
                },
            ),
        ]
    )
    return repository, knowledge, access


def test_intent_router_covers_numeric_and_comparison_queries() -> None:
    assert route_evidence_intent("What is the Recall table value?", "general_fact") == "numeric_table"
    assert route_evidence_intent("比较两篇论文的结果", "general_fact") == "cross_paper_comparison"
    assert route_evidence_intent("plain factual question", "claim_verification") == "claim_verification"


def test_scope_bound_search_only_reads_selected_sources(tmp_path: Path) -> None:
    repository, knowledge, access = _ready_scope(tmp_path)
    service = ScopeBoundEvidenceService(
        repository,
        ResearchRetrievalService(knowledge, graph_enabled=False),
    )
    result = service.search(
        access,
        EvidenceSearchRequest(
            scope_id="scope-1",
            query="What recall table value does the method report?",
            top_k=5,
            candidate_k=10,
        ),
    )
    assert result.routed_intent == "numeric_table"
    assert result.evidence
    assert {card.paper_id for card in result.evidence} == {"paper-selected"}
    assert all(card.scope_id == "scope-1" for card in result.evidence)
    assert all(card.scope_version == 1 for card in result.evidence)
    assert result.retrieval_rounds <= 2
    read = service.read_evidence(access, "scope-1", result.evidence[0].evidence_id)
    assert read.evidence_id == result.evidence[0].evidence_id
    verified = service.verify_citation(
        access,
        "scope-1",
        "The method reports recall table value 95 percent.",
        [result.evidence[0].evidence_id],
    )
    assert verified.resolved_evidence_ids == (result.evidence[0].evidence_id,)
    assert not verified.missing_evidence_ids

    escaped = service.search(
        access,
        EvidenceSearchRequest(
            scope_id="scope-1",
            query="forbidden secret target",
            top_k=5,
            candidate_k=10,
        ),
    )
    assert all(card.paper_id != "paper-outside" for card in escaped.evidence)
    assert escaped.retrieval_rounds == 2
    with pytest.raises(KeyError, match="outside"):
        service.read_evidence(access, "scope-1", "evidence-outside")


def test_cross_paper_gap_triggers_one_bounded_rewrite(tmp_path: Path) -> None:
    repository, knowledge, access = _ready_scope(tmp_path)
    service = ScopeBoundEvidenceService(
        repository,
        ResearchRetrievalService(knowledge, graph_enabled=False),
    )
    result = service.search(
        access,
        EvidenceSearchRequest(
            scope_id="scope-1",
            query="Compare the retrieval results across papers",
            top_k=5,
            candidate_k=10,
        ),
    )
    assert result.routed_intent == "cross_paper_comparison"
    assert result.retrieval_rounds == 2
    assert result.rewritten_query is not None
    assert not result.confidence.sufficient
    assert "not enough selected papers are represented" in result.confidence.reasons

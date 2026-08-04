from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from taskforge.context import ContextAssembler
from taskforge.knowledge import AccessContext, InMemoryKnowledgeStore, KnowledgeChunk
from taskforge.memory import (
    InMemoryMemoryStore,
    MemoryItem,
    MemoryProvenance,
    MemoryScope,
)

NOW = datetime(2026, 8, 4, 8, 0, tzinfo=UTC)


def test_knowledge_search_enforces_tenant_acl_expiry_and_latest_version() -> None:
    store = InMemoryKnowledgeStore(
        [
            KnowledgeChunk("old", "acme", "refund policy old", "kb://refund", version="1", version_order=1),
            KnowledgeChunk("new", "acme", "refund policy requires approval", "kb://refund", version="2", version_order=2),
            KnowledgeChunk("private", "acme", "refund policy private", "kb://private", acl=frozenset({"user:u2"})),
            KnowledgeChunk("expired", "acme", "refund policy expired", "kb://expired", valid_until=NOW),
            KnowledgeChunk("other-tenant", "other", "refund policy secret", "kb://other"),
        ]
    )

    hits = store.search("refund policy", AccessContext("acme", user_id="u1"), now=NOW, top_k=10)

    assert [hit.chunk.chunk_id for hit in hits] == ["new"]


def test_same_item_ids_are_isolated_per_tenant_in_both_stores() -> None:
    knowledge = InMemoryKnowledgeStore(
        [
            KnowledgeChunk("shared-id", "acme", "acme refund", "kb://acme"),
            KnowledgeChunk("shared-id", "other", "other refund", "kb://other"),
        ]
    )
    memory = InMemoryMemoryStore(
        [
            MemoryItem("shared-id", "acme", "acme refund preference"),
            MemoryItem("shared-id", "other", "other refund preference"),
        ]
    )

    assert knowledge.get("shared-id", AccessContext("acme")).text == "acme refund"
    assert knowledge.get("shared-id", AccessContext("other")).text == "other refund"
    assert memory.get("shared-id", AccessContext("acme")).content == "acme refund preference"
    assert memory.get("shared-id", AccessContext("other")).content == "other refund preference"


def test_knowledge_hybrid_scores_are_separate_and_order_is_deterministic() -> None:
    store = InMemoryKnowledgeStore(
        [
            KnowledgeChunk("b", "acme", "refund guidance", "kb://b"),
            KnowledgeChunk("a", "acme", "refund guidance", "kb://a"),
            KnowledgeChunk("semantic", "acme", "unrelated words", "kb://semantic"),
        ]
    )
    principal = AccessContext("acme")

    lexical = store.search("refund", principal)
    hybrid = store.search("refund", principal, semantic_scores={"semantic": 1.0}, semantic_weight=0.9, lexical_weight=0.1)

    assert [hit.chunk.chunk_id for hit in lexical] == ["a", "b"]
    assert hybrid[0].chunk.chunk_id == "semantic"
    assert hybrid[0].lexical_score == 0
    assert hybrid[0].semantic_score == 1


def test_latest_version_filter_keeps_every_chunk_in_the_selected_version() -> None:
    store = InMemoryKnowledgeStore(
        [
            KnowledgeChunk("old", "acme", "refund legacy", "kb://refund", version="1", version_order=1),
            KnowledgeChunk("new-1", "acme", "refund approval", "kb://refund", version="2", version_order=2),
            KnowledgeChunk("new-2", "acme", "refund evidence", "kb://refund", version="2", version_order=2),
        ]
    )

    hits = store.search("refund", AccessContext("acme"), top_k=10)

    assert {hit.chunk.chunk_id for hit in hits} == {"new-1", "new-2"}


def test_memory_recall_enforces_all_scopes_tenant_and_expiry() -> None:
    provenance = MemoryProvenance(source_type="user_statement", source_id="turn-1")
    store = InMemoryMemoryStore(
        [
            MemoryItem("tenant", "acme", "prefers concise reports", provenance=provenance),
            MemoryItem("org", "acme", "org prefers concise reports", MemoryScope.ORG, "o1", provenance),
            MemoryItem("user", "acme", "user prefers concise reports", MemoryScope.USER, "u1", provenance),
            MemoryItem("agent", "acme", "agent prefers concise reports", MemoryScope.AGENT, "a1", provenance),
            MemoryItem("task", "acme", "task prefers concise reports", MemoryScope.TASK, "t1", provenance),
            MemoryItem("wrong-user", "acme", "other prefers concise reports", MemoryScope.USER, "u2", provenance),
            MemoryItem("expired", "acme", "expired prefers concise reports", expires_at=NOW, provenance=provenance),
            MemoryItem("other", "other", "secret prefers concise reports", provenance=provenance),
        ]
    )
    principal = AccessContext("acme", user_id="u1", org_id="o1", agent_id="a1", task_id="t1")

    hits = store.recall("concise reports", principal, now=NOW, top_k=20)

    assert {hit.item.memory_id for hit in hits} == {"tenant", "org", "user", "agent", "task"}
    assert all(hit.item.provenance.source_id == "turn-1" for hit in hits)


def test_memory_order_uses_relevance_then_importance_and_is_stable() -> None:
    store = InMemoryMemoryStore(
        [
            MemoryItem("low", "acme", "concise report", importance=0.1, updated_at=NOW),
            MemoryItem("b", "acme", "concise report", importance=0.9, updated_at=NOW),
            MemoryItem("a", "acme", "concise report", importance=0.9, updated_at=NOW),
        ]
    )

    hits = store.recall("concise report", AccessContext("acme"), now=NOW)

    assert [hit.item.memory_id for hit in hits] == ["a", "b", "low"]


def test_memory_forget_respects_visibility() -> None:
    store = InMemoryMemoryStore(
        [
            MemoryItem("mine", "acme", "private preference", MemoryScope.USER, "u1"),
            MemoryItem("shared", "acme", "tenant policy", MemoryScope.TENANT),
        ]
    )

    assert store.forget("shared", AccessContext("acme", user_id="u1")) is False
    assert store.get("shared", AccessContext("acme", user_id="u1")) is not None
    assert store.forget("mine", AccessContext("acme", user_id="u2")) is False
    assert store.get("mine", AccessContext("acme", user_id="u1")) is not None
    assert store.forget("mine", AccessContext("acme", user_id="u1")) is True
    assert store.get("mine", AccessContext("acme", user_id="u1")) is None


def test_context_assembler_combines_task_profile_with_citations_and_hard_budget() -> None:
    knowledge = InMemoryKnowledgeStore(
        [
            KnowledgeChunk("allowed", "acme", "refund approval " * 30, "kb://support", metadata={"knowledge_base_id": "support"}),
            KnowledgeChunk("blocked-source", "acme", "refund approval hidden", "kb://finance", metadata={"knowledge_base_id": "finance"}),
        ]
    )
    memory = InMemoryMemoryStore(
        [
            MemoryItem(
                "preference",
                "acme",
                "refund answers should be concise",
                MemoryScope.USER,
                "u1",
                MemoryProvenance(source_type="user_statement", source_uri="turn://7"),
                importance=1.0,
                updated_at=NOW,
            )
        ]
    )
    assembler = ContextAssembler(knowledge, memory, default_char_budget=220)
    task = {"id": "t1", "tenant_id": "acme", "user_id": "u1", "goal": "refund approval", "knowledge_bases": ["support"]}
    profile = {"id": "a1", "knowledge_bases": ["support", "finance"], "memory_scopes": ["user"]}

    result = assembler.assemble("refund", profile, task, now=NOW)

    assert result.used_chars == len(result.text) <= 220
    assert result.truncated is True
    assert result.citations
    assert all(f"[{citation.label}]" in result.text for citation in result.citations)
    assert all(citation.item_id != "blocked-source" for citation in result.citations)
    assert {citation.kind for citation in result.citations} == {"knowledge", "memory"}


def test_context_assembler_rejects_identity_conflicts() -> None:
    assembler = ContextAssembler()

    with pytest.raises(PermissionError, match="tenant_id"):
        assembler.assemble(
            "refund",
            task={"tenant_id": "other", "user_id": "u1"},
            principal=AccessContext("acme", user_id="u1"),
        )


def test_validity_boundaries_are_half_open() -> None:
    chunk = KnowledgeChunk(
        "scheduled",
        "acme",
        "refund policy",
        "kb://scheduled",
        valid_from=NOW,
        valid_until=NOW + timedelta(hours=1),
    )
    principal = AccessContext("acme")

    assert chunk.is_visible_to(principal, NOW)
    assert not chunk.is_visible_to(principal, NOW + timedelta(hours=1))

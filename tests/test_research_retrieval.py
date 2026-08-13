from __future__ import annotations

import pytest

from taskforge.builtins import agent_profiles, create_tool_registry
from taskforge.domain import RunState, RunStatus, Task, ToolRequest
from taskforge.knowledge import (
    AccessContext,
    InMemoryKnowledgeStore,
    KnowledgeChunk,
    KnowledgeHit,
)
from taskforge.memory import InMemoryMemoryStore
from taskforge.research_retrieval import (
    ResearchQuery,
    ResearchRetrievalService,
    _Candidate,
)


def test_mapping_request_is_accepted_by_native_adapter() -> None:
    service = ResearchRetrievalService(store())
    result = service.search({"query": "recall", "top_k": 3}, principal())
    assert len(result.evidence) <= 3


def test_search_returns_bounded_cards_and_read_returns_authoritative_text() -> None:
    long_text = "recall evidence " * 600
    knowledge = InMemoryKnowledgeStore(
        [
            KnowledgeChunk(
                chunk_id="long-paper",
                tenant_id="tenant-a",
                text=long_text,
                source_uri="paper://long",
                document_id="long-paper",
                metadata={"evidence_id": "long:result", "title": "Long paper"},
            )
        ]
    )
    service = ResearchRetrievalService(knowledge)
    search = service.search("recall evidence", principal())
    assert len(search.evidence[0].text) <= service.SEARCH_SNIPPET_CHARS
    read = service.read_evidence("long:result", principal())
    assert len(read.text) > len(search.evidence[0].text)
    assert len(read.text) <= service.READ_TEXT_CHARS


@pytest.mark.asyncio
async def test_native_paper_tool_rejects_unbound_scope_and_generic_search_still_works() -> None:
    knowledge = store()
    registry = create_tool_registry(
        workspace_root=".",
        artifact_root=".taskforge/test-artifacts",
        knowledge_store=knowledge,
        memory_store=InMemoryMemoryStore(),
    )
    profile = next(item for item in agent_profiles() if item.id == "research-agent")
    task = Task(tenant_id="tenant-a", user_id="researcher", goal="research recall")
    state = RunState(
        task_id=task.id,
        agent_profile_id=profile.id,
        status=RunStatus.RUNNING,
        step_budget=6,
    )
    unbound = await registry.execute(
        ToolRequest(call_id="paper-search-1", name="paper_search", arguments={"query": "recall"}),
        task,
        profile,
        state,
    )
    assert unbound.ok is False
    assert "unknown_tool" in (unbound.error or "")

    generic = await registry.execute(
        ToolRequest(
            call_id="knowledge-search-1",
            name="knowledge_search",
            arguments={"query": "recall", "limit": 5},
        ),
        task,
        profile,
        state,
    )
    assert generic.ok
    assert generic.output["hits"][0]["evidence_id"]


def principal() -> AccessContext:
    return AccessContext(tenant_id="tenant-a", user_id="researcher")


def store() -> InMemoryKnowledgeStore:
    return InMemoryKnowledgeStore(
        [
            KnowledgeChunk(
                chunk_id="paper-a-method",
                tenant_id="tenant-a",
                text="Paper A uses retrieval augmented generation and reports recall 0.91.",
                source_uri="paper://a",
                document_id="paper-a",
                metadata={
                    "evidence_id": "paper-a:method",
                    "title": "Paper A",
                    "section": "Method",
                    "knowledge_base_id": "taskforge",
                },
            ),
            KnowledgeChunk(
                chunk_id="paper-a-table",
                tenant_id="tenant-a",
                text="Table 2 reports recall 0.91 on the test set.",
                source_uri="paper://a",
                document_id="paper-a",
                metadata={
                    "evidence_id": "paper-a:table",
                    "title": "Paper A",
                    "kind": "table",
                    "table_rows": ["recall 0.91"],
                    "page": 7,
                    "knowledge_base_id": "taskforge",
                },
            ),
            KnowledgeChunk(
                chunk_id="paper-b-result",
                tenant_id="tenant-a",
                text="Paper B reports recall 0.88 with a different retriever.",
                source_uri="paper://b",
                document_id="paper-b",
                metadata={
                    "evidence_id": "paper-b:result",
                    "title": "Paper B",
                    "section": "Results",
                    "knowledge_base_id": "taskforge",
                },
            ),
            KnowledgeChunk(
                chunk_id="secret",
                tenant_id="tenant-b",
                text="Paper secret recall 1.0.",
                source_uri="paper://secret",
                document_id="paper-secret",
                metadata={"evidence_id": "secret:1"},
            ),
        ]
    )


def test_unified_search_returns_evidence_ids_and_acl_scopes_results() -> None:
    service = ResearchRetrievalService(store())
    result = service.search("retrieval recall", principal())

    assert result.evidence
    assert all(item.evidence_id != "secret:1" for item in result.evidence)
    assert result.evidence[0].source.startswith("paper://")
    assert "graph_feature_rerank" in result.evidence[0].retrieval_sources
    assert result.retrieval_rounds == 1


def test_comparison_query_activates_source_coverage_operator() -> None:
    service = ResearchRetrievalService(store())
    result = service.search(
        ResearchQuery(query="compare retrieval augmented generation across papers", mode="rigorous"),
        principal(),
    )

    assert "source_coverage" in result.activated_operators
    assert result.coverage.source_count >= 1
    assert result.retrieval_rounds == 2


def test_read_and_citation_verification_reject_forged_evidence() -> None:
    service = ResearchRetrievalService(store())
    evidence = service.read_evidence("paper-a:method", principal())
    assert evidence.chunk_id == "paper-a-method"

    verification = service.verify_citation(
        "Paper A reports recall 0.91.",
        ["paper-a:method"],
        principal(),
    )
    assert verification.verified
    assert service.verify_citation("unsupported", ["forged:id"], principal()).verified is False
    with pytest.raises(KeyError):
        service.read_evidence("forged:id", principal())


def test_query_budget_is_strict() -> None:
    with pytest.raises(ValueError):
        ResearchQuery(query="recall", top_k=20, candidate_k=10)


def test_dense_index_is_reused_for_an_unchanged_scoped_corpus() -> None:
    class CountingEmbedder:
        def __init__(self) -> None:
            self.document_calls = 0

        def embed_documents(self, texts):  # type: ignore[no-untyped-def]
            self.document_calls += 1
            return [[1.0, float(index + 1)] for index, _ in enumerate(texts)]

        def embed_query(self, text):  # type: ignore[no-untyped-def]
            return [1.0, 1.0]

    embedder = CountingEmbedder()
    service = ResearchRetrievalService(
        store(),
        dense_embedder=embedder,
        graph_enabled=False,
    )

    service.search("retrieval recall", principal())
    service.search("different retrieval question", principal())

    assert embedder.document_calls == 1


def test_learned_reranker_scores_the_full_candidate_pool() -> None:
    class RecordingReranker:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def score(self, query, documents):  # type: ignore[no-untyped-def]
            self.batch_sizes.append(len(documents))
            return [float(index) for index, _ in enumerate(documents)]

    knowledge = InMemoryKnowledgeStore(
        [
            KnowledgeChunk(
                chunk_id=f"candidate-{index:02d}",
                tenant_id="tenant-a",
                text=f"retrieval evidence candidate {index}",
                source_uri="paper://candidate-pool",
                document_id="candidate-pool",
            )
            for index in range(30)
        ]
    )
    reranker = RecordingReranker()
    service = ResearchRetrievalService(
        knowledge,
        reranker=reranker,
        graph_enabled=False,
    )

    result = service.search(
        ResearchQuery(query="retrieval evidence", top_k=10, candidate_k=30),
        principal(),
    )

    assert len(result.evidence) == 10
    assert reranker.batch_sizes == [30]


def test_intent_rank_fusion_preserves_candidates_and_blends_orders() -> None:
    service = ResearchRetrievalService(
        store(),
        graph_enabled=False,
        intent_rank_fusion_weight=0.45,
    )
    chunks = [
        KnowledgeChunk(
            chunk_id=f"fusion-{index}",
            tenant_id="tenant-a",
            text=f"fusion candidate {index}",
            source_uri="paper://fusion",
            document_id="fusion",
        )
        for index in range(3)
    ]
    def candidates(order: list[int]) -> list[_Candidate]:
        return [
            _Candidate(
                hit=KnowledgeHit(
                    chunk=chunks[index],
                    score=float(len(order) - rank),
                    lexical_score=0.0,
                ),
                sources=("test",),
            )
            for rank, index in enumerate(order)
        ]

    fused = service._intent_rank_fuse(candidates([0, 1, 2]), candidates([2, 1, 0]))
    assert {item.hit.chunk.chunk_id for item in fused} == {
        "fusion-0",
        "fusion-1",
        "fusion-2",
    }
    assert fused[0].hit.chunk.chunk_id == "fusion-0"
    assert "intent_rank_fusion" in fused[0].sources

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
from taskforge.rag_experiment_profile import resolve_rag_experiment_profile
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


def test_search_presents_support_from_late_in_a_child() -> None:
    prefix = "background context " * 80
    support = "The decisive experiment reports recall 0.91."
    knowledge = InMemoryKnowledgeStore(
        [
            KnowledgeChunk(
                chunk_id="late-support",
                tenant_id="tenant-a",
                text=prefix + support,
                source_uri="paper://late-support",
                document_id="late-support-paper",
            )
        ]
    )
    service = ResearchRetrievalService(knowledge, graph_enabled=False)

    result = service.search("decisive experiment recall", principal())

    assert support in result.evidence[0].text


def test_hierarchical_search_indexes_child_and_read_expands_to_parent() -> None:
    parent = KnowledgeChunk(
        chunk_id="parent-1",
        tenant_id="tenant-a",
        text="needle evidence supports recall 0.91. exclusive parent statement.",
        source_uri="paper://hierarchical",
        document_id="hierarchical-paper",
        metadata={"retrieval_role": "parent"},
    )
    child = KnowledgeChunk(
        chunk_id="child-1",
        tenant_id="tenant-a",
        text="needle evidence supports recall 0.91.",
        source_uri="paper://hierarchical",
        document_id="hierarchical-paper",
        metadata={
            "retrieval_role": "child",
            "parent_chunk_id": "parent-1",
            "evidence_id": "hierarchical:evidence",
        },
    )
    service = ResearchRetrievalService(InMemoryKnowledgeStore([parent, child]))

    result = service.search("needle evidence", principal())
    read = service.read_evidence("hierarchical:evidence", principal())

    assert result.candidate_count == 1
    assert [item.chunk_id for item in result.evidence] == ["child-1"]
    assert read.chunk_id == "child-1"
    assert read.parent_chunk_id == "parent-1"
    assert read.presentation_strategy == "parent_context_for_child"
    assert "exclusive parent statement" in read.text
    assert not service.verify_citation(
        "exclusive parent statement",
        ["hierarchical:evidence"],
        principal(),
    ).verified


def test_parent_aware_rerank_uses_context_but_returns_raw_child_text() -> None:
    optimized_profile = resolve_rag_experiment_profile("optimized", "c")
    class RecordingReranker:
        def __init__(self) -> None:
            self.documents: list[list[str]] = []

        def score(self, query, documents):  # type: ignore[no-untyped-def]
            values = list(documents)
            self.documents.append(values)
            if len(self.documents) == 1:
                return [1.0 if "alpha" in value else 0.5 for value in values]
            return [
                0.0 if "Child evidence:\nbaseline alpha" in value else 2.0
                for value in values
            ]

    parent = KnowledgeChunk(
        chunk_id="parent-context",
        tenant_id="tenant-a",
        text="baseline alpha\n\nbaseline beta with the decisive section context",
        source_uri="paper://parent-aware",
        document_id="parent-aware-paper",
        metadata={
            "retrieval_role": "parent",
            "chunking_mode": "parent_child",
            **optimized_profile.metadata(),
        },
    )
    child_alpha = KnowledgeChunk(
        chunk_id="child-alpha",
        tenant_id="tenant-a",
        text="baseline alpha",
        source_uri="paper://parent-aware",
        document_id="parent-aware-paper",
        metadata={
            "retrieval_role": "child",
            "chunking_mode": "parent_child",
            "parent_chunk_id": "parent-context",
            "next_chunk_id": "child-beta",
            "title": "A Parent-Aware Paper",
            "heading_path": ["Experiments", "Baselines"],
            "retrieval_text": (
                "Document: A Parent-Aware Paper\n\n"
                "Section: Experiments > Baselines\n\nContent:\nbaseline alpha"
            ),
            **optimized_profile.metadata(),
        },
    )
    child_beta = KnowledgeChunk(
        chunk_id="child-beta",
        tenant_id="tenant-a",
        text="baseline beta with the decisive section context",
        source_uri="paper://parent-aware",
        document_id="parent-aware-paper",
        metadata={
            "retrieval_role": "child",
            "chunking_mode": "parent_child",
            "parent_chunk_id": "parent-context",
            "previous_chunk_id": "child-alpha",
            "title": "A Parent-Aware Paper",
            "heading_path": ["Experiments", "Baselines"],
            "retrieval_text": (
                "Document: A Parent-Aware Paper\n\n"
                "Section: Experiments > Baselines\n\n"
                "Content:\nbaseline beta with the decisive section context"
            ),
            **optimized_profile.metadata(),
        },
    )
    reranker = RecordingReranker()
    service = ResearchRetrievalService(
        InMemoryKnowledgeStore([parent, child_alpha, child_beta]),
        reranker=reranker,
        graph_enabled=False,
        operator_budget_standard=0,
        operator_budget_rigorous=0,
        parent_child_score_weight=0.2,
        parent_context_score_weight=0.8,
        parent_retrieval_score_weight=0.0,
        experiment_profile=optimized_profile,
    )

    result = service.search(
        ResearchQuery(query="baseline", top_k=2, candidate_k=10),
        principal(),
    )

    assert len(reranker.documents) == 2
    assert all("Document: A Parent-Aware Paper" in value for value in reranker.documents[0])
    assert any("Previous context:" in value for value in reranker.documents[1])
    assert result.evidence[0].chunk_id == "child-beta"
    assert result.evidence[0].text == child_beta.text
    assert "Document:" not in result.evidence[0].text
    assert result.trace is not None
    assert result.trace.reranked_hits[0].parent_context_used is True
    assert result.trace.reranked_hits[0].context_score is not None
    assert result.trace.reranked_hits[0].parent_rank_before is not None
    assert result.trace.reranked_hits[0].parent_rank_after == 1


def test_parent_aware_rerank_skips_underspecified_referential_queries() -> None:
    class RecordingReranker:
        def __init__(self) -> None:
            self.calls = 0

        def score(self, query, documents):  # type: ignore[no-untyped-def]
            self.calls += 1
            return [1.0 for _ in documents]

    profile = resolve_rag_experiment_profile("optimized", "c")
    parent = KnowledgeChunk(
        chunk_id="guard-parent",
        tenant_id="tenant-a",
        text="The performance results report accuracy 0.91.",
        source_uri="paper://guard",
        document_id="guard-paper",
        metadata={"retrieval_role": "parent", **profile.metadata()},
    )
    child = KnowledgeChunk(
        chunk_id="guard-child",
        tenant_id="tenant-a",
        text="The performance results report accuracy 0.91.",
        source_uri="paper://guard",
        document_id="guard-paper",
        metadata={
            "retrieval_role": "child",
            "parent_chunk_id": "guard-parent",
            **profile.metadata(),
        },
    )
    reranker = RecordingReranker()
    service = ResearchRetrievalService(
        InMemoryKnowledgeStore([parent, child]),
        reranker=reranker,
        graph_enabled=False,
        operator_budget_standard=0,
        experiment_profile=profile,
    )

    result = service.search(
        ResearchQuery(
            query="What were their performance results?",
            top_k=1,
            candidate_k=10,
        ),
        principal(),
    )

    assert reranker.calls == 1
    assert "parent_aware_referential_guard" in result.evidence[0].retrieval_sources


def test_contextual_child_rerank_uses_bounded_same_parent_neighbors() -> None:
    class RecordingReranker:
        def __init__(self) -> None:
            self.documents: list[str] = []

        def score(self, query, documents):  # type: ignore[no-untyped-def]
            self.documents = list(documents)
            return [
                2.0 if "[Target Child]\ntarget decisive evidence" in value else 0.0
                for value in self.documents
            ]

    profile = resolve_rag_experiment_profile("optimized", "a")
    shared = {
        "tenant_id": "tenant-a",
        "source_uri": "paper://contextual-child",
        "document_id": "contextual-child-paper",
    }
    previous = KnowledgeChunk(
        chunk_id="context-previous",
        text=(
            "PREVIOUS_START_SHOULD_NOT_APPEAR "
            + "background " * 40
            + "previous tail signal"
        ),
        metadata={
            "retrieval_role": "child",
            "parent_chunk_id": "context-parent",
            "next_chunk_id": "context-target",
            **profile.metadata(),
        },
        **shared,
    )
    target = KnowledgeChunk(
        chunk_id="context-target",
        text="target decisive evidence remains the authoritative citation",
        metadata={
            "retrieval_role": "child",
            "parent_chunk_id": "context-parent",
            "previous_chunk_id": "context-previous",
            "next_chunk_id": "context-next",
            "heading_path": ["Methods", "Evaluation"],
            **profile.metadata(),
        },
        **shared,
    )
    following = KnowledgeChunk(
        chunk_id="context-next",
        text=(
            "next head signal "
            + "background " * 40
            + " NEXT_END_SHOULD_NOT_APPEAR"
        ),
        metadata={
            "retrieval_role": "child",
            "parent_chunk_id": "context-parent",
            "previous_chunk_id": "context-target",
            **profile.metadata(),
        },
        **shared,
    )
    reranker = RecordingReranker()
    service = ResearchRetrievalService(
        InMemoryKnowledgeStore([previous, target, following]),
        reranker=reranker,
        graph_enabled=False,
        operator_budget_standard=0,
        contextual_child_rerank_enabled=True,
        contextual_child_neighbor_tokens=16,
        contextual_child_max_tokens=256,
        experiment_profile=profile,
    )

    result = service.search(
        ResearchQuery(query="target decisive", top_k=1, candidate_k=10),
        principal(),
    )

    target_view = next(
        value
        for value in reranker.documents
        if "[Target Child]\ntarget decisive evidence" in value
    )
    assert "[Section]\nMethods > Evaluation" in target_view
    assert "[Previous Context]" in target_view
    assert "previous tail signal" in target_view
    assert "PREVIOUS_START_SHOULD_NOT_APPEAR" not in target_view
    assert "[Next Context]\nnext head signal" in target_view
    assert "NEXT_END_SHOULD_NOT_APPEAR" not in target_view
    assert "Document:" not in target_view
    assert result.evidence[0].chunk_id == "context-target"
    assert result.evidence[0].text == target.text
    assert "contextual_child_rerank" in result.evidence[0].retrieval_sources

    current = ResearchRetrievalService(
        InMemoryKnowledgeStore([]),
        contextual_child_rerank_enabled=True,
    )
    assert current.contextual_child_rerank_enabled is False


def test_multi_query_rrf_recovers_synonym_variant_and_traces_all_queries() -> None:
    knowledge = InMemoryKnowledgeStore(
        [
            KnowledgeChunk(
                chunk_id="synonym-child",
                tenant_id="tenant-a",
                text="The car speed was measured on the highway benchmark.",
                source_uri="paper://synonym",
                document_id="synonym-paper",
            )
        ]
    )
    service = ResearchRetrievalService(knowledge, graph_enabled=False)

    baseline = service.search("automobile velocity", principal())
    expanded = service.search(
        ResearchQuery(
            query="automobile velocity",
            query_variants=("car speed", "highway benchmark keywords"),
        ),
        principal(),
    )

    assert not baseline.evidence
    assert [item.chunk_id for item in expanded.evidence] == ["synonym-child"]
    assert expanded.trace is not None
    assert expanded.trace.query_variants == [
        "automobile velocity",
        "car speed",
        "highway benchmark keywords",
    ]
    assert "query_variant_1" in expanded.evidence[0].retrieval_sources


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


def test_comparison_query_activates_per_source_operator() -> None:
    service = ResearchRetrievalService(store())
    result = service.search(
        ResearchQuery(query="compare retrieval augmented generation across papers", mode="rigorous"),
        principal(),
    )

    assert "per_source_comparison" in result.activated_operators
    assert result.coverage.source_count >= 1
    assert result.retrieval_rounds == 2


def test_experimental_setup_gap_targets_experiment_sections_once() -> None:
    knowledge = InMemoryKnowledgeStore(
        [
            KnowledgeChunk(
                chunk_id="overview",
                tenant_id="tenant-a",
                text=(
                    "The experimental setup and hyperparameters are discussed "
                    "elsewhere; this overview asks what setup is used."
                ),
                source_uri="paper://setup",
                document_id="setup",
                metadata={"heading": "Overview", "kind": "paragraph"},
            ),
            KnowledgeChunk(
                chunk_id="training",
                tenant_id="tenant-a",
                text="Training uses batch size 32 and Adam.",
                source_uri="paper://setup",
                document_id="setup",
                metadata={"heading": "Experimental Setup", "kind": "paragraph"},
            ),
        ]
    )
    service = ResearchRetrievalService(knowledge, graph_enabled=False)

    result = service.search(
        ResearchQuery(
            query="What experimental setup and hyperparameters are used?",
            top_k=1,
            candidate_k=10,
            mode="rigorous",
        ),
        principal(),
    )

    assert result.retrieval_rounds <= 2
    assert "experiment_section" in result.activated_operators


def test_visual_query_reports_unparsed_visual_gap() -> None:
    knowledge = InMemoryKnowledgeStore(
        [
            KnowledgeChunk(
                chunk_id="figure",
                tenant_id="tenant-a",
                text="Figure 2 architecture diagram.",
                source_uri="paper://visual",
                document_id="visual",
                metadata={
                    "kind": "image",
                    "block_types": ["image"],
                    "visual_artifact_ids": ["C:/cache/figure.png"],
                    "visual_pending": True,
                    "visual_text_ready": False,
                    "page": 2,
                },
            )
        ]
    )

    result = ResearchRetrievalService(knowledge, graph_enabled=False).search(
        ResearchQuery(
            query="What does Figure 2 show?",
            top_k=1,
            candidate_k=10,
            mode="rigorous",
        ),
        principal(),
    )

    assert result.coverage.unresolved_visual_count == 1
    assert not result.coverage.sufficient
    assert "visual_evidence" in result.activated_operators


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


def test_cjk_queries_use_the_optional_multilingual_route_and_separate_cache() -> None:
    class CountingEmbedder:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name
            self.document_calls = 0

        def embed_documents(self, texts):  # type: ignore[no-untyped-def]
            self.document_calls += 1
            return [[1.0, float(index + 1)] for index, _ in enumerate(texts)]

        def embed_query(self, text):  # type: ignore[no-untyped-def]
            return [1.0, 1.0]

    class RecordingReranker:
        def __init__(self) -> None:
            self.calls = 0

        def score(self, query, documents):  # type: ignore[no-untyped-def]
            self.calls += 1
            return [float(index) for index, _ in enumerate(documents)]

    english = CountingEmbedder("english-model")
    multilingual = CountingEmbedder("multilingual-model")
    english_reranker = RecordingReranker()
    multilingual_reranker = RecordingReranker()
    knowledge = InMemoryKnowledgeStore(
        [
            KnowledgeChunk(
                chunk_id="zh-child",
                tenant_id="tenant-a",
                text="本文方法使用分层检索，并在测试集上报告召回率。",
                source_uri="paper://zh",
                document_id="zh-paper",
            ),
            KnowledgeChunk(
                chunk_id="en-child",
                tenant_id="tenant-a",
                text="The English baseline reports retrieval recall.",
                source_uri="paper://en",
                document_id="en-paper",
            ),
        ]
    )
    service = ResearchRetrievalService(
        knowledge,
        dense_embedder=english,
        reranker=english_reranker,
        multilingual_dense_embedder=multilingual,
        multilingual_reranker=multilingual_reranker,
        graph_enabled=False,
    )

    result = service.search(
        ResearchQuery(query="方法的召回率是多少", top_k=1, candidate_k=10),
        principal(),
    )

    assert result.retrieval_route == "multilingual"
    assert multilingual.document_calls == 1
    assert multilingual_reranker.calls == 1
    assert english.document_calls == 0
    assert english_reranker.calls == 0


def test_cjk_route_is_explicitly_reported_before_optional_models_are_configured() -> None:
    knowledge = InMemoryKnowledgeStore(
        [
            KnowledgeChunk(
                chunk_id="zh-child",
                tenant_id="tenant-a",
                text="本文报告了实验结果和召回率。",
                source_uri="paper://zh",
                document_id="zh-paper",
            )
        ]
    )
    result = ResearchRetrievalService(knowledge, graph_enabled=False).search(
        ResearchQuery(query="实验结果是什么", top_k=1, candidate_k=10),
        principal(),
    )

    assert result.retrieval_route == "multilingual_fallback"


def test_dual_route_does_not_use_english_reranker_for_cjk_without_multilingual_model() -> None:
    class RecordingReranker:
        def __init__(self) -> None:
            self.calls = 0

        def score(self, query, documents):  # type: ignore[no-untyped-def]
            self.calls += 1
            return [1.0 for _ in documents]

    english_reranker = RecordingReranker()
    knowledge = InMemoryKnowledgeStore(
        [
            KnowledgeChunk(
                chunk_id="flat-zh",
                tenant_id="tenant-a",
                text="中文论文方法与实验结果。",
                source_uri="paper://dual-zh",
                document_id="dual-zh",
                metadata={
                    "retrieval_role": "child",
                    "hybrid_route": "flat_primary",
                    "parent_chunk_id": "flat-zh",
                },
            ),
            KnowledgeChunk(
                chunk_id="child-zh",
                tenant_id="tenant-a",
                text="中文论文方法实验结果辅助片段。",
                source_uri="paper://dual-zh",
                document_id="dual-zh",
                metadata={
                    "retrieval_role": "child",
                    "hybrid_route": "child_aux",
                    "parent_chunk_id": "parent-zh",
                },
            ),
        ]
    )
    service = ResearchRetrievalService(
        knowledge,
        reranker=english_reranker,
        graph_enabled=False,
        operator_budget_standard=0,
        operator_budget_rigorous=0,
        dual_route_enabled=True,
    )

    result = service.search(
        ResearchQuery(query="方法实验结果是什么", top_k=1, candidate_k=10),
        principal(),
    )

    assert result.retrieval_route == "multilingual_fallback"
    assert english_reranker.calls == 0
    assert result.evidence[0].retrieval_sources
    assert "dual_route_score_fusion" in result.evidence[0].retrieval_sources


def test_dual_route_limits_multilingual_rerank_budget() -> None:
    class RecordingReranker:
        def __init__(self) -> None:
            self.document_batches: list[list[str]] = []

        def score(self, query, documents):  # type: ignore[no-untyped-def]
            values = list(documents)
            self.document_batches.append(values)
            return [float(index) for index, _ in enumerate(values)]

    chunks = []
    for lane in ("flat_primary", "child_aux"):
        for index in range(4):
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{lane}-{index}",
                    tenant_id="tenant-a",
                    text=f"中文方法证据 {lane} {index}。",
                    source_uri="paper://dual-budget",
                    document_id="dual-budget",
                    metadata={
                        "retrieval_role": "child",
                        "hybrid_route": lane,
                        "parent_chunk_id": f"parent-{index}",
                    },
                )
            )
    reranker = RecordingReranker()
    service = ResearchRetrievalService(
        InMemoryKnowledgeStore(chunks),
        multilingual_reranker=reranker,
        graph_enabled=False,
        operator_budget_standard=0,
        operator_budget_rigorous=0,
        dual_route_enabled=True,
        dual_route_rerank_candidate_k=3,
        dual_route_tail_rerank_candidate_k=0,
    )

    result = service.search(
        ResearchQuery(query="中文方法证据", top_k=3, candidate_k=10),
        principal(),
    )

    assert result.retrieval_route == "multilingual"
    assert len(reranker.document_batches) == 1
    assert len(reranker.document_batches[0]) == 3


def test_english_route_keeps_english_models_when_multilingual_models_are_configured() -> None:
    class CountingEmbedder:
        def __init__(self) -> None:
            self.document_calls = 0

        def embed_documents(self, texts):  # type: ignore[no-untyped-def]
            self.document_calls += 1
            return [[1.0, float(index + 1)] for index, _ in enumerate(texts)]

        def embed_query(self, text):  # type: ignore[no-untyped-def]
            return [1.0, 1.0]

    class RecordingReranker:
        def __init__(self) -> None:
            self.calls = 0

        def score(self, query, documents):  # type: ignore[no-untyped-def]
            self.calls += 1
            return [float(index) for index, _ in enumerate(documents)]

    english = CountingEmbedder()
    multilingual = CountingEmbedder()
    english_reranker = RecordingReranker()
    multilingual_reranker = RecordingReranker()
    knowledge = InMemoryKnowledgeStore(
        [
            KnowledgeChunk(
                chunk_id="en-child",
                tenant_id="tenant-a",
                text="The English baseline reports retrieval recall.",
                source_uri="paper://en",
                document_id="en-paper",
            )
        ]
    )
    service = ResearchRetrievalService(
        knowledge,
        dense_embedder=english,
        reranker=english_reranker,
        multilingual_dense_embedder=multilingual,
        multilingual_reranker=multilingual_reranker,
        graph_enabled=False,
    )

    result = service.search(
        ResearchQuery(query="What is the retrieval recall?", top_k=1, candidate_k=10),
        principal(),
    )

    assert result.retrieval_route == "english"
    assert english.document_calls == 1
    assert english_reranker.calls == 1
    assert multilingual.document_calls == 0
    assert multilingual_reranker.calls == 0


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
    assert result.trace is not None
    assert len(result.trace.candidate_hits) == 30
    assert len(result.trace.reranked_hits) == 30
    assert len(result.trace.returned_hits) == 10


def test_dual_route_merges_lanes_and_falls_back_to_flat_on_coverage_gap() -> None:
    class ChildPreferringReranker:
        def score(self, query, documents):  # type: ignore[no-untyped-def]
            return [
                1.0 if "child auxiliary" in document else 0.1
                for document in documents
            ]

    chunks = [
        KnowledgeChunk(
            chunk_id="hybrid-parent",
            tenant_id="tenant-a",
            text="parent context for child auxiliary",
            source_uri="paper://hybrid",
            document_id="hybrid",
            metadata={"retrieval_role": "parent"},
        ),
        KnowledgeChunk(
            chunk_id="flat-primary",
            tenant_id="tenant-a",
            text="needle answer 42 flat primary",
            source_uri="paper://hybrid",
            document_id="hybrid",
            metadata={
                "retrieval_role": "child",
                "hybrid_route": "flat_primary",
                "parent_chunk_id": "flat-primary",
                "kind": "table",
            },
        ),
        KnowledgeChunk(
            chunk_id="child-auxiliary",
            tenant_id="tenant-a",
            text="child auxiliary needle 41",
            source_uri="paper://hybrid",
            document_id="hybrid",
            metadata={
                "retrieval_role": "child",
                "hybrid_route": "child_aux",
                "parent_chunk_id": "hybrid-parent",
            },
        ),
    ]
    service = ResearchRetrievalService(
        InMemoryKnowledgeStore(chunks),
        reranker=ChildPreferringReranker(),
        graph_enabled=False,
        operator_budget_standard=0,
        operator_budget_rigorous=0,
        dual_route_enabled=True,
        dual_route_flat_candidate_k=10,
        dual_route_child_candidate_k=10,
        dual_route_flat_head_k=1,
        multilingual_reranker=ChildPreferringReranker(),
    )

    result = service.search(
        ResearchQuery(query="needle answer 42", top_k=1, candidate_k=10),
        principal(),
    )

    assert result.trace is not None
    assert any(
        "dual_route_flat" in hit.retrieval_sources
        for hit in result.trace.candidate_hits
    )
    assert any(
        "dual_route_child" in hit.retrieval_sources
        for hit in result.trace.candidate_hits
    )
    assert result.evidence[0].chunk_id == "flat-primary"
    assert "dual_route_flat_fallback" in result.evidence[0].retrieval_sources


def test_dual_route_deduplicates_cross_lane_spans_and_caps_siblings() -> None:
    flat = KnowledgeChunk(
        chunk_id="flat-span",
        tenant_id="tenant-a",
        text="same scientific paragraph with complete context",
        source_uri="paper://dedupe",
        document_id="dedupe",
        metadata={
            "hybrid_route": "flat_primary",
            "block_ids": ["b1", "b2"],
            "retrieval_role": "child",
            "parent_chunk_id": "flat-span",
        },
    )
    child = KnowledgeChunk(
        chunk_id="child-span",
        tenant_id="tenant-a",
        text="same scientific paragraph with complete context",
        source_uri="paper://dedupe",
        document_id="dedupe",
        metadata={
            "hybrid_route": "child_aux",
            "block_ids": ["b1"],
            "retrieval_role": "child",
            "parent_chunk_id": "parent-span",
        },
    )
    candidates = [
        _Candidate(
            KnowledgeHit(chunk=flat, score=0.4, lexical_score=0.4),
            sources=("dual_route_flat",),
        ),
        _Candidate(
            KnowledgeHit(chunk=child, score=0.8, lexical_score=0.8),
            sources=("dual_route_child",),
        ),
    ]

    deduped = ResearchRetrievalService._dual_route_dedupe(candidates)

    assert [item.hit.chunk.chunk_id for item in deduped] == ["child-span"]
    assert "dual_route_cross_granularity_dedupe" in deduped[0].sources


def test_dual_route_tail_pass_only_triggers_for_an_unstable_visible_head() -> None:
    chunk = KnowledgeChunk(
        chunk_id="tail-check",
        tenant_id="tenant-a",
        text="evidence",
        source_uri="paper://tail-check",
        document_id="tail-check",
    )
    weak = [
        _Candidate(
            KnowledgeHit(chunk=chunk, score=score, lexical_score=score),
            sources=(),
        )
        for score in (0.9, 0.8, 0.7, 0.2)
    ]
    stable = [
        _Candidate(
            KnowledgeHit(chunk=chunk, score=score, lexical_score=score),
            sources=(),
        )
        for score in (0.95, 0.72, 0.42, 0.10)
    ]

    assert ResearchRetrievalService._dual_route_needs_tail_rerank(weak, 3)
    assert not ResearchRetrievalService._dual_route_needs_tail_rerank(stable, 3)


def test_final_evidence_demotes_bibliography_and_prefers_tighter_duplicate() -> None:
    abstract = (
        "Abstract This paper designs a retrieval augmented generation system "
        "with vector search, reranking, and grounded response generation. "
    ) * 5
    chunks = [
        KnowledgeChunk(
            chunk_id="references",
            tenant_id="tenant-a",
            text=(
                "[1] Retrieval augmented generation survey. "
                "[2] Retrieval augmented generation methods. "
                "[3] Retrieval augmented generation systems."
            ),
            source_uri="paper://quality",
            document_id="quality",
            metadata={"section": "References"},
        ),
        KnowledgeChunk(
            chunk_id="wrapped-abstract",
            tenant_id="tenant-a",
            text=f"Paper title Authors Keywords RAG retrieval {abstract}",
            source_uri="paper://quality",
            document_id="quality",
        ),
        KnowledgeChunk(
            chunk_id="clean-abstract",
            tenant_id="tenant-a",
            text=abstract,
            source_uri="paper://quality",
            document_id="quality",
        ),
        KnowledgeChunk(
            chunk_id="method",
            tenant_id="tenant-a",
            text="The retriever uses a dense index and a cross-encoder reranker.",
            source_uri="paper://quality",
            document_id="quality",
            metadata={"section": "Method"},
        ),
    ]
    candidates = [
        _Candidate(KnowledgeHit(chunk=chunks[0], score=0.99, lexical_score=0.99), sources=()),
        _Candidate(KnowledgeHit(chunk=chunks[1], score=0.95, lexical_score=0.95), sources=()),
        _Candidate(KnowledgeHit(chunk=chunks[2], score=0.90, lexical_score=0.90), sources=()),
        _Candidate(KnowledgeHit(chunk=chunks[3], score=0.80, lexical_score=0.80), sources=()),
    ]

    reranked = ResearchRetrievalService._paper_quality_rerank(
        "How is the RAG system designed?",
        candidates,
    )
    final = ResearchRetrievalService._dedupe_final_evidence(reranked, top_k=3)

    assert [item.hit.chunk.chunk_id for item in final] == [
        "clean-abstract",
        "method",
        "references",
    ]
    assert final[-1].hit.score < 0.1
    assert "near_duplicate_dedupe" in final[0].sources


def test_query_excerpt_avoids_reference_list_when_answer_text_follows() -> None:
    text = (
        "References [1] RAG survey. [2] RAG overview. [3] RAG benchmark. "
        * 12
        + "Method The RAG design uses vector retrieval followed by evidence reranking. "
        * 8
    )

    excerpt, _, _, _ = ResearchRetrievalService._query_excerpt(
        "How is the RAG design implemented?",
        text,
    )

    assert "vector retrieval followed by evidence reranking" in excerpt


def test_empty_table_shell_is_never_returned_as_citation_ready_evidence() -> None:
    shell = KnowledgeChunk(
        chunk_id="empty-table-shell",
        tenant_id="tenant-a",
        text="| relevant doc 1 |\n| --- |\n| relevant doc 2 |\n| relevant doc 3 |",
        source_uri="paper://table-quality",
        document_id="table-quality",
        metadata={"kind": "table", "section": "LLM"},
    )
    metric = KnowledgeChunk(
        chunk_id="metric-table",
        tenant_id="tenant-a",
        text="| Model | Accuracy |\n| --- | --- |\n| RAG | 91% |",
        source_uri="paper://table-quality",
        document_id="table-quality",
        metadata={"kind": "table", "section": "Results"},
    )
    candidates = [
        _Candidate(
            KnowledgeHit(chunk=shell, score=0.95, lexical_score=0.95),
            sources=(),
        ),
        _Candidate(
            KnowledgeHit(chunk=metric, score=0.80, lexical_score=0.80),
            sources=(),
        ),
    ]

    reranked = ResearchRetrievalService._paper_quality_rerank(
        "Which relevant documents and accuracy are reported?",
        candidates,
    )
    final = ResearchRetrievalService._dedupe_final_evidence(reranked, top_k=2)

    assert [item.hit.chunk.chunk_id for item in final] == ["metric-table"]
    assert ResearchRetrievalService._low_information_table(shell)
    assert not ResearchRetrievalService._low_information_table(metric)


def test_dual_route_reranker_failure_rolls_back_to_flat() -> None:
    class ExplodingReranker:
        def score(self, query, documents):  # type: ignore[no-untyped-def]
            raise RuntimeError("model unavailable")

    chunks = [
        KnowledgeChunk(
            chunk_id="flat-table",
            tenant_id="tenant-a",
            text="value 42 flat answer",
            source_uri="paper://reranker-failure",
            document_id="reranker-failure",
            metadata={
                "retrieval_role": "child",
                "hybrid_route": "flat_primary",
                "parent_chunk_id": "flat-table",
                "kind": "table",
            },
        ),
        KnowledgeChunk(
            chunk_id="child-generic",
            tenant_id="tenant-a",
            text="value child auxiliary context",
            source_uri="paper://reranker-failure",
            document_id="reranker-failure",
            metadata={
                "retrieval_role": "child",
                "hybrid_route": "child_aux",
                "parent_chunk_id": "parent-generic",
            },
        ),
    ]
    service = ResearchRetrievalService(
        InMemoryKnowledgeStore(chunks),
        graph_enabled=False,
        operator_budget_standard=0,
        operator_budget_rigorous=0,
        dual_route_enabled=True,
        multilingual_reranker=ExplodingReranker(),
        dual_route_flat_head_k=1,
    )

    result = service.search(
        ResearchQuery(query="value 42 是什么", top_k=1, candidate_k=10),
        principal(),
    )

    assert result.evidence[0].chunk_id == "flat-table"
    assert "dual_route_reranker_fallback" in result.evidence[0].retrieval_sources


def test_search_returns_query_centered_window_and_trace_span() -> None:
    # Query-centred slicing is only needed for unusually long/atomic children.
    text = ("irrelevant prefix material " * 180) + "needle answer is forty two"
    knowledge = InMemoryKnowledgeStore(
        [
            KnowledgeChunk(
                chunk_id="late-answer",
                tenant_id="tenant-a",
                text=text,
                source_uri="paper://late",
                document_id="late",
            )
        ]
    )
    result = ResearchRetrievalService(knowledge, graph_enabled=False).search(
        "needle answer",
        principal(),
    )

    assert "needle answer" in result.evidence[0].text
    assert result.evidence[0].text_start > 0
    assert result.evidence[0].presentation_strategy == "query_centered_lexical_window"
    assert result.trace is not None
    assert result.trace.returned_hits[0].text_start == result.evidence[0].text_start


def test_intent_fusion_runs_before_top_k_truncation() -> None:
    class ReverseReranker:
        def score(self, query, documents):  # type: ignore[no-untyped-def]
            return [float(len(documents) - index) for index, _ in enumerate(documents)]

    chunks = [
        KnowledgeChunk(
            chunk_id=f"generic-{index:02d}",
            tenant_id="tenant-a",
            text=f"experiments candidate generic detail {index}",
            source_uri="paper://intent",
            document_id="intent",
        )
        for index in range(10)
    ]
    chunks.append(
        KnowledgeChunk(
            chunk_id="method-target",
            tenant_id="tenant-a",
            text="Method ::: Experiments ::: the requested algorithm detail",
            source_uri="paper://intent",
            document_id="intent",
        )
    )
    service = ResearchRetrievalService(
        InMemoryKnowledgeStore(chunks),
        reranker=ReverseReranker(),
        graph_enabled=False,
        intent_section_fusion_enabled=True,
        intent_rank_fusion_weight=0.9,
    )

    result = service.search(
        ResearchQuery(
            query="which method was used in experiments",
            top_k=10,
            candidate_k=11,
        ),
        principal(),
    )

    assert result.trace is not None
    assert len(result.trace.reranked_hits) == 11
    assert "method-target" in {item.chunk_id for item in result.evidence}


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

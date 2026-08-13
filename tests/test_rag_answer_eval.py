from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest

from taskforge.builtins import create_tool_registry
from taskforge.domain import ModelTurn, ToolRequest
from taskforge.knowledge import InMemoryKnowledgeStore
from taskforge.memory import InMemoryMemoryStore
from taskforge.rag_answer_eval import (
    ANSWER_EVAL_METADATA_FIELD_WEIGHTS,
    OnlineModelPrice,
    RAGAnswerEvalConfig,
    _failure_stage,
    _generate_answer,
    _parse_online_cited_answer,
    _query_time_window,
    run_rag_answer_eval,
)
from taskforge.rag_answer_gate import load_answer_eval_run
from taskforge.rag_baseline import LockedSplitManifest, sha256_file
from taskforge.rag_evaluation import RAGEvalCase, load_multihop_rag_dataset
from taskforge.rag_experiment import (
    ExperimentDatasetConfig,
    ExperimentRetrievalConfig,
)
from taskforge.rag_online_gate import (
    OnlineGateThresholds,
    evaluate_online_answer_run,
)

FIXED_TIME = datetime(2026, 8, 5, 10, 0, tzinfo=UTC)


def test_query_slot_context_is_scoped_to_provided_tatqa_naive_mode() -> None:
    with pytest.raises(ValueError, match="TAT-QA dataset"):
        RAGAnswerEvalConfig(model="fake", tatqa_query_slot_context=True)
    with pytest.raises(ValueError, match="provided_hybrid_context"):
        RAGAnswerEvalConfig(
            model="fake",
            dataset=ExperimentDatasetConfig(kind="tatqa_locked"),
            tatqa_query_slot_context=True,
        )
    with pytest.raises(ValueError, match="naive mode"):
        RAGAnswerEvalConfig(
            model="fake",
            dataset=ExperimentDatasetConfig(
                kind="tatqa_locked",
                tatqa_context_mode="provided_hybrid_context",
            ),
            mode="agentic",
            tatqa_query_slot_context=True,
        )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "exact_match": 0.0,
                "candidate_full_recall": False,
                "top10_full_recall": False,
                "presented_full_recall": False,
                "answer": "42",
                "parse_error": None,
                "execution_error": None,
            },
            "candidate_missing",
        ),
        (
            {
                "exact_match": 0.0,
                "candidate_full_recall": True,
                "top10_full_recall": False,
                "presented_full_recall": False,
                "answer": "42",
                "parse_error": None,
                "execution_error": None,
            },
            "top10_ranking_failure",
        ),
        (
            {
                "exact_match": 0.0,
                "candidate_full_recall": True,
                "top10_full_recall": True,
                "presented_full_recall": False,
                "answer": "42",
                "parse_error": None,
                "execution_error": None,
            },
            "context_coverage_failure",
        ),
        (
            {
                "exact_match": 0.0,
                "candidate_full_recall": True,
                "top10_full_recall": True,
                "presented_full_recall": True,
                "answer": "41",
                "parse_error": None,
                "execution_error": None,
            },
            "reasoning_failure",
        ),
        (
            {
                "exact_match": 0.0,
                "candidate_full_recall": True,
                "top10_full_recall": True,
                "presented_full_recall": True,
                "answer": "",
                "parse_error": "invalid_json",
                "execution_error": None,
            },
            "format_or_scale_failure",
        ),
    ],
)
def test_failure_stage_attributes_first_observable_stage(
    kwargs: dict[str, object], expected: str
) -> None:
    assert _failure_stage(**kwargs) == expected


class FakeAnswerProvider:
    """Returns a scripted final answer per question; never touches a network."""

    def __init__(self, answers: dict[str, str]) -> None:
        self._answers = answers

    async def complete(self, *, task, profile, context, tools):
        return ModelTurn(
            kind="final",
            final_answer=self._answers.get(task.goal, "UNKNOWN"),
        )

    async def aclose(self) -> None:
        pass


class StepLimitProvider:
    """Uses the whole step budget on searches, then answers when forced."""

    def __init__(self, final_answer: str) -> None:
        self._final = final_answer
        self.calls = 0

    async def complete(self, *, task, profile, context, tools):
        self.calls += 1
        if self.calls % 5 != 0:
            return ModelTurn(
                kind="tool",
                tool_requests=[
                    ToolRequest(
                        call_id=f"ks-{self.calls}",
                        name="knowledge_search",
                        arguments={"query": task.goal, "limit": 5},
                    )
                ],
            )
        return ModelTurn(kind="final", final_answer=self._final)

    async def aclose(self) -> None:
        pass


class FakeAgenticProvider:
    """First proposes knowledge_search, then returns a final answer."""

    def __init__(self, final_answer: str) -> None:
        self._final = final_answer
        self.calls = 0

    async def complete(self, *, task, profile, context, tools):
        self.calls += 1
        if self.calls % 2 == 1:
            return ModelTurn(
                kind="tool",
                tool_requests=[
                    ToolRequest(
                        call_id=f"ks-{self.calls}",
                        name="knowledge_search",
                        arguments={"query": task.goal, "limit": 5},
                    )
                ],
            )
        return ModelTurn(kind="final", final_answer=self._final)

    async def aclose(self) -> None:
        pass


def _write_fixture(repository: Path) -> None:
    corpus_path = repository / ".taskforge" / "eval-cache" / "corpus.json"
    queries_path = repository / ".taskforge" / "eval-cache" / "MultiHopRAG.json"
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_text(
        json.dumps(
            [
                {
                    "title": "Apple news",
                    "author": "u1",
                    "source": "The Verge",
                    "published_at": "2024-01-01T00:00:00+00:00",
                    "category": "technology",
                    "url": "https://ex.com/a",
                    "body": "Apple announced a new device in The Verge report.",
                },
                {
                    "title": "Apple probe",
                    "author": "u2",
                    "source": "TechCrunch",
                    "published_at": "2024-01-02T00:00:00+00:00",
                    "category": "technology",
                    "url": "https://ex.com/b",
                    "body": "Apple faces a new investigation per TechCrunch.",
                },
            ]
        ),
        encoding="utf-8",
    )
    queries_path.write_text(
        json.dumps(
            [
                {
                    "query": "Who reported the Apple device news?",
                    "answer": "The Verge",
                    "question_type": "inference_query",
                    "evidence_list": [{"url": "https://ex.com/a", "fact": "a"}],
                },
                {
                    "query": "Which outlet covered the Apple probe?",
                    "answer": "TechCrunch",
                    "question_type": "inference_query",
                    "evidence_list": [{"url": "https://ex.com/b", "fact": "b"}],
                },
            ]
        ),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_answer_eval_scores_retrieved_answers_end_to_end(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    queries_path = repository / ".taskforge" / "eval-cache" / "MultiHopRAG.json"
    corpus_path = repository / ".taskforge" / "eval-cache" / "corpus.json"
    split_path = repository / "eval" / "splits" / "locked.json"
    _write_fixture(repository)
    split_path.parent.mkdir(parents=True)
    dataset = load_multihop_rag_dataset(queries_path, corpus_path)
    split = LockedSplitManifest(
        split_id="fixture-locked",
        dataset="MultiHop-RAG",
        source_split="fixture",
        source_sha256=sha256_file(queries_path),
        selection={"locked": True},
        case_ids=[case.case_id for case in dataset.cases],
        category_counts=dict(Counter(case.category for case in dataset.cases)),
    )
    split_path.write_text(split.model_dump_json(), encoding="utf-8")

    config = RAGAnswerEvalConfig(
        dataset=ExperimentDatasetConfig(
            kind="multihop_rag_locked",
            multihop_rag_queries_path=".taskforge/eval-cache/MultiHopRAG.json",
            multihop_rag_corpus_path=".taskforge/eval-cache/corpus.json",
            multihop_rag_locked_split_path="eval/splits/locked.json",
        ),
        retrieval=ExperimentRetrievalConfig(top_k=[1, 2], candidate_k=4),
        retriever="bm25_source_coverage_rrf",
        model="fake",
        answer_contract="cited_v1",
        evidence_top_k=2,
    )
    relevant_by_query = {
        case.query: case.relevant_ids[0] for case in dataset.cases
    }
    provider = FakeAnswerProvider(
        {
            "Who reported the Apple device news?": json.dumps(
                {
                    "answer": "The Verge",
                    "citation_ids": [
                        relevant_by_query["Who reported the Apple device news?"]
                    ],
                }
            ),
            "Which outlet covered the Apple probe?": json.dumps(
                {
                    "answer": "TechCrunch",
                    "citation_ids": [
                        relevant_by_query["Which outlet covered the Apple probe?"]
                    ],
                }
            ),
        }
    )

    rows, metrics, manifest = await run_rag_answer_eval(
        output_dir=tmp_path / "answer-run",
        config=config,
        provider=provider,
        repository_root=repository,
        created_at=FIXED_TIME,
    )

    assert metrics["total_cases"] == 2
    assert metrics["exact_match_accuracy"] == 1.0
    assert metrics["avg_token_f1"] == 1.0
    assert metrics["grounding"]["status"] == "measured_strict_gold_evidence"
    assert metrics["grounding"]["summary"]["citation_precision"] == 1.0
    assert metrics["grounding"]["summary"]["strict_unsupported_claim_rate"] == 0.0
    assert metrics["execution_error_cases"] == 0
    assert metrics["evidence_retrieval"]["definition"].startswith("retrieved Top-K")
    assert "candidate_retrieval" in metrics
    assert "presented_context" in metrics
    assert {row["case_id"] for row in rows} == {
        case.case_id for case in dataset.cases
    }
    assert all(row["grounding"]["strict_supported_claim"] for row in rows)
    assert all(
        set(row["citation_ids"]).issubset(row["presented_evidence_ids"])
        for row in rows
    )
    assert manifest["schema_version"] == "1.3"
    assert manifest["answer_contract"] == "cited_v1"
    assert manifest["prompt"]["sha256"]
    assert manifest["sample"]["case_ids"] == [case.case_id for case in dataset.cases]
    assert manifest["run_id"]
    loaded = load_answer_eval_run(tmp_path / "answer-run", label="fixture")
    assert loaded.case_ids == tuple(case.case_id for case in dataset.cases)
    assert loaded.answer_contract == "cited_v1"


@pytest.mark.asyncio
async def test_answer_eval_scores_wrong_answer_as_zero(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    queries_path = repository / ".taskforge" / "eval-cache" / "MultiHopRAG.json"
    corpus_path = repository / ".taskforge" / "eval-cache" / "corpus.json"
    split_path = repository / "eval" / "splits" / "locked.json"
    _write_fixture(repository)
    split_path.parent.mkdir(parents=True)
    dataset = load_multihop_rag_dataset(queries_path, corpus_path)
    split = LockedSplitManifest(
        split_id="fixture-locked",
        dataset="MultiHop-RAG",
        source_split="fixture",
        source_sha256=sha256_file(queries_path),
        selection={"locked": True},
        case_ids=[case.case_id for case in dataset.cases],
        category_counts=dict(Counter(case.category for case in dataset.cases)),
    )
    split_path.write_text(split.model_dump_json(), encoding="utf-8")
    config = RAGAnswerEvalConfig(
        dataset=ExperimentDatasetConfig(
            kind="multihop_rag_locked",
            multihop_rag_queries_path=".taskforge/eval-cache/MultiHopRAG.json",
            multihop_rag_corpus_path=".taskforge/eval-cache/corpus.json",
            multihop_rag_locked_split_path="eval/splits/locked.json",
        ),
        retrieval=ExperimentRetrievalConfig(top_k=[1, 2], candidate_k=4),
        retriever="bm25_source_coverage_rrf",
        model="fake",
    )
    provider = FakeAnswerProvider({})

    rows, metrics, _ = await run_rag_answer_eval(
        output_dir=tmp_path / "answer-run-2",
        config=config,
        provider=provider,
        repository_root=repository,
        created_at=FIXED_TIME,
    )

    assert metrics["exact_match_accuracy"] == 0.0
    assert metrics["avg_token_f1"] == 0.0
    assert all(row["generated_answer"] == "UNKNOWN" for row in rows)


@pytest.mark.asyncio
async def test_agentic_mode_runs_multi_turn_retrieval_through_the_runtime(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    queries_path = repository / ".taskforge" / "eval-cache" / "MultiHopRAG.json"
    corpus_path = repository / ".taskforge" / "eval-cache" / "corpus.json"
    split_path = repository / "eval" / "splits" / "locked.json"
    _write_fixture(repository)
    split_path.parent.mkdir(parents=True)
    dataset = load_multihop_rag_dataset(queries_path, corpus_path)
    split = LockedSplitManifest(
        split_id="fixture-locked",
        dataset="MultiHop-RAG",
        source_split="fixture",
        source_sha256=sha256_file(queries_path),
        selection={"locked": True},
        case_ids=[case.case_id for case in dataset.cases],
        category_counts=dict(Counter(case.category for case in dataset.cases)),
    )
    split_path.write_text(split.model_dump_json(), encoding="utf-8")
    config = RAGAnswerEvalConfig(
        dataset=ExperimentDatasetConfig(
            kind="multihop_rag_locked",
            multihop_rag_queries_path=".taskforge/eval-cache/MultiHopRAG.json",
            multihop_rag_corpus_path=".taskforge/eval-cache/corpus.json",
            multihop_rag_locked_split_path="eval/splits/locked.json",
        ),
        retrieval=ExperimentRetrievalConfig(top_k=[1, 2], candidate_k=4),
        retriever="bm25_source_coverage_rrf",
        mode="agentic",
        model="fake",
        agent_max_steps=4,
    )
    provider = FakeAgenticProvider("The Verge")

    rows, metrics, _ = await run_rag_answer_eval(
        output_dir=tmp_path / "answer-agentic",
        config=config,
        provider=provider,
        repository_root=repository,
        created_at=FIXED_TIME,
    )

    assert metrics["mode"] == "agentic"
    assert metrics["total_cases"] == 2
    assert all(row["mode"] == "agentic" for row in rows)
    assert all(row["steps"] >= 1 for row in rows)
    # First query's gold is "The Verge" -> correct; second is "TechCrunch" -> wrong.
    assert metrics["exact_match_accuracy"] == 0.5


def test_answer_eval_defaults_to_citation_metadata_field_weights() -> None:
    config = RAGAnswerEvalConfig(model="fake")

    assert config.retrieval.bm25_field_weights == ANSWER_EVAL_METADATA_FIELD_WEIGHTS
    assert config.answer_contract == "bare_v1"
    assert config.agentic_host_fallback is False


class CalculatorAnswerProvider:
    def __init__(
        self, *, declared_used: bool = True, correct_on_retry: bool = False
    ) -> None:
        self.calls = 0
        self.contexts: list[object] = []
        self.declared_used = declared_used
        self.correct_on_retry = correct_on_retry

    async def complete(self, *, task, profile, context, tools):
        self.calls += 1
        self.contexts.append(context)
        if self.calls == 1:
            assert [tool["name"] for tool in tools] == ["calculator"]
            return ModelTurn(
                kind="tool",
                tool_requests=[
                    ToolRequest(
                        call_id="calc-1",
                        name="calculator",
                        arguments={"expression": "(15+18+18)/3"},
                    )
                ],
                metadata={
                    "usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 5,
                        "total_tokens": 55,
                    }
                },
            )
        return ModelTurn(
            kind="final",
            final_answer=json.dumps(
                {
                    "answer": "17",
                    "derivation": "Calculator result: (15+18+18)/3 = 17.",
                    "cited_evidence_ids": ["doc-1"],
                    "calculator_used": (
                        True
                        if self.correct_on_retry and self.calls >= 3
                        else self.declared_used
                    ),
                    "abstained": False,
                }
            ),
            metadata={
                "usage": {
                    "prompt_tokens": 70,
                    "completion_tokens": 20,
                    "total_tokens": 90,
                    "prompt_cache_hit_tokens": 30,
                    "prompt_cache_miss_tokens": 40,
                }
            },
        )


def _test_calculator_registry(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return create_tool_registry(
        workspace_root=workspace,
        artifact_root=tmp_path / "artifacts",
        knowledge_store=InMemoryKnowledgeStore([]),
        memory_store=InMemoryMemoryStore(),
    )


@pytest.mark.asyncio
async def test_online_contract_executes_host_calculator_and_replays_receipt(
    tmp_path: Path,
) -> None:
    provider = CalculatorAnswerProvider()
    case = RAGEvalCase(
        case_id="case-calc",
        dataset="fixture",
        query="What is the average of 15, 18, and 18?",
        relevant_ids=["doc-1"],
        category="arithmetic",
        answer=17,
    )

    result = await _generate_answer(
        provider,
        case,
        [("doc-1", "evidence_id: doc-1\nValues are 15, 18, and 18.")],
        model="fake",
        max_evidence_chars=2_000,
        contract="online_cited_v1",
        calculator_registry=_test_calculator_registry(tmp_path),
    )

    assert result.answer == "17"
    assert result.calculator_used is True
    assert result.abstained is False
    assert result.parse_error is None
    assert result.provider_calls == 2
    assert result.usage == {
        "input_tokens": 120,
        "output_tokens": 25,
        "total_tokens": 145,
        "input_cache_hit_tokens": 30,
        "input_cache_miss_tokens": 40,
    }
    receipt = result.calculator_receipts[0]
    assert receipt["ok"] is True
    assert receipt["output"] == {"value": 17.0}
    second_context = provider.contexts[1]
    assert second_context["trajectory"][0]["tool_results"][0]["call_id"] == "calc-1"
    assert set(second_context["assembled"]) == {"evidence", "question"}


def test_online_contract_rejects_invalid_shape_and_abstention() -> None:
    assert _parse_online_cited_answer("not-json")[-1] == "invalid_json"
    valid_abstention = json.dumps(
        {
            "answer": "The inspected evidence is insufficient.",
            "derivation": "No evidence.",
            "cited_evidence_ids": ["inspected-evidence"],
            "calculator_used": False,
            "abstained": True,
        }
    )
    assert _parse_online_cited_answer(valid_abstention)[-1] is None


@pytest.mark.asyncio
async def test_online_contract_rejects_declared_calculator_mismatch(
    tmp_path: Path,
) -> None:
    provider = CalculatorAnswerProvider(declared_used=False)
    case = RAGEvalCase(
        case_id="case-calc",
        dataset="fixture",
        query="Average?",
        relevant_ids=["doc-1"],
        category="arithmetic",
        answer=17,
    )
    result = await _generate_answer(
        provider,
        case,
        [("doc-1", "evidence_id: doc-1\n15, 18, 18")],
        model="fake",
        max_evidence_chars=500,
        contract="online_cited_v1",
        calculator_registry=_test_calculator_registry(tmp_path),
    )
    assert result.parse_error == "calculator_usage_mismatch"
    assert result.contract_retry_count == 1


@pytest.mark.asyncio
async def test_online_contract_retries_same_provider_for_contract_correction(
    tmp_path: Path,
) -> None:
    provider = CalculatorAnswerProvider(
        declared_used=False,
        correct_on_retry=True,
    )
    case = RAGEvalCase(
        case_id="case-calc",
        dataset="fixture",
        query="Average?",
        relevant_ids=["doc-1"],
        category="arithmetic",
        answer=17,
    )
    result = await _generate_answer(
        provider,
        case,
        [("doc-1", "evidence_id: doc-1\n15, 18, 18")],
        model="fake",
        max_evidence_chars=500,
        contract="online_cited_v1",
        calculator_registry=_test_calculator_registry(tmp_path),
    )
    assert result.parse_error is None
    assert result.calculator_used is True
    assert result.contract_retry_count == 1
    assert result.provider_calls == 3


class CapturingOnlineProvider:
    def __init__(self, evidence_id: str) -> None:
        self.evidence_id = evidence_id
        self.contexts: list[object] = []

    async def complete(self, *, task, profile, context, tools):
        self.contexts.append(context)
        assert [tool["name"] for tool in tools] == ["calculator"]
        return ModelTurn(
            kind="final",
            final_answer=json.dumps(
                {
                    "answer": "42",
                    "derivation": "The cited table reports revenue of 42.",
                    "cited_evidence_ids": [self.evidence_id],
                    "calculator_used": False,
                    "abstained": False,
                }
            ),
            metadata={
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                }
            },
        )


@pytest.mark.asyncio
async def test_frozen_pair_online_eval_reuses_offline_stage_and_writes_audit_artifacts(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    input_path = repository / ".taskforge" / "eval-cache" / "tatqa.json"
    split_path = repository / "eval" / "splits" / "heldout.json"
    input_path.parent.mkdir(parents=True)
    split_path.parent.mkdir(parents=True)
    input_path.write_text(
        json.dumps(
            [
                {
                    "table": {
                        "uid": "online-fixture",
                        "table": [["Metric", "2024"], ["Revenue", "42"]],
                    },
                    "paragraphs": [
                        {
                            "uid": "paragraph-1",
                            "order": 1,
                            "text": "The table reports annual revenue.",
                        }
                    ],
                    "questions": [
                        {
                            "uid": "question-1",
                            "order": 1,
                            "question": "What was revenue in 2024?",
                            "answer": ["42"],
                            "derivation": "",
                            "answer_type": "span",
                            "answer_from": "table",
                            "rel_paragraphs": [],
                            "req_comparison": False,
                            "scale": "",
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    case_id = "tatqa:question-1"
    table_id = "tatqa:online-fixture:table"
    split = LockedSplitManifest(
        split_id="online-heldout-fixture",
        dataset="TAT-QA",
        source_split="official_train",
        source_sha256=sha256_file(input_path),
        selection={"locked": True},
        case_ids=[case_id],
        category_counts={"table": 1},
    )
    split_path.write_text(split.model_dump_json(), encoding="utf-8")
    price = OnlineModelPrice(
        model="deepseek-v4-flash",
        input_cache_hit_per_million=0.0028,
        input_cache_miss_per_million=0.14,
        output_per_million=0.28,
        source_url="https://api-docs.deepseek.com/quick_start/pricing/",
        retrieved_at="2026-08-10",
    )
    config = RAGAnswerEvalConfig(
        dataset=ExperimentDatasetConfig(
            kind="tatqa_locked",
            tatqa_input_path=".taskforge/eval-cache/tatqa.json",
            tatqa_locked_split_path="eval/splits/heldout.json",
            tatqa_context_mode="provided_hybrid_context",
        ),
        retrieval=ExperimentRetrievalConfig(),
        retriever="tatqa_frozen_pair_rerank",
        model="deepseek-v4-flash",
        answer_contract="online_cited_v1",
        evidence_top_k=10,
        thinking_mode="disabled",
        json_mode=True,
        price_table=price,
        tatqa_query_slot_context=True,
        tatqa_query_slot_k=2,
    )
    provider = CapturingOnlineProvider(table_id)
    output = tmp_path / "online-pair-run"

    rows, metrics, manifest = await run_rag_answer_eval(
        output_dir=output,
        config=config,
        provider=provider,
        repository_root=repository,
        created_at=FIXED_TIME,
    )

    assert rows[0]["retrieved_ids"][0] == table_id
    assert rows[0]["presented_evidence_ids"][0] == table_id
    assert rows[0]["grounding"]["strict_supported_claim"] is True
    assert rows[0]["estimated_cost"]["amount"] > 0
    assert metrics["avg_token_f1"] == 1.0
    assert metrics["online_safety"]["invalid_evidence_ids"] == 0
    assert metrics["online_safety"]["fallback_cases"] == 0
    assert manifest["config"]["effective"]["retrieval"]["candidate_k"] == 50
    assert manifest["config"]["effective"]["retrieval"]["top_k"] == [1, 5, 10]
    assert manifest["config"]["effective"]["retrieval"]["stages"] == [
        "bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_pair_rerank_rrf"
    ]
    assert manifest["tools"]["names"] == ["calculator"]
    assert manifest["index"]["identity"]["offline_stage_metrics"]
    assert set(provider.contexts[0]["assembled"]) == {"evidence", "question"}
    assert (
        "TAT-QA query slot plan"
        in provider.contexts[0]["assembled"]["evidence"][0]
    )
    assert (
        "Full retrieved evidence"
        in provider.contexts[0]["assembled"]["evidence"][0]
    )
    assert rows[0]["tatqa_query_slot_context"] is True
    for artifact in ("manifest.json", "predictions.jsonl", "metrics.json", "failures.jsonl", "costs.jsonl"):
        assert (output / artifact).is_file()
    gate = evaluate_online_answer_run(
        output,
        profile="canary20",
        thresholds=OnlineGateThresholds(expected_cases=1, gate_quality=False),
    )
    assert gate["gates"]["sample_size"]["passed"] is True
    assert gate["gates"]["usage_and_cost_trace"]["passed"] is True
    assert gate["gates"]["real_provider"]["passed"] is False
    assert gate["passed"] is False


@pytest.mark.asyncio
async def test_cited_generation_records_only_evidence_inside_the_context_budget() -> None:
    question = "Who published the report?"
    provider = FakeAnswerProvider(
        {
            question: json.dumps(
                {"answer": "The Verge", "citation_ids": ["doc-1"]}
            )
        }
    )
    case = RAGEvalCase(
        case_id="case-1",
        dataset="fixture",
        query=question,
        relevant_ids=["doc-1"],
        category="text",
        answer="The Verge",
    )

    result = await _generate_answer(
        provider,
        case,
        [
            ("doc-1", "evidence_id: doc-1\n" + "A" * 700),
            ("doc-2", "evidence_id: doc-2\nThe second document"),
        ],
        model="fake",
        max_evidence_chars=500,
        contract="cited_v1",
    )

    assert result.answer == "The Verge"
    assert result.citation_ids == ["doc-1"]
    assert result.presented_evidence_ids == ["doc-1"]
    assert result.parse_error is None


@pytest.mark.asyncio
async def test_cited_generation_does_not_repair_invalid_json() -> None:
    question = "Who published the report?"
    provider = FakeAnswerProvider({question: "The Verge [doc-1]"})
    case = RAGEvalCase(
        case_id="case-1",
        dataset="fixture",
        query=question,
        relevant_ids=["doc-1"],
        category="text",
        answer="The Verge",
    )

    result = await _generate_answer(
        provider,
        case,
        [("doc-1", "evidence_id: doc-1\nThe Verge published it")],
        model="fake",
        max_evidence_chars=500,
        contract="cited_v1",
    )

    assert result.answer == ""
    assert result.citation_ids == []
    assert result.parse_error == "invalid_json"


@pytest.mark.asyncio
async def test_agentic_mode_forces_answer_after_step_limit(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    queries_path = repository / ".taskforge" / "eval-cache" / "MultiHopRAG.json"
    corpus_path = repository / ".taskforge" / "eval-cache" / "corpus.json"
    split_path = repository / "eval" / "splits" / "locked.json"
    _write_fixture(repository)
    split_path.parent.mkdir(parents=True)
    dataset = load_multihop_rag_dataset(queries_path, corpus_path)
    split = LockedSplitManifest(
        split_id="fixture-locked",
        dataset="MultiHop-RAG",
        source_split="fixture",
        source_sha256=sha256_file(queries_path),
        selection={"locked": True},
        case_ids=[case.case_id for case in dataset.cases],
        category_counts=dict(Counter(case.category for case in dataset.cases)),
    )
    split_path.write_text(split.model_dump_json(), encoding="utf-8")
    config = RAGAnswerEvalConfig(
        dataset=ExperimentDatasetConfig(
            kind="multihop_rag_locked",
            multihop_rag_queries_path=".taskforge/eval-cache/MultiHopRAG.json",
            multihop_rag_corpus_path=".taskforge/eval-cache/corpus.json",
            multihop_rag_locked_split_path="eval/splits/locked.json",
        ),
        retrieval=ExperimentRetrievalConfig(top_k=[1, 2], candidate_k=4),
        retriever="bm25",
        mode="agentic",
        model="fake",
        agent_max_steps=4,
        agentic_host_fallback=True,
        max_cases=1,
    )
    provider = StepLimitProvider("The Verge")

    rows, metrics, _ = await run_rag_answer_eval(
        output_dir=tmp_path / "answer-step-limit",
        config=config,
        provider=provider,
        repository_root=repository,
        created_at=FIXED_TIME,
    )

    assert metrics["total_cases"] == 1
    assert metrics["exact_match_accuracy"] == 1.0
    assert rows[0]["generated_answer"] == "The Verge"
    assert rows[0]["steps"] == 4
    assert rows[0]["retrieved_ids"]
    assert rows[0]["fallback_used"] is True
    assert rows[0]["execution_error"] == "step_budget_exhausted"


def test_query_time_window_between_dates() -> None:
    after, before = _query_time_window(
        "Between November 9, 2023 and November 15, 2023, did Polygon change?"
    )
    assert after == datetime(2023, 11, 9, tzinfo=UTC)
    assert before == datetime(2023, 11, 15, tzinfo=UTC)


def test_query_time_window_single_after_bound() -> None:
    after, before = _query_time_window(
        "After the TechCrunch report on October 7, 2023, what changed?"
    )
    assert after == datetime(2023, 10, 7, tzinfo=UTC)
    assert before is None


def test_query_time_window_single_before_bound() -> None:
    after, before = _query_time_window(
        "Has the portrayal remained consistent before October 22, 2023?"
    )
    assert after is None
    assert before == datetime(2023, 10, 22, tzinfo=UTC)


def test_query_time_window_leaves_dates_queries_unfiltered() -> None:
    assert _query_time_window("What is the capital of France?") == (None, None)
    assert _query_time_window("Who won the 2023 championship?") == (None, None)

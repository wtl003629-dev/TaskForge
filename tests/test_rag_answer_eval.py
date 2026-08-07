from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest

from taskforge.domain import ModelTurn, ToolRequest
from taskforge.rag_answer_eval import (
    ANSWER_EVAL_METADATA_FIELD_WEIGHTS,
    RAGAnswerEvalConfig,
    _query_time_window,
    run_rag_answer_eval,
)
from taskforge.rag_baseline import LockedSplitManifest, sha256_file
from taskforge.rag_evaluation import load_multihop_rag_dataset
from taskforge.rag_experiment import (
    ExperimentDatasetConfig,
    ExperimentRetrievalConfig,
)

FIXED_TIME = datetime(2026, 8, 5, 10, 0, tzinfo=UTC)


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
        retriever="bm25",
        model="fake",
        evidence_top_k=2,
    )
    provider = FakeAnswerProvider(
        {
            "Who reported the Apple device news?": "The Verge",
            "Which outlet covered the Apple probe?": "TechCrunch",
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
    assert {row["case_id"] for row in rows} == {
        case.case_id for case in dataset.cases
    }
    assert manifest["run_id"]


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
        retriever="bm25",
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
        retriever="bm25",
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

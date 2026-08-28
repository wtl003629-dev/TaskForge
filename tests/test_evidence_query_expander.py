from __future__ import annotations

import json

import httpx
import pytest

from taskforge.literature.evidence_query_expander import (
    EvidenceQueryExpansionError,
    OpenAICompatibleEvidenceQueryExpander,
    protected_query_terms,
)


def test_protected_query_terms_keep_entities_numbers_and_negation() -> None:
    assert protected_query_terms(
        "Compare QASPER Recall@10, not Recall@50, versus RAGAS"
    ) == (
        "10",
        "50",
        "compare",
        "not",
        "versus",
        "qasper",
        "recall",
        "ragas",
    )


@pytest.mark.asyncio
async def test_expander_returns_constrained_semantic_and_keyword_queries() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = {
            "synonym_query": "Compare QASPER Recall 10 not 50 versus RAGAS retrieval",
            "keyword_query": "QASPER RAGAS Compare versus Recall 10 not 50 evidence",
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(payload)}}]},
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    expander = OpenAICompatibleEvidenceQueryExpander(
        api_key="test",
        model="test-model",
        base_url="https://llm.test",
        client=http,
    )

    values = await expander.expand(
        "Compare QASPER Recall 10 not 50 versus RAGAS",
        "cross_paper_comparison",
    )

    assert values[0].startswith("Compare QASPER")
    assert "RAGAS" in values[1]
    await http.aclose()


@pytest.mark.asyncio
async def test_expander_rejects_dropped_numeric_constraint() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = {
            "synonym_query": "Compare QASPER retrieval versus RAGAS",
            "keyword_query": "QASPER RAGAS retrieval comparison",
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(payload)}}]},
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    expander = OpenAICompatibleEvidenceQueryExpander(
        api_key="test",
        model="test-model",
        base_url="https://llm.test",
        client=http,
    )

    with pytest.raises(EvidenceQueryExpansionError, match="dropped"):
        await expander.expand(
            "Compare QASPER Recall 10 versus RAGAS",
            "cross_paper_comparison",
        )
    await http.aclose()


@pytest.mark.asyncio
async def test_expander_reports_reasoning_budget_exhaustion() -> None:
    calls = 0
    budgets: list[int] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        budgets.append(body["max_tokens"])
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "content": "",
                            "reasoning_content": "unfinished reasoning",
                        },
                    }
                ]
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    expander = OpenAICompatibleEvidenceQueryExpander(
        api_key="test",
        model="reasoning-model",
        base_url="https://llm.test",
        client=http,
    )

    with pytest.raises(EvidenceQueryExpansionError, match="finish_reason=length"):
        await expander.expand("Which experiments were carried out?", "experimental_setup")
    assert calls == 3
    assert budgets == [2_000, 2_000, 4_000]
    await http.aclose()


@pytest.mark.asyncio
async def test_expander_repairs_a_dropped_protected_term() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        request_data = json.loads(json.loads(request.content)["messages"][1]["content"])
        if calls == 1:
            result = {
                "synonym_query": "Which reward learning algorithm is adapted?",
                "keyword_query": "reward learning algorithm adaptation",
            }
        else:
            assert "repair_feedback" in request_data
            result = {
                "synonym_query": "What RL reward learning algorithm was adapted?",
                "keyword_query": "RL reward learning algorithm adaptation",
            }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(result)}}]},
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    expander = OpenAICompatibleEvidenceQueryExpander(
        api_key="test",
        model="test-model",
        base_url="https://llm.test",
        client=http,
    )

    synonym, keyword = await expander.expand(
        "Which RL reward learning algorithm is adapted?",
        "experimental_setup",
    )

    assert calls == 2
    assert "RL" in synonym and "RL" in keyword
    await http.aclose()

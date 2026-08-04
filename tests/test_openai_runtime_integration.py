from __future__ import annotations

import json

import httpx

from taskforge.builtins import create_tool_registry
from taskforge.checkpoints import SQLiteCheckpointStore
from taskforge.context import ContextAssembler
from taskforge.domain import AgentProfile, RunStatus, Task
from taskforge.knowledge import InMemoryKnowledgeStore
from taskforge.memory import InMemoryMemoryStore
from taskforge.openai_provider import OpenAIResponsesProvider
from taskforge.runtime import AgentRuntime
from taskforge.tooling import CapabilityPolicy


async def test_native_function_call_executes_and_continues_with_previous_response(tmp_path) -> None:
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        if len(payloads) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "resp-tool",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "calc-1",
                            "name": "calculator",
                            "arguments": '{"expression":"(6 * 7) + 1"}',
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "resp-final",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "计算结果是 43。"}],
                    }
                ],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIResponsesProvider(
        api_key="test-key",
        enabled=True,
        model="gpt-test",
        client=client,
    )
    knowledge = InMemoryKnowledgeStore()
    memory = InMemoryMemoryStore()
    registry = create_tool_registry(
        workspace_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        knowledge_store=knowledge,
        memory_store=memory,
    )
    runtime = AgentRuntime(
        provider=provider,
        registry=registry,
        policy=CapabilityPolicy(registry),
        checkpoint=SQLiteCheckpointStore(tmp_path / "state.sqlite3"),
        context=ContextAssembler(knowledge, memory),
    )
    task = Task(tenant_id="local", user_id="demo", goal="计算 6 * 7 + 1")
    profile = AgentProfile(
        id="calculator-agent",
        name="Calculator",
        instructions="Use the calculator and report its receipt.",
        model="gpt-test",
        allowed_tools=["calculator"],
        max_steps=3,
    )

    try:
        state = await runtime.run(task, profile)
    finally:
        await client.aclose()

    assert state.status is RunStatus.COMPLETED
    assert state.final_answer == "计算结果是 43。"
    assert state.steps[0].tool_results[0].output == {"value": 43}
    assert payloads[1]["previous_response_id"] == "resp-tool"
    continuation = payloads[1]["input"]
    assert continuation[0]["type"] == "function_call_output"
    assert continuation[0]["call_id"] == "calc-1"
    assert '"value":43' in continuation[0]["output"]

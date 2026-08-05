from __future__ import annotations

import json

import httpx

from taskforge.builtins import create_tool_registry
from taskforge.checkpoints import SQLiteCheckpointStore
from taskforge.context import ContextAssembler
from taskforge.domain import AgentProfile, RunStatus, Task
from taskforge.knowledge import InMemoryKnowledgeStore
from taskforge.memory import InMemoryMemoryStore
from taskforge.openai_provider import OpenAIChatCompletionsProvider
from taskforge.runtime import AgentRuntime
from taskforge.tooling import CapabilityPolicy


async def test_native_function_call_executes_and_continues_with_full_history(
    tmp_path,
) -> None:
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        if len(payloads) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-tool",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "calc-1",
                                        "type": "function",
                                        "function": {
                                            "name": "calculator",
                                            "arguments": '{"expression":"(6 * 7) + 1"}',
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-final",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "计算结果是 43。"},
                    }
                ],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIChatCompletionsProvider(
        api_key="test-key",
        enabled=True,
        model="deepseek-test",
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
        model="deepseek-test",
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
    assert "previous_response_id" not in payloads[1]
    messages = payloads[1]["messages"]
    assistant = next(message for message in messages if message.get("tool_calls"))
    assert assistant["tool_calls"][0]["id"] == "calc-1"
    tool_message = next(message for message in messages if message["role"] == "tool")
    assert tool_message["tool_call_id"] == "calc-1"
    assert '"value":43' in tool_message["content"]

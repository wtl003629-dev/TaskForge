from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pytest

from taskforge.checkpoints import SQLiteCheckpointStore
from taskforge.context import ContextAssembler
from taskforge.domain import (
    AgentProfile,
    ApprovalResponse,
    ModelTurn,
    PolicyDecision,
    RunState,
    RunStatus,
    Task,
    ToolRequest,
    ToolResult,
)
from taskforge.providers import (
    OpenAIResponsesAdapter,
    ProviderResponseError,
    ScriptedProvider,
    build_openai_responses_payload,
    parse_json_action,
)
from taskforge.runtime import AgentRuntime
from taskforge.tooling import CapabilityPolicy, ToolRegistry, ToolRisk, ToolSpec


@dataclass(frozen=True)
class Assembled:
    text: str
    citations: tuple[str, ...] = ()


class FakeContext:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def assemble(self, **kwargs: Any) -> Assembled:
        self.calls.append(kwargs)
        return Assembled(text=f"evidence for {kwargs['query']}", citations=("doc:1",))


class FakeCheckpoint:
    def __init__(self) -> None:
        self.states: list[RunState] = []

    async def save(self, state: RunState) -> None:
        self.states.append(state.model_copy(deep=True))


class FakePolicy:
    def __init__(self, decisions: Mapping[str, PolicyDecision] | None = None) -> None:
        self.decisions = dict(decisions or {})
        self.calls: list[str] = []

    async def evaluate(
        self,
        task: Task,
        profile: AgentProfile,
        request: ToolRequest,
    ) -> PolicyDecision:
        del task, profile
        self.calls.append(request.call_id)
        return self.decisions.get(request.name, PolicyDecision.allow())


class FakeRegistry:
    def __init__(self, *, fail: bool = False, mismatched_id: bool = False) -> None:
        self.fail = fail
        self.mismatched_id = mismatched_id
        self.calls: list[ToolRequest] = []

    def list_specs(self, profile: AgentProfile) -> Sequence[Mapping[str, Any]]:
        return [
            {
                "name": name,
                "description": f"controlled {name}",
                "parameters": {"type": "object", "properties": {}},
            }
            for name in profile.allowed_tools
        ]

    async def execute(
        self,
        request: ToolRequest,
        task: Task,
        profile: AgentProfile,
        state: RunState,
    ) -> ToolResult:
        del task, profile, state
        self.calls.append(request.model_copy(deep=True))
        if self.fail:
            raise TimeoutError("controlled tool timed out")
        return ToolResult(
            call_id="wrong" if self.mismatched_id else request.call_id,
            ok=True,
            output={"echo": deepcopy(request.arguments)},
        )


def make_task() -> Task:
    return Task(id="task-1", tenant_id="tenant-1", user_id="user-1", goal="research")


def test_one_model_turn_has_a_bounded_tool_fanout() -> None:
    with pytest.raises(ValueError):
        ModelTurn(
            kind="tool",
            tool_requests=[
                ToolRequest(call_id=f"call-{index}", name="lookup")
                for index in range(17)
            ],
        )


def make_profile(
    *,
    max_steps: int = 5,
    allowed_tools: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentProfile:
    return AgentProfile(
        id="profile-1",
        name="researcher",
        instructions="Use evidence and controlled tools.",
        allowed_tools=allowed_tools or ["lookup", "sensitive"],
        max_steps=max_steps,
        metadata=metadata or {},
    )


def make_runtime(
    provider: ScriptedProvider,
    *,
    registry: FakeRegistry | None = None,
    policy: FakePolicy | None = None,
    checkpoint: FakeCheckpoint | None = None,
) -> tuple[AgentRuntime, FakeRegistry, FakePolicy, FakeCheckpoint, FakeContext]:
    registry = registry or FakeRegistry()
    policy = policy or FakePolicy()
    checkpoint = checkpoint or FakeCheckpoint()
    context = FakeContext()
    return (
        AgentRuntime(
            provider=provider,
            registry=registry,
            policy=policy,
            checkpoint=checkpoint,
            context=context,
        ),
        registry,
        policy,
        checkpoint,
        context,
    )


def test_json_action_parser_is_structured_and_rejects_surrounding_prose() -> None:
    tool = parse_json_action(
        {
            "action": "lookup",
            "call_id": "call-1",
            "arguments": {"query": "bounded loops"},
            "idempotency_key": "lookup:bounded-loops",
        }
    )
    assert tool.kind == "tool"
    assert tool.tool_requests == [
        ToolRequest(
            call_id="call-1",
            name="lookup",
            arguments={"query": "bounded loops"},
            idempotency_key="lookup:bounded-loops",
        )
    ]
    assert parse_json_action('{"action":"final","answer":"done"}').final_answer == "done"
    with pytest.raises(ProviderResponseError):
        parse_json_action('First do this: {"action":"final","answer":"done"}')


def test_openai_responses_adapter_is_a_pure_bidirectional_conversion() -> None:
    neutral_tools = [
        {
            "name": "lookup",
            "description": "Find governed evidence",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]
    payload = build_openai_responses_payload(
        model="gpt-test",
        instructions="Use tools.",
        input_items=[{"role": "user", "content": "find it"}],
        tools=neutral_tools,
    )
    assert payload["tools"][0]["type"] == "function"
    assert payload["tools"][0]["parameters"]["required"] == ["query"]
    assert "input_schema" not in payload["tools"][0]

    turn = OpenAIResponsesAdapter.parse_response(
        {
            "id": "resp-1",
            "model": "gpt-test",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call-native",
                    "name": "lookup",
                    "arguments": (
                        '{"query":"runtime","idempotency_key":"lookup:runtime"}'
                    ),
                }
            ],
        }
    )
    assert turn.kind == "tool"
    assert turn.provider_response_id == "resp-1"
    assert turn.tool_requests[0].arguments == {
        "query": "runtime",
        "idempotency_key": "lookup:runtime",
    }
    assert turn.tool_requests[0].idempotency_key == "lookup:runtime"

    final = OpenAIResponsesAdapter.parse_response(
        {
            "id": "resp-2",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "finished"}],
                }
            ],
        }
    )
    assert final.kind == "final"
    assert final.final_answer == "finished"


@pytest.mark.asyncio
async def test_runtime_executes_tool_then_final_and_checkpoints_trajectory() -> None:
    provider = ScriptedProvider(
        [
            {
                "action": "lookup",
                "call_id": "call-1",
                "arguments": {"query": "agent runtime"},
            },
            {"action": "final", "answer": "evidence-backed result"},
        ]
    )
    runtime, registry, policy, checkpoint, context = make_runtime(provider)

    state = await runtime.run(make_task(), make_profile())

    assert state.status == RunStatus.COMPLETED
    assert state.final_answer == "evidence-backed result"
    assert len(registry.calls) == 1
    assert policy.calls == ["call-1"]
    assert state.receipts["call-1"].ok is True
    assert [saved.status for saved in checkpoint.states][0] == RunStatus.RUNNING
    assert checkpoint.states[-1].status == RunStatus.COMPLETED
    assert context.calls[0]["query"] == "research"
    second_context = provider.calls[1].context
    assert second_context["assembled"]["citations"] == ("doc:1",)
    assert second_context["trajectory"][0]["tool_results"][0]["output"] == {
        "echo": {"query": "agent runtime"}
    }


@pytest.mark.asyncio
async def test_research_trajectory_is_compacted_only_for_provider_context() -> None:
    provider = ScriptedProvider(
        [
            {
                "action": "paper_search",
                "call_id": "paper-1",
                "arguments": {"query": "recall"},
            },
            {
                "action": "paper_search",
                "call_id": "paper-2",
                "arguments": {"query": "precision"},
            },
            {"action": "final", "answer": "finished"},
        ]
    )
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="paper_search",
            description="search",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        lambda *_: {
            "query": "recall",
            "evidence": [
                {
                    "evidence_id": "E1",
                    "source": "paper://1",
                    "text": "long evidence text " * 100,
                }
            ],
            "coverage": {"source_count": 1},
        },
    )
    runtime, _, _, _, _ = make_runtime(provider, registry=registry)
    profile = make_profile(
        allowed_tools=["paper_search"],
        metadata={"compact_tool_trajectory": True},
    )
    state = await runtime.run(make_task(), profile)
    assert state.status == RunStatus.COMPLETED
    historical = provider.calls[2].context["trajectory"][0]["tool_results"][0]["output"]
    assert historical["receipt_type"] == "research.search.v1"
    assert historical["evidence_ids"] == ["E1"]
    assert len(json.dumps(historical, ensure_ascii=False)) < 500
    latest = provider.calls[2].context["trajectory"][1]["tool_results"][0]["output"]
    assert "evidence" in latest
    durable = state.receipts["paper-1"].output
    assert len(durable["evidence"][0]["text"]) > 500


@pytest.mark.asyncio
async def test_approval_pauses_before_execution_and_resumes_exact_call() -> None:
    provider = ScriptedProvider(
        [
            {
                "action": "sensitive",
                "call_id": "approval-call",
                "arguments": {"record": 7},
                "idempotency_key": "record:7",
            },
            {"action": "final", "answer": "approved work completed"},
        ]
    )
    policy = FakePolicy(
        {"sensitive": PolicyDecision.approval("human confirmation required")}
    )
    runtime, registry, _, checkpoint, _ = make_runtime(provider, policy=policy)

    paused = await runtime.run(make_task(), make_profile())
    assert paused.status == RunStatus.WAITING_APPROVAL
    assert paused.pending_approval is not None
    assert paused.pending_approval.request.call_id == "approval-call"
    assert registry.calls == []
    assert checkpoint.states[-1].status == RunStatus.WAITING_APPROVAL

    with pytest.raises(ValueError, match="does not match"):
        await runtime.run(
            make_task(),
            make_profile(),
            paused,
            approval=ApprovalResponse(call_id="other", approved=True),
        )
    assert registry.calls == []

    resumed = await runtime.run(
        make_task(),
        make_profile(),
        paused,
        approval=ApprovalResponse(call_id="approval-call", approved=True),
    )
    assert resumed.status == RunStatus.COMPLETED
    assert len(registry.calls) == 1
    assert resumed.receipts["approval-call"].ok is True


@pytest.mark.asyncio
async def test_approved_call_is_not_executed_after_capability_revocation() -> None:
    provider = ScriptedProvider(
        [
            {
                "action": "sensitive",
                "call_id": "approval-call",
                "arguments": {"record": 7},
                "idempotency_key": "record:7",
            },
            {"action": "final", "answer": "revocation observed"},
        ]
    )
    policy = FakePolicy(
        {"sensitive": PolicyDecision.approval("human confirmation required")}
    )
    runtime, registry, _, _, _ = make_runtime(provider, policy=policy)
    paused = await runtime.run(make_task(), make_profile())

    revoked = make_profile().model_copy(update={"allowed_tools": ["lookup"]})
    resumed = await runtime.run(
        make_task(),
        revoked,
        paused,
        approval=ApprovalResponse(call_id="approval-call", approved=True),
    )

    assert resumed.status == RunStatus.COMPLETED
    assert registry.calls == []
    assert resumed.receipts["approval-call"].error == "approval_invalidated"


@pytest.mark.asyncio
async def test_mid_turn_approval_resumes_remaining_calls_without_replay() -> None:
    provider = ScriptedProvider(
        [
            {
                "tool_calls": [
                    {"call_id": "before", "name": "lookup", "arguments": {"n": 1}},
                    {
                        "call_id": "approve",
                        "name": "sensitive",
                        "arguments": {"n": 2},
                        "idempotency_key": "sensitive:2",
                    },
                    {"call_id": "after", "name": "lookup", "arguments": {"n": 3}},
                ]
            },
            {"action": "final", "answer": "all calls processed"},
        ]
    )
    policy = FakePolicy({"sensitive": PolicyDecision.approval("confirm")})
    runtime, registry, _, _, _ = make_runtime(provider, policy=policy)

    paused = await runtime.run(make_task(), make_profile())
    assert [call.call_id for call in registry.calls] == ["before"]
    assert paused.pending_approval is not None
    assert paused.pending_approval.request_index == 1

    completed = await runtime.run(
        make_task(),
        make_profile(),
        paused,
        approval=ApprovalResponse(call_id="approve", approved=True),
    )
    assert completed.status == RunStatus.COMPLETED
    assert [call.call_id for call in registry.calls] == ["before", "approve", "after"]
    assert [result.call_id for result in completed.steps[0].tool_results] == [
        "before",
        "approve",
        "after",
    ]


@pytest.mark.asyncio
async def test_duplicate_call_receipt_is_reused_but_changed_arguments_fail_closed() -> None:
    identical = ScriptedProvider(
        [
            {"action": "lookup", "call_id": "same", "arguments": {"q": 1}},
            {"action": "lookup", "call_id": "same", "arguments": {"q": 1}},
            {"action": "final", "answer": "done"},
        ]
    )
    runtime, registry, _, _, _ = make_runtime(identical)
    reused = await runtime.run(make_task(), make_profile())
    assert reused.status == RunStatus.COMPLETED
    assert len(registry.calls) == 1
    assert reused.steps[1].tool_results[0].metadata["request_fingerprint"]

    changed = ScriptedProvider(
        [
            {"action": "lookup", "call_id": "same", "arguments": {"q": 1}},
            {"action": "lookup", "call_id": "same", "arguments": {"q": 2}},
        ]
    )
    runtime, registry, _, _, _ = make_runtime(changed)
    failed = await runtime.run(make_task(), make_profile())
    assert failed.status == RunStatus.FAILED
    assert failed.error is not None
    assert failed.error.code == "call_id_reused_with_different_request"
    assert len(registry.calls) == 1


@pytest.mark.asyncio
async def test_provider_failure_is_terminal_but_tool_failure_is_recoverable_observation() -> None:
    provider = ScriptedProvider([RuntimeError("provider unavailable")])
    runtime, _, _, _, _ = make_runtime(provider)
    failed = await runtime.run(make_task(), make_profile())
    assert failed.status == RunStatus.FAILED
    assert failed.error is not None
    assert failed.error.stage == "provider"
    assert failed.error.retryable is False
    assert failed.steps[-1].error == failed.error

    recovery_provider = ScriptedProvider(
        [
            {"action": "lookup", "call_id": "timeout", "arguments": {}},
            {"action": "final", "answer": "used a fallback"},
        ]
    )
    runtime, _, _, _, _ = make_runtime(
        recovery_provider,
        registry=FakeRegistry(fail=True),
    )
    recovered = await runtime.run(make_task(), make_profile())
    assert recovered.status == RunStatus.COMPLETED
    assert recovered.steps[0].tool_results[0].ok is False
    assert "TimeoutError" in (recovered.steps[0].tool_results[0].error or "")
    assert recovery_provider.calls[1].context["trajectory"][0]["tool_results"][0][
        "ok"
    ] is False


@pytest.mark.asyncio
async def test_tool_result_integrity_and_step_budget_fail_deterministically() -> None:
    mismatch_provider = ScriptedProvider(
        [{"action": "lookup", "call_id": "expected", "arguments": {}}]
    )
    runtime, _, _, _, _ = make_runtime(
        mismatch_provider,
        registry=FakeRegistry(mismatched_id=True),
    )
    mismatch = await runtime.run(make_task(), make_profile())
    assert mismatch.status == RunStatus.FAILED
    assert mismatch.error is not None
    assert mismatch.error.stage == "runtime"
    assert "call_id" in mismatch.error.message

    budget_provider = ScriptedProvider(
        [
            {"action": "lookup", "call_id": "one", "arguments": {}},
            {"action": "lookup", "call_id": "two", "arguments": {}},
            {"action": "final", "answer": "must not be reached"},
        ]
    )
    runtime, registry, _, checkpoint, _ = make_runtime(budget_provider)
    limited = await runtime.run(make_task(), make_profile(max_steps=2))
    assert limited.status == RunStatus.STEP_LIMIT
    assert limited.error is not None
    assert limited.error.code == "step_budget_exhausted"
    assert len(registry.calls) == 2
    assert len(budget_provider.calls) == 2
    assert checkpoint.states[-1].status == RunStatus.STEP_LIMIT


@pytest.mark.asyncio
async def test_runtime_ports_integrate_with_real_registry_context_and_sqlite(tmp_path) -> None:
    task = make_task()
    profile = AgentProfile(
        id="profile-real",
        name="calculator",
        instructions="Use the governed calculator.",
        allowed_tools=["double"],
        max_steps=3,
    )
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="double",
            description="Double one integer",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            risk=ToolRisk.COMPUTE,
        ),
        lambda arguments, *_: arguments["value"] * 2,
    )
    provider = ScriptedProvider(
        [
            {
                "action": "double",
                "call_id": "double-1",
                "arguments": {"value": 21},
            },
            {"action": "final", "answer": "42"},
        ]
    )
    checkpoint = SQLiteCheckpointStore(tmp_path / "taskforge.db")
    checkpoint.save_task(task)
    checkpoint.save_profile(profile)
    runtime = AgentRuntime(
        provider=provider,
        registry=registry,
        policy=CapabilityPolicy(registry),
        checkpoint=checkpoint,
        context=ContextAssembler(),
    )

    completed = await runtime.run(task, profile)

    assert completed.status == RunStatus.COMPLETED
    assert completed.receipts["double-1"].output == 42
    persisted = checkpoint.load(completed.run_id)
    assert persisted == completed
    assert provider.calls[1].context["assembled"]["retrieval_query"] == task.goal


@pytest.mark.asyncio
async def test_host_tool_call_limits_cap_same_turn_fanout_and_hide_exhausted_tool(tmp_path) -> None:
    task = make_task()
    profile = AgentProfile(
        id="profile-limited",
        name="limited researcher",
        instructions="Use one lookup and stop.",
        allowed_tools=["lookup"],
        max_steps=2,
        metadata={"tool_call_limits": {"lookup": 1}},
    )
    registry = ToolRegistry()
    calls: list[str] = []
    registry.register(
        ToolSpec(
            name="lookup",
            description="Bounded lookup",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        lambda arguments, *_: calls.append(arguments["query"]) or arguments,
    )
    provider = ScriptedProvider(
        [
            ModelTurn(
                kind="tool",
                tool_requests=[
                    ToolRequest(call_id="first", name="lookup", arguments={"query": "one"}),
                    ToolRequest(call_id="second", name="lookup", arguments={"query": "two"}),
                ],
            ),
            ModelTurn(kind="final", final_answer="done"),
        ]
    )
    checkpoint = SQLiteCheckpointStore(tmp_path / "limited.db")
    checkpoint.save_task(task)
    checkpoint.save_profile(profile)
    runtime = AgentRuntime(
        provider=provider,
        registry=registry,
        policy=CapabilityPolicy(registry),
        checkpoint=checkpoint,
        context=ContextAssembler(),
    )

    completed = await runtime.run(task, profile)

    assert completed.status == RunStatus.COMPLETED
    assert calls == ["one"]
    assert completed.steps[0].tool_results[1].error == "tool_call_limit_exceeded"
    assert provider.calls[1].tools == ()


@pytest.mark.asyncio
async def test_successful_host_terminal_tool_completes_without_extra_model_turn(tmp_path) -> None:
    task = make_task()
    profile = AgentProfile(
        id="profile-terminal",
        name="structured role",
        instructions="Submit the durable result.",
        allowed_tools=["submit"],
        max_steps=1,
        metadata={"terminal_tools": ["submit"]},
    )
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="submit",
            description="Submit durable structured output",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            risk=ToolRisk.COMPUTE,
        ),
        lambda arguments, *_: arguments,
    )
    provider = ScriptedProvider(
        [
            ModelTurn(
                kind="tool",
                tool_requests=[
                    ToolRequest(call_id="submit-1", name="submit", arguments={"value": "done"})
                ],
            )
        ]
    )
    checkpoint = SQLiteCheckpointStore(tmp_path / "terminal.db")
    checkpoint.save_task(task)
    checkpoint.save_profile(profile)
    runtime = AgentRuntime(
        provider=provider,
        registry=registry,
        policy=CapabilityPolicy(registry),
        checkpoint=checkpoint,
        context=ContextAssembler(),
    )

    completed = await runtime.run(task, profile)

    assert completed.status == RunStatus.COMPLETED
    assert completed.final_answer == "Host accepted submit receipt."
    assert len(provider.calls) == 1

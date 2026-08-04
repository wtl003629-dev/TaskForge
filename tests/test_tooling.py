import asyncio

import pytest

from taskforge.domain import AgentProfile, RunState, Task, ToolRequest
from taskforge.tooling import CapabilityPolicy, ToolRegistry, ToolRisk, ToolSpec


def profile(*tools: str) -> AgentProfile:
    return AgentProfile(name="test", instructions="test", allowed_tools=list(tools))


def task() -> Task:
    return Task(tenant_id="tenant-a", user_id="user-a", goal="test")


def state(agent: AgentProfile, current_task: Task) -> RunState:
    return RunState(
        task_id=current_task.id,
        agent_profile_id=agent.id,
        step_budget=agent.max_steps,
    )


def echo_spec(**overrides):
    values = {
        "name": "utility.echo",
        "description": "Return a supplied value",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    }
    values.update(overrides)
    return ToolSpec(**values)


def test_strict_provider_schema_rejects_implicit_optional_properties() -> None:
    with pytest.raises(ValueError, match="strict function schemas"):
        ToolSpec(
            name="utility.optional",
            description="Bad strict schema",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "additionalProperties": False,
            },
        )

    with pytest.raises(ValueError, match="additionalProperties=false"):
        ToolSpec(
            name="utility.open",
            description="Bad open schema",
            parameters={"type": "object", "properties": {}, "additionalProperties": True},
        )


async def test_registry_validates_schema_and_caches_call_id() -> None:
    registry = ToolRegistry()
    calls = 0

    async def handler(arguments, _task, _profile, _state):
        nonlocal calls
        calls += 1
        return {"echo": arguments["value"]}

    registry.register(echo_spec(), handler)
    agent = profile("utility.echo")
    current_task = task()
    run = state(agent, current_task)
    request = ToolRequest(call_id="call-1", name="utility.echo", arguments={"value": "ok"})

    first = await registry.execute(request, current_task, agent, run)
    second = await registry.execute(request, current_task, agent, run)
    invalid = await registry.execute(
        ToolRequest(call_id="call-2", name="utility.echo", arguments={"value": 3}),
        current_task,
        agent,
        run,
    )
    assert first.ok and second.ok
    assert calls == 1
    assert invalid.ok is False and invalid.error.startswith("invalid_arguments")


async def test_policy_requires_approval_and_idempotency_for_writes() -> None:
    registry = ToolRegistry()
    registry.register(
        echo_spec(risk=ToolRisk.WRITE, side_effecting=True, requires_approval=True),
        lambda arguments, *_: arguments,
    )
    agent = profile("utility.echo")
    current_task = task()
    policy = CapabilityPolicy(registry)

    missing_key = await policy.evaluate(
        current_task,
        agent,
        ToolRequest(call_id="one", name="utility.echo", arguments={"value": "x"}),
    )
    approval = await policy.evaluate(
        current_task,
        agent,
        ToolRequest(
            call_id="two",
            name="utility.echo",
            arguments={"value": "x"},
            idempotency_key="task:write:x",
        ),
    )
    assert not missing_key.allowed and not missing_key.requires_approval
    assert not approval.allowed and approval.requires_approval


async def test_idempotency_key_replays_prior_receipt_without_second_effect() -> None:
    registry = ToolRegistry()
    effects = []
    registry.register(
        echo_spec(risk=ToolRisk.WRITE, side_effecting=True, requires_approval=True),
        lambda arguments, *_: effects.append(arguments["value"]) or {"saved": True},
    )
    agent = profile("utility.echo")
    current_task = task()
    run = state(agent, current_task)
    first = ToolRequest(
        call_id="one",
        name="utility.echo",
        arguments={"value": "x"},
        idempotency_key="same-effect",
    )
    second = first.model_copy(update={"call_id": "two"})

    assert (await registry.execute(first, current_task, agent, run)).ok
    replay = await registry.execute(second, current_task, agent, run)
    assert replay.ok
    assert replay.metadata["idempotent_replay_of"] == "one"
    assert effects == ["x"]


async def test_timeout_and_output_cap_are_receipts_not_exceptions() -> None:
    registry = ToolRegistry()

    async def slow(*_):
        await asyncio.sleep(0.05)
        return "late"

    registry.register(echo_spec(timeout_seconds=0.01), slow)
    registry.register(
        ToolSpec(
            name="utility.large",
            description="Return a large value",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            max_output_chars=256,
        ),
        lambda *_: "x" * 1_000,
    )
    agent = profile("utility.echo", "utility.large")
    current_task = task()
    run = state(agent, current_task)

    timed_out = await registry.execute(
        ToolRequest(call_id="slow", name="utility.echo", arguments={"value": "x"}),
        current_task,
        agent,
        run,
    )
    large = await registry.execute(
        ToolRequest(call_id="large", name="utility.large", arguments={}),
        current_task,
        agent,
        run,
    )
    assert timed_out.error == "tool_timeout"
    assert large.ok and large.metadata["truncated"] is True

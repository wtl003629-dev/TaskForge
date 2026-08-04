from __future__ import annotations

from taskforge.builtins import (
    agent_profiles,
    create_tool_registry,
    seed_local_knowledge,
)
from taskforge.domain import RunState, RunStatus, Task, ToolRequest
from taskforge.memory import InMemoryMemoryStore
from taskforge.tooling import CapabilityPolicy


def _setup(tmp_path):
    (tmp_path / "README.md").write_text("TaskForge permission governed Agent runtime", encoding="utf-8")
    (tmp_path / "sample.py").write_text("class AgentRuntime:\n    pass\n", encoding="utf-8")
    knowledge = seed_local_knowledge(tmp_path, tenant_id="local")
    registry = create_tool_registry(
        workspace_root=tmp_path,
        artifact_root=tmp_path / ".taskforge" / "artifacts",
        knowledge_store=knowledge,
        memory_store=InMemoryMemoryStore(),
    )
    profiles = {profile.id: profile for profile in agent_profiles()}
    task = Task(tenant_id="local", user_id="demo", goal="inspect AgentRuntime")
    state = RunState(
        task_id=task.id,
        agent_profile_id="repo-agent",
        status=RunStatus.RUNNING,
        step_budget=8,
    )
    return registry, profiles, task, state


async def test_workspace_tools_are_bound_to_host_root(tmp_path) -> None:
    registry, profiles, task, state = _setup(tmp_path)
    request = ToolRequest(
        call_id="grep-1",
        name="workspace_grep",
        arguments={
            "pattern": "AgentRuntime",
            "include": "*.py",
            "regex": False,
            "case_sensitive": False,
            "limit": 20,
        },
    )

    result = await registry.execute(request, task, profiles["repo-agent"], state)

    assert result.ok
    assert result.output["matches"][0]["path"] == "sample.py"


async def test_knowledge_tool_applies_profile_base_filter(tmp_path) -> None:
    registry, profiles, task, state = _setup(tmp_path)
    profile = profiles["research-agent"]
    state.agent_profile_id = profile.id
    result = await registry.execute(
        ToolRequest(
            call_id="knowledge-1",
            name="knowledge_search",
            arguments={"query": "permission Agent", "limit": 5},
        ),
        task,
        profile,
        state,
    )

    assert result.ok
    assert result.output["hits"][0]["source"] == "README.md"


async def test_artifact_write_needs_idempotency_and_human_approval(tmp_path) -> None:
    registry, profiles, task, state = _setup(tmp_path)
    profile = profiles["repo-agent"]
    request = ToolRequest(
        call_id="write-1",
        name="artifact_write",
        idempotency_key="task-report-0001",
        arguments={
            "filename": "report.md",
            "content": "# Evidence\n",
            "idempotency_key": "task-report-0001",
        },
    )

    decision = await CapabilityPolicy(registry).evaluate(task, profile, request)

    assert decision.requires_approval
    assert not (tmp_path / ".taskforge" / "artifacts" / state.run_id / "report.md").exists()

    result = await registry.execute(request, task, profile, state)
    assert result.ok
    assert result.output["artifact"]["source"].endswith("/report.md")
    assert (tmp_path / ".taskforge" / "artifacts" / state.run_id / "report.md").is_file()


async def test_memory_write_is_approved_and_identity_bound(tmp_path) -> None:
    registry, profiles, task, state = _setup(tmp_path)
    profile = profiles["research-agent"]
    state.agent_profile_id = profile.id
    request = ToolRequest(
        call_id="remember-1",
        name="memory_remember",
        idempotency_key="remember-preference-1",
        arguments={
            "scope": "user",
            "content": "Prefer concise evidence-backed reports",
            "importance": 0.8,
            "expires_in_days": 90,
            "idempotency_key": "remember-preference-1",
        },
    )

    decision = await CapabilityPolicy(registry).evaluate(task, profile, request)
    assert decision.requires_approval

    result = await registry.execute(request, task, profile, state)
    assert result.ok
    assert result.output["scope_id"] == task.user_id

    recalled = await registry.execute(
        ToolRequest(
            call_id="recall-after-write",
            name="memory_recall",
            arguments={"query": "concise evidence", "limit": 5},
        ),
        task,
        profile,
        state,
    )
    assert recalled.ok
    assert recalled.output["hits"][0]["memory_id"] == result.output["memory_id"]
    assert recalled.output["hits"][0]["scope"] == "user"


def test_profiles_prove_configuration_driven_specialisation() -> None:
    profiles = agent_profiles()

    assert {profile.id for profile in profiles} == {"research-agent", "repo-agent", "document-agent"}
    assert len({profile.model for profile in profiles}) == 1
    assert "workspace_grep" in next(p for p in profiles if p.id == "repo-agent").allowed_tools
    assert "workspace_grep" not in next(p for p in profiles if p.id == "research-agent").allowed_tools

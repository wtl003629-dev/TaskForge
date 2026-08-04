from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from taskforge.app import create_app
from taskforge.config import Settings
from taskforge.domain import ModelTurn, ToolRequest
from taskforge.memory import MemoryItem, MemoryScope
from taskforge.providers import ScriptedProvider
from taskforge.worker import DurableWorker


def make_settings(tmp_path: Path, **changes) -> Settings:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    (workspace / "docs").mkdir(exist_ok=True)
    (workspace / "README.md").write_text(
        "TaskForge is a permission governed Agent runtime.", encoding="utf-8"
    )
    (workspace / "AGENTS.md").write_text(
        "The host validates tool proposals and approvals.", encoding="utf-8"
    )
    (workspace / "docs" / "ARCHITECTURE.md").write_text(
        "AgentRuntime checkpoints every durable transition.", encoding="utf-8"
    )
    (workspace / "sample.py").write_text(
        "class AgentRuntime:\n    pass\n", encoding="utf-8"
    )
    values = {
        "sqlite_path": tmp_path / "state" / "taskforge.sqlite3",
        "context_sqlite_path": tmp_path / "state" / "context.sqlite3",
        "operations_sqlite_path": tmp_path / "state" / "operations.sqlite3",
        "orchestration_sqlite_path": tmp_path / "state" / "orchestration.sqlite3",
        "review_case_sqlite_path": tmp_path / "state" / "review-cases.sqlite3",
        "workspace_root": workspace,
        "artifact_root": tmp_path / "artifacts",
        "provider": "demo",
    }
    values.update(changes)
    return Settings(_env_file=None, **values)


def create_waiting_run(client: TestClient, profile_id: str = "research-agent", **kwargs):
    response = client.post(
        "/api/runs",
        json={
            "goal": kwargs.pop("goal", "Inspect TaskForge AgentRuntime architecture"),
            "agent_profile_id": profile_id,
            **kwargs,
        },
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["status"] == "waiting_approval"
    assert payload["pending_approval"]["request"]["name"] == "artifact_write"
    return payload


def test_health_and_agent_catalog_do_not_expose_instructions(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        health = client.get("/health")
        agents = client.get("/api/agents")

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "provider": "demo",
        "execution": "offline-deterministic-demo",
    }
    assert agents.status_code == 200
    payload = agents.json()
    assert {item["id"] for item in payload} == {
        "research-agent",
        "repo-agent",
        "document-agent",
    }
    assert all("instructions" not in item for item in payload)
    assert "instructions" not in json.dumps(payload, ensure_ascii=False).casefold()


def test_skill_pack_is_a_backend_enforced_tool_subset(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        catalog = client.get("/api/agents").json()
        research = next(item for item in catalog if item["id"] == "research-agent")
        research_pack = next(
            item for item in research["skill_packs"] if item["id"] == "research"
        )
        assert set(research_pack["tools"]) < set(research["allowed_tools"])
        assert "memory_remember" not in research_pack["tools"]

        waiting = client.post(
            "/api/runs",
            json={
                "goal": "Find AgentRuntime with the smallest grep capability.",
                "agent_profile_id": "repo-agent",
                "skill_pack_id": "grep",
            },
        )
        unknown = client.post(
            "/api/runs",
            json={
                "goal": "Attempt an unknown pack",
                "agent_profile_id": "repo-agent",
                "skill_pack_id": "not-configured",
            },
        )

    assert waiting.status_code == 201, waiting.text
    run = waiting.json()
    assert run["status"] == "waiting_approval"
    assert run["agent_profile_id"] == "repo-agent--skill--grep"
    stored = app.state.container.store.load_profile(run["agent_profile_id"])
    assert stored.allowed_tools == ["workspace_grep", "artifact_write"]
    assert stored.metadata["base_agent_profile_id"] == "repo-agent"
    assert stored.metadata["selected_skill_pack_id"] == "grep"
    assert unknown.status_code == 422


def test_queued_skill_packs_have_independent_profile_snapshots(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        first = client.post(
            "/api/runs",
            json={
                "goal": "Research governed evidence",
                "agent_profile_id": "research-agent",
                "skill_pack_id": "research",
                "execution_mode": "queued",
            },
        )
        second = client.post(
            "/api/runs",
            json={
                "goal": "Prepare an evidence report",
                "agent_profile_id": "research-agent",
                "skill_pack_id": "reporting",
                "execution_mode": "queued",
            },
        )

    assert first.status_code == second.status_code == 202
    first_id = first.json()["agent_profile_id"]
    second_id = second.json()["agent_profile_id"]
    assert first_id != second_id
    assert app.state.container.store.load_profile(first_id).allowed_tools == [
        "calculator",
        "knowledge_search",
        "memory_recall",
        "artifact_write",
    ]
    assert app.state.container.store.load_profile(second_id).allowed_tools == [
        "knowledge_search",
        "memory_recall",
        "artifact_write",
    ]


@pytest.mark.parametrize(
    ("profile_id", "first_tool"),
    [
        ("research-agent", "knowledge_search"),
        ("document-agent", "knowledge_search"),
        ("repo-agent", "workspace_grep"),
    ],
)
def test_all_profiles_complete_real_approval_flow(
    tmp_path: Path,
    profile_id: str,
    first_tool: str,
) -> None:
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        waiting = create_waiting_run(client, profile_id)
        run_id = waiting["run_id"]
        assert waiting["steps"][0]["model_turn"]["tool_requests"][0]["name"] == first_tool
        read_receipt = waiting["steps"][0]["tool_results"][0]
        assert read_receipt["ok"] is True

        pending = waiting["pending_approval"]["request"]
        key = pending["idempotency_key"]
        assert key == pending["arguments"]["idempotency_key"]
        target = settings.artifact_root / run_id / f"{profile_id}-report.md"
        assert not target.exists()

        approved = client.post(
            f"/api/runs/{run_id}/approve",
            json={"call_id": pending["call_id"], "approved": True, "reason": "reviewed"},
        )

        assert approved.status_code == 200, approved.text
        complete = approved.json()
        assert complete["status"] == "completed"
        assert "deterministic offline demo provider" in complete["final_answer"]
        receipt = complete["receipts"][pending["call_id"]]
        assert receipt["ok"] is True
        assert receipt["output"]["artifact"]["source"].endswith(
            f"/{profile_id}-report.md"
        )
        assert target.is_file()
        report = target.read_text(encoding="utf-8")
        assert "actual bounded receipt" in report
        assert first_tool in report


def test_rejected_approval_finishes_without_side_effect(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        waiting = create_waiting_run(client)
        run_id = waiting["run_id"]
        pending = waiting["pending_approval"]["request"]

        rejected = client.post(
            f"/api/runs/{run_id}/approve",
            json={
                "call_id": pending["call_id"],
                "approved": False,
                "reason": "operator rejected the write",
            },
        )

    assert rejected.status_code == 200, rejected.text
    payload = rejected.json()
    assert payload["status"] == "completed"
    assert payload["receipts"][pending["call_id"]]["error"] == "approval_denied"
    assert "approval_denied" in payload["final_answer"]
    assert not (settings.artifact_root / run_id).exists()


def test_run_and_approval_are_owner_isolated(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    owner = {"X-TaskForge-Tenant": "tenant-a", "X-TaskForge-User": "alice"}
    stranger = {"X-TaskForge-Tenant": "tenant-a", "X-TaskForge-User": "mallory"}
    with TestClient(create_app(settings)) as client:
        created = client.post(
            "/api/runs",
            headers=owner,
            json={"goal": "Find AgentRuntime", "agent_profile_id": "repo-agent"},
        )
        assert created.status_code == 201
        run = created.json()
        run_id = run["run_id"]
        call_id = run["pending_approval"]["request"]["call_id"]

        assert client.get(f"/api/runs/{run_id}", headers=stranger).status_code == 404
        assert client.get(
            "/api/metrics", headers=stranger, params={"run_id": run_id}
        ).status_code == 404
        assert client.get(
            f"/api/runs/{run_id}/audit", headers=stranger
        ).status_code == 404
        assert (
            client.post(
                f"/api/runs/{run_id}/approve",
                headers=stranger,
                json={"call_id": call_id, "approved": True},
            ).status_code
            == 404
        )
        assert client.get(f"/api/runs/{run_id}", headers=owner).status_code == 200


def test_restart_recovers_task_profile_run_and_pending_approval(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as first_client:
        waiting = create_waiting_run(first_client, "document-agent")
    run_id = waiting["run_id"]
    call_id = waiting["pending_approval"]["request"]["call_id"]

    with TestClient(create_app(settings)) as restarted_client:
        persisted = restarted_client.get(f"/api/runs/{run_id}")
        assert persisted.status_code == 200
        assert persisted.json()["status"] == "waiting_approval"

        resumed = restarted_client.post(
            f"/api/runs/{run_id}/approve",
            json={"call_id": call_id, "approved": True},
        )

    assert resumed.status_code == 200, resumed.text
    assert resumed.json()["status"] == "completed"
    assert (settings.artifact_root / run_id / "document-agent-report.md").is_file()


@pytest.mark.asyncio
async def test_concurrent_approvals_execute_artifact_handler_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings)
    replace_calls = 0
    original_replace = os.replace

    def counted_replace(source, target) -> None:
        nonlocal replace_calls
        replace_calls += 1
        original_replace(source, target)

    monkeypatch.setattr("taskforge.builtins.os.replace", counted_replace)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/api/runs",
            json={
                "goal": "Inspect TaskForge architecture",
                "agent_profile_id": "research-agent",
            },
        )
        assert created.status_code == 201
        waiting = created.json()
        run_id = waiting["run_id"]
        call_id = waiting["pending_approval"]["request"]["call_id"]

        async def approve() -> httpx.Response:
            return await client.post(
                f"/api/runs/{run_id}/approve",
                json={"call_id": call_id, "approved": True},
            )

        responses = await asyncio.gather(approve(), approve())

    assert sorted(response.status_code for response in responses) == [200, 409]
    assert replace_calls == 1
    assert (settings.artifact_root / run_id / "research-agent-report.md").is_file()


def test_unknown_profile_run_and_call_id_are_client_errors(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        unknown_profile = client.post(
            "/api/runs",
            json={"goal": "test", "agent_profile_id": "missing-agent"},
        )
        assert unknown_profile.status_code == 404
        assert client.get("/api/runs/missing-run").status_code == 404

        waiting = create_waiting_run(client)
        wrong_call = client.post(
            f"/api/runs/{waiting['run_id']}/approve",
            json={"call_id": "not-the-pending-call", "approved": True},
        )
        assert wrong_call.status_code == 409


def test_identity_and_host_roots_cannot_be_overridden_in_body(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        identity = client.post(
            "/api/runs",
            json={
                "goal": "test",
                "agent_profile_id": "research-agent",
                "tenant_id": "attacker",
                "user_id": "attacker",
            },
        )
        root = client.post(
            "/api/runs",
            json={
                "goal": "test",
                "agent_profile_id": "research-agent",
                "metadata": {"nested": {"artifactRoot": str(tmp_path / "escape")}},
            },
        )
        skill_pack = client.post(
            "/api/runs",
            json={
                "goal": "test",
                "agent_profile_id": "research-agent",
                "metadata": {"skill_pack_id": "reporting"},
            },
        )

    assert identity.status_code == 422
    assert root.status_code == 422
    assert skill_pack.status_code == 422


def test_user_memory_is_persistent_searchable_and_deletable(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    owner = {"X-TaskForge-Tenant": "tenant-a", "X-TaskForge-User": "alice"}
    stranger = {"X-TaskForge-Tenant": "tenant-a", "X-TaskForge-User": "mallory"}
    with TestClient(create_app(settings)) as client:
        created = client.post(
            "/api/memory",
            headers=owner,
            json={
                "content": "Prefer concise evidence-backed architecture reports",
                "importance": 0.9,
                "expires_in_days": 90,
            },
        )
        assert created.status_code == 201, created.text
        memory_id = created.json()["memory_id"]
        assert client.get(
            "/api/memory", headers=stranger, params={"query": "architecture"}
        ).json() == []
        assert client.delete(f"/api/memory/{memory_id}", headers=stranger).status_code == 404

    with TestClient(create_app(settings)) as restarted:
        recalled = restarted.get(
            "/api/memory", headers=owner, params={"query": "architecture"}
        )
        assert recalled.status_code == 200
        assert recalled.json()[0]["memory_id"] == memory_id
        assert recalled.json()[0]["provenance"] == "user_api"
        assert restarted.delete(f"/api/memory/{memory_id}", headers=owner).status_code == 204
        assert restarted.get(
            "/api/memory", headers=owner, params={"query": "architecture"}
        ).json() == []


def test_user_cannot_delete_shared_tenant_memory(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    api = create_app(settings)
    api.state.container.memory_store.remember(
        MemoryItem(
            "shared-policy",
            "tenant-a",
            "shared tenant policy",
            MemoryScope.TENANT,
        )
    )
    headers = {"X-TaskForge-Tenant": "tenant-a", "X-TaskForge-User": "mallory"}
    with TestClient(api) as client:
        visible = client.get(
            "/api/memory", headers=headers, params={"query": "shared"}
        )
        deleted = client.delete("/api/memory/shared-policy", headers=headers)

    assert visible.status_code == 200
    assert visible.json()[0]["memory_id"] == "shared-policy"
    assert deleted.status_code == 404


def test_queued_run_worker_audit_metrics_and_inline_approval(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    api = create_app(settings)
    with TestClient(api) as client:
        queued = client.post(
            "/api/runs",
            json={
                "goal": "Inspect durable TaskForge execution",
                "agent_profile_id": "research-agent",
                "execution_mode": "queued",
            },
        )
        assert queued.status_code == 202, queued.text
        pending = queued.json()
        assert pending["status"] == "pending"
        run_id = pending["run_id"]

        job = client.get(f"/api/runs/{run_id}/job")
        assert job.status_code == 200
        assert job.json()["status"] == "queued"
        assert "lease_token" not in job.json()
        assert "owner" not in job.json()
        assert "lease_version" not in job.json()

        container = api.state.container
        worker = DurableWorker(
            owner="api-test-worker",
            operations=container.operations,
            checkpoints=container.store,
            runtime=container.runtime,
            lease_seconds=10,
        )
        outcome = asyncio.run(worker.run_once())
        assert outcome is not None
        assert outcome.outcome == "waiting_approval"

        waiting = client.get(f"/api/runs/{run_id}").json()
        assert waiting["status"] == "waiting_approval"
        assert client.get(f"/api/runs/{run_id}/job").json()["status"] == "completed"
        call_id = waiting["pending_approval"]["request"]["call_id"]
        approved = client.post(
            f"/api/runs/{run_id}/approve",
            json={"call_id": call_id, "approved": True, "reason": "reviewed"},
        )
        assert approved.status_code == 200
        assert approved.json()["status"] == "completed"

        audit = client.get(f"/api/runs/{run_id}/audit")
        assert audit.status_code == 200
        actions = [event["action"] for event in audit.json()]
        assert "run.enqueue" in actions
        assert "run.execute" in actions
        assert "run.approve" in actions
        assert "tool.execute" in actions

        metrics = client.get("/api/metrics", params={"run_id": run_id})
        assert metrics.status_code == 200
        assert metrics.json()["run_count"] == 1
        assert metrics.json()["run_success_count"] == 1
        assert metrics.json()["tool_count"] == 2


def test_inline_audit_records_usage_and_safety_consistently(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    provider = ScriptedProvider(
        [
            ModelTurn(
                kind="tool",
                tool_requests=[ToolRequest(call_id="denied", name="not_allowed")],
                metadata={"usage": {"input_tokens": 2, "total_tokens": 2}},
            ),
            ModelTurn(
                kind="final",
                final_answer="denial observed",
                metadata={"usage": {"output_tokens": 3, "total_tokens": 3}},
            ),
        ]
    )
    with TestClient(create_app(settings, provider=provider)) as client:
        created = client.post(
            "/api/runs",
            json={"goal": "attempt denied capability", "agent_profile_id": "research-agent"},
        )
        assert created.status_code == 201
        run_id = created.json()["run_id"]
        metrics = client.get("/api/metrics", params={"run_id": run_id}).json()
        audit = client.get(f"/api/runs/{run_id}/audit").json()

    assert metrics["total_tokens"] == 5
    assert metrics["safety_violation_count"] == 1
    assert any(
        event["action"] == "tool.execute" and event["safety_violation"]
        for event in audit
    )


def test_mcp_config_is_host_owned_and_status_is_sanitised(tmp_path: Path) -> None:
    assert Settings(_env_file=None, mcp_config_path="").mcp_config_path is None
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "profile_ids": ["research-agent"],
                        "server": {
                            "namespace": "docs",
                            "endpoint": "https://example.com/mcp",
                            "enabled": False,
                            "allowed_tools": [],
                            "tool_policies": {},
                            "secret_env_var": "PRIVATE_MCP_TOKEN",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    settings = make_settings(tmp_path, mcp_config_path=config_path)
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/mcp/servers")

    assert response.status_code == 200
    assert response.json() == [
        {
            "namespace": "docs",
            "enabled": False,
            "profile_ids": ["research-agent"],
            "configured_tools": [],
            "mounted_tools": [],
        }
    ]
    assert "example.com" not in response.text
    assert "PRIVATE_MCP_TOKEN" not in response.text


@pytest.mark.parametrize(
    "overrides",
    [
        {"openai_api_key": None, "openai_model": "gpt-test"},
        {"openai_api_key": "sk-test", "openai_model": None},
    ],
)
def test_openai_selection_fails_fast_when_required_configuration_is_missing(
    tmp_path: Path,
    overrides: dict,
) -> None:
    settings = make_settings(tmp_path, provider="openai", **overrides)
    with pytest.raises(ValueError, match="TASKFORGE_OPENAI"):
        create_app(settings)

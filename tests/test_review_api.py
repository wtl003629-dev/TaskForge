from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from taskforge.app import ReviewExecutionDisclosure, create_app
from taskforge.config import Settings
from taskforge.domain import ModelTurn
from taskforge.providers import ScriptedProvider


OWNER = {"X-TaskForge-Tenant": "tenant-a", "X-TaskForge-User": "alice"}
STRANGER = {"X-TaskForge-Tenant": "tenant-a", "X-TaskForge-User": "mallory"}
OTHER_TENANT = {"X-TaskForge-Tenant": "tenant-b", "X-TaskForge-User": "alice"}


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    (workspace / "policy.txt").write_text(
        "Enterprise changes require evidence and a human final decision.",
        encoding="utf-8",
    )
    state = tmp_path / "state"
    values: dict[str, object] = {
        "sqlite_path": state / "taskforge.sqlite3",
        "context_sqlite_path": state / "context.sqlite3",
        "operations_sqlite_path": state / "operations.sqlite3",
        "orchestration_sqlite_path": state / "orchestration.sqlite3",
        "review_case_sqlite_path": state / "review-cases.sqlite3",
        "workspace_root": workspace,
        "artifact_root": tmp_path / "artifacts",
        "provider": "demo",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def case_body(*, title: str = "Payment cluster migration") -> dict:
    return {
        "kind": "enterprise_change",
        "title": title,
        "submission": {
            "request_summary": "Move the payment service to the new cluster.",
            "business_justification": "The old cluster reaches end of support.",
            "attributes": {"change_window": "2026-08-10T02:00:00Z"},
            "evidence_refs": [
                {
                    "evidence_id": "change-ticket-17",
                    "source_type": "document",
                    "locator": "case://change-ticket-17",
                    "title": "Approved change request",
                    "version": "3",
                    "excerpt": (
                        "The payment service migration is approved for the stated "
                        "window, has a tested rollback plan, and requires a human "
                        "reviewer before production execution."
                    ),
                }
            ],
        },
    }


def create_case(
    client: TestClient,
    *,
    headers: dict[str, str] = OWNER,
    key: str = "create-payment-migration",
    body: dict | None = None,
) -> dict:
    response = client.post(
        "/api/review-cases",
        headers={**headers, "Idempotency-Key": key},
        json=body or case_body(),
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_demo_review_case_requires_a_human_final_decision(tmp_path: Path) -> None:
    with TestClient(create_app(make_settings(tmp_path))) as client:
        created = create_case(client)
        case_id = created["case"]["case_id"]

        assert created["case"]["conversation_id"] == case_id
        assert created["case"]["status"] == "draft"
        assert created["plan"] is None
        assert created["execution"] == {
            "provider": "demo",
            "mode": "offline-deterministic-demo",
            "provider_configured": False,
            "contract_tested_mock": True,
            "live_smoke_verified": False,
            "business_e2e_verified": False,
            "recommendation_authority": "model_untrusted",
            "final_decision_authority": "human",
        }
        assert "idempotency_key" not in created["audit_events"][0]
        assert "request_hash" not in created["audit_events"][0]

        started_response = client.post(
            f"/api/review-cases/{case_id}/submit-and-start",
            headers={**OWNER, "Idempotency-Key": "start-payment-migration"},
        )
        assert started_response.status_code == 200, started_response.text
        started = started_response.json()
        assert started["case"]["status"] == "running"
        assert len(started["plan"]["slots"]) == 4
        assert "objective" not in started["plan"]
        assert "client_idempotency_key" not in started["plan"]
        assert "request_hash" not in started["plan"]

        run_response = client.post(
            f"/api/review-cases/{case_id}/run-until-review",
            headers=OWNER,
            json={"max_iterations": 4},
        )
        assert run_response.status_code == 200, run_response.text
        waiting = run_response.json()
        assert waiting["case"]["status"] == "waiting_human_review"
        assert waiting["case"]["human_decision"] is None
        recommendation = waiting["case"]["recommendation"]
        assert recommendation["authority"] == "model_untrusted"
        assert "deterministic offline demo provider" in recommendation["summary"]
        assert len(waiting["role_runs"]) == 4
        assert all(item["status"] == "succeeded" for item in waiting["role_runs"])
        assert all("runtime_run_id" not in item for item in waiting["role_runs"])
        assert all(item["runtime_metrics"]["step_count"] >= 3 for item in waiting["role_runs"])
        assert all(
            item["runtime_metrics"]["tool_failure_count"] == 0
            for item in waiting["role_runs"]
        )
        assert all(
            item["runtime_metrics"]["safety_violation_count"] == 0
            for item in waiting["role_runs"]
        )
        # The demo role claims cite genuinely retrieved evidence, so host
        # citation-grounded verification promotes them to verified/tool facts.
        assert all(item["status"] == "verified" for item in waiting["shared_facts"])
        assert all(item["authority"] == "tool" for item in waiting["shared_facts"])
        assert len(waiting["handoffs"]) == 4
        assert {item["to_slot_id"] for item in waiting["handoffs"]} == {
            "compliance",
            "risk",
            "decision",
        }

        revision = waiting["case"]["revision"]
        decision_body = {
            "expected_revision": revision,
            "outcome": "approved",
            "rationale": "The accountable owner confirmed the rollback plan.",
            "evidence_ref_ids": ["change-ticket-17"],
            "display_name": "Alice Reviewer",
        }
        decided_response = client.post(
            f"/api/review-cases/{case_id}/decision",
            headers={**OWNER, "Idempotency-Key": "human-decision-1"},
            json=decision_body,
        )
        assert decided_response.status_code == 200, decided_response.text
        decided = decided_response.json()
        assert decided["case"]["status"] == "approved"
        assert decided["case"]["human_decision"]["actor"] == {
            "actor_user_id": "alice",
            "display_name": "Alice Reviewer",
            "authority": "human",
        }
        assert decided["audit_events"][-1]["actor_authority"] == "human"

        replay = client.post(
            f"/api/review-cases/{case_id}/decision",
            headers={**OWNER, "Idempotency-Key": "human-decision-1"},
            json=decision_body,
        )
        assert replay.status_code == 200
        assert replay.json()["case"]["revision"] == decided["case"]["revision"]

        inbox = client.get(
            "/api/review-cases",
            headers=OWNER,
            params={"status": "approved"},
        )
        assert inbox.status_code == 200
        assert [item["case_id"] for item in inbox.json()["items"]] == [case_id]


def test_openai_credentials_only_disclose_configured_not_verified(
    tmp_path: Path,
) -> None:
    settings = make_settings(
        tmp_path,
        provider="openai",
        openai_api_key="sk-contract-only-not-live",
        openai_model="gpt-contract-test",
    )
    # Creating the provider and reading disclosure must not make a model call.
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/review-cases", headers=OWNER)

    assert response.status_code == 200, response.text
    assert response.json()["execution"] == {
        "provider": "openai",
        "mode": "configured-provider",
        "provider_configured": True,
        "contract_tested_mock": True,
        "live_smoke_verified": False,
        "business_e2e_verified": False,
        "recommendation_authority": "model_untrusted",
        "final_decision_authority": "human",
    }


@pytest.mark.parametrize(
    "field_name",
    ["live_smoke_verified", "business_e2e_verified"],
)
def test_execution_disclosure_cannot_claim_unpersisted_live_verification(
    field_name: str,
) -> None:
    payload = {
        "provider": "openai",
        "mode": "configured-provider",
        "provider_configured": True,
        "contract_tested_mock": True,
        field_name: True,
    }
    with pytest.raises(ValidationError):
        ReviewExecutionDisclosure.model_validate(payload)


def test_review_case_routes_are_exactly_owner_scoped(tmp_path: Path) -> None:
    with TestClient(create_app(make_settings(tmp_path))) as client:
        created = create_case(client)
        case_id = created["case"]["case_id"]

        for headers in (STRANGER, OTHER_TENANT):
            assert client.get(
                f"/api/review-cases/{case_id}", headers=headers
            ).status_code == 404
            assert client.get("/api/review-cases", headers=headers).json()["items"] == []
            assert client.post(
                f"/api/review-cases/{case_id}/submit-and-start",
                headers={**headers, "Idempotency-Key": "foreign-start"},
            ).status_code == 404
            assert client.post(
                f"/api/review-cases/{case_id}/execute-next",
                headers=headers,
            ).status_code == 404
            assert client.post(
                f"/api/review-cases/{case_id}/run-until-review",
                headers=headers,
                json={"max_iterations": 4},
            ).status_code == 404
            assert client.post(
                f"/api/review-cases/{case_id}/decision",
                headers={**headers, "Idempotency-Key": "foreign-decision"},
                json={
                    "expected_revision": 1,
                    "outcome": "approved",
                    "rationale": "Attempted lateral decision.",
                },
            ).status_code == 404

        assert client.get(f"/api/review-cases/{case_id}", headers=OWNER).status_code == 200


def test_review_case_api_rejects_authority_injection_and_conflicts(
    tmp_path: Path,
) -> None:
    with TestClient(create_app(make_settings(tmp_path))) as client:
        injected = {
            **case_body(),
            "tenant_id": "attacker-tenant",
            "owner_user_id": "attacker",
            "conversation_id": "attacker-conversation",
            "case_id": "attacker-case",
            "status": "approved",
            "recommendation": {"authority": "model_untrusted"},
        }
        assert client.post(
            "/api/review-cases",
            headers={**OWNER, "Idempotency-Key": "injected-create"},
            json=injected,
        ).status_code == 422
        assert client.post(
            "/api/review-cases",
            headers=OWNER,
            json=case_body(),
        ).status_code == 422

        created = create_case(client)
        case_id = created["case"]["case_id"]
        replay = client.post(
            "/api/review-cases",
            headers={**OWNER, "Idempotency-Key": "create-payment-migration"},
            json=case_body(),
        )
        assert replay.status_code == 201
        assert replay.json()["case"]["case_id"] == case_id

        changed = client.post(
            "/api/review-cases",
            headers={**OWNER, "Idempotency-Key": "create-payment-migration"},
            json=case_body(title="Changed title under reused key"),
        )
        assert changed.status_code == 409

        invalid_decision = client.post(
            f"/api/review-cases/{case_id}/decision",
            headers={**OWNER, "Idempotency-Key": "invalid-decision"},
            json={
                "expected_revision": 1,
                "outcome": "escalate",
                "rationale": "A model recommendation cannot be a final decision.",
                "actor_user_id": "attacker",
                "authority": "model_untrusted",
            },
        )
        assert invalid_decision.status_code == 422


def test_review_case_state_survives_app_restart(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        created = create_case(client, key="restart-create")
        case_id = created["case"]["case_id"]
        started = client.post(
            f"/api/review-cases/{case_id}/submit-and-start",
            headers={**OWNER, "Idempotency-Key": "restart-start"},
        )
        assert started.status_code == 200
        assert started.json()["case"]["status"] == "running"

    with TestClient(create_app(settings)) as restarted:
        persisted = restarted.get(f"/api/review-cases/{case_id}", headers=OWNER)
        assert persisted.status_code == 200
        assert persisted.json()["case"]["status"] == "running"
        assert len(persisted.json()["plan"]["slots"]) == 4
        replay = restarted.post(
            f"/api/review-cases/{case_id}/submit-and-start",
            headers={**OWNER, "Idempotency-Key": "restart-start"},
        )
        assert replay.status_code == 200
        assert replay.json()["case"]["revision"] == 3
        assert len(replay.json()["audit_events"]) == 3


@pytest.mark.asyncio
async def test_concurrent_human_decisions_commit_exactly_once(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/api/review-cases",
            headers={**OWNER, "Idempotency-Key": "concurrent-create"},
            json=case_body(),
        )
        assert created.status_code == 201, created.text
        case_id = created.json()["case"]["case_id"]
        started = await client.post(
            f"/api/review-cases/{case_id}/submit-and-start",
            headers={**OWNER, "Idempotency-Key": "concurrent-start"},
        )
        assert started.status_code == 200, started.text
        waiting = await client.post(
            f"/api/review-cases/{case_id}/run-until-review",
            headers=OWNER,
            json={"max_iterations": 4},
        )
        assert waiting.status_code == 200, waiting.text
        revision = waiting.json()["case"]["revision"]

        async def decide(outcome: str, key: str) -> httpx.Response:
            return await client.post(
                f"/api/review-cases/{case_id}/decision",
                headers={**OWNER, "Idempotency-Key": key},
                json={
                    "expected_revision": revision,
                    "outcome": outcome,
                    "rationale": f"Concurrent human decision: {outcome}.",
                    "evidence_ref_ids": ["change-ticket-17"],
                },
            )

        approved, rejected = await asyncio.gather(
            decide("approved", "concurrent-approve"),
            decide("rejected", "concurrent-reject"),
        )
        assert sorted([approved.status_code, rejected.status_code]) == [200, 409]
        final = await client.get(f"/api/review-cases/{case_id}", headers=OWNER)
        assert final.json()["case"]["status"] in {"approved", "rejected"}
        assert len(final.json()["audit_events"]) == 5


def test_review_api_persists_failed_case_after_role_attempt_exhaustion(
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider(
        [
            ModelTurn(kind="final", final_answer="No structured receipt one."),
            ModelTurn(kind="final", final_answer="No structured receipt two."),
        ]
    )
    with TestClient(create_app(make_settings(tmp_path), provider=provider)) as client:
        created = create_case(client, key="api-exhaustion-create")
        case_id = created["case"]["case_id"]
        started = client.post(
            f"/api/review-cases/{case_id}/submit-and-start",
            headers={**OWNER, "Idempotency-Key": "api-exhaustion-start"},
        )
        assert started.status_code == 200, started.text

        response = client.post(
            f"/api/review-cases/{case_id}/run-until-review",
            headers=OWNER,
            json={"max_iterations": 2},
        )
        assert response.status_code == 200, response.text
        detail = response.json()
        assert detail["case"]["status"] == "failed"
        assert "intake" in detail["case"]["failure"]["reason"]
        assert detail["plan"]["status"] == "failed"
        assert [run["attempt"] for run in detail["role_runs"]] == [1, 2]
        assert all(run["status"] == "failed" for run in detail["role_runs"])
        assert detail["audit_events"][-1]["event_type"] == "case_failed"
        assert detail["execution"] == {
            "provider": "ScriptedProvider",
            "mode": "injected-test-provider",
            "provider_configured": False,
            "contract_tested_mock": False,
            "live_smoke_verified": False,
            "business_e2e_verified": False,
            "recommendation_authority": "model_untrusted",
            "final_decision_authority": "human",
        }

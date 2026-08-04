"""Offline HTTP-contract tests for the OpenAI Responses review workflow.

These tests exercise the real ``OpenAIResponsesProvider`` adapter through an
``httpx.MockTransport``.  They make no network calls and provide no evidence
about live OpenAI availability, model reasoning quality, or model accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from taskforge.builtins import create_tool_registry
from taskforge.case_profiles import ENTERPRISE_REVIEW_ROLES, enterprise_review_profiles
from taskforge.case_runtime import CaseAgentExecutor
from taskforge.checkpoints import SQLiteCheckpointStore
from taskforge.context import ContextAssembler
from taskforge.knowledge import InMemoryKnowledgeStore, KnowledgeChunk
from taskforge.memory import InMemoryMemoryStore
from taskforge.openai_provider import OpenAIResponsesProvider
from taskforge.orchestration import (
    FactStatus,
    OrchestrationAccess,
    PlanStatus,
    RoleRunStatus,
    SQLiteOrchestrationStore,
)
from taskforge.review_cases import (
    CaseKind,
    CaseStatus,
    CaseSubmission,
    EvidenceRef,
    RecommendationOutcome,
    SQLiteReviewCaseStore,
)
from taskforge.review_service import (
    RecommendationEvidenceError,
    ReviewCaseCoordinator,
)
from taskforge.runtime import AgentRuntime
from taskforge.tooling import CapabilityPolicy


TENANT_ID = "tenant-contract"
USER_ID = "reviewer-contract"
EVIDENCE_ID = "change-ticket-17"
MODEL_ID = "gpt-contract-mock"


def _function_call_response(
    *,
    response_id: str,
    call_id: str,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": response_id,
        "model": MODEL_ID,
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": json.dumps(
                    arguments,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        ],
    }


def _final_response(*, response_id: str, role_id: str) -> dict[str, Any]:
    return {
        "id": response_id,
        "model": MODEL_ID,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": f"Mock contract final acknowledged for {role_id}.",
                    }
                ],
            }
        ],
    }


class MockResponsesReviewContract:
    """Stateful local Responses endpoint; this is not a model simulator."""

    _CLAIMS: dict[str, tuple[str, Any]] = {
        "intake_analyst": ("intake.scope", "payment-service"),
        "compliance_reviewer": ("compliance.result", "controls-present"),
        "risk_reviewer": ("risk.level", "medium"),
        "decision_synthesizer": ("decision.outcome", "approve"),
    }

    def __init__(self, *, decision_evidence_ref: str = EVIDENCE_ID) -> None:
        self.decision_evidence_ref = decision_evidence_ref
        self.phase_by_role = {role_id: 0 for role_id in ENTERPRISE_REVIEW_ROLES}
        self.payloads: list[dict[str, Any]] = []
        self.retrieval_receipts: dict[str, dict[str, Any]] = {}
        self.submit_receipts: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _role_from_payload(payload: dict[str, Any]) -> str:
        instructions = payload.get("instructions")
        assert isinstance(instructions, str)
        matches = [
            role_id
            for role_id in ENTERPRISE_REVIEW_ROLES
            if f"fixed {role_id!r} role" in instructions
        ]
        assert len(matches) == 1
        return matches[0]

    @staticmethod
    def _continuation_receipt(payload: dict[str, Any]) -> dict[str, Any]:
        items = payload.get("input")
        assert isinstance(items, list) and len(items) == 1
        assert items[0]["type"] == "function_call_output"
        output = json.loads(items[0]["output"])
        assert output["call_id"] == items[0]["call_id"]
        return output

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("https://api.openai.com/v1/responses")
        assert request.headers["Authorization"] == "Bearer sk-contract-not-real"
        payload = json.loads(request.content)
        self.payloads.append(payload)
        assert payload["model"] == MODEL_ID
        assert all(tool["type"] == "function" for tool in payload["tools"])
        tool_names = {tool["name"] for tool in payload["tools"]}
        assert {"knowledge_search", "submit_role_result"} <= tool_names

        role_id = self._role_from_payload(payload)
        phase = self.phase_by_role[role_id]

        if phase == 0:
            # A fresh role must send the bounded case task, not a fabricated
            # continuation. The returned function call is only a proposal.
            assert "previous_response_id" not in payload
            assert isinstance(payload["input"], str)
            assert "CASE_INPUT_JSON=" in payload["input"]
            assert EVIDENCE_ID in payload["input"]
            response = _function_call_response(
                response_id=f"resp-{role_id}-knowledge",
                call_id=f"knowledge-{role_id}",
                name="knowledge_search",
                arguments={"query": "payment change policy evidence", "limit": 5},
            )
        elif phase == 1:
            assert payload["previous_response_id"] == f"resp-{role_id}-knowledge"
            receipt = self._continuation_receipt(payload)
            assert receipt["call_id"] == f"knowledge-{role_id}"
            assert receipt["ok"] is True
            hits = receipt["output"]["hits"]
            hits_by_evidence_id = {
                hit["evidence_id"]: hit
                for hit in hits
                if isinstance(hit.get("evidence_id"), str)
            }
            assert hits_by_evidence_id[EVIDENCE_ID]["chunk_id"] == EVIDENCE_ID
            assert (
                hits_by_evidence_id[EVIDENCE_ID]["source"]
                == "case://change-ticket-17"
            )
            if role_id == "decision_synthesizer":
                assert self.decision_evidence_ref in hits_by_evidence_id
            self.retrieval_receipts[role_id] = receipt

            fact_key, value = self._CLAIMS[role_id]
            evidence_ref = (
                self.decision_evidence_ref
                if role_id == "decision_synthesizer"
                else EVIDENCE_ID
            )
            response = _function_call_response(
                response_id=f"resp-{role_id}-submit",
                call_id=f"submit-{role_id}",
                name="submit_role_result",
                arguments={
                    "claims": [
                        {
                            "fact_key": fact_key,
                            "value": value,
                            "evidence_refs": [evidence_ref],
                            "confidence": 0.9,
                        }
                    ],
                    "summary": f"Mock Responses contract result for {role_id}.",
                    "handoff_summary": (
                        f"Mock contract handoff from {role_id}; human authority retained."
                    ),
                },
            )
        elif phase == 2:
            assert payload["previous_response_id"] == f"resp-{role_id}-submit"
            receipt = self._continuation_receipt(payload)
            assert receipt["call_id"] == f"submit-{role_id}"
            assert receipt["ok"] is True
            structured = receipt["output"]
            assert structured["receipt_type"] == "taskforge.role_result.v1"
            assert structured["binding"]["role_id"] == role_id
            assert structured["submission"]["claims"][0]["fact_key"] == self._CLAIMS[
                role_id
            ][0]
            self.submit_receipts[role_id] = receipt
            response = _final_response(
                response_id=f"resp-{role_id}-final",
                role_id=role_id,
            )
        else:  # pragma: no cover - any extra provider call is a contract failure
            raise AssertionError(f"unexpected extra Responses call for {role_id}")

        self.phase_by_role[role_id] = phase + 1
        return httpx.Response(200, json=response)


def _submission() -> CaseSubmission:
    return CaseSubmission(
        request_summary="Move the payment service to the new production cluster.",
        business_justification="The old cluster reaches end of support this quarter.",
        attributes={"change_window": "2026-08-10T02:00:00Z"},
        evidence_refs=[
            EvidenceRef(
                evidence_id=EVIDENCE_ID,
                source_type="document",
                locator="case://change-ticket-17",
                title="Approved change request",
                version="3",
                excerpt=(
                    "Payment change policy evidence confirms the approved change "
                    "ticket and rollback plan."
                ),
            )
        ],
    )


def _coordinator(
    tmp_path: Path,
    provider: OpenAIResponsesProvider,
) -> tuple[
    ReviewCaseCoordinator,
    SQLiteOrchestrationStore,
    InMemoryKnowledgeStore,
]:
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
        checkpoint=SQLiteCheckpointStore(tmp_path / "runtime.sqlite3"),
        context=ContextAssembler(knowledge, memory),
    )
    orchestration_store = SQLiteOrchestrationStore(tmp_path / "orchestration.sqlite3")
    case_store = SQLiteReviewCaseStore(tmp_path / "review-cases.sqlite3")
    profiles = {
        profile.id: profile for profile in enterprise_review_profiles(model=MODEL_ID)
    }
    executor = CaseAgentExecutor(
        store=orchestration_store,
        runtime=runtime,
        user_id=USER_ID,
        profiles=profiles,
    )
    return (
        ReviewCaseCoordinator(
            case_store=case_store,
            orchestration_store=orchestration_store,
            executor=executor,
            tenant_id=TENANT_ID,
            user_id=USER_ID,
        ),
        orchestration_store,
        knowledge,
    )


def _seed_case_evidence(
    knowledge: InMemoryKnowledgeStore,
    case_id: str,
    *,
    include_unsubmitted_evidence: bool = False,
) -> None:
    knowledge.upsert(
        KnowledgeChunk(
            chunk_id=EVIDENCE_ID,
            tenant_id=TENANT_ID,
            text=(
                "Payment change policy evidence confirms the approved change "
                "ticket, rollback plan, and human decision requirement."
            ),
            source_uri="case://change-ticket-17",
            document_id="change-ticket-17",
            version="3",
            version_order=3,
            metadata={
                "knowledge_base_id": f"enterprise-review:{case_id}",
                "evidence_id": EVIDENCE_ID,
            },
        )
    )
    if include_unsubmitted_evidence:
        knowledge.upsert(
            KnowledgeChunk(
                chunk_id="fabricated-policy-reference",
                tenant_id=TENANT_ID,
                text=(
                    "Payment change policy evidence from another retrieved item "
                    "that was not submitted with this review case."
                ),
                source_uri="case://fabricated-policy-reference",
                document_id="fabricated-policy-reference",
                metadata={
                    "knowledge_base_id": f"enterprise-review:{case_id}",
                    "evidence_id": "fabricated-policy-reference",
                },
            )
        )


async def _run_review(
    tmp_path: Path,
    contract: MockResponsesReviewContract,
) -> tuple[
    ReviewCaseCoordinator,
    SQLiteOrchestrationStore,
    Any,
    httpx.AsyncClient,
]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(contract))
    provider = OpenAIResponsesProvider(
        api_key="sk-contract-not-real",
        enabled=True,
        model=MODEL_ID,
        client=client,
    )
    coordinator, orchestration_store, knowledge = _coordinator(tmp_path, provider)
    draft = coordinator.create_draft(
        kind=CaseKind.ENTERPRISE_CHANGE,
        title="Payment cluster migration",
        submission=_submission(),
        idempotency_key="openai-contract-create",
    )
    _seed_case_evidence(knowledge, draft.case_id)
    coordinator.submit_and_start(
        draft.case_id,
        idempotency_key="openai-contract-start",
    )
    try:
        finished = await coordinator.run_until_pause_or_review(
            draft.case_id,
            max_iterations=4,
        )
    except Exception:
        await client.aclose()
        raise
    return coordinator, orchestration_store, finished, client


@pytest.mark.asyncio
async def test_mock_openai_responses_contract_runs_four_role_review_chain(
    tmp_path: Path,
) -> None:
    """Protocol coverage only: this does not test a live model or its quality."""

    contract = MockResponsesReviewContract()
    coordinator, orchestration_store, finished, client = await _run_review(
        tmp_path,
        contract,
    )
    try:
        assert finished.review_case.status is CaseStatus.WAITING_HUMAN_REVIEW
        assert finished.review_case.human_decision is None
        recommendation = finished.review_case.recommendation
        assert recommendation is not None
        assert recommendation.authority == "model_untrusted"
        assert recommendation.outcome is RecommendationOutcome.APPROVE
        assert [item.evidence_id for item in recommendation.evidence_refs] == [
            EVIDENCE_ID
        ]
        assert recommendation.evidence_refs[0] == _submission().evidence_refs[0]
        assert finished.plan is not None
        assert finished.plan.status is PlanStatus.COMPLETED

        access = OrchestrationAccess(
            tenant_id=TENANT_ID,
            user_id=USER_ID,
            conversation_id=finished.review_case.case_id,
        )
        runs = orchestration_store.list_role_runs(access, finished.plan.plan_id)
        assert len(runs) == 4
        assert all(run.status is RoleRunStatus.SUCCEEDED for run in runs)
        facts = orchestration_store.list_shared_facts(access)
        assert len(facts) == 4
        assert all(fact.status is FactStatus.VERIFIED for fact in facts)
        assert all(fact.authority == "tool" for fact in facts)
        handoffs = orchestration_store.list_handoffs(access, finished.plan.plan_id)
        assert len(handoffs) == 4
        assert {handoff.to_slot_id for handoff in handoffs} == {
            "compliance",
            "risk",
            "decision",
        }

        assert len(contract.payloads) == 12
        assert contract.phase_by_role == {
            role_id: 3 for role_id in ENTERPRISE_REVIEW_ROLES
        }
        assert set(contract.retrieval_receipts) == set(ENTERPRISE_REVIEW_ROLES)
        assert set(contract.submit_receipts) == set(ENTERPRISE_REVIEW_ROLES)
        # The provider returned an approve recommendation, but the host still
        # stopped at human review instead of granting decision authority.
        assert coordinator.get_state(finished.review_case.case_id).review_case.status is (
            CaseStatus.WAITING_HUMAN_REVIEW
        )
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_mock_openai_contract_unbound_decision_evidence_fails_closed(
    tmp_path: Path,
) -> None:
    """HTTP schema success cannot bypass exact submitted-evidence binding."""

    contract = MockResponsesReviewContract(
        decision_evidence_ref="fabricated-policy-reference"
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(contract))
    provider = OpenAIResponsesProvider(
        api_key="sk-contract-not-real",
        enabled=True,
        model=MODEL_ID,
        client=client,
    )
    coordinator, orchestration_store, knowledge = _coordinator(tmp_path, provider)
    draft = coordinator.create_draft(
        kind=CaseKind.ENTERPRISE_CHANGE,
        title="Payment cluster migration",
        submission=_submission(),
        idempotency_key="openai-negative-create",
    )
    _seed_case_evidence(
        knowledge,
        draft.case_id,
        include_unsubmitted_evidence=True,
    )
    coordinator.submit_and_start(
        draft.case_id,
        idempotency_key="openai-negative-start",
    )
    try:
        with pytest.raises(
            RecommendationEvidenceError,
            match="exactly match a submitted evidence_id or locator",
        ):
            await coordinator.run_until_pause_or_review(
                draft.case_id,
                max_iterations=4,
            )

        persisted = coordinator.get_state(draft.case_id)
        assert persisted.review_case.status is CaseStatus.RUNNING
        assert persisted.review_case.recommendation is None
        assert persisted.plan is not None
        assert persisted.plan.status is PlanStatus.RUNNING
        runs = orchestration_store.list_role_runs(
            OrchestrationAccess(
                tenant_id=TENANT_ID,
                user_id=USER_ID,
                conversation_id=draft.case_id,
            ),
            persisted.plan.plan_id,
        )
        assert len(runs) == 4
        assert all(run.status is RoleRunStatus.SUCCEEDED for run in runs)
        assert len(contract.payloads) == 12
    finally:
        await client.aclose()

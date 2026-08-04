from __future__ import annotations

import pytest

from taskforge.case_profiles import enterprise_review_profiles
from taskforge.case_runtime import RoleResultSubmission
from taskforge.demo import DemoProvider
from taskforge.domain import Task


TOOLS = [
    {"name": "knowledge_search"},
    {"name": "submit_role_result"},
]


@pytest.mark.asyncio
async def test_demo_case_mode_collects_evidence_submits_structure_then_final() -> None:
    provider = DemoProvider()
    profile = enterprise_review_profiles(model="demo")[0]
    task = Task(
        id="task-1",
        tenant_id="tenant-a",
        user_id="alice",
        goal="Review change request",
    )

    first = await provider.complete(task=task, profile=profile, context={}, tools=TOOLS)
    assert first.kind == "tool"
    assert first.tool_requests[0].name == "knowledge_search"

    knowledge_step = {
        "tool_requests": [first.tool_requests[0].model_dump(mode="json")],
        "tool_results": [
            {
                "call_id": first.tool_requests[0].call_id,
                "ok": True,
                "output": {
                    "hits": [
                        {
                            "chunk_id": "policy-42",
                            "source": "policy.pdf",
                            "text": "approval control",
                        }
                    ]
                },
                "error": None,
                "metadata": {"tool": "knowledge_search"},
            }
        ],
    }
    second = await provider.complete(
        task=task,
        profile=profile,
        context={"trajectory": [knowledge_step]},
        tools=TOOLS,
    )
    assert second.tool_requests[0].name == "submit_role_result"
    submission = RoleResultSubmission.model_validate(second.tool_requests[0].arguments)
    assert submission.claims[0].evidence_refs == ["policy-42"]
    assert submission.claims[0].value["offline_demo"] is True

    submit_step = {
        "tool_requests": [second.tool_requests[0].model_dump(mode="json")],
        "tool_results": [
            {
                "call_id": second.tool_requests[0].call_id,
                "ok": True,
                "output": {"receipt_type": "taskforge.role_result.v1"},
                "error": None,
                "metadata": {"tool": "submit_role_result"},
            }
        ],
    }
    third = await provider.complete(
        task=task,
        profile=profile,
        context={"trajectory": [knowledge_step, submit_step]},
        tools=TOOLS,
    )
    assert third.kind == "final"
    assert "No live model was called" in third.final_answer


@pytest.mark.asyncio
async def test_demo_case_fallback_evidence_is_explicitly_unverified() -> None:
    provider = DemoProvider()
    profile = enterprise_review_profiles(model="demo")[1]
    task = Task(
        id="task-2",
        tenant_id="tenant-a",
        user_id="alice",
        goal="Review policy",
    )
    second = await provider.complete(
        task=task,
        profile=profile,
        context={
            "trajectory": [
                {
                    "tool_requests": [
                        {"call_id": "lookup", "name": "knowledge_search"}
                    ],
                    "tool_results": [
                        {"call_id": "lookup", "ok": True, "output": {"hits": []}}
                    ],
                }
            ]
        },
        tools=TOOLS,
    )
    claim = RoleResultSubmission.model_validate(
        second.tool_requests[0].arguments
    ).claims[0]
    assert claim.evidence_refs == ["demo:offline-unverified-evidence"]


@pytest.mark.asyncio
async def test_demo_decision_uses_exact_host_case_evidence_and_escalates_to_human() -> None:
    provider = DemoProvider()
    profile = enterprise_review_profiles(model="demo")[3]
    task = Task(
        id="task-decision",
        tenant_id="tenant-a",
        user_id="alice",
        workspace_id="case-123",
        goal=(
            "Case objective: host review directive\n"
            'CASE_INPUT_JSON={"case_id":"case-123","evidence_ids":["ev-1","ev-2"]}'
            "\nAssigned role: decision_synthesizer"
        ),
    )
    knowledge_step = {
        "tool_requests": [
            {"call_id": "lookup", "name": "knowledge_search"}
        ],
        "tool_results": [
            {
                "call_id": "lookup",
                "ok": True,
                "output": {
                    "hits": [
                        {"chunk_id": "chunk-a", "evidence_id": "ev-1"},
                        {"chunk_id": "chunk-b", "evidence_id": "ev-2"},
                        {"chunk_id": "unbound-policy-chunk"},
                    ]
                },
            }
        ],
    }

    turn = await provider.complete(
        task=task,
        profile=profile,
        context={"trajectory": [knowledge_step]},
        tools=TOOLS,
    )
    submission = RoleResultSubmission.model_validate(turn.tool_requests[0].arguments)
    claim = submission.claims[0]
    assert claim.fact_key == "decision.outcome"
    assert claim.value == "escalate"
    assert claim.evidence_refs == ["ev-1", "ev-2"]


@pytest.mark.asyncio
async def test_demo_rejects_foreign_case_input_binding() -> None:
    provider = DemoProvider()
    profile = enterprise_review_profiles(model="demo")[3]
    task = Task(
        id="task-foreign",
        tenant_id="tenant-a",
        user_id="alice",
        workspace_id="case-owner",
        goal='CASE_INPUT_JSON={"case_id":"case-foreign","evidence_ids":["ev-1"]}',
    )
    step = {
        "tool_requests": [{"call_id": "lookup", "name": "knowledge_search"}],
        "tool_results": [
            {"call_id": "lookup", "ok": True, "output": {"hits": []}}
        ],
    }
    turn = await provider.complete(
        task=task,
        profile=profile,
        context={"trajectory": [step]},
        tools=TOOLS,
    )
    claim = RoleResultSubmission.model_validate(turn.tool_requests[0].arguments).claims[0]
    assert claim.evidence_refs == ["demo:offline-unverified-evidence"]

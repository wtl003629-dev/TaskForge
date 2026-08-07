from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from taskforge.case_profiles import (
    ResearchSurveyDepth,
    enterprise_review_profiles,
    research_survey_profiles,
)
from taskforge.case_runtime import (
    SUBMIT_ROLE_RESULT,
    CaseAgentExecutor,
    RoleResultSubmission,
)
from taskforge.checkpoints import SQLiteCheckpointStore
from taskforge.domain import ModelTurn, ToolRequest
from taskforge.orchestration import (
    FactStatus,
    OrchestrationAccess,
    OrchestrationNotFoundError,
    PlanStatus,
    SQLiteOrchestrationStore,
)
from taskforge.providers import ScriptedProvider
from taskforge.review_cases import (
    CaseAccess,
    CaseKind,
    CaseStatus,
    CaseSubmission,
    EvidenceRef,
    RecommendationOutcome,
    ReviewCaseNotFoundError,
    SQLiteReviewCaseStore,
)
from taskforge.review_service import (
    RecommendationEvidenceError,
    ReviewCaseCoordinator,
    ReviewRunLimitError,
)
from taskforge.runtime import AgentRuntime
from taskforge.tooling import CapabilityPolicy, ToolRegistry, ToolRisk, ToolSpec


class StaticContext:
    def assemble(self, **kwargs: Any) -> dict[str, Any]:
        return {"query": kwargs["query"], "scope": "review-service-test"}


def _submission() -> CaseSubmission:
    return CaseSubmission(
        request_summary="Move the payment service to the new production cluster.",
        business_justification="The old cluster reaches end of support this quarter.",
        attributes={"change_window": "2026-08-10T02:00:00Z"},
        evidence_refs=[
            EvidenceRef(
                evidence_id="change-ticket-17",
                source_type="document",
                locator="kb://change/ticket-17",
                title="Approved change request",
                version="3",
                excerpt=(
                    "The payment migration has an approved change ticket, "
                    "documented rollback plan, and named human reviewer."
                ),
            )
        ],
    )


def _research_script(
    *,
    verdict: str = "accept",
    depth: ResearchSurveyDepth = ResearchSurveyDepth.RIGOROUS,
) -> list[ModelTurn]:
    claim_sets = {
        ResearchSurveyDepth.MINIMAL: [
            ("planner.sub_questions", ["sub-q1", "sub-q2"]),
            ("survey.verdict", verdict),
        ],
        ResearchSurveyDepth.STANDARD: [
            ("planner.sub_questions", ["sub-q1", "sub-q2"]),
            ("evaluator.source_gaps", ["no authoritative baseline"]),
            ("survey.verdict", verdict),
        ],
        ResearchSurveyDepth.RIGOROUS: [
            ("planner.sub_questions", ["sub-q1", "sub-q2"]),
            ("evaluator.source_gaps", ["no authoritative baseline"]),
            ("writer.section", "Survey section synthesizing retrieved sources."),
            ("survey.verdict", verdict),
        ],
    }
    claims = claim_sets[depth]
    turns: list[ModelTurn] = []
    for index, (key, value) in enumerate(claims, start=1):
        turns.append(
            ModelTurn(
                kind="tool",
                tool_requests=[
                    ToolRequest(
                        call_id=f"knowledge-{index}",
                        name="knowledge_search",
                        arguments={"query": "research methods", "limit": 5},
                    )
                ],
            )
        )
        turns.append(_role_turn(index, key, value))
        turns.append(
            ModelTurn(kind="final", final_answer=f"Research role {index} submitted.")
        )
    return turns


def _role_turn(
    index: int,
    fact_key: str,
    value: Any,
    *,
    evidence_ref: str = "change-ticket-17",
) -> ModelTurn:
    return ModelTurn(
        kind="tool",
        tool_requests=[
            ToolRequest(
                call_id=f"submit-{index}",
                name=SUBMIT_ROLE_RESULT,
                arguments={
                    "claims": [
                        {
                            "fact_key": fact_key,
                            "value": value,
                            "evidence_refs": [evidence_ref],
                            "confidence": 0.9 - index / 100,
                        }
                    ],
                    "summary": f"Structured review result {index}",
                    "handoff_summary": f"Bounded rationale {index}",
                },
            )
        ],
    )


def _script(
    *,
    decision_evidence: str = "change-ticket-17",
    decision_value: Any = "approve",
) -> list[ModelTurn]:
    claims = [
        ("intake.scope", "payment-service"),
        ("compliance.result", "controls-present"),
        ("risk.level", "medium"),
        ("decision.outcome", decision_value),
    ]
    turns: list[ModelTurn] = []
    for index, (key, value) in enumerate(claims, start=1):
        turns.append(
            ModelTurn(
                kind="tool",
                tool_requests=[
                    ToolRequest(
                        call_id=f"knowledge-{index}",
                        name="knowledge_search",
                        arguments={"query": "payment change ticket", "limit": 5},
                    )
                ],
            )
        )
        turns.append(
            _role_turn(
                index,
                key,
                value,
                evidence_ref=(decision_evidence if index == 4 else "change-ticket-17"),
            )
        )
        turns.append(
            ModelTurn(kind="final", final_answer=f"Role {index} submitted its receipt.")
        )
    return turns


def _coordinator(
    tmp_path: Path,
    provider: ScriptedProvider,
    *,
    tenant_id: str = "tenant-a",
    user_id: str = "user-a",
    knowledge_hits: list[dict[str, Any]] | None = None,
    stores: tuple[SQLiteReviewCaseStore, SQLiteOrchestrationStore] | None = None,
) -> tuple[ReviewCaseCoordinator, SQLiteReviewCaseStore, SQLiteOrchestrationStore]:
    if stores is None:
        case_store = SQLiteReviewCaseStore(tmp_path / "cases.db")
        orchestration_store = SQLiteOrchestrationStore(tmp_path / "orchestration.db")
    else:
        case_store, orchestration_store = stores
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="knowledge_search",
            description="Return one host-bound review evidence candidate.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query", "limit"],
                "additionalProperties": False,
            },
            risk=ToolRisk.READ,
        ),
        lambda *_: {
            "hits": (
                knowledge_hits
                if knowledge_hits is not None
                else [
                    {
                        "chunk_id": "review-evidence-chunk",
                        "evidence_id": "change-ticket-17",
                        "source": "kb://change/ticket-17",
                        "text": "Approved payment migration and rollback plan.",
                    }
                ]
            )
        },
    )
    runtime = AgentRuntime(
        provider=provider,
        registry=registry,
        policy=CapabilityPolicy(registry),
        checkpoint=SQLiteCheckpointStore(
            tmp_path / f"runtime-{tenant_id}-{user_id}.db"
        ),
        context=StaticContext(),
    )
    profiles = {
        profile.id: profile
        for profile in (
            enterprise_review_profiles(model="scripted")
            + research_survey_profiles(model="scripted")
        )
    }
    executor = CaseAgentExecutor(
        store=orchestration_store,
        runtime=runtime,
        user_id=user_id,
        profiles=profiles,
    )
    return (
        ReviewCaseCoordinator(
            case_store=case_store,
            orchestration_store=orchestration_store,
            executor=executor,
            tenant_id=tenant_id,
            user_id=user_id,
        ),
        case_store,
        orchestration_store,
    )


@pytest.mark.asyncio
async def test_four_role_runtime_dag_hands_only_untrusted_recommendation_to_human(
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider(_script())
    coordinator, _, orchestration_store = _coordinator(tmp_path, provider)
    draft = coordinator.create_draft(
        kind=CaseKind.ENTERPRISE_CHANGE,
        title="Payment cluster migration",
        submission=_submission(),
        idempotency_key="create-payment-migration",
    )
    assert draft.conversation_id == draft.case_id

    started = coordinator.submit_and_start(
        draft.case_id, idempotency_key="start-payment-migration"
    )
    assert started.review_case.status == CaseStatus.RUNNING
    assert started.plan is not None
    assert "Move the payment service" in started.plan.objective
    assert "end of support" in started.plan.objective
    assert "change-ticket-17" in started.plan.objective

    finished = await coordinator.run_until_pause_or_review(
        draft.case_id, max_iterations=4
    )

    assert finished.iterations == 4
    assert finished.plan is not None
    assert finished.plan.status == PlanStatus.COMPLETED
    assert finished.review_case.status == CaseStatus.WAITING_HUMAN_REVIEW
    assert finished.review_case.human_decision is None
    recommendation = finished.review_case.recommendation
    assert recommendation is not None
    assert recommendation.authority == "model_untrusted"
    assert recommendation.outcome == RecommendationOutcome.APPROVE
    assert [item.evidence_id for item in recommendation.evidence_refs] == [
        "change-ticket-17"
    ]
    assert len(provider.calls) == 12

    access = OrchestrationAccess(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=draft.case_id,
    )
    facts = orchestration_store.list_shared_facts(access, current_only=True)
    assert len(facts) == 4
    assert all(fact.status == FactStatus.VERIFIED for fact in facts)
    assert all(fact.authority == "tool" and fact.verifier_ref is not None for fact in facts)
    all_history = orchestration_store.list_shared_facts(access, current_only=False)
    assert len(all_history) == 8
    assert {fact.status for fact in all_history} == {
        FactStatus.PROPOSED,
        FactStatus.VERIFIED,
    }
    role_runs = orchestration_store.list_role_runs(access, finished.plan.plan_id)
    assert all(
        "change-ticket-17" in (role_run.output or {}).get("retrieved_evidence_refs", [])
        for role_run in role_runs
    )


@pytest.mark.asyncio
async def test_saga_and_completed_review_replay_without_second_model_run(
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider(_script())
    coordinator, case_store, _ = _coordinator(tmp_path, provider)
    first = coordinator.create_draft(
        kind=CaseKind.ENTERPRISE_ADMISSION,
        title="Vendor admission",
        submission=_submission(),
        idempotency_key="vendor-17",
    )
    replay = coordinator.create_draft(
        kind=CaseKind.ENTERPRISE_ADMISSION,
        title="Vendor admission",
        submission=_submission(),
        idempotency_key="vendor-17",
    )
    assert replay.case_id == first.case_id

    coordinator.submit_and_start(first.case_id, idempotency_key="submit-vendor-17")
    completed = await coordinator.run_until_pause(first.case_id, max_iterations=4)
    calls = len(provider.calls)

    saga_replay = coordinator.submit_and_start(
        first.case_id, idempotency_key="a-different-retry-token"
    )
    execution_replay = await coordinator.run_until_pause(
        first.case_id, max_iterations=4
    )
    assert saga_replay.review_case.revision == completed.review_case.revision
    assert execution_replay.review_case.revision == completed.review_case.revision
    assert len(provider.calls) == calls

    access = CaseAccess(
        tenant_id="tenant-a",
        owner_user_id="user-a",
        actor_user_id="user-a",
        conversation_id=first.case_id,
    )
    events = case_store.list_audit_events(access, first.case_id)
    assert [event.event_type.value for event in events].count(
        "model_recommendation_recorded"
    ) == 1


def test_same_tenant_wrong_user_and_wrong_conversation_cannot_read_case_or_plan(
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider([])
    coordinator, case_store, orchestration_store = _coordinator(tmp_path, provider)
    draft = coordinator.create_draft(
        kind=CaseKind.ENTERPRISE_CHANGE,
        title="Scoped change",
        submission=_submission(),
        idempotency_key="scoped-change",
    )
    started = coordinator.submit_and_start(draft.case_id, idempotency_key="start-scoped")
    assert started.plan is not None

    wrong_user, _, _ = _coordinator(
        tmp_path,
        ScriptedProvider([]),
        user_id="user-b",
        stores=(case_store, orchestration_store),
    )
    with pytest.raises(ReviewCaseNotFoundError):
        wrong_user.get_state(draft.case_id)

    with pytest.raises(ReviewCaseNotFoundError):
        case_store.get_case(
            CaseAccess(
                tenant_id="tenant-a",
                owner_user_id="user-a",
                actor_user_id="user-a",
                conversation_id="another-conversation",
            ),
            draft.case_id,
        )
    with pytest.raises(OrchestrationNotFoundError):
        orchestration_store.get_plan(
            OrchestrationAccess(
                tenant_id="tenant-a",
                user_id="user-a",
                conversation_id="another-conversation",
            ),
            started.plan.plan_id,
        )


@pytest.mark.asyncio
async def test_decision_with_zero_exact_evidence_matches_fails_closed(
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider(_script(decision_evidence="change-ticket-17#page=1"))
    coordinator, _, _ = _coordinator(tmp_path, provider)
    draft = coordinator.create_draft(
        kind=CaseKind.ENTERPRISE_CHANGE,
        title="Unbound evidence change",
        submission=_submission(),
        idempotency_key="unbound-change",
    )
    coordinator.submit_and_start(draft.case_id, idempotency_key="start-unbound")

    with pytest.raises(RecommendationEvidenceError, match="knowledge_search|exactly"):
        await coordinator.run_until_pause_or_review(draft.case_id, max_iterations=4)

    state = coordinator.get_state(draft.case_id)
    assert state.review_case.status == CaseStatus.RUNNING
    assert state.review_case.recommendation is None
    assert state.plan is not None and state.plan.status == PlanStatus.RUNNING


@pytest.mark.asyncio
async def test_submitted_evidence_id_without_a_retrieval_receipt_cannot_support_decision(
    tmp_path: Path,
) -> None:
    """A caller-provided evidence ID is not proof the decision role retrieved it."""

    provider = ScriptedProvider(_script())
    coordinator, _, _ = _coordinator(tmp_path, provider, knowledge_hits=[])
    draft = coordinator.create_draft(
        kind=CaseKind.ENTERPRISE_CHANGE,
        title="Evidence receipt is mandatory",
        submission=_submission(),
        idempotency_key="missing-retrieval-receipt",
    )
    coordinator.submit_and_start(
        draft.case_id,
        idempotency_key="start-missing-retrieval-receipt",
    )

    with pytest.raises(
        RecommendationEvidenceError,
        match="durable knowledge retrieval|successful knowledge_search",
    ):
        await coordinator.run_until_pause_or_review(
            draft.case_id,
            max_iterations=4,
        )

    state = coordinator.get_state(draft.case_id)
    assert state.review_case.status == CaseStatus.RUNNING
    assert state.review_case.recommendation is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision_value", "expected"),
    [
        ("reject", RecommendationOutcome.REJECT),
        ("needs_revision", RecommendationOutcome.ESCALATE),
        ("APPROVE", RecommendationOutcome.ESCALATE),
    ],
)
async def test_outcome_mapping_is_exact_and_unknown_values_escalate(
    tmp_path: Path,
    decision_value: str,
    expected: RecommendationOutcome,
) -> None:
    provider = ScriptedProvider(
        _script(
            decision_value=decision_value,
            decision_evidence="kb://change/ticket-17",
        )
    )
    coordinator, _, _ = _coordinator(tmp_path, provider)
    draft = coordinator.create_draft(
        kind=CaseKind.ENTERPRISE_CHANGE,
        title=f"Outcome mapping {decision_value}",
        submission=_submission(),
        idempotency_key=f"outcome-{decision_value}",
    )
    coordinator.submit_and_start(draft.case_id, idempotency_key="start-outcome")

    finished = await coordinator.run_until_pause_or_review(
        draft.case_id, max_iterations=4
    )
    assert finished.review_case.recommendation is not None
    assert finished.review_case.recommendation.outcome == expected
    assert finished.review_case.recommendation.evidence_refs[0].locator == (
        "kb://change/ticket-17"
    )


@pytest.mark.asyncio
async def test_run_loop_enforces_callers_iteration_ceiling(tmp_path: Path) -> None:
    provider = ScriptedProvider(_script())
    coordinator, _, _ = _coordinator(tmp_path, provider)
    draft = coordinator.create_draft(
        kind=CaseKind.ENTERPRISE_CHANGE,
        title="Bounded review",
        submission=_submission(),
        idempotency_key="bounded-review",
    )
    coordinator.submit_and_start(draft.case_id, idempotency_key="start-bounded")

    with pytest.raises(ReviewRunLimitError):
        await coordinator.run_until_pause_or_review(draft.case_id, max_iterations=3)
    assert len(provider.calls) == 9


@pytest.mark.asyncio
async def test_required_role_attempt_exhaustion_fails_plan_and_case(
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider(
        [
            ModelTurn(kind="final", final_answer="Missing structured receipt one."),
            ModelTurn(kind="final", final_answer="Missing structured receipt two."),
        ]
    )
    coordinator, case_store, orchestration_store = _coordinator(tmp_path, provider)
    draft = coordinator.create_draft(
        kind=CaseKind.ENTERPRISE_CHANGE,
        title="Exhausted intake review",
        submission=_submission(),
        idempotency_key="exhausted-intake",
    )
    coordinator.submit_and_start(draft.case_id, idempotency_key="start-exhausted")

    failed = await coordinator.run_until_pause_or_review(
        draft.case_id,
        max_iterations=2,
    )

    assert failed.review_case.status == CaseStatus.FAILED
    assert failed.review_case.failure is not None
    assert "intake" in failed.review_case.failure.reason
    assert failed.plan is not None and failed.plan.status == PlanStatus.FAILED
    runs = orchestration_store.list_role_runs(
        OrchestrationAccess(
            tenant_id="tenant-a",
            user_id="user-a",
            conversation_id=draft.case_id,
        ),
        failed.plan.plan_id,
    )
    assert [(run.attempt, run.status.value) for run in runs] == [
        (1, "failed"),
        (2, "failed"),
    ]
    events = case_store.list_audit_events(
        CaseAccess(
            tenant_id="tenant-a",
            owner_user_id="user-a",
            conversation_id=draft.case_id,
        ),
        draft.case_id,
    )
    assert events[-1].event_type.value == "case_failed"

    replay = coordinator.get_state(draft.case_id)
    assert replay.review_case.status == CaseStatus.FAILED
    assert replay.plan is not None and replay.plan.status == PlanStatus.FAILED
    assert len(provider.calls) == 2


def test_decision_citation_may_reference_host_created_shared_fact() -> None:
    output = {
        "retrieved_evidence_refs": ["change-ticket-17", "case://change-ticket-17"],
        "role_result": {
            "claims": [
                {
                    "fact_key": "decision.outcome",
                    "value": "approve",
                    "evidence_refs": ["change-ticket-17", "fact:abc-123"],
                    "confidence": 0.9,
                }
            ],
            "summary": "Approval recommended.",
            "handoff_summary": "Approval rationale.",
        },
    }
    role_result = RoleResultSubmission.model_validate(output["role_result"])
    ReviewCaseCoordinator._require_retrieved_recommendation_evidence(
        output, role_result, frozenset({"abc-123"})
    )
    bound = ReviewCaseCoordinator._bind_recommendation_evidence(
        [_submission().evidence_refs[0]], role_result
    )
    assert [item.evidence_id for item in bound] == ["change-ticket-17"]


def test_decision_citation_of_unretrieved_submitted_evidence_still_fails() -> None:
    output = {
        "retrieved_evidence_refs": ["case://change-ticket-17"],
        "role_result": {
            "claims": [
                {
                    "fact_key": "decision.outcome",
                    "value": "approve",
                    "evidence_refs": ["change-ticket-17", "case://change-ticket-17"],
                    "confidence": 0.9,
                }
            ],
            "summary": "Approval recommended.",
            "handoff_summary": "Approval rationale.",
        },
    }
    role_result = RoleResultSubmission.model_validate(output["role_result"])
    with pytest.raises(RecommendationEvidenceError, match="knowledge_search"):
        ReviewCaseCoordinator._require_retrieved_recommendation_evidence(
            output, role_result, frozenset()
        )


@pytest.mark.asyncio
async def test_research_survey_reaches_human_review_with_verified_verdict(
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider(_research_script(verdict="accept"))
    coordinator, _, _ = _coordinator(tmp_path, provider)
    draft = coordinator.create_draft(
        kind=CaseKind.RESEARCH_SURVEY,
        title="RAG evaluation methods survey",
        submission=_submission(),
        idempotency_key="research-survey-1",
    )
    coordinator.submit_and_start(draft.case_id, idempotency_key="start-survey-1")

    finished = await coordinator.run_until_pause_or_review(
        draft.case_id, max_iterations=4
    )

    assert finished.review_case.status == CaseStatus.WAITING_HUMAN_REVIEW
    assert finished.plan is not None
    assert [slot.role_id for slot in finished.plan.slots] == [
        "retrieval_planner",
        "source_evaluator",
        "synthesis_writer",
        "critical_reviewer",
    ]
    recommendation = finished.review_case.recommendation
    assert recommendation is not None
    assert recommendation.outcome == RecommendationOutcome.APPROVE
    assert recommendation.evidence_refs
    assert all(item.evidence_id == "change-ticket-17" or item.evidence_id == "kb://change/ticket-17"
               for item in recommendation.evidence_refs)


@pytest.mark.parametrize(
    ("depth", "expected_slots"),
    [
        (ResearchSurveyDepth.MINIMAL, ["retrieval_planner", "synthesis_writer"]),
        (
            ResearchSurveyDepth.STANDARD,
            ["retrieval_planner", "source_evaluator", "synthesis_writer"],
        ),
    ],
)
@pytest.mark.asyncio
async def test_research_survey_shallow_depth_reaches_human_review_with_writer_verdict(
    tmp_path: Path,
    depth: ResearchSurveyDepth,
    expected_slots: list[str],
) -> None:
    provider = ScriptedProvider(_research_script(verdict="accept", depth=depth))
    coordinator, _, _ = _coordinator(tmp_path, provider)
    draft = coordinator.create_draft(
        kind=CaseKind.RESEARCH_SURVEY,
        title="Shallow survey",
        submission=_submission(),
        idempotency_key=f"research-survey-{depth.value}",
        survey_depth=depth,
    )
    coordinator.submit_and_start(draft.case_id, idempotency_key=f"start-{depth.value}")

    finished = await coordinator.run_until_pause_or_review(
        draft.case_id, max_iterations=4
    )

    assert finished.review_case.status == CaseStatus.WAITING_HUMAN_REVIEW
    assert finished.plan is not None
    assert [slot.role_id for slot in finished.plan.slots] == expected_slots
    recommendation = finished.review_case.recommendation
    assert recommendation is not None
    assert recommendation.outcome == RecommendationOutcome.APPROVE
    assert recommendation.evidence_refs

from __future__ import annotations

from pathlib import Path

import pytest

from taskforge.case_runtime import SUBMIT_ROLE_RESULT
from taskforge.domain import ModelTurn, ToolRequest
from taskforge.providers import ScriptedProvider
from taskforge.review_ab import (
    ArmExecution,
    build_report,
    load_review_benchmark,
    multi_business_e2e_passed,
    run_multi_arm,
    run_single_arm,
    score_execution,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "eval" / "review_ab_cases.json"


def test_review_ab_dataset_is_balanced_and_gold_evidence_exists() -> None:
    benchmark = load_review_benchmark(DATASET)

    assert len(benchmark.cases) == 20
    assert {category: sum(case.category == category for case in benchmark.cases) for category in {
        "approve", "reject", "escalate", "adversarial"
    }} == {"approve": 5, "reject": 5, "escalate": 5, "adversarial": 5}
    for case in benchmark.cases:
        available = {document.evidence_id for document in case.documents}
        assert set(case.gold.required_evidence_ids) <= available


def test_score_requires_outcome_findings_evidence_and_no_forbidden_claims() -> None:
    case = load_review_benchmark(DATASET).cases[0]
    passing = ArmExecution(
        arm="single",
        case_id=case.id,
        status="completed",
        outcome="approve",
        text=(
            "The change was approved by CAB. The test gate passed. Rollback was "
            "rehearsed. The monitoring plan defines cutover steps and success "
            "thresholds; current policy permits rollout because controls are satisfied."
        ),
        cited_evidence_ids=[
            "chg-001", "test-001", "rollback-001", "implementation-001", "policy-001"
        ],
        duration_ms=1,
    )
    score = score_execution(case, passing)
    assert score.quality_passed is True

    failing = passing.model_copy(
        update={
            "outcome": "reject",
            "text": passing.text + " No rollback plan.",
            "cited_evidence_ids": ["chg-001"],
        }
    )
    score = score_execution(case, failing)
    assert score.outcome_correct is False
    assert score.evidence_recall < 1
    assert score.forbidden_claim_count == 1
    assert score.quality_passed is False


def _single_script() -> list[ModelTurn]:
    return [
        ModelTurn(
            kind="tool",
            tool_requests=[
                ToolRequest(
                    call_id="single-search",
                    name="knowledge_search",
                    arguments={"query": "approval tests rollback monitoring policy", "limit": 5},
                )
            ],
        ),
        ModelTurn(
            kind="tool",
            tool_requests=[
                ToolRequest(
                    call_id="single-submit",
                    name="submit_review_result",
                    arguments={
                        "outcome": "approve",
                        "findings": [
                            {
                                "code": "approval_present",
                                "summary": "CAB approved the change.",
                                "evidence_ids": ["chg-001"],
                            },
                            {
                                "code": "tests_passed",
                                "summary": "Tests passed.",
                                "evidence_ids": ["test-001"],
                            },
                            {
                                "code": "rollback_rehearsed",
                                "summary": "Rollback was rehearsed.",
                                "evidence_ids": ["rollback-001"],
                            },
                            {
                                "code": "implementation_and_monitoring_ready",
                                "summary": "The monitoring plan defines cutover steps and success thresholds.",
                                "evidence_ids": ["implementation-001"],
                            },
                            {
                                "code": "current_policy_satisfied",
                                "summary": "Current policy permits rollout because controls are satisfied.",
                                "evidence_ids": ["policy-001"],
                            },
                        ],
                        "summary": "All required controls are evidenced.",
                        "rationale": "Approve as a non-authoritative recommendation.",
                        "confidence": 0.9,
                    },
                )
            ],
        ),
        ModelTurn(kind="final", final_answer="Submitted for human review."),
    ]


def _multi_script() -> list[ModelTurn]:
    role_claims = [
        ("intake.scope", "CAB approved the bounded payment API change.", ["chg-001"]),
        ("compliance.result", "Tests passed; current policy permits rollout because controls are satisfied.", ["test-001"]),
        ("risk.rollback", "Rollback was rehearsed; the monitoring plan defines cutover steps and success thresholds.", ["rollback-001", "implementation-001"]),
        ("decision.outcome", "approve", ["policy-001"]),
    ]
    turns: list[ModelTurn] = []
    for index, (fact_key, value, evidence_ids) in enumerate(role_claims, start=1):
        turns.append(
            ModelTurn(
                kind="tool",
                tool_requests=[
                    ToolRequest(
                        call_id=f"search-{index}",
                        name="knowledge_search",
                        arguments={"query": "approval tests rollback monitoring policy", "limit": 5},
                    )
                ],
            )
        )
        turns.append(
            ModelTurn(
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
                                    "evidence_refs": evidence_ids,
                                    "confidence": 0.9,
                                }
                            ],
                            "summary": str(value),
                            "handoff_summary": str(value),
                        },
                    )
                ],
            )
        )
        turns.append(ModelTurn(kind="final", final_answer="Role submitted."))
    return turns


@pytest.mark.asyncio
async def test_both_arms_execute_same_case_through_real_runtime_contracts(
    tmp_path: Path,
) -> None:
    case = load_review_benchmark(DATASET).cases[0]
    single = await run_single_arm(
        case,
        provider=ScriptedProvider(_single_script()),
        model="scripted",
        workdir=tmp_path / "single",
    )
    multi = await run_multi_arm(
        case,
        provider=ScriptedProvider(_multi_script()),
        model="scripted",
        workdir=tmp_path / "multi",
    )

    assert single.status == "completed"
    assert single.outcome == "approve"
    assert multi.status == "waiting_human_review"
    assert multi.outcome == "approve"
    assert multi.details["succeeded_role_count"] == 4
    assert multi_business_e2e_passed(multi) is True

    report = build_report(
        provider="scripted",
        model="scripted",
        dataset=str(DATASET),
        cases=[case],
        executions={case.id: {"single": single, "multi": multi}},
    )
    assert report.summary["arms"]["single"]["outcome_accuracy"] == 1
    assert report.summary["arms"]["multi"]["outcome_accuracy"] == 1
    assert report.cases[0].scores["single"].quality_passed is True
    assert report.cases[0].scores["multi"].quality_passed is True


def test_business_e2e_allows_a_failed_retry_before_four_successes() -> None:
    execution = ArmExecution(
        arm="multi",
        case_id="retry-case",
        status="waiting_human_review",
        outcome="escalate",
        safety_violation_count=0,
        duration_ms=1,
        details={
            "role_run_count": 5,
            "succeeded_role_count": 4,
            "human_review_reached": True,
        },
    )
    assert multi_business_e2e_passed(execution) is True

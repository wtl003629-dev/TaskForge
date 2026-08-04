from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier

import pytest
from pydantic import ValidationError

from taskforge.orchestration import (
    ExecutionClaimUnavailableError,
    FactRuleError,
    FactStatus,
    IdempotencyConflictError,
    InvalidTransitionError,
    OrchestrationAccess,
    OrchestrationNotFoundError,
    PlanStatus,
    RoleNotAllowedError,
    RoleRun,
    RoleRunExecutionClaim,
    RoleRunStatus,
    SlotNotReadyError,
    SpeakerSlot,
    SQLiteOrchestrationStore,
    VersionConflictError,
)

NOW = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)
ACCESS = OrchestrationAccess(
    tenant_id="tenant-a",
    user_id="user-a",
    conversation_id="conversation-a",
)
OTHER_TENANT_ACCESS = OrchestrationAccess(
    tenant_id="tenant-b",
    user_id="user-a",
    conversation_id="conversation-a",
)
OTHER_USER_ACCESS = OrchestrationAccess(
    tenant_id="tenant-a",
    user_id="user-b",
    conversation_id="conversation-a",
)
OTHER_CONVERSATION_ACCESS = OrchestrationAccess(
    tenant_id="tenant-a",
    user_id="user-a",
    conversation_id="conversation-b",
)


def slots() -> list[SpeakerSlot]:
    return [
        SpeakerSlot(
            slot_id="research",
            role_id="researcher",
            agent_profile_id="profile-research",
            instruction="Collect cited evidence.",
            order=1,
        ),
        SpeakerSlot(
            slot_id="critique",
            role_id="critic",
            agent_profile_id="profile-critic",
            instruction="Challenge unsupported claims.",
            depends_on=["research"],
            order=2,
        ),
        SpeakerSlot(
            slot_id="edit",
            role_id="editor",
            agent_profile_id="profile-editor",
            instruction="Produce the final report.",
            depends_on=["critique"],
            order=3,
        ),
    ]


def create_plan(
    store: SQLiteOrchestrationStore,
    *,
    key: str = "client-key",
    objective: str = "Prepare an evidence-backed report",
):
    return store.create_plan(
        ACCESS,
        objective=objective,
        strategy="model_router",
        allowed_role_ids=["researcher", "critic", "editor"],
        slots=slots(),
        client_idempotency_key=key,
        now=NOW,
    )


def succeed_run(
    store: SQLiteOrchestrationStore,
    run: RoleRun,
    *,
    output: dict | None = None,
) -> RoleRun:
    running = store.transition_role_run(
        ACCESS,
        run.role_run_id,
        expected_version=run.version,
        status=RoleRunStatus.RUNNING,
        now=NOW,
    )
    return store.transition_role_run(
        ACCESS,
        running.role_run_id,
        expected_version=running.version,
        status=RoleRunStatus.SUCCEEDED,
        output=output or {"artifact_id": f"artifact-{run.slot_id}"},
        now=NOW,
    )


def test_plan_creation_is_client_idempotent_and_request_hash_bound(tmp_path: Path) -> None:
    store = SQLiteOrchestrationStore(tmp_path / "orchestration.db")
    first = create_plan(store)
    replay = store.create_plan(
        ACCESS,
        objective="Prepare an evidence-backed report",
        strategy="model_router",
        allowed_role_ids=["editor", "critic", "researcher"],
        slots=list(reversed(slots())),
        client_idempotency_key="client-key",
        now=NOW + timedelta(seconds=1),
    )

    assert replay.plan_id == first.plan_id
    assert replay.request_hash == first.request_hash
    assert replay.created_at == first.created_at
    assert first.owner_user_id == ACCESS.user_id
    with pytest.raises(IdempotencyConflictError):
        create_plan(store, objective="A different request")

    # Idempotency keys are owner scoped rather than globally shared.
    other = store.create_plan(
        OTHER_USER_ACCESS,
        objective="A different request",
        allowed_role_ids=["researcher"],
        slots=[slots()[0]],
        client_idempotency_key="client-key",
        now=NOW,
    )
    assert other.tenant_id == first.tenant_id
    assert other.owner_user_id == OTHER_USER_ACCESS.user_id
    assert other.plan_id != first.plan_id


def test_concurrent_plan_and_slot_materialization_are_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "orchestration.db"
    first_store = SQLiteOrchestrationStore(path)
    second_store = SQLiteOrchestrationStore(path)
    plan_barrier = Barrier(2)

    def create_concurrently(store: SQLiteOrchestrationStore):
        plan_barrier.wait(timeout=5)
        return create_plan(store, key="concurrent-plan")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(create_concurrently, first_store),
            pool.submit(create_concurrently, second_store),
        ]
        plans = [future.result(timeout=5) for future in futures]
    assert plans[0].plan_id == plans[1].plan_id

    run_barrier = Barrier(2)

    def materialize(store: SQLiteOrchestrationStore):
        run_barrier.wait(timeout=5)
        return store.create_role_run(
            ACCESS,
            plans[0].plan_id,
            "research",
            expected_plan_version=1,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(materialize, first_store),
            pool.submit(materialize, second_store),
        ]
        runs = [future.result(timeout=5) for future in futures]
    assert runs[0].role_run_id == runs[1].role_run_id
    assert len(first_store.list_role_runs(ACCESS, plans[0].plan_id)) == 1


def test_concurrent_role_run_execution_claim_has_one_owner(tmp_path: Path) -> None:
    path = tmp_path / "orchestration.db"
    first_store = SQLiteOrchestrationStore(path)
    second_store = SQLiteOrchestrationStore(path)
    plan = create_plan(first_store, key="execution-claim")
    run = first_store.create_role_run(
        ACCESS,
        plan.plan_id,
        "research",
        expected_plan_version=plan.version,
        now=NOW,
    )
    claim_barrier = Barrier(2)

    def claim(store: SQLiteOrchestrationStore, token: str):
        claim_barrier.wait(timeout=5)
        try:
            return store.claim_role_run_execution(
                ACCESS,
                run.role_run_id,
                claim_token=token,
                lease_seconds=30,
                now=NOW,
            )
        except ExecutionClaimUnavailableError as exc:
            return exc

    tokens = ("executor-token-a", "executor-token-b")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(claim, first_store, tokens[0]),
            pool.submit(claim, second_store, tokens[1]),
        ]
        outcomes = [future.result(timeout=5) for future in futures]

    winners = [item for item in outcomes if isinstance(item, RoleRunExecutionClaim)]
    losers = [
        item for item in outcomes if isinstance(item, ExecutionClaimUnavailableError)
    ]
    assert len(winners) == 1
    assert len(losers) == 1
    assert winners[0].claim_token in tokens
    assert winners[0].role_run_id == run.role_run_id


def test_active_execution_claim_fences_transitions_and_terminal_clears_it(
    tmp_path: Path,
) -> None:
    store = SQLiteOrchestrationStore(tmp_path / "orchestration.db")
    plan = create_plan(store, key="transition-fence")
    run = store.create_role_run(
        ACCESS,
        plan.plan_id,
        "research",
        expected_plan_version=plan.version,
        now=NOW,
    )
    claim_token = "executor-token-a"
    store.claim_role_run_execution(
        ACCESS,
        run.role_run_id,
        claim_token=claim_token,
        lease_seconds=30,
        now=NOW,
    )

    with pytest.raises(ExecutionClaimUnavailableError, match="active execution claim"):
        store.transition_role_run(
            ACCESS,
            run.role_run_id,
            expected_version=run.version,
            status=RoleRunStatus.RUNNING,
            now=NOW,
        )
    with pytest.raises(ExecutionClaimUnavailableError, match="active execution claim"):
        store.transition_role_run(
            ACCESS,
            run.role_run_id,
            expected_version=run.version,
            status=RoleRunStatus.RUNNING,
            execution_claim_token="executor-token-b",
            now=NOW,
        )

    running = store.transition_role_run(
        ACCESS,
        run.role_run_id,
        expected_version=run.version,
        status=RoleRunStatus.RUNNING,
        execution_claim_token=claim_token,
        now=NOW,
    )
    succeeded = store.transition_role_run(
        ACCESS,
        run.role_run_id,
        expected_version=running.version,
        status=RoleRunStatus.SUCCEEDED,
        output={"artifact_id": "artifact-research"},
        execution_claim_token=claim_token,
        now=NOW + timedelta(seconds=1),
    )

    assert succeeded.status == RoleRunStatus.SUCCEEDED
    assert store.release_role_run_execution(
        ACCESS,
        run.role_run_id,
        claim_token=claim_token,
    ) is False


def test_execution_claim_renewal_and_expiry_takeover_use_token_cas(
    tmp_path: Path,
) -> None:
    store = SQLiteOrchestrationStore(tmp_path / "orchestration.db")
    plan = create_plan(store, key="claim-takeover")
    run = store.create_role_run(
        ACCESS,
        plan.plan_id,
        "research",
        expected_plan_version=plan.version,
        now=NOW,
    )
    first_token = "executor-token-a"
    second_token = "executor-token-b"
    first = store.claim_role_run_execution(
        ACCESS,
        run.role_run_id,
        claim_token=first_token,
        lease_seconds=15,
        now=NOW,
    )
    renewed = store.renew_role_run_execution(
        ACCESS,
        run.role_run_id,
        claim_token=first_token,
        lease_seconds=30,
        now=NOW + timedelta(seconds=10),
    )

    assert renewed.acquired_at == first.acquired_at
    assert renewed.expires_at == NOW + timedelta(seconds=40)
    with pytest.raises(ExecutionClaimUnavailableError, match="already claimed"):
        store.claim_role_run_execution(
            ACCESS,
            run.role_run_id,
            claim_token=second_token,
            lease_seconds=30,
            now=NOW + timedelta(seconds=39),
        )

    takeover_at = NOW + timedelta(seconds=40)
    takeover = store.claim_role_run_execution(
        ACCESS,
        run.role_run_id,
        claim_token=second_token,
        lease_seconds=30,
        now=takeover_at,
    )
    assert takeover.claim_token == second_token
    assert takeover.acquired_at == takeover_at
    assert takeover.expires_at == takeover_at + timedelta(seconds=30)

    with pytest.raises(ExecutionClaimUnavailableError, match="active execution claim"):
        store.transition_role_run(
            ACCESS,
            run.role_run_id,
            expected_version=run.version,
            status=RoleRunStatus.RUNNING,
            execution_claim_token=first_token,
            now=takeover_at,
        )
    with pytest.raises(ExecutionClaimUnavailableError, match="cannot be renewed"):
        store.renew_role_run_execution(
            ACCESS,
            run.role_run_id,
            claim_token=first_token,
            lease_seconds=30,
            now=takeover_at,
        )
    running = store.transition_role_run(
        ACCESS,
        run.role_run_id,
        expected_version=run.version,
        status=RoleRunStatus.RUNNING,
        execution_claim_token=second_token,
        now=takeover_at,
    )
    assert running.status == RoleRunStatus.RUNNING


def test_plan_validates_allowlist_dag_and_uses_version_cas(tmp_path: Path) -> None:
    store = SQLiteOrchestrationStore(tmp_path / "orchestration.db")
    with pytest.raises(ValidationError, match="acyclic"):
        store.create_plan(
            ACCESS,
            objective="cycle",
            allowed_role_ids=["one", "two"],
            slots=[
                SpeakerSlot(
                    slot_id="one",
                    role_id="one",
                    agent_profile_id="p1",
                    instruction="one",
                    depends_on=["two"],
                ),
                SpeakerSlot(
                    slot_id="two",
                    role_id="two",
                    agent_profile_id="p2",
                    instruction="two",
                    depends_on=["one"],
                ),
            ],
            client_idempotency_key="cycle",
        )
    with pytest.raises(ValidationError, match="not allowed"):
        store.create_plan(
            ACCESS,
            objective="role escape",
            allowed_role_ids=["researcher"],
            slots=[slots()[1]],
            client_idempotency_key="role-escape",
        )

    plan = create_plan(store)
    running = store.transition_plan(
        ACCESS,
        plan.plan_id,
        expected_version=1,
        status=PlanStatus.RUNNING,
        now=NOW,
    )
    assert running.version == 2
    with pytest.raises(VersionConflictError):
        store.transition_plan(
            ACCESS,
            plan.plan_id,
            expected_version=1,
            status=PlanStatus.COMPLETED,
        )
    with pytest.raises(InvalidTransitionError):
        store.transition_plan(
            ACCESS,
            plan.plan_id,
            expected_version=2,
            status=PlanStatus.READY,
        )
    with pytest.raises(OrchestrationNotFoundError):
        store.get_plan(OTHER_TENANT_ACCESS, plan.plan_id)


def test_fixed_dag_readiness_model_role_guard_and_bounded_retries(tmp_path: Path) -> None:
    store = SQLiteOrchestrationStore(tmp_path / "orchestration.db")
    plan = create_plan(store)

    assert [slot.slot_id for slot in store.next_ready_slots(ACCESS, plan.plan_id)] == [
        "research"
    ]
    proposed = store.validate_model_role_proposal(
        ACCESS,
        plan.plan_id,
        "researcher",
        expected_plan_version=1,
    )
    assert proposed.slot_id == "research"
    with pytest.raises(RoleNotAllowedError):
        store.validate_model_role_proposal(ACCESS, plan.plan_id, "administrator")
    with pytest.raises(SlotNotReadyError):
        store.validate_model_role_proposal(ACCESS, plan.plan_id, "critic")

    research = store.create_role_run(
        ACCESS,
        plan.plan_id,
        "research",
        expected_plan_version=1,
        now=NOW,
    )
    replay = store.create_role_run(
        ACCESS,
        plan.plan_id,
        "research",
        expected_plan_version=1,
        now=NOW,
    )
    assert replay.role_run_id == research.role_run_id
    assert store.next_ready_slots(ACCESS, plan.plan_id) == []
    research = succeed_run(store, research)
    assert research.status == RoleRunStatus.SUCCEEDED
    assert [slot.slot_id for slot in store.next_ready_slots(ACCESS, plan.plan_id)] == [
        "critique"
    ]

    critique = store.create_role_run(
        ACCESS,
        plan.plan_id,
        "critique",
        expected_plan_version=1,
    )
    critique = store.transition_role_run(
        ACCESS,
        critique.role_run_id,
        expected_version=1,
        status=RoleRunStatus.RUNNING,
    )
    critique = store.transition_role_run(
        ACCESS,
        critique.role_run_id,
        expected_version=2,
        status=RoleRunStatus.FAILED,
        error="provider unavailable",
    )
    assert critique.status == RoleRunStatus.FAILED
    assert [slot.slot_id for slot in store.next_ready_slots(ACCESS, plan.plan_id)] == [
        "critique"
    ]
    retry = store.create_role_run(
        ACCESS,
        plan.plan_id,
        "critique",
        expected_plan_version=1,
    )
    assert retry.attempt == 2
    retry = store.transition_role_run(
        ACCESS,
        retry.role_run_id,
        expected_version=1,
        status=RoleRunStatus.FAILED,
        error="retry exhausted",
    )
    assert retry.status == RoleRunStatus.FAILED
    assert store.next_ready_slots(ACCESS, plan.plan_id) == []


def test_shared_facts_are_proposed_then_versioned_verified_truth(tmp_path: Path) -> None:
    store = SQLiteOrchestrationStore(tmp_path / "orchestration.db")
    plan = create_plan(store)
    research = store.create_role_run(
        ACCESS, plan.plan_id, "research", expected_plan_version=1
    )
    with pytest.raises(FactRuleError, match="succeeded"):
        store.propose_shared_fact(
            ACCESS,
            "report.title",
            "Draft",
            source_role_run_id=research.role_run_id,
        )
    research = succeed_run(store, research)

    proposed = store.propose_shared_fact(
        ACCESS,
        "report.title",
        "Evidence report",
        source_role_run_id=research.role_run_id,
        now=NOW,
    )
    assert proposed.status == FactStatus.PROPOSED
    assert proposed.authority == "model" and proposed.version == 1
    with pytest.raises(FactRuleError, match="models cannot"):
        store.verify_shared_fact(
            ACCESS,
            "report.title",
            expected_version=1,
            verifier="model",  # type: ignore[arg-type]
            verifier_ref="model-output",
        )
    receipt = store.record_host_verification_receipt(
        ACCESS,
        "report.title",
        "Evidence report",
        authority="tool",
        receipt_id="receipt-1",
        evidence_ref="tool-run:receipt-1",
        now=NOW,
    )
    replayed_receipt = store.record_host_verification_receipt(
        ACCESS,
        "report.title",
        "Evidence report",
        authority="tool",
        receipt_id="receipt-1",
        evidence_ref="tool-run:receipt-1",
        now=NOW + timedelta(seconds=5),
    )
    assert replayed_receipt == receipt
    with pytest.raises(IdempotencyConflictError, match="another payload"):
        store.record_host_verification_receipt(
            ACCESS,
            "report.title",
            "Evidence report",
            authority="tool",
            receipt_id="receipt-1",
            evidence_ref="tool-run:different",
            now=NOW + timedelta(seconds=5),
        )
    verified = store.verify_shared_fact(
        ACCESS,
        "report.title",
        expected_version=1,
        verifier="tool",
        verifier_ref="receipt-1",
        now=NOW,
    )
    assert verified.status == FactStatus.VERIFIED
    assert verified.version == 2
    assert verified.supersedes_fact_id == proposed.fact_id

    with pytest.raises(VersionConflictError):
        store.propose_shared_fact(
            ACCESS,
            "report.title",
            "Stale update",
            source_role_run_id=research.role_run_id,
            expected_version=1,
        )
    update = store.propose_shared_fact(
        ACCESS,
        "report.title",
        "Revised evidence report",
        source_role_run_id=research.role_run_id,
        expected_version=2,
    )
    assert update.version == 3 and update.status == FactStatus.PROPOSED
    # A newer model proposal does not erase the last verified authority.
    authoritative = store.list_shared_facts(ACCESS, verified_only=True)
    assert [(fact.value, fact.version) for fact in authoritative] == [
        ("Evidence report", 2)
    ]
    latest = store.list_shared_facts(ACCESS)
    assert latest[0].version == 3 and latest[0].status == FactStatus.PROPOSED


def test_handoff_is_fixed_dependency_and_references_only_verified_facts(
    tmp_path: Path,
) -> None:
    store = SQLiteOrchestrationStore(tmp_path / "orchestration.db")
    plan = create_plan(store)
    research = succeed_run(
        store,
        store.create_role_run(
            ACCESS, plan.plan_id, "research", expected_plan_version=1
        ),
    )
    proposed = store.propose_shared_fact(
        ACCESS,
        "claim.one",
        {"claim": "verified evidence exists"},
        source_role_run_id=research.role_run_id,
    )
    with pytest.raises(FactRuleError, match="verified"):
        store.create_handoff(
            ACCESS,
            plan.plan_id,
            from_role_run_id=research.role_run_id,
            to_slot_id="critique",
            summary="Review the evidence.",
            shared_fact_ids=[proposed.fact_id],
        )
    store.record_host_verification_receipt(
        ACCESS,
        "claim.one",
        {"claim": "verified evidence exists"},
        authority="user",
        receipt_id="user-confirmation-1",
        evidence_ref="approval:user-confirmation-1",
    )
    verified = store.verify_shared_fact(
        ACCESS,
        "claim.one",
        expected_version=1,
        verifier="user",
        verifier_ref="user-confirmation-1",
    )
    handoff = store.create_handoff(
        ACCESS,
        plan.plan_id,
        from_role_run_id=research.role_run_id,
        to_slot_id="critique",
        summary="Review the evidence.",
        shared_fact_ids=[verified.fact_id],
    )
    replay = store.create_handoff(
        ACCESS,
        plan.plan_id,
        from_role_run_id=research.role_run_id,
        to_slot_id="critique",
        summary="Review the evidence.",
        shared_fact_ids=[verified.fact_id],
    )
    assert replay.handoff_id == handoff.handoff_id
    with pytest.raises(IdempotencyConflictError):
        store.create_handoff(
            ACCESS,
            plan.plan_id,
            from_role_run_id=research.role_run_id,
            to_slot_id="critique",
            summary="Changed replay payload.",
            shared_fact_ids=[verified.fact_id],
        )
    with pytest.raises(SlotNotReadyError, match="depend"):
        store.create_handoff(
            ACCESS,
            plan.plan_id,
            from_role_run_id=research.role_run_id,
            to_slot_id="edit",
            summary="Skip the critic.",
        )


def test_private_memory_is_tenant_conversation_and_role_isolated_and_durable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "orchestration.db"
    store = SQLiteOrchestrationStore(path)
    plan = create_plan(store)
    research = succeed_run(
        store,
        store.create_role_run(
            ACCESS, plan.plan_id, "research", expected_plan_version=1
        ),
    )
    memory = store.remember_private(
        ACCESS,
        "researcher",
        "The user prefers primary sources.",
        kind="preference",
        provenance_role_run_id=research.role_run_id,
        extractor_version="extractor-v2",
        expires_at=NOW + timedelta(days=30),
        now=NOW,
    )
    replay = store.remember_private(
        ACCESS,
        "researcher",
        "The user prefers primary sources.",
        kind="preference",
        provenance_role_run_id=research.role_run_id,
        extractor_version="extractor-v2",
        expires_at=NOW + timedelta(days=30),
        now=NOW,
    )
    assert replay.memory_id == memory.memory_id
    assert store.list_private_memories(
        ACCESS, "researcher", now=NOW
    ) == [memory]
    assert store.list_private_memories(
        ACCESS, "critic", now=NOW
    ) == []
    assert store.list_private_memories(
        OTHER_TENANT_ACCESS, "researcher", now=NOW
    ) == []
    with pytest.raises(OrchestrationNotFoundError):
        store.get_private_memory(
            ACCESS,
            "critic",
            memory.memory_id,
            now=NOW,
        )
    with pytest.raises(RoleNotAllowedError, match="provenance"):
        store.remember_private(
            ACCESS,
            "critic",
            "Attempt to copy another role's memory.",
            provenance_role_run_id=research.role_run_id,
        )
    assert store.list_private_memories(
        ACCESS,
        "researcher",
        now=NOW + timedelta(days=31),
    ) == []

    reopened = SQLiteOrchestrationStore(path)
    assert reopened.get_private_memory(
        ACCESS,
        "researcher",
        memory.memory_id,
        now=NOW,
    ).content == memory.content
    with pytest.raises(OrchestrationNotFoundError):
        reopened.get_role_run(OTHER_TENANT_ACCESS, research.role_run_id)


def test_plan_completion_requires_required_dag_and_terminal_plan_fences_runs(
    tmp_path: Path,
) -> None:
    store = SQLiteOrchestrationStore(tmp_path / "orchestration.db")
    plan = create_plan(store)
    running_plan = store.transition_plan(
        ACCESS,
        plan.plan_id,
        expected_version=1,
        status=PlanStatus.RUNNING,
    )
    with pytest.raises(InvalidTransitionError, match="required slots"):
        store.transition_plan(
            ACCESS,
            plan.plan_id,
            expected_version=running_plan.version,
            status=PlanStatus.COMPLETED,
        )

    for slot_id in ("research", "critique", "edit"):
        run = store.create_role_run(
            ACCESS,
            plan.plan_id,
            slot_id,
            expected_plan_version=running_plan.version,
        )
        succeed_run(store, run)
    completed = store.transition_plan(
        ACCESS,
        plan.plan_id,
        expected_version=running_plan.version,
        status=PlanStatus.COMPLETED,
    )
    assert completed.status == PlanStatus.COMPLETED

    other = create_plan(store, key="cancelled")
    pending = store.create_role_run(
        ACCESS, other.plan_id, "research", expected_plan_version=1
    )
    store.transition_plan(
        ACCESS,
        other.plan_id,
        expected_version=1,
        status=PlanStatus.CANCELLED,
    )
    with pytest.raises(InvalidTransitionError, match="terminal speaker plan"):
        store.transition_role_run(
            ACCESS,
            pending.role_run_id,
            expected_version=1,
            status=RoleRunStatus.RUNNING,
        )


def test_plan_budget_covers_required_dependency_closure(tmp_path: Path) -> None:
    store = SQLiteOrchestrationStore(tmp_path / "orchestration.db")
    with pytest.raises(ValidationError, match="minimum required DAG"):
        store.create_plan(
            ACCESS,
            objective="Impossible budget",
            allowed_role_ids=["researcher", "critic", "editor"],
            slots=slots(),
            client_idempotency_key="impossible",
            max_role_runs=2,
        )


def test_role_run_replay_returns_terminal_materialization(tmp_path: Path) -> None:
    store = SQLiteOrchestrationStore(tmp_path / "orchestration.db")
    plan = create_plan(store)
    original = store.create_role_run(
        ACCESS,
        plan.plan_id,
        "research",
        expected_plan_version=1,
        role_run_id="stable-role-run",
    )
    succeeded = succeed_run(store, original)
    explicit_replay = store.create_role_run(
        ACCESS,
        plan.plan_id,
        "research",
        expected_plan_version=1,
        role_run_id="stable-role-run",
    )
    materialization_replay = store.create_role_run(
        ACCESS,
        plan.plan_id,
        "research",
        expected_plan_version=1,
    )
    assert explicit_replay == succeeded
    assert materialization_replay == succeeded
    with pytest.raises(IdempotencyConflictError, match="explicit RoleRun identity"):
        store.create_role_run(
            ACCESS,
            plan.plan_id,
            "research",
            expected_plan_version=1,
            role_run_id="different-explicit-role-run",
        )


def test_handoff_rejects_empty_and_superseded_fact_sets(tmp_path: Path) -> None:
    store = SQLiteOrchestrationStore(tmp_path / "orchestration.db")
    plan = create_plan(store)
    research = succeed_run(
        store,
        store.create_role_run(
            ACCESS, plan.plan_id, "research", expected_plan_version=1
        ),
    )
    with pytest.raises(FactRuleError, match="at least one"):
        store.create_handoff(
            ACCESS,
            plan.plan_id,
            from_role_run_id=research.role_run_id,
            to_slot_id="critique",
            summary="Untrusted assertion without evidence.",
        )

    first = store.propose_shared_fact(
        ACCESS,
        "claim.current",
        "first",
        source_role_run_id=research.role_run_id,
    )
    store.record_host_verification_receipt(
        ACCESS,
        "claim.current",
        "first",
        authority="tool",
        receipt_id="first-receipt",
        evidence_ref="tool:first",
    )
    old_verified = store.verify_shared_fact(
        ACCESS,
        "claim.current",
        expected_version=first.version,
        verifier="tool",
        verifier_ref="first-receipt",
    )
    newer = store.propose_shared_fact(
        ACCESS,
        "claim.current",
        "second",
        source_role_run_id=research.role_run_id,
        expected_version=old_verified.version,
    )
    store.record_host_verification_receipt(
        ACCESS,
        "claim.current",
        "second",
        authority="tool",
        receipt_id="second-receipt",
        evidence_ref="tool:second",
    )
    store.verify_shared_fact(
        ACCESS,
        "claim.current",
        expected_version=newer.version,
        verifier="tool",
        verifier_ref="second-receipt",
    )
    with pytest.raises(FactRuleError, match="superseded"):
        store.create_handoff(
            ACCESS,
            plan.plan_id,
            from_role_run_id=research.role_run_id,
            to_slot_id="critique",
            summary="Old claim.",
            shared_fact_ids=[old_verified.fact_id],
        )


def test_verification_receipts_are_value_bound_and_single_use(tmp_path: Path) -> None:
    store = SQLiteOrchestrationStore(tmp_path / "orchestration.db")
    plan = create_plan(store)
    research = succeed_run(
        store,
        store.create_role_run(
            ACCESS, plan.plan_id, "research", expected_plan_version=1
        ),
    )
    proposed = store.propose_shared_fact(
        ACCESS,
        "claim.bound",
        {"approved": True},
        source_role_run_id=research.role_run_id,
    )
    store.record_host_verification_receipt(
        ACCESS,
        "claim.bound",
        {"approved": False},
        authority="user",
        receipt_id="wrong-value",
        evidence_ref="approval:wrong",
    )
    with pytest.raises(FactRuleError, match="does not bind"):
        store.verify_shared_fact(
            ACCESS,
            "claim.bound",
            expected_version=proposed.version,
            verifier="user",
            verifier_ref="wrong-value",
        )


@pytest.mark.parametrize(
    "intruder",
    [OTHER_USER_ACCESS, OTHER_CONVERSATION_ACCESS],
    ids=["same-tenant-wrong-user", "same-tenant-wrong-conversation"],
)
def test_plan_role_run_and_handoff_apis_enforce_full_owner_scope(
    tmp_path: Path,
    intruder: OrchestrationAccess,
) -> None:
    store = SQLiteOrchestrationStore(tmp_path / "orchestration.db")
    plan = create_plan(store)
    research = succeed_run(
        store,
        store.create_role_run(
            ACCESS, plan.plan_id, "research", expected_plan_version=1
        ),
    )
    proposed = store.propose_shared_fact(
        ACCESS,
        "claim.scope",
        "owner-only evidence",
        source_role_run_id=research.role_run_id,
    )
    store.record_host_verification_receipt(
        ACCESS,
        "claim.scope",
        "owner-only evidence",
        authority="system",
        receipt_id="scope-receipt",
        evidence_ref="system:scope-receipt",
    )
    verified = store.verify_shared_fact(
        ACCESS,
        "claim.scope",
        expected_version=proposed.version,
        verifier="system",
        verifier_ref="scope-receipt",
    )
    handoff = store.create_handoff(
        ACCESS,
        plan.plan_id,
        from_role_run_id=research.role_run_id,
        to_slot_id="critique",
        summary="Owner-scoped handoff.",
        shared_fact_ids=[verified.fact_id],
    )
    assert store.list_handoffs(ACCESS, plan.plan_id) == [handoff]

    plan_calls = [
        lambda: store.get_plan(intruder, plan.plan_id),
        lambda: store.next_ready_slots(intruder, plan.plan_id),
        lambda: store.list_role_runs(intruder, plan.plan_id),
        lambda: store.list_handoffs(intruder, plan.plan_id),
        lambda: store.transition_plan(
            intruder,
            plan.plan_id,
            expected_version=plan.version,
            status=PlanStatus.RUNNING,
        ),
        lambda: store.create_role_run(
            intruder,
            plan.plan_id,
            "critique",
            expected_plan_version=plan.version,
        ),
        lambda: store.create_handoff(
            intruder,
            plan.plan_id,
            from_role_run_id=research.role_run_id,
            to_slot_id="critique",
            summary="Cross-scope replay.",
            shared_fact_ids=[verified.fact_id],
        ),
    ]
    for call in plan_calls:
        with pytest.raises(OrchestrationNotFoundError):
            call()

    with pytest.raises(OrchestrationNotFoundError):
        store.get_role_run(intruder, research.role_run_id)
    with pytest.raises(OrchestrationNotFoundError):
        store.transition_role_run(
            intruder,
            research.role_run_id,
            expected_version=research.version,
            status=RoleRunStatus.CANCELLED,
        )


def test_private_memory_enforces_owner_conversation_and_role_capability(
    tmp_path: Path,
) -> None:
    store = SQLiteOrchestrationStore(tmp_path / "orchestration.db")
    owner_memory = store.remember_private(
        ACCESS,
        "researcher",
        "Owner-only private note.",
        provenance_key="scope-note",
        now=NOW,
    )

    for intruder in (OTHER_USER_ACCESS, OTHER_CONVERSATION_ACCESS):
        assert store.list_private_memories(intruder, "researcher", now=NOW) == []
        with pytest.raises(OrchestrationNotFoundError):
            store.get_private_memory(
                intruder,
                "researcher",
                owner_memory.memory_id,
                now=NOW,
            )
        isolated = store.remember_private(
            intruder,
            "researcher",
            "Owner-only private note.",
            provenance_key="scope-note",
            now=NOW,
        )
        assert isolated.memory_id != owner_memory.memory_id

    role_access = ACCESS.model_copy(
        update={"allowed_role_ids": ("researcher",)},
        deep=True,
    )
    assert store.get_private_memory(
        role_access, "researcher", owner_memory.memory_id, now=NOW
    ) == owner_memory
    with pytest.raises(RoleNotAllowedError, match="caller access scope"):
        store.list_private_memories(role_access, "critic", now=NOW)
    with pytest.raises(RoleNotAllowedError, match="caller access scope"):
        store.get_private_memory(
            role_access, "critic", owner_memory.memory_id, now=NOW
        )
    with pytest.raises(RoleNotAllowedError, match="caller access scope"):
        store.remember_private(
            role_access,
            "critic",
            "The model attempted to switch role IDs.",
            provenance_key="role-escalation",
            now=NOW,
        )

    with pytest.raises(TypeError, match="trusted OrchestrationAccess"):
        store.get_plan("tenant-a", "untrusted-object-id")  # type: ignore[arg-type]


def test_role_run_apis_enforce_reduced_role_capability(tmp_path: Path) -> None:
    store = SQLiteOrchestrationStore(tmp_path / "orchestration.db")
    plan = create_plan(store)
    critic_access = ACCESS.model_copy(
        update={"allowed_role_ids": ("critic",)},
        deep=True,
    )

    assert store.next_ready_slots(critic_access, plan.plan_id) == []
    with pytest.raises(RoleNotAllowedError, match="caller access scope"):
        store.create_role_run(
            critic_access,
            plan.plan_id,
            "research",
            expected_plan_version=plan.version,
            now=NOW,
        )

    research = store.create_role_run(
        ACCESS,
        plan.plan_id,
        "research",
        expected_plan_version=plan.version,
        now=NOW,
    )
    assert store.list_role_runs(critic_access, plan.plan_id) == []
    with pytest.raises(RoleNotAllowedError, match="caller access scope"):
        store.get_role_run(critic_access, research.role_run_id)
    with pytest.raises(RoleNotAllowedError, match="caller access scope"):
        store.transition_role_run(
            critic_access,
            research.role_run_id,
            expected_version=research.version,
            status=RoleRunStatus.RUNNING,
            now=NOW,
        )

    succeed_run(store, research)
    ready = store.next_ready_slots(critic_access, plan.plan_id)
    assert [slot.role_id for slot in ready] == ["critic"]
    critique = store.create_role_run(
        critic_access,
        plan.plan_id,
        "critique",
        expected_plan_version=plan.version,
        now=NOW,
    )
    assert store.get_role_run(critic_access, critique.role_run_id) == critique


def test_memory_expiry_replay_conflicts_and_connections_close(tmp_path: Path) -> None:
    database = tmp_path / "orchestration.db"
    store = SQLiteOrchestrationStore(database)
    memory = store.remember_private(
        ACCESS,
        "researcher",
        "Retention-bound preference",
        provenance_key="manual-1",
        expires_at=NOW + timedelta(days=30),
        now=NOW,
    )
    assert memory.expires_at == NOW + timedelta(days=30)
    with pytest.raises(IdempotencyConflictError, match="expiry policy"):
        store.remember_private(
            ACCESS,
            "researcher",
            "Retention-bound preference",
            provenance_key="manual-1",
            expires_at=NOW + timedelta(days=1),
            now=NOW,
        )
    store.list_private_memories(ACCESS, "researcher", now=NOW)
    database.unlink()
    assert not database.exists()


def test_json_payloads_reject_surrogates_and_excessive_depth(tmp_path: Path) -> None:
    store = SQLiteOrchestrationStore(tmp_path / "orchestration.db")
    with pytest.raises(ValueError, match="Unicode scalar"):
        store.create_plan(
            ACCESS,
            objective="bad-\ud800",
            allowed_role_ids=["researcher"],
            slots=[slots()[0]],
            client_idempotency_key="surrogate",
        )
    nested: object = "leaf"
    for _ in range(55):
        nested = {"child": nested}
    plan = create_plan(store, key="nested")
    research = succeed_run(
        store,
        store.create_role_run(
            ACCESS, plan.plan_id, "research", expected_plan_version=1
        ),
    )
    with pytest.raises(ValueError, match="nesting depth"):
        store.propose_shared_fact(
            ACCESS,
            "too.deep",
            nested,
            source_role_run_id=research.role_run_id,
        )

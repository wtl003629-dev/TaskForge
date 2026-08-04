from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier

import pytest
from pydantic import ValidationError

from taskforge.review_cases import (
    AuditEventType,
    CaseAccess,
    CaseAccessDeniedError,
    CaseDecisionRuleError,
    CaseIdempotencyConflictError,
    CaseInvalidTransitionError,
    CaseKind,
    CaseRevisionConflictError,
    CaseStatus,
    CaseSubmission,
    EvidenceRef,
    HostActor,
    HumanActor,
    ModelRecommendation,
    RecommendationOutcome,
    ReviewCaseNotFoundError,
    SQLiteReviewCaseStore,
)


NOW = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)
ACCESS = CaseAccess(
    tenant_id="tenant-a",
    owner_user_id="owner-a",
    conversation_id="conversation-a",
)
EVIDENCE = EvidenceRef(
    evidence_id="evidence-policy-v7",
    source_type="document",
    locator="kb://change-policy/v7#page=12",
    title="Production change policy",
    version="7",
    checksum_sha256="a" * 64,
    page_number=12,
    excerpt=(
        "Production changes require an approved ticket, rollback plan, "
        "segregation of duties, and a human final decision."
    ),
)


def submission(*, evidence: bool = True) -> CaseSubmission:
    return CaseSubmission(
        request_summary="Admit the payment gateway change into production.",
        business_justification="The change removes a reconciliation bottleneck.",
        attributes={
            "system": "payments",
            "change_window": "2026-08-09T02:00:00Z",
            "risk_level": "high",
        },
        evidence_refs=[EVIDENCE] if evidence else [],
    )


def recommendation(
    *, evidence_ref: EvidenceRef = EVIDENCE,
    outcome: RecommendationOutcome = RecommendationOutcome.ESCALATE,
) -> ModelRecommendation:
    return ModelRecommendation(
        recommendation_id="recommendation-1",
        model_run_id="agent-run-1",
        model_id="provider/model-1",
        outcome=outcome,
        summary="Escalate because rollback ownership needs confirmation.",
        rationale="The submitted policy requires a named rollback owner.",
        confidence=0.82,
        evidence_refs=[evidence_ref],
        produced_at=NOW + timedelta(minutes=3),
    )


def create_case(
    store: SQLiteReviewCaseStore,
    *,
    access: CaseAccess = ACCESS,
    key: str = "create-case-1",
    with_evidence: bool = True,
):
    return store.create_case(
        access,
        kind=CaseKind.ENTERPRISE_CHANGE,
        title="Payment gateway production admission",
        submission=submission(evidence=with_evidence),
        idempotency_key=key,
        now=NOW,
    )


def advance_to_human_review(
    store: SQLiteReviewCaseStore,
    *,
    access: CaseAccess = ACCESS,
):
    draft = create_case(store, access=access)
    submitted = store.submit_case(
        access,
        draft.case_id,
        expected_revision=draft.revision,
        idempotency_key="submit-case-1",
        now=NOW + timedelta(minutes=1),
    )
    running = store.start_case(
        access,
        draft.case_id,
        expected_revision=submitted.revision,
        idempotency_key="start-case-1",
        actor=HostActor(actor_id="review-worker-1", authority="system"),
        now=NOW + timedelta(minutes=2),
    )
    waiting = store.submit_model_recommendation(
        access,
        draft.case_id,
        expected_revision=running.revision,
        idempotency_key="recommend-case-1",
        recommendation=recommendation(),
        now=NOW + timedelta(minutes=4),
    )
    return draft, submitted, running, waiting


def test_create_update_and_commands_are_revisioned_and_idempotent(
    tmp_path: Path,
) -> None:
    store = SQLiteReviewCaseStore(tmp_path / "review-cases.db")
    created = create_case(store)
    replay = store.create_case(
        ACCESS,
        kind=CaseKind.ENTERPRISE_CHANGE,
        title="Payment gateway production admission",
        submission=submission(),
        idempotency_key="create-case-1",
        now=NOW + timedelta(hours=1),
    )

    assert replay == created
    assert created.status == CaseStatus.DRAFT
    assert created.revision == 1
    with pytest.raises(CaseIdempotencyConflictError):
        store.create_case(
            ACCESS,
            kind=CaseKind.ENTERPRISE_CHANGE,
            title="Different semantic request",
            submission=submission(),
            idempotency_key="create-case-1",
            now=NOW,
        )

    updated = store.update_draft(
        ACCESS,
        created.case_id,
        expected_revision=1,
        idempotency_key="update-case-1",
        title="Updated gateway admission",
        now=NOW + timedelta(minutes=1),
    )
    update_replay = store.update_draft(
        ACCESS,
        created.case_id,
        expected_revision=1,
        idempotency_key="update-case-1",
        title="Updated gateway admission",
        now=NOW + timedelta(hours=2),
    )
    assert update_replay == updated
    assert updated.revision == 2
    assert updated.title == "Updated gateway admission"

    with pytest.raises(CaseRevisionConflictError):
        store.update_draft(
            ACCESS,
            created.case_id,
            expected_revision=1,
            idempotency_key="stale-update",
            title="Lost update",
            now=NOW + timedelta(minutes=2),
        )

    events = store.list_audit_events(ACCESS, created.case_id)
    assert [event.event_type for event in events] == [
        AuditEventType.CREATED,
        AuditEventType.DRAFT_UPDATED,
    ]
    assert [event.revision for event in events] == [1, 2]


def test_full_lifecycle_keeps_model_recommendation_non_authoritative(
    tmp_path: Path,
) -> None:
    store = SQLiteReviewCaseStore(tmp_path / "review-cases.db")
    draft, submitted, running, waiting = advance_to_human_review(store)

    assert [
        draft.status,
        submitted.status,
        running.status,
        waiting.status,
    ] == [
        CaseStatus.DRAFT,
        CaseStatus.SUBMITTED,
        CaseStatus.RUNNING,
        CaseStatus.WAITING_HUMAN_REVIEW,
    ]
    assert waiting.recommendation is not None
    assert waiting.recommendation.authority == "model_untrusted"
    assert waiting.human_decision is None

    with pytest.raises(TypeError, match="HumanActor"):
        store.decide_case(
            ACCESS,
            waiting.case_id,
            expected_revision=waiting.revision,
            idempotency_key="model-cannot-decide",
            outcome=CaseStatus.APPROVED,
            human_actor=HostActor(actor_id="agent-run-1", authority="system"),  # type: ignore[arg-type]
            rationale="A model final answer is not a human decision.",
            now=NOW + timedelta(minutes=5),
        )

    with pytest.raises(CaseAccessDeniedError):
        store.decide_case(
            ACCESS,
            waiting.case_id,
            expected_revision=waiting.revision,
            idempotency_key="wrong-human",
            outcome=CaseStatus.APPROVED,
            human_actor=HumanActor(actor_user_id="someone-else"),
            rationale="Wrong identity.",
            now=NOW + timedelta(minutes=5),
        )

    approved = store.decide_case(
        ACCESS,
        waiting.case_id,
        expected_revision=waiting.revision,
        idempotency_key="human-approve-1",
        outcome=CaseStatus.APPROVED,
        human_actor=HumanActor(actor_user_id="owner-a", display_name="Alice"),
        rationale="Rollback owner was confirmed out of band.",
        evidence_ref_ids=[EVIDENCE.evidence_id],
        now=NOW + timedelta(minutes=5),
    )

    assert approved.status == CaseStatus.APPROVED
    assert approved.recommendation is not None
    assert approved.human_decision is not None
    assert approved.human_decision.actor.authority == "human"
    assert approved.human_decision.outcome == CaseStatus.APPROVED
    with pytest.raises(CaseInvalidTransitionError):
        store.fail_case(
            ACCESS,
            approved.case_id,
            expected_revision=approved.revision,
            idempotency_key="terminal-failure",
            actor=HostActor(actor_id="worker", authority="system"),
            reason="Cannot mutate terminal state.",
            now=NOW + timedelta(minutes=6),
        )

    events = store.list_audit_events(ACCESS, approved.case_id)
    assert [event.event_type for event in events] == [
        AuditEventType.CREATED,
        AuditEventType.SUBMITTED,
        AuditEventType.STARTED,
        AuditEventType.RECOMMENDATION_RECORDED,
        AuditEventType.APPROVED,
    ]
    assert events[-2].actor_authority == "model_untrusted"
    assert events[-1].actor_authority == "human"


def test_evidence_is_required_and_model_citations_are_candidate_bound(
    tmp_path: Path,
) -> None:
    store = SQLiteReviewCaseStore(tmp_path / "review-cases.db")
    empty = create_case(store, key="empty-evidence", with_evidence=False)
    with pytest.raises(CaseDecisionRuleError, match="evidence"):
        store.submit_case(
            ACCESS,
            empty.case_id,
            expected_revision=empty.revision,
            idempotency_key="submit-empty",
            now=NOW + timedelta(minutes=1),
        )

    draft = create_case(store)
    submitted = store.submit_case(
        ACCESS,
        draft.case_id,
        expected_revision=1,
        idempotency_key="submit-evidence",
        now=NOW + timedelta(minutes=1),
    )
    running = store.start_case(
        ACCESS,
        draft.case_id,
        expected_revision=submitted.revision,
        idempotency_key="start-evidence",
        actor=HostActor(actor_id="worker", authority="system"),
        now=NOW + timedelta(minutes=2),
    )
    fabricated = EvidenceRef(
        evidence_id=EVIDENCE.evidence_id,
        source_type="document",
        locator="kb://attacker-controlled-document",
        checksum_sha256="b" * 64,
        excerpt="Attacker-controlled evidence content.",
    )
    with pytest.raises(CaseDecisionRuleError, match="exactly match"):
        store.submit_model_recommendation(
            ACCESS,
            draft.case_id,
            expected_revision=running.revision,
            idempotency_key="fabricated-citation",
            recommendation=recommendation(evidence_ref=fabricated),
            now=NOW + timedelta(minutes=3),
        )
    assert store.get_case(ACCESS, draft.case_id).status == CaseStatus.RUNNING


def test_tenant_owner_and_conversation_are_isolated_and_admin_is_explicit(
    tmp_path: Path,
) -> None:
    store = SQLiteReviewCaseStore(tmp_path / "review-cases.db")
    review_case = create_case(store)
    wrong_scopes = [
        CaseAccess(
            tenant_id="tenant-b",
            owner_user_id="owner-a",
            conversation_id="conversation-a",
        ),
        CaseAccess(
            tenant_id="tenant-a",
            owner_user_id="owner-b",
            conversation_id="conversation-a",
        ),
        CaseAccess(
            tenant_id="tenant-a",
            owner_user_id="owner-a",
            conversation_id="conversation-b",
        ),
    ]
    for scope in wrong_scopes:
        with pytest.raises(ReviewCaseNotFoundError):
            store.get_case(scope, review_case.case_id)
        assert store.list_cases(scope) == []

    role_without_override = CaseAccess(
        tenant_id="tenant-a",
        owner_user_id="owner-a",
        conversation_id="conversation-a",
        actor_user_id="admin-a",
        actor_roles=frozenset({"case_admin"}),
    )
    with pytest.raises(CaseAccessDeniedError):
        store.get_case(role_without_override, review_case.case_id)

    explicit_admin = role_without_override.model_copy(
        update={"admin_override": True}
    )
    assert store.get_case(explicit_admin, review_case.case_id).case_id == review_case.case_id

    with pytest.raises(ValidationError, match="case_admin"):
        CaseAccess(
            tenant_id="tenant-a",
            owner_user_id="owner-a",
            conversation_id="conversation-a",
            actor_user_id="outsider",
            admin_override=True,
        )


def test_owner_inbox_spans_conversations_without_crossing_owner_scope(
    tmp_path: Path,
) -> None:
    store = SQLiteReviewCaseStore(tmp_path / "review-cases.db")
    first = create_case(store)
    second_access = CaseAccess(
        tenant_id="tenant-a",
        owner_user_id="owner-a",
        conversation_id="conversation-b",
    )
    second = store.create_case(
        second_access,
        kind=CaseKind.ENTERPRISE_ADMISSION,
        title="Second owner case",
        submission=submission(),
        idempotency_key="create-second-conversation",
        now=NOW,
    )
    other_owner = CaseAccess(
        tenant_id="tenant-a",
        owner_user_id="owner-b",
        conversation_id="conversation-private",
    )
    store.create_case(
        other_owner,
        kind=CaseKind.ENTERPRISE_CHANGE,
        title="Other owner private case",
        submission=submission(),
        idempotency_key="create-other-owner",
        now=NOW,
    )

    inbox = store.list_owned_cases(ACCESS)

    assert {item.case_id for item in inbox} == {first.case_id, second.case_id}
    assert all(item.owner_user_id == "owner-a" for item in inbox)
    assert store.list_cases(ACCESS) == [first]


def test_concurrent_revision_cas_allows_exactly_one_submit(tmp_path: Path) -> None:
    path = tmp_path / "review-cases.db"
    first_store = SQLiteReviewCaseStore(path)
    second_store = SQLiteReviewCaseStore(path)
    review_case = create_case(first_store)
    barrier = Barrier(2)

    def submit(store: SQLiteReviewCaseStore, key: str):
        barrier.wait(timeout=5)
        try:
            return store.submit_case(
                ACCESS,
                review_case.case_id,
                expected_revision=1,
                idempotency_key=key,
                now=NOW + timedelta(minutes=1),
            )
        except CaseRevisionConflictError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [
            future.result(timeout=5)
            for future in (
                pool.submit(submit, first_store, "submit-concurrent-a"),
                pool.submit(submit, second_store, "submit-concurrent-b"),
            )
        ]

    assert sum(hasattr(result, "status") for result in results) == 1
    assert sum(isinstance(result, CaseRevisionConflictError) for result in results) == 1
    persisted = first_store.get_case(ACCESS, review_case.case_id)
    assert persisted.status == CaseStatus.SUBMITTED
    assert persisted.revision == 2
    assert len(first_store.list_audit_events(ACCESS, review_case.case_id)) == 2


def test_failure_is_host_attributed_and_terminal(tmp_path: Path) -> None:
    store = SQLiteReviewCaseStore(tmp_path / "review-cases.db")
    draft = create_case(store)
    submitted = store.submit_case(
        ACCESS,
        draft.case_id,
        expected_revision=1,
        idempotency_key="submit-before-failure",
        now=NOW + timedelta(minutes=1),
    )
    failed = store.fail_case(
        ACCESS,
        draft.case_id,
        expected_revision=submitted.revision,
        idempotency_key="fail-case-1",
        actor=HostActor(actor_id="queue-worker-2", authority="system"),
        reason="The bounded processing budget was exhausted.",
        now=NOW + timedelta(minutes=2),
    )

    assert failed.status == CaseStatus.FAILED
    assert failed.failure is not None
    assert failed.failure.actor.authority == "system"
    assert failed.human_decision is None
    assert failed.resolved_at == NOW + timedelta(minutes=2)


@pytest.mark.parametrize(
    "attributes",
    [
        {"unsafe": "\ud800"},
        {"non_finite": float("nan")},
        {"too_large": "x" * 1_000_001},
    ],
)
def test_submission_rejects_unsafe_or_unbounded_json(attributes: dict) -> None:
    with pytest.raises(ValidationError):
        CaseSubmission(
            request_summary="Review this request.",
            business_justification="A bounded justification.",
            attributes=attributes,
        )


def test_submission_rejects_excessive_json_depth() -> None:
    nested: dict = {}
    cursor = nested
    for _ in range(55):
        child: dict = {}
        cursor["next"] = child
        cursor = child
    with pytest.raises(ValidationError, match="nesting depth"):
        CaseSubmission(
            request_summary="Review this request.",
            business_justification="A bounded justification.",
            attributes=nested,
        )


def test_connections_close_after_transaction_errors(tmp_path: Path) -> None:
    path = tmp_path / "review-cases.db"
    store = SQLiteReviewCaseStore(path)
    review_case = create_case(store)
    store.update_draft(
        ACCESS,
        review_case.case_id,
        expected_revision=1,
        idempotency_key="first-update",
        title="First update",
        now=NOW + timedelta(minutes=1),
    )
    with pytest.raises(CaseRevisionConflictError):
        store.update_draft(
            ACCESS,
            review_case.case_id,
            expected_revision=1,
            idempotency_key="stale-update-after-error",
            title="Stale update",
            now=NOW + timedelta(minutes=2),
        )

    # On Windows this fails with PermissionError if the transaction connection
    # leaked; on other platforms it still verifies the repository owns no open
    # handle needed after a failed command.
    path.unlink()
    assert not path.exists()

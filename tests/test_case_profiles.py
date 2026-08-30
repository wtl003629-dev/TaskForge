from __future__ import annotations

from taskforge.case_profiles import (
    ENTERPRISE_REVIEW_ROLES,
    RESEARCH_SURVEY_ROLES,
    ResearchSurveyDepth,
    enterprise_review_profiles,
    enterprise_review_slots,
    research_survey_profiles,
    research_survey_slots,
)


def test_enterprise_profiles_bind_exactly_one_fixed_role_without_write_tools() -> None:
    profiles = enterprise_review_profiles(model="test-model")

    assert len(profiles) == 4
    assert {profile.metadata["role_id"] for profile in profiles} == set(
        ENTERPRISE_REVIEW_ROLES
    )
    assert all(profile.model == "test-model" for profile in profiles)
    assert all(profile.knowledge_base_ids == ["enterprise-review"] for profile in profiles)
    assert all("artifact_write" not in profile.allowed_tools for profile in profiles)
    assert all(profile.metadata["human_decision_required"] is True for profile in profiles)


def test_enterprise_slots_form_expected_parallel_then_join_dag() -> None:
    slots = enterprise_review_slots()
    by_id = {slot.slot_id: slot for slot in slots}

    assert [slot.slot_id for slot in slots] == ["intake", "compliance", "risk", "decision"]
    assert by_id["intake"].depends_on == []
    assert by_id["compliance"].depends_on == ["intake"]
    assert by_id["risk"].depends_on == ["intake"]
    assert set(by_id["decision"].depends_on) == {"compliance", "risk"}
    assert {slot.role_id for slot in slots} == set(ENTERPRISE_REVIEW_ROLES)
    assert len({slot.agent_profile_id for slot in slots}) == 4


def test_research_survey_rigorous_keeps_full_chain() -> None:
    slots = research_survey_slots(ResearchSurveyDepth.RIGOROUS)

    assert [slot.slot_id for slot in slots] == [
        "planner",
        "evaluator",
        "writer",
        "critic",
    ]
    by_id = {slot.slot_id: slot for slot in slots}
    assert by_id["planner"].depends_on == []
    assert by_id["evaluator"].depends_on == ["planner"]
    assert by_id["writer"].depends_on == ["evaluator"]
    assert by_id["critic"].depends_on == ["writer"]
    # The critic owns the verdict only at rigorous depth.
    assert "survey.verdict" in by_id["critic"].instruction
    assert "survey.verdict" not in by_id["writer"].instruction
    assert "protocol 只能放在 research_payload 内部" in by_id["writer"].instruction
    assert "exactly one strongest Evidence ID" in by_id["critic"].instruction


def test_research_survey_standard_drops_critic_writer_owns_verdict() -> None:
    slots = research_survey_slots(ResearchSurveyDepth.STANDARD)

    assert [slot.slot_id for slot in slots] == ["planner", "evaluator", "writer"]
    by_id = {slot.slot_id: slot for slot in slots}
    assert by_id["writer"].depends_on == ["evaluator"]
    assert "survey.verdict" in by_id["writer"].instruction


def test_research_survey_minimal_is_planner_and_writer() -> None:
    slots = research_survey_slots(ResearchSurveyDepth.MINIMAL)

    assert [slot.slot_id for slot in slots] == ["planner", "writer"]
    by_id = {slot.slot_id: slot for slot in slots}
    assert by_id["writer"].depends_on == ["planner"]
    assert "survey.verdict" in by_id["writer"].instruction


def test_research_survey_profiles_follow_depth_selection() -> None:
    minimal = research_survey_profiles(model="test-model", depth=ResearchSurveyDepth.MINIMAL)
    standard = research_survey_profiles(model="test-model", depth=ResearchSurveyDepth.STANDARD)
    rigorous = research_survey_profiles(model="test-model", depth=ResearchSurveyDepth.RIGOROUS)

    assert [p.metadata["role_id"] for p in minimal] == [
        "retrieval_planner",
        "synthesis_writer",
    ]
    assert [p.metadata["role_id"] for p in standard] == [
        "retrieval_planner",
        "source_evaluator",
        "synthesis_writer",
    ]
    assert [p.metadata["role_id"] for p in rigorous] == list(RESEARCH_SURVEY_ROLES)


def test_paper_protocol_scopes_retrieval_to_evaluator() -> None:
    profiles = research_survey_profiles(model="test-model", protocol="paper")
    by_role = {profile.metadata["role_id"]: profile for profile in profiles}
    assert "paper_search" not in by_role["retrieval_planner"].allowed_tools
    assert "paper_search" in by_role["source_evaluator"].allowed_tools
    assert "paper_search" not in by_role["synthesis_writer"].allowed_tools
    assert "paper_search" not in by_role["critical_reviewer"].allowed_tools
    assert by_role["synthesis_writer"].max_steps == 3
    assert all(
        profile.metadata["research_protocol"] == "paper"
        for profile in profiles
    )


def test_paper_text_cannot_grant_scope_mutation_capabilities() -> None:
    profiles = research_survey_profiles(model="test-model", protocol="paper")
    forbidden = {
        "scope_create",
        "scope_update",
        "scope_confirm",
        "research_scope_create",
        "research_scope_update",
    }
    assert all(forbidden.isdisjoint(profile.allowed_tools) for profile in profiles)
    assert profiles[0].allowed_tools == []

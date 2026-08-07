"""Host-owned profiles and fixed DAG for the enterprise review showcase."""

from __future__ import annotations

from enum import Enum

from .domain import AgentProfile
from .orchestration import SpeakerSlot

ENTERPRISE_REVIEW_ROLES = (
    "intake_analyst",
    "compliance_reviewer",
    "risk_reviewer",
    "decision_synthesizer",
)


class ResearchSurveyDepth(str, Enum):
    """How much of the research survey DAG a case materialises.

    The four role steps form a dependency chain ``planner -> evaluator ->
    writer -> critic``; each depth truncates the *tail* while keeping the
    chain valid, and the terminal step always submits the ``survey.verdict``.
    """

    MINIMAL = "minimal"  # planner -> writer
    STANDARD = "standard"  # planner -> evaluator -> writer
    RIGOROUS = "rigorous"  # planner -> evaluator -> writer -> critic


def enterprise_review_profiles(*, model: str) -> list[AgentProfile]:
    """Return role-bound profiles; callers may not substitute role metadata."""

    common_tools = ["knowledge_search", "calculator", "memory_recall"]
    common_scopes = ["tenant", "user", "task"]
    definitions = [
        (
            "case-intake-agent",
            "材料受理分析员",
            "intake_analyst",
            "抽取申请范围、变更对象、约束与缺失材料；每个结论必须给出证据引用。",
            "结构化受理材料并识别缺口，不做最终批准。",
        ),
        (
            "case-compliance-agent",
            "合规审查员",
            "compliance_reviewer",
            "逐条对照适用政策与控制要求；区分满足、不满足和证据不足，不得编造条款。",
            "依据版本化政策检查合规性，只提交待验证建议。",
        ),
        (
            "case-risk-agent",
            "风险评估员",
            "risk_reviewer",
            "识别安全、隐私、运营与回滚风险，评估影响和缓解措施；不替代人工决策。",
            "形成带证据的风险登记与缓解建议。",
        ),
        (
            "case-decision-agent",
            "决策汇总员",
            "decision_synthesizer",
            "汇总上游事实、争议和证据缺口，给出 approve/reject/escalate 建议；"
            "明确说明这只是模型建议，最终决定必须由人工作出。",
            "汇总多角色结论，生成可审计的人工决策建议。",
        ),
    ]
    return [
        AgentProfile(
            id=profile_id,
            name=name,
            instructions=instructions,
            model=model,
            allowed_tools=list(common_tools),
            knowledge_base_ids=["enterprise-review"],
            memory_scopes=list(common_scopes),
            max_steps=7,
            metadata={
                "role_id": role_id,
                "description": description,
                "domain": "enterprise_change_review",
                "human_decision_required": True,
            },
        )
        for profile_id, name, role_id, instructions, description in definitions
    ]


def enterprise_review_slots() -> list[SpeakerSlot]:
    """Return the fixed review DAG; model routing can only select ready roles."""

    return [
        SpeakerSlot(
            slot_id="intake",
            role_id="intake_analyst",
            agent_profile_id="case-intake-agent",
            instruction="Extract scope, requested change, constraints, evidence inventory, and missing items.",
            order=10,
        ),
        SpeakerSlot(
            slot_id="compliance",
            role_id="compliance_reviewer",
            agent_profile_id="case-compliance-agent",
            instruction="Check the intake against current policy evidence and enumerate compliance gaps.",
            depends_on=["intake"],
            order=20,
        ),
        SpeakerSlot(
            slot_id="risk",
            role_id="risk_reviewer",
            agent_profile_id="case-risk-agent",
            instruction="Assess security, privacy, operational, and rollback risks with mitigations.",
            depends_on=["intake"],
            order=30,
        ),
        SpeakerSlot(
            slot_id="decision",
            role_id="decision_synthesizer",
            agent_profile_id="case-decision-agent",
            instruction=(
                "Synthesize verified facts and clearly labelled unverified recommendations into an "
                "approve, reject, or escalate recommendation for human review."
            ),
            depends_on=["compliance", "risk"],
            order=40,
        ),
    ]


RESEARCH_SURVEY_ROLES = (
    "retrieval_planner",
    "source_evaluator",
    "synthesis_writer",
    "critical_reviewer",
)


def research_survey_profiles(
    *, model: str, depth: ResearchSurveyDepth = ResearchSurveyDepth.RIGOROUS
) -> list[AgentProfile]:
    """Return role-bound research profiles; callers may not substitute metadata.

    ``depth`` selects which roles participate: minimal keeps planner+writer,
    standard adds the source evaluator, rigorous keeps the full four-role
    chain (including the critical reviewer that owns the verdict).
    """

    common_tools = ["knowledge_search", "calculator", "memory_recall"]
    common_scopes = ["tenant", "user", "task"]
    include_roles = {
        ResearchSurveyDepth.MINIMAL: {"retrieval_planner", "synthesis_writer"},
        ResearchSurveyDepth.STANDARD: {
            "retrieval_planner",
            "source_evaluator",
            "synthesis_writer",
        },
        ResearchSurveyDepth.RIGOROUS: set(RESEARCH_SURVEY_ROLES),
    }[depth]
    definitions = [
        (
            "case-research-planner-agent",
            "检索规划员",
            "retrieval_planner",
            "把研究问题拆成可检索的子问题，规划检索策略并执行检索，登记检索到的来源清单；"
            "每个结论必须给出证据引用，不得编造来源。",
            "产出检索计划与来源清单，不直接给出综述结论。",
        ),
        (
            "case-research-evaluator-agent",
            "来源甄别员",
            "source_evaluator",
            "对检索到的来源评估可信度、相关性与时效性，标注权威来源与存疑来源，识别证据缺口；"
            "所有判断必须基于本次真实检索到的来源。",
            "形成可信度评估与证据缺口清单。",
        ),
        (
            "case-research-writer-agent",
            "综合综述员",
            "synthesis_writer",
            "综合检索到的来源撰写分章节综述；每条结论必须引用真实检索到的来源（evidence_id 或 source），"
            "不得编造引用、作者或文献。区分已验证事实与模型推断。",
            "产出带引用的综述初稿。",
        ),
        (
            "case-research-critic-agent",
            "批判审查员",
            "critical_reviewer",
            "反向审查综述稿：检查每条结论是否超出现有引用能支撑的范围、是否忽略相反证据、章节是否有缺口；"
            "给出 survey.verdict（accept / needs_revision / more_evidence）建议，最终由人复核。",
            "反向质疑综述并给出人工复核建议。",
        ),
    ]
    selected = [
        item for item in definitions if item[2] in include_roles
    ]
    return [
        AgentProfile(
            id=profile_id,
            name=name,
            instructions=instructions,
            model=model,
            allowed_tools=list(common_tools),
            knowledge_base_ids=["enterprise-review"],
            memory_scopes=list(common_scopes),
            max_steps=7,
            metadata={
                "role_id": role_id,
                "description": description,
                "domain": "research_survey",
                "human_decision_required": True,
            },
        )
        for profile_id, name, role_id, instructions, description in selected
    ]


def research_survey_slots(
    depth: ResearchSurveyDepth = ResearchSurveyDepth.RIGOROUS,
) -> list[SpeakerSlot]:
    """Return the research DAG truncated at ``depth``.

    ``plan -> evaluate -> write -> critique`` under rigorous; standard drops
    the critic (writer becomes the terminal verdict owner); minimal also drops
    the evaluator and makes the writer depend directly on the planner.
    """

    planner = SpeakerSlot(
        slot_id="planner",
        role_id="retrieval_planner",
        agent_profile_id="case-research-planner-agent",
        instruction=(
            "Decompose the research question into searchable sub-questions, "
            "plan the retrieval strategy, and record the retrieved source "
            "inventory with exact evidence references."
        ),
        order=10,
    )
    writer_depends = ["evaluator"] if depth != ResearchSurveyDepth.MINIMAL else ["planner"]
    # When the critic is absent, the writer is the terminal step and must own
    # the verdict; otherwise the critic does.
    writer_owns_verdict = depth != ResearchSurveyDepth.RIGOROUS
    writer_verdict = (
        " As the final step, submit one survey.verdict claim (accept / "
        "needs_revision / more_evidence) after finishing the survey; final "
        "authority is human."
        if writer_owns_verdict
        else ""
    )
    writer = SpeakerSlot(
        slot_id="writer",
        role_id="synthesis_writer",
        agent_profile_id="case-research-writer-agent",
        instruction=(
            "Write the survey in sections; every claim must cite a real "
            "retrieved source (evidence_id or source). Never invent citations."
            + writer_verdict
        ),
        depends_on=writer_depends,
        order=30,
    )
    if depth == ResearchSurveyDepth.MINIMAL:
        return [planner, writer]
    evaluator = SpeakerSlot(
        slot_id="evaluator",
        role_id="source_evaluator",
        agent_profile_id="case-research-evaluator-agent",
        instruction=(
            "Assess credibility, relevance, and recency of the retrieved "
            "sources; flag authoritative vs questionable sources and evidence gaps."
        ),
        depends_on=["planner"],
        order=20,
    )
    if depth == ResearchSurveyDepth.STANDARD:
        return [planner, evaluator, writer]
    critic = SpeakerSlot(
        slot_id="critic",
        role_id="critical_reviewer",
        agent_profile_id="case-research-critic-agent",
        instruction=(
            "Critically review the survey: do claims exceed what the cited "
            "sources support? Are counter-evidence and section gaps missed? "
            "Submit one survey.verdict claim (accept / needs_revision / "
            "more_evidence); final authority is human."
        ),
        depends_on=["writer"],
        order=40,
    )
    return [planner, evaluator, writer, critic]


__all__ = [
    "ENTERPRISE_REVIEW_ROLES",
    "RESEARCH_SURVEY_ROLES",
    "ResearchSurveyDepth",
    "enterprise_review_profiles",
    "enterprise_review_slots",
    "research_survey_profiles",
    "research_survey_slots",
]

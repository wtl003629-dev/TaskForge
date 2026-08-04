"""Host-owned profiles and fixed DAG for the enterprise review showcase."""

from __future__ import annotations

from .domain import AgentProfile
from .orchestration import SpeakerSlot

ENTERPRISE_REVIEW_ROLES = (
    "intake_analyst",
    "compliance_reviewer",
    "risk_reviewer",
    "decision_synthesizer",
)


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


__all__ = [
    "ENTERPRISE_REVIEW_ROLES",
    "enterprise_review_profiles",
    "enterprise_review_slots",
]

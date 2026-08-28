"""Host-owned profiles and fixed DAG for the enterprise review showcase."""

from __future__ import annotations

from enum import Enum
from typing import Literal

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

    common_tools = [
        "knowledge_search",
        "paper_search",
        "paper_read",
        "citation_verify",
        "calculator",
        "memory_recall",
    ]
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
    *,
    model: str,
    depth: ResearchSurveyDepth = ResearchSurveyDepth.RIGOROUS,
    protocol: Literal["legacy", "paper"] = "legacy",
) -> list[AgentProfile]:
    """Return role-bound research profiles; callers may not substitute metadata.

    ``depth`` selects which roles participate: minimal keeps planner+writer,
    standard adds the source evaluator, rigorous keeps the full four-role
    chain (including the critical reviewer that owns the verdict).
    """

    if protocol not in {"legacy", "paper"}:
        raise ValueError("research protocol must be legacy or paper")
    # The legacy ``knowledge_search`` capability stays available for the
    # existing review-case adapter.  The production paper protocol is split
    # by role: only the evaluator discovers sources; downstream roles resolve
    # IDs and verify claims instead of launching duplicate searches.
    role_tools = (
        {
            "retrieval_planner": [],
            "source_evaluator": [
                "paper_search",
                "scope_expansion_request",
            ],
            "synthesis_writer": [
                "paper_read",
                "citation_verify",
            ],
            "critical_reviewer": [
                "paper_read",
                "citation_verify",
                "scope_expansion_request",
            ],
        }
        if protocol == "paper"
        else {
            "retrieval_planner": ["knowledge_search", "calculator", "memory_recall"],
            "source_evaluator": [
                "knowledge_search",
                "calculator",
                "memory_recall",
            ],
            "synthesis_writer": ["knowledge_search", "calculator", "memory_recall"],
            "critical_reviewer": ["knowledge_search", "calculator", "memory_recall"],
        }
    )
    if protocol == "paper" and depth == ResearchSurveyDepth.MINIMAL:
        # Without an evaluator, the writer is the bounded fallback discovery
        # role. Standard and rigorous keep discovery exclusive to evaluator.
        role_tools["synthesis_writer"].insert(0, "paper_search")
    role_steps = {
        "retrieval_planner": 1,
        "source_evaluator": 2,
        # The paper Writer may use paper_read and citation_verify before its
        # mandatory submit_role_result call; each model turn consumes one step.
        "synthesis_writer": 3,
        "critical_reviewer": 3,
    }
    research_tool_limits = {
        "retrieval_planner": {},
        "source_evaluator": {
            "paper_search": 1,
            "scope_expansion_request": 1,
        },
        "synthesis_writer": {
            "paper_search": 1,
            "paper_read": 1,
            "citation_verify": 1,
        },
        "critical_reviewer": {
            "paper_read": 1,
            "citation_verify": 1,
            "scope_expansion_request": 1,
        },
    }
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
            "把研究问题压缩成最多 3 个子问题、4 项证据要求和 5 个短纲要条目；"
            "使用 Host 已绑定的 Scope 元数据立即提交结构化计划，不执行检索，不输出长篇解释。",
            "产出检索计划与来源清单，不直接给出综述结论。",
        ),
        (
            "case-research-evaluator-agent",
            "来源甄别员",
            "source_evaluator",
            "只执行一次覆盖研究问题的 paper_search，并直接筛选其有界证据片段；"
            "随后只提交 Evidence ID 账本并将 evidence_cards 设为空列表；Host 会从工具回执拼接证据卡，"
            "不要为每个子问题重复搜索。",
            "形成可信度评估与证据缺口清单。",
        ),
        (
            "case-research-writer-agent",
            "综合综述员",
            "synthesis_writer",
            "只消费上游 EvidenceCard，最多读取 1 条最高价值证据并核验 1 条核心论断，"
            "生成不超过 6 条 ClaimRecord；"
            "每条结论引用真实 evidence_id，不重复检索，不输出完整论文原文。",
            "产出带引用的综述初稿。",
        ),
        (
            "case-research-critic-agent",
            "批判审查员",
            "critical_reviewer",
            "只审查高风险或未验证 ClaimRecord，最多读取 1 条证据并核验 1 条核心论断，"
            "输出不超过 6 个差异补丁；"
            "不得重写全文。给出 survey.verdict，最终由人复核。",
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
            allowed_tools=list(role_tools[role_id]),
            knowledge_base_ids=["enterprise-review"],
            memory_scopes=list(common_scopes),
            max_steps=role_steps[role_id] if protocol == "paper" else 7,
            metadata={
                "role_id": role_id,
                "description": description,
                "domain": "research_survey",
                "human_decision_required": True,
                "compact_tool_trajectory": True,
                "communication_protocol": "research.blackboard.delta.v1",
                "research_protocol": protocol,
                "tool_call_limits": (
                    research_tool_limits[role_id] if protocol == "paper" else {}
                ),
                **(
                    {
                        "thinking_mode": (
                            "disabled"
                            if role_id
                            in {
                                "retrieval_planner",
                                "source_evaluator",
                                "synthesis_writer",
                            }
                            else "enabled"
                        )
                    }
                    if protocol == "paper"
                    else {}
                ),
                **(
                    {"terminal_tools": ["submit_role_result"]}
                    if protocol == "paper"
                    else {}
                ),
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
            "Decompose the research question into at most 3 short sub-questions, "
            "4 short evidence requirements, and 5 short outline items. Do not run "
            "paper_search in this role; source discovery belongs to the "
            "source_evaluator role. Submit research_payload using "
            "research.planner_handoff.v1 with a structured ResearchPlan."
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
            "Write the survey from the upstream evidence-card blackboard; every "
            "claim must cite a real retrieved source (evidence_id or source). "
            "Before the claim manifest, fill direct_answer with the shortest "
            "standalone answer to the user's question. Prefer one exact noun "
            "phrase copied verbatim from a cited evidence snippet; do not replace "
            "specific names with a broad category. It must normally be at most 12 "
            "tokens and answer only what was asked, not summarize the evidence. "
            "For yes/no questions use exactly Yes, No, or Unanswerable. For "
            "how-many or numeric questions use the number form exactly as written "
            "in the source, without adding a second form. For what-is-baseline, "
            "what-system, or list questions return only the specific names/items. "
            "For task questions return only the task name. For comparison "
            "questions return only the requested comparison. Do not put citations, "
            "rationale, methods, or extra paper details in direct_answer. The "
            "claim manifest is still the complete cited explanation shown to the "
            "user. "
            "Use at most one paper_read and one citation_verify call for the "
            "highest-value cited claim, and never start "
            "a duplicate paper_search in this role. Emit no more than 6 claims. "
            "DraftArtifact.claim_ids must exactly equal the ClaimManifest claim_id "
            "values in the same order; never list a draft claim without its full "
            "ClaimRecord. "
            "Submit research_payload "
            "using research.writer_handoff.v1 with DraftArtifact, direct_answer, "
            "and ClaimManifest."
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
            "Run the bounded unified paper retrieval loop, assess credibility, "
            "relevance, and recency, and publish evidence IDs, coverage gaps, "
            "and verification receipts for downstream roles. Use exactly one "
            "broad paper_search with top_k at most 8 and intent matching the user "
            "request; its evidence cards already contain the bounded "
            "snippets needed for screening, so do not read full passages. Do not issue "
            "one search per sub-question and never call undeclared tools such as "
            "paper_info. Set evidence_cards to an empty list "
            "because the Host joins cards from receipts. Then immediately submit "
            "research.evaluator_handoff.v1 with EvidenceLedger and bounded "
            "EvidenceCards; never include full paper text."
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
            "Critically review the claim manifest: do claims exceed what the "
            "cited sources support? Are counter-evidence and section gaps missed? "
            "Treat direct_answer as a compact answer contract: do not rewrite it "
            "for style, and preserve its yes/no, numeric, or list form unless a "
            "specific cited fact is wrong. If no concrete factual defect exists, "
            "accept the Writer answer without patches. "
            "Use at most one paper_read and one citation_verify call for the highest-"
            "risk cited claim; do not search "
            "for new papers or rewrite the draft, and emit no more than 6 patches. "
            "Every patch claim_id must be copied exactly from the upstream "
            "ClaimManifest; never patch a draft-only or invented claim ID. "
            "Use scope_expansion_request only for genuinely new paper IDs not already "
            "listed in the bound Scope. Missing evidence for an already selected paper "
            "is an ingestion/retrieval gap: report it in a patch and do not request "
            "Scope expansion. Never mutate Scope. Submit "
            "research.critic_handoff.v1 with ReviewPatch entries and a verdict. "
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

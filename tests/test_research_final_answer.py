from __future__ import annotations

import pytest
from pydantic import ValidationError

from taskforge.app import _research_scope_notice
from taskforge.research_protocol import (
    CriticHandoff,
    DraftArtifact,
    ReviewPatch,
    WriterHandoff,
    project_final_research_answer,
)


def test_field_wide_question_gets_host_owned_scope_notice() -> None:
    limited, note = _research_scope_notice("RAG 的最新技术有哪些？", 2)

    assert limited is True
    assert "2 篇已选论文" in note
    assert "不能代表整个领域" in note


def test_specific_question_does_not_get_field_wide_scope_notice() -> None:
    assert _research_scope_notice("论文中的忠实性分数如何计算？", 2) == (False, "")


def _writer() -> WriterHandoff:
    return WriterHandoff(
        direct_answer="dataset A with 90 percent accuracy",
        draft=DraftArtifact(
            draft_id="draft-1",
            claim_ids=["claim-1", "claim-2", "claim-3"],
            section_count=1,
        ),
        claim_manifest=[
            {
                "claim_id": "claim-1",
                "claim_text": "The model uses dataset A.",
                "evidence_ids": ["evidence-1"],
            },
            {
                "claim_id": "claim-2",
                "claim_text": "It reports 90 percent accuracy.",
                "evidence_ids": ["evidence-2"],
            },
            {
                "claim_id": "claim-3",
                "claim_text": "An unsupported conclusion.",
                "evidence_ids": ["evidence-3"],
            },
        ],
    )


def test_final_answer_applies_critic_patches_and_preserves_writer_evidence() -> None:
    critic = CriticHandoff(
        patches=[
            ReviewPatch(
                claim_id="claim-1",
                action="keep",
                reason="The cited paragraph supports the dataset.",
            ),
            ReviewPatch(
                claim_id="claim-2",
                action="revise",
                replacement="It reports 89 percent accuracy.",
                reason="The draft rounded the reported value incorrectly.",
            ),
            ReviewPatch(
                claim_id="claim-3",
                action="request_evidence",
                reason="No selected paragraph supports this conclusion.",
            ),
        ],
        verdict="needs_revision",
    )

    answer = project_final_research_answer(_writer(), critic)

    assert answer.answer == (
        "The model uses dataset A. [1]\n\nIt reports 89 percent accuracy. [2]"
    )
    assert answer.direct_answer == "dataset A with 90 percent accuracy"
    assert answer.evidence_ids == ["evidence-1", "evidence-2"]
    assert answer.included_claim_ids == ["claim-1", "claim-2"]
    assert answer.removed_claim_ids == ["claim-3"]
    assert answer.unresolved_claim_ids == ["claim-3"]


def test_final_answer_hides_internal_ids_and_projects_numbered_citations() -> None:
    writer = WriterHandoff(
        direct_answer="从已选论文看，规范约束是其中一个方向。",
        draft=DraftArtifact(
            draft_id="draft-public",
            claim_ids=["claim-public"],
            section_count=1,
        ),
        claim_manifest=[
            {
                "claim_id": "claim-public",
                "claim_text": (
                    "Paper-d1fce1cccb6949ed95693f3a933b9fe2 提出规范约束型 RAG"
                    "（evidence_id: evidence:scope-demo:v1:chunk-1）。"
                ),
                "paper_ids": ["paper-d1fce1cccb6949ed95693f3a933b9fe2"],
                "evidence_ids": ["evidence:scope-demo:v1:chunk-1"],
            }
        ],
    )

    answer = project_final_research_answer(
        writer,
        CriticHandoff(patches=[], verdict="accept"),
    )

    assert answer.answer == "该论文提出规范约束型 RAG。 [1]"
    assert "paper-" not in answer.answer.casefold()
    assert "evidence:" not in answer.answer.casefold()
    assert answer.evidence_ids == ["evidence:scope-demo:v1:chunk-1"]


def test_final_answer_deduplicates_same_paper_citations_within_claim() -> None:
    writer = WriterHandoff(
        direct_answer="The selected paper supports one bounded direction.",
        draft=DraftArtifact(
            draft_id="draft-dedup",
            claim_ids=["claim-dedup"],
            section_count=1,
        ),
        claim_manifest=[
            {
                "claim_id": "claim-dedup",
                "claim_text": "The method uses retrieval-grounded generation.",
                "paper_ids": ["paper-1"],
                "evidence_ids": ["evidence-abstract", "evidence-keywords", "evidence-body"],
            }
        ],
    )

    answer = project_final_research_answer(
        writer,
        CriticHandoff(patches=[], verdict="accept"),
        evidence_paper_ids={
            "evidence-abstract": "paper-1",
            "evidence-keywords": "paper-1",
            "evidence-body": "paper-1",
        },
    )

    assert answer.answer.endswith("[1]")
    assert answer.evidence_ids == ["evidence-abstract"]
    assert answer.cited_paper_count == 1


def test_final_answer_uses_neutral_selected_paper_scope_wording() -> None:
    writer = WriterHandoff(
        direct_answer="受限于仅2篇中文文献，结论只覆盖局部方向。",
        draft=DraftArtifact(
            draft_id="draft-scope",
            claim_ids=["claim-supported", "claim-scope"],
            section_count=1,
        ),
        claim_manifest=[
            {
                "claim_id": "claim-supported",
                "claim_text": "本报告仅基于2篇中文论文，并支持一个局部方向。",
                "paper_ids": ["paper-1"],
                "evidence_ids": ["evidence-1"],
            },
            {
                "claim_id": "claim-scope",
                "claim_text": "本报告仅基于2篇中文论文。",
                "evidence_ids": [],
            }
        ],
    )

    answer = project_final_research_answer(
        writer,
        CriticHandoff(patches=[], verdict="more_evidence"),
        scope_limited=True,
        scope_note="当前证据不足以代表整个领域。",
    )

    assert answer.answer == "本报告仅基于2篇已选论文，并支持一个局部方向。 [1]"
    assert answer.direct_answer == "当前证据不足以代表整个领域。"
    assert answer.scope_limited is True
    assert answer.scope_note == "当前证据不足以代表整个领域。"
    assert answer.removed_claim_ids == ["claim-scope"]


def test_scope_limited_answer_removes_cited_gap_and_false_cross_paper_claims() -> None:
    writer = WriterHandoff(
        direct_answer="The bounded corpus cannot establish the field-wide latest state.",
        draft=DraftArtifact(
            draft_id="draft-grounding-guards",
            claim_ids=["claim-supported", "claim-gap", "claim-both"],
            section_count=1,
        ),
        claim_manifest=[
            {
                "claim_id": "claim-supported",
                "claim_text": "论文 A 描述了一种合规型 RAG 应用。",
                "evidence_ids": ["evidence-a"],
            },
            {
                "claim_id": "claim-gap",
                "claim_text": "当前已选论文未命名任何专用评测基准。",
                "evidence_ids": ["evidence-a-gap"],
            },
            {
                "claim_id": "claim-both",
                "claim_text": "两篇论文均指出该技术属于近期进展。",
                "evidence_ids": ["evidence-a-recent"],
            },
        ],
    )

    answer = project_final_research_answer(
        writer,
        CriticHandoff(patches=[], verdict="accept"),
        evidence_paper_ids={
            "evidence-a": "paper-a",
            "evidence-a-gap": "paper-a",
            "evidence-a-recent": "paper-a",
        },
        scope_limited=True,
        scope_note="当前证据不足以代表整个领域。",
    )

    assert answer.answer == "论文 A 描述了一种合规型 RAG 应用。 [1]"
    assert answer.evidence_ids == ["evidence-a"]
    assert answer.removed_claim_ids == ["claim-gap", "claim-both"]


@pytest.mark.parametrize(
    "patch, message",
    [
        (
            ReviewPatch(
                claim_id="unknown",
                action="keep",
                reason="Invalid claim ID.",
            ),
            "unknown claim",
        ),
        (
            ReviewPatch(
                claim_id="claim-1",
                action="revise",
                reason="Missing replacement.",
            ),
            "non-empty replacement",
        ),
    ],
)
def test_final_answer_rejects_invalid_critic_patch(
    patch: ReviewPatch,
    message: str,
) -> None:
    critic = CriticHandoff(patches=[patch], verdict="needs_revision")

    with pytest.raises(ValueError, match=message):
        project_final_research_answer(_writer(), critic)


def test_final_answer_rejects_projection_without_supported_claims() -> None:
    critic = CriticHandoff(
        patches=[
            ReviewPatch(
                claim_id=claim_id,
                action="remove",
                reason="Remove from the final answer.",
            )
            for claim_id in ("claim-1", "claim-2", "claim-3")
        ],
        verdict="more_evidence",
    )

    with pytest.raises(ValueError, match="removed every answer claim"):
        project_final_research_answer(_writer(), critic)


def test_writer_handoff_rejects_draft_claim_without_manifest_record() -> None:
    with pytest.raises(ValidationError, match="exactly match claim_manifest"):
        WriterHandoff(
            draft=DraftArtifact(
                draft_id="draft-bad",
                claim_ids=["claim-present", "claim-missing"],
                section_count=1,
            ),
            claim_manifest=[
                {
                    "claim_id": "claim-present",
                    "claim_text": "Supported claim.",
                    "evidence_ids": ["evidence-1"],
                }
            ],
        )

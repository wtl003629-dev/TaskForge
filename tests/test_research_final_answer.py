from __future__ import annotations

import pytest
from pydantic import ValidationError

from taskforge.research_protocol import (
    CriticHandoff,
    DraftArtifact,
    ReviewPatch,
    WriterHandoff,
    project_final_research_answer,
)


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

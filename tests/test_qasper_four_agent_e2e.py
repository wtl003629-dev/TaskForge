from scripts.evaluate_qasper_four_agent_e2e import (
    _candidate_child_recall,
    _deterministic_candidate_metrics,
    _presented_window_recall,
    _window_citation_metrics,
    _writer_selected_context_recall,
    _writer_selected_evidence,
)
from taskforge.qasper_alignment import GoldAlignedSpan, GoldAlignment
from taskforge.rag_evaluation import (
    GoldEvidenceSet,
    GoldEvidenceUnit,
    QasperGoldLabels,
    RAGEvalCase,
)


def _case() -> RAGEvalCase:
    unit = GoldEvidenceUnit(
        unit_id="gold-1",
        text="the decisive result is forty two percent on the held out set",
        alternative_paragraph_ids=["paragraph-1"],
    )
    return RAGEvalCase(
        case_id="case-1",
        dataset="qasper",
        query="What is the decisive result?",
        relevant_ids=["paragraph-1"],
        category="fact",
        qasper_gold=QasperGoldLabels(
            evidence_sets=[GoldEvidenceSet(annotation_id="ann-1", units=[unit])]
        ),
    )


def _retrieval_row(text: str) -> dict[str, object]:
    return {
        "retrieved_evidence": [
            {
                "evidence_id": "evidence-1",
                "chunk_id": "child-1",
                "text": text,
            }
        ],
        "gold_alignments": {
            "gold-1": GoldAlignment(
                gold_unit_id="gold-1",
                status="exact",
                aligned_child_spans=[
                    GoldAlignedSpan(
                        child_id="child-1",
                        gold_token_start=0,
                        gold_token_end=11,
                        matched_tokens=11,
                        score=1.0,
                    )
                ],
                normalized_coverage=1.0,
                alignment_score=1.0,
                gold_token_count=11,
            ).model_dump(mode="json")
        },
    }


def test_visibility_recall_separates_child_hit_from_missing_window() -> None:
    case = _case()
    row = _retrieval_row("an unrelated prefix from the same retrieved child")

    assert _candidate_child_recall(case, row, top_k=1) == 1.0
    assert _presented_window_recall(case, row, top_k=1) == 0.0


def test_visibility_recall_obeys_writer_initial_context_budget() -> None:
    case = _case()
    gold = "the decisive result is forty two percent on the held out set"
    row = _retrieval_row("x " * 410 + gold)

    assert _presented_window_recall(case, row, top_k=1) == 1.0
    assert _presented_window_recall(case, row, top_k=1, text_chars=800) == 0.0


def test_citation_support_uses_presented_text_not_only_child_id() -> None:
    case = _case()
    presented = _retrieval_row("unrelated text from the same child")[
        "retrieved_evidence"
    ]

    metrics = _window_citation_metrics(["evidence-1"], presented, case)

    assert metrics["citation_validity"] == 1.0
    assert metrics["gold_content_citation_precision"] == 0.0
    assert metrics["gold_evidence_unit_coverage"] == 0.0


def test_writer_context_preserves_receipt_selection() -> None:
    case = _case()
    gold = "the decisive result is forty two percent on the held out set"
    presented = [
        {"evidence_id": "evidence-1", "text": "unrelated ranked evidence"},
        {"evidence_id": "evidence-2", "text": gold},
    ]
    agent_row = {
        "role_runs": [
            {
                "role_id": "source_evaluator",
                "role_result": {
                    "research_payload": {
                        "ledger": {
                            "evidence_ids": ["invented", "evidence-2"]
                        }
                    }
                },
            }
        ]
    }

    selected = _writer_selected_evidence(agent_row, presented)

    assert [item["evidence_id"] for item in selected] == ["evidence-2"]
    assert _writer_selected_context_recall(case, agent_row, presented) == 1.0


def test_deterministic_candidate_metrics_include_strict_grounded_exact_match() -> None:
    case = _case()
    gold = "the decisive result is forty two percent on the held out set"
    presented = _retrieval_row(gold)["retrieved_evidence"]
    alignments = {
        "gold-1": GoldAlignment.model_validate(
            _retrieval_row(gold)["gold_alignments"]["gold-1"]
        )
    }

    metrics = _deterministic_candidate_metrics(
        answer=gold,
        direct_answer="42 percent",
        citations=["evidence-1"],
        references=["42 percent"],
        presented=presented,
        case=case,
        alignments=alignments,
    )

    assert metrics["exact_match"] == 1.0
    assert metrics["strict_exact_and_gold_citation"] == 1.0
    assert metrics["scoring_answer"] == "42 percent"

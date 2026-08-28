from __future__ import annotations

import json

import pytest

from scripts.evaluate_qasper_answer_e2e import (
    _validate_retrieval_report,
    citation_metrics,
    parse_semantic_judgement,
    qasper_answer_references,
    qasper_gold_evidence_texts,
    qasper_reference_answer,
)
from taskforge.qasper_alignment import GoldAlignedSpan, GoldAlignment
from taskforge.rag_evaluation import (
    GoldEvidenceSet,
    GoldEvidenceUnit,
    QasperGoldLabels,
)


def test_citation_metrics_require_presented_id_and_gold_content_overlap() -> None:
    labels = QasperGoldLabels(
        evidence_sets=[
            GoldEvidenceSet(
                annotation_id="a-1",
                units=[
                    GoldEvidenceUnit(
                        unit_id="gold-a",
                        text="alpha beta gamma delta epsilon",
                        alternative_paragraph_ids=["p-a"],
                    ),
                    GoldEvidenceUnit(
                        unit_id="gold-b",
                        text="one two three four five",
                        alternative_paragraph_ids=["p-b"],
                    ),
                ],
            )
        ]
    )
    alignments = {
        "gold-a": GoldAlignment(
            gold_unit_id="gold-a",
            status="exact",
            aligned_child_spans=[
                GoldAlignedSpan(
                    child_id="child-a",
                    gold_token_start=0,
                    gold_token_end=5,
                    matched_tokens=5,
                    score=1.0,
                )
            ],
            normalized_coverage=1.0,
            alignment_score=1.0,
            gold_token_count=5,
        ),
        "gold-b": GoldAlignment(
            gold_unit_id="gold-b",
            status="exact",
            aligned_child_spans=[
                GoldAlignedSpan(
                    child_id="child-b",
                    gold_token_start=0,
                    gold_token_end=5,
                    matched_tokens=5,
                    score=1.0,
                )
            ],
            normalized_coverage=1.0,
            alignment_score=1.0,
            gold_token_count=5,
        ),
    }
    metrics = citation_metrics(
        ["e-1", "invented", "e-2"],
        [
            {"evidence_id": "e-1", "chunk_id": "child-a"},
            {"evidence_id": "e-2", "chunk_id": "unrelated"},
        ],
        labels,
        alignments,
    )

    assert metrics == {
        "citation_count": 3,
        "valid_citation_count": 2,
        "gold_supported_citation_count": 1,
        "invalid_citation_ids": ["invented"],
        "citation_validity": pytest.approx(2 / 3),
        "gold_content_citation_precision": pytest.approx(1 / 3),
        "gold_evidence_unit_coverage": pytest.approx(1 / 2),
        "covered_gold_unit_ids": ["gold-a"],
        "selected_gold_annotation_id": "a-1",
    }


def test_parse_semantic_judgement_is_strict() -> None:
    judgement = parse_semantic_judgement(
        json.dumps(
            {
                "answer_verdict": "correct",
                "citation_verdict": "fully_supported",
                "critical_error": False,
                "missing_key_points": [],
                "contradictions": [],
                "rationale": "The answer and cited passage match the reference.",
            }
        )
    )

    assert judgement.answer_verdict == "correct"
    assert judgement.citation_verdict == "fully_supported"

    with pytest.raises(ValueError):
        parse_semantic_judgement('{"answer_verdict":"maybe"}')


def test_qasper_reference_answer_restores_yes_no_labels() -> None:
    assert qasper_reference_answer(1.0) == "Yes"
    assert qasper_reference_answer(0.0) == "No"
    assert qasper_reference_answer("1.0") == "1.0"


def test_qasper_uses_all_answerable_annotations() -> None:
    question = {
        "answers": [
            {
                "answer": {
                    "unanswerable": False,
                    "free_form_answer": "First phrasing",
                    "extractive_spans": [],
                    "yes_no": None,
                    "evidence": ["Evidence A"],
                }
            },
            {
                "answer": {
                    "unanswerable": False,
                    "free_form_answer": "",
                    "extractive_spans": ["Second", "phrasing"],
                    "yes_no": None,
                    "evidence": ["Evidence B", "Evidence A"],
                }
            },
            {"answer": {"unanswerable": True, "evidence": ["ignored"]}},
        ]
    }

    assert qasper_answer_references(question) == [
        "First phrasing",
        "Second; phrasing",
    ]
    assert qasper_gold_evidence_texts(question) == ["Evidence A", "Evidence B"]


def test_answer_eval_rejects_failed_alignment_for_retrieved_evidence() -> None:
    report = {
        "schema_version": "2.1",
        "evaluation_type": "qasper_real_pdf_upload_retrieval",
        "alignment_gate": {"passed": False},
    }

    with pytest.raises(ValueError, match="alignment gate"):
        _validate_retrieval_report(report, evidence_source="retrieved")

    _validate_retrieval_report(report, evidence_source="oracle")


def test_answer_eval_rejects_legacy_page_proxy_report() -> None:
    report = {
        "schema_version": "2.3",
        "evaluation_type": "qasper_direct_pdf_upload_retrieval",
    }

    with pytest.raises(ValueError, match="page-proxy"):
        _validate_retrieval_report(report, evidence_source="oracle")


def test_bilingual_paper_smoke_requires_an_explicit_track() -> None:
    report = {
        "schema_version": "2.3",
        "evaluation_type": "bilingual_mixed_paper_corpus_cross_language_queries",
        "benchmark_track": "scholarly_paper_fulltext_retrieval",
        "alignment_gate": {"passed": True},
    }

    with pytest.raises(ValueError, match="strict QASPER"):
        _validate_retrieval_report(report, evidence_source="retrieved")

    _validate_retrieval_report(
        report,
        evidence_source="retrieved",
        benchmark_track="bilingual_paper_smoke",
    )

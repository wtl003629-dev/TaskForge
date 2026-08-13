from __future__ import annotations

import json

import pytest

from taskforge.rag_evaluation import (
    CitedAnswerPrediction,
    EvalCorpusDocument,
    RAGEvalCase,
    RetrievalPrediction,
    answer_exact_match,
    answer_token_f1,
    evaluate_answer_grounding,
    evaluate_hierarchical_retrieval,
    evaluate_retrieval,
    load_mmlongbench_cases,
    load_multihop_rag_dataset,
    load_qasper_dataset,
    load_tatqa_dataset,
)


def test_retrieval_metrics_score_missing_cases_as_zero() -> None:
    cases = [
        RAGEvalCase(
            case_id="one",
            dataset="synthetic",
            query="first",
            relevant_ids=["a", "b"],
            category="table",
        ),
        RAGEvalCase(
            case_id="two",
            dataset="synthetic",
            query="second",
            relevant_ids=["c"],
            category="text",
        ),
    ]
    report = evaluate_retrieval(
        cases,
        [RetrievalPrediction(case_id="one", retrieved_ids=["x", "b", "a"])],
        ks=(1, 2, 3),
    )

    assert report.summary.total_cases == 2
    assert report.summary.missing_predictions == 1
    assert report.summary.recall_at_k == {"1": 0.0, "2": 0.25, "3": 0.5}
    assert report.summary.mrr_at_k["2"] == pytest.approx(0.25)
    assert report.summary.ndcg_at_k["3"] == pytest.approx(0.3467, abs=0.001)
    assert report.summary.by_category_recall_at_k["table"]["3"] == 1.0
    assert report.cases[1].missing_prediction is True


def test_retrieval_evaluator_rejects_ambiguous_predictions() -> None:
    case = RAGEvalCase(
        case_id="one",
        dataset="synthetic",
        query="q",
        relevant_ids=["a"],
        category="text",
    )
    with pytest.raises(ValueError, match="unknown case"):
        evaluate_retrieval(
            [case],
            [RetrievalPrediction(case_id="other", retrieved_ids=[])],
        )
    with pytest.raises(ValueError, match="duplicates"):
        RetrievalPrediction(case_id="one", retrieved_ids=["a", "a"])


def test_hierarchical_retrieval_reports_parent_and_weak_operand_metrics() -> None:
    case = RAGEvalCase(
        case_id="one",
        dataset="TAT-QA",
        query="What was revenue?",
        relevant_ids=["table"],
        category="table",
        answer=["$1,496.5"],
        metadata={"derivation": ""},
    )
    documents = [
        EvalCorpusDocument(
            document_id="table",
            text="Revenue | $1,496.5",
            source_uri="tatqa://table",
            metadata={
                "kind": "table",
                "parent_document_id": "context",
                "table_rows": [["Metric", "2025"], ["Revenue", "$1,496.5"]],
            },
        )
    ]
    report = evaluate_hierarchical_retrieval(
        [case],
        [
            RetrievalPrediction(
                case_id="one",
                retrieved_ids=["table"],
                retrieved_parent_ids=["context"],
                retrieved_complete_table_ids=["table"],
            )
        ],
        documents,
        retrieved_texts_by_case={"one": ["Revenue | $1,496.5"]},
        ks=(1,),
    )

    assert report["parent_recall_at_k"] == {"1": 1.0}
    assert report["table_recall_at_k"] == {"1": 1.0}
    assert report["row_recall_at_k"] == {"1": 1.0}
    assert report["cell_recall_at_k"] == {"1": 1.0}
    assert report["full_evidence_recall_at_k"] == {"1": 1.0}
    assert report["weak_operand_recall_at_k"] == {"1": 1.0}
    assert "official TAT-QA cell recall" in report["weak_operand_definition"]


def test_tatqa_adapter_preserves_table_and_paragraph_evidence(tmp_path) -> None:
    payload = [
        {
            "table": {
                "uid": "table-1",
                "table": [["Metric", "2025"], ["Revenue", "42"]],
            },
            "paragraphs": [
                {"uid": "p-1", "order": 1, "text": "Revenue increased."}
            ],
            "questions": [
                {
                    "uid": "q-1",
                    "question": "What was revenue?",
                    "answer": ["42"],
                    "answer_type": "span",
                    "answer_from": "table-text",
                    "rel_paragraphs": ["1"],
                    "derivation": "",
                    "scale": "million",
                }
            ],
        }
    ]
    path = tmp_path / "tatqa.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = load_tatqa_dataset(path)

    assert dataset.license == "CC BY 4.0"
    assert len(dataset.documents) == 2
    assert dataset.cases[0].relevant_ids == [
        "tatqa:table-1:table",
        "tatqa:table-1:paragraph:1",
    ]
    assert dataset.cases[0].category == "table"
    assert dataset.documents[0].metadata["parent_document_id"] == (
        "tatqa:table-1:context"
    )
    assert dataset.documents[1].metadata["parent_document_id"] == (
        "tatqa:table-1:context"
    )
    assert dataset.cases[0].metadata["parent_document_id"] == (
        "tatqa:table-1:context"
    )


def test_multihop_rag_adapter_normalizes_cross_document_cases(tmp_path) -> None:
    corpus = [
        {
            "title": "Alpha",
            "author": "u1",
            "source": "Ex",
            "published_at": "2024-01-01T00:00:00+00:00",
            "category": "technology",
            "url": "https://ex.com/a",
            "body": "Alpha article body.",
        },
        {
            "title": "Beta",
            "author": "u2",
            "source": "Ex",
            "published_at": "2024-01-02T00:00:00+00:00",
            "category": "technology",
            "url": "https://ex.com/b",
            "body": "Beta article body.",
        },
    ]
    queries = [
        {
            "query": "Compare Alpha and Beta.",
            "answer": "equal",
            "question_type": "comparison_query",
            "evidence_list": [
                {"url": "https://ex.com/a", "fact": "Alpha fact"},
                {"url": "https://ex.com/b", "fact": "Beta fact"},
            ],
        },
        {
            "query": "Unanswerable.",
            "answer": "Insufficient information.",
            "question_type": "null_query",
            "evidence_list": [],
        },
    ]
    corpus_path = tmp_path / "corpus.json"
    queries_path = tmp_path / "queries.json"
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    queries_path.write_text(json.dumps(queries), encoding="utf-8")

    dataset = load_multihop_rag_dataset(queries_path, corpus_path)

    assert dataset.dataset == "MultiHop-RAG"
    assert dataset.license == "ODC-BY"
    assert len(dataset.documents) == 2
    # null_query rows are unanswerable and carry no evidence.
    assert len(dataset.cases) == 1
    case = dataset.cases[0]
    assert case.category == "comparison_query"
    assert case.answer == "equal"
    assert set(case.relevant_ids) == {doc.document_id for doc in dataset.documents}
    assert all(case_id.startswith("multihop:") for case_id in [case.case_id])


def test_qasper_adapter_uses_only_exact_full_text_evidence(tmp_path) -> None:
    evidence = "The method improves recall by 4 points."
    payload = {
        "paper-1": {
            "title": "A paper",
            "abstract": "Abstract.",
            "full_text": [
                {"section_name": "Results", "paragraphs": [evidence]},
            ],
            "qas": [
                {
                    "question": "What improves recall?",
                    "question_id": "q1",
                    "answers": [
                        {
                            "answer": {
                                "unanswerable": False,
                                "extractive_spans": ["4 points"],
                                "free_form_answer": "",
                                "yes_no": None,
                                "evidence": [evidence],
                            }
                        }
                    ],
                },
                {
                    "question": "Missing evidence?",
                    "question_id": "q2",
                    "answers": [
                        {
                            "answer": {
                                "unanswerable": False,
                                "extractive_spans": ["not present"],
                                "free_form_answer": "",
                                "yes_no": None,
                                "evidence": ["not a paragraph"],
                            }
                        }
                    ],
                },
            ],
        }
    }
    path = tmp_path / "qasper.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    dataset = load_qasper_dataset(path)
    assert dataset.dataset == "QASPER"
    assert len(dataset.cases) == 1
    assert dataset.cases[0].relevant_ids[0].endswith("paragraph:0")
    assert all("not a paragraph" not in doc.text for doc in dataset.documents)
    evidence_document = next(
        doc for doc in dataset.documents if doc.document_id.endswith("paragraph:0")
    )
    assert evidence_document.metadata["paper_title"] == "A paper"
    assert evidence_document.metadata["section_title"] == "Results"
    assert evidence_document.metadata["section_id"].endswith("section:0")
    assert evidence_document.metadata["parent_document_id"] == "qasper:paper-1:paper"
    assert evidence_document.metadata["sentence_spans"] == [
        {"start": 0, "end": len(evidence)}
    ]
    assert evidence_document.metadata["char_start"] == 0
    assert evidence_document.metadata["char_end"] == len(evidence)


def test_mmlongbench_adapter_maps_evidence_pages_without_pdf_download(tmp_path) -> None:
    path = tmp_path / "samples.json"
    path.write_text(
        json.dumps(
            [
                {
                    "doc_id": "report.pdf",
                    "doc_type": "report",
                    "question": "Compare the two pages.",
                    "answer": "42%",
                    "evidence_pages": "[3, 5]",
                    "evidence_sources": "['Pure-text (Table)']",
                    "answer_format": "Float",
                }
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_mmlongbench_cases(path)

    assert dataset.documents == []
    assert dataset.cases[0].category == "cross-page-table"
    assert dataset.cases[0].relevant_ids == [
        "mmlongbench:report.pdf:page:3",
        "mmlongbench:report.pdf:page:5",
    ]


def test_answer_metrics_are_deterministic_and_multi_answer_aware() -> None:
    assert answer_exact_match("$1,496.5", ["1496.5", "other"]) == 1.0
    assert answer_token_f1("fixed price contract", "fixed contract") == pytest.approx(0.8)


def _grounding_case(*, relevant_ids: list[str] | None = None) -> RAGEvalCase:
    return RAGEvalCase(
        case_id="grounding-one",
        dataset="synthetic",
        query="Who reported the result?",
        relevant_ids=relevant_ids or ["gold-a", "gold-b"],
        category="multi-hop",
        answer="The Verge",
    )


def test_answer_grounding_scores_perfect_citations_without_a_judge() -> None:
    report = evaluate_answer_grounding(
        [_grounding_case()],
        [
            CitedAnswerPrediction(
                case_id="grounding-one",
                answer="The Verge",
                retrieved_ids=["gold-a", "gold-b", "distractor"],
                presented_evidence_ids=["gold-a", "gold-b", "distractor"],
                citation_ids=["gold-a", "gold-b"],
            )
        ],
    )

    row = report.cases[0]
    assert row.citation_precision == 1.0
    assert row.citation_recall == 1.0
    assert row.retrieval_to_citation_coverage == 1.0
    assert row.strict_supported_claim is True
    assert report.summary.strict_unsupported_claim_rate == 0.0


def test_answer_grounding_keeps_irrelevant_citations_in_precision_denominator() -> None:
    report = evaluate_answer_grounding(
        [_grounding_case(relevant_ids=["gold-a"])],
        [
            CitedAnswerPrediction(
                case_id="grounding-one",
                answer="The Verge",
                retrieved_ids=["gold-a", "distractor"],
                presented_evidence_ids=["gold-a", "distractor"],
                citation_ids=["gold-a", "distractor"],
            )
        ],
    )

    row = report.cases[0]
    assert row.citation_precision == 0.5
    assert row.citation_recall == 1.0
    assert row.invalid_citation_ids == ["distractor"]


def test_answer_grounding_rejects_forged_gold_id_not_retrieved_by_host() -> None:
    report = evaluate_answer_grounding(
        [_grounding_case()],
        [
            CitedAnswerPrediction(
                case_id="grounding-one",
                answer="The Verge",
                retrieved_ids=["gold-a"],
                presented_evidence_ids=["gold-a"],
                citation_ids=["gold-b"],
            )
        ],
    )

    row = report.cases[0]
    assert row.valid_citation_ids == []
    assert row.invalid_citation_ids == ["gold-b"]
    assert row.citation_precision == 0.0
    assert row.citation_recall == 0.0
    assert row.retrieval_to_citation_coverage == 0.0
    assert row.unsupported_claim is True


def test_answer_grounding_scores_partial_gold_evidence_adoption() -> None:
    report = evaluate_answer_grounding(
        [_grounding_case()],
        [
            CitedAnswerPrediction(
                case_id="grounding-one",
                answer="The Verge",
                retrieved_ids=["gold-a", "gold-b"],
                presented_evidence_ids=["gold-a", "gold-b"],
                citation_ids=["gold-a"],
            )
        ],
    )

    row = report.cases[0]
    assert row.citation_precision == 1.0
    assert row.citation_recall == 0.5
    assert row.retrieval_to_citation_coverage == 0.5
    assert row.strict_supported_claim is True


def test_answer_grounding_scores_no_citations_as_unsupported() -> None:
    report = evaluate_answer_grounding(
        [_grounding_case(relevant_ids=["gold-a"])],
        [
            CitedAnswerPrediction(
                case_id="grounding-one",
                answer="The Verge",
                retrieved_ids=["gold-a"],
                presented_evidence_ids=["gold-a"],
                citation_ids=[],
            )
        ],
    )

    row = report.cases[0]
    assert row.citation_precision == 0.0
    assert row.citation_recall == 0.0
    assert row.retrieval_to_citation_coverage == 0.0
    assert row.unsupported_claim is True


def test_answer_grounding_marks_rtc_not_applicable_without_correct_retrieval() -> None:
    report = evaluate_answer_grounding(
        [_grounding_case(relevant_ids=["gold-a"])],
        [
            CitedAnswerPrediction(
                case_id="grounding-one",
                answer="The Verge",
                retrieved_ids=["distractor"],
                presented_evidence_ids=["distractor"],
                citation_ids=["distractor"],
            )
        ],
    )

    assert report.cases[0].retrieval_to_citation_coverage is None
    assert report.summary.retrieval_to_citation_coverage is None
    assert report.summary.retrieval_to_citation_eligible_cases == 0


def test_cited_answer_prediction_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="citation_ids must not contain duplicates"):
        CitedAnswerPrediction(
            case_id="grounding-one",
            citation_ids=["gold-a", "gold-a"],
        )


def test_cited_answer_prediction_requires_presented_ids_to_be_retrieved() -> None:
    with pytest.raises(ValueError, match="must be a subset"):
        CitedAnswerPrediction(
            case_id="grounding-one",
            retrieved_ids=["gold-a"],
            presented_evidence_ids=["gold-b"],
        )


def test_answer_grounding_scores_missing_prediction_as_unsupported() -> None:
    report = evaluate_answer_grounding([_grounding_case()], [])

    row = report.cases[0]
    assert row.missing_prediction is True
    assert row.citation_precision == 0.0
    assert row.citation_recall == 0.0
    assert row.retrieval_to_citation_coverage is None
    assert row.unsupported_claim is True
    assert report.summary.missing_predictions == 1
    assert report.summary.strict_unsupported_claim_rate == 1.0


def test_answer_grounding_requires_parse_success_and_exact_match_for_support() -> None:
    case = _grounding_case(relevant_ids=["gold-a"])
    parsed_wrong = CitedAnswerPrediction(
        case_id="grounding-one",
        answer="TechCrunch",
        retrieved_ids=["gold-a"],
        presented_evidence_ids=["gold-a"],
        citation_ids=["gold-a"],
    )
    parse_failed = parsed_wrong.model_copy(
        update={"answer": "The Verge", "parse_error": "invalid_json"}
    )

    wrong_report = evaluate_answer_grounding([case], [parsed_wrong])
    failed_report = evaluate_answer_grounding([case], [parse_failed])

    assert wrong_report.cases[0].exact_match == 0.0
    assert wrong_report.cases[0].unsupported_claim is True
    assert failed_report.cases[0].exact_match == 1.0
    assert failed_report.cases[0].parse_failed is True
    assert failed_report.cases[0].unsupported_claim is True

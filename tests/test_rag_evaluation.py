from __future__ import annotations

import json

import pytest

from taskforge.rag_evaluation import (
    RAGEvalCase,
    RetrievalPrediction,
    answer_exact_match,
    answer_token_f1,
    evaluate_retrieval,
    load_mmlongbench_cases,
    load_multihop_rag_dataset,
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

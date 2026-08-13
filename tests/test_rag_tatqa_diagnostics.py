from __future__ import annotations

from taskforge.rag_evaluation import EvalCorpusDocument, RAGEvalCase
from taskforge.rag_tatqa_diagnostics import (
    TATQARetrievalRow,
    build_tatqa_query_plan,
    diagnose_tatqa_retrieval,
)


def _case(answer_type: str = "arithmetic") -> RAGEvalCase:
    return RAGEvalCase(
        case_id="tatqa:q1",
        dataset="TAT-QA",
        query="What was the percentage increase from 2020 to 2021, at least 10%?",
        relevant_ids=["table-1"],
        category=answer_type,
        metadata={"answer_type": answer_type, "scale": "percent"},
    )


def _documents() -> list[EvalCorpusDocument]:
    return [
        EvalCorpusDocument(
            document_id="table-1",
            text="Year | Revenue\n2020 | 10\n2021 | 12",
            source_uri="tatqa://t1/table",
            metadata={
                "kind": "table",
                "parent_document_id": "parent-1",
                "table_uid": "t1",
            },
        ),
        EvalCorpusDocument(
            document_id="paragraph-1",
            text="The company grew.",
            source_uri="tatqa://t1/paragraph/1",
            metadata={"kind": "paragraph", "parent_document_id": "parent-1"},
        ),
    ]


def test_query_plan_is_deterministic_and_contains_numeric_constraints() -> None:
    plan = build_tatqa_query_plan(_case())
    assert plan["operation"] == "arithmetic"
    assert plan["years"] == [2020, 2021]
    assert plan["comparator"] == "gte"
    assert plan["thresholds"] == ["10%"]
    assert plan["scale"] == "percent"


def test_oracles_separate_parent_access_from_real_child_recall() -> None:
    case = _case()
    rows = [
        TATQARetrievalRow(
            case_id=case.case_id,
            category=case.category,
            relevant_ids=("table-1",),
            retrieved_ids=("paragraph-1",),
            retrieved_parent_ids=("parent-1",),
        )
    ]
    report = diagnose_tatqa_retrieval(
        [case], _documents(), rows, top_k=1, candidate_k=1
    )
    aggregate = report["aggregate"]
    assert aggregate["real_recall_at_10"] == 0.0
    assert aggregate["gold_parent_to_real_child"] == 1.0
    assert aggregate["gold_section_to_real_unit"] == 0.0
    assert aggregate["oracle_top10_from_candidate"] == 0.0
    assert aggregate["candidate_state_counts"] == {
        "zero": 1,
        "partial": 0,
        "complete": 0,
    }
    assert aggregate["candidate_any_hit_rate"] == 0.0
    assert aggregate["candidate_all_evidence_rate"] == 0.0
    assert aggregate["candidate_missing_reason_counts"] == {
        "section_kind_unreached": 1
    }
    assert report["per_case"][0]["candidate_missing_reasons"] == {
        "table-1": "section_kind_unreached"
    }


def test_diagnostics_separate_partial_multi_evidence_from_complete_candidates() -> None:
    documents = [
        *_documents(),
        EvalCorpusDocument(
            document_id="table-2",
            text="Year | Cost\n2020 | 4\n2021 | 5",
            source_uri="tatqa://t2/table",
            metadata={
                "kind": "table",
                "parent_document_id": "parent-2",
                "table_uid": "t2",
            },
        ),
    ]
    case = _case().model_copy(
        update={"relevant_ids": ["table-1", "table-2"]}
    )
    rows = [
        TATQARetrievalRow(
            case_id=case.case_id,
            category=case.category,
            relevant_ids=("table-1", "table-2"),
            retrieved_ids=("table-1",),
            retrieved_parent_ids=("parent-1",),
        )
    ]

    report = diagnose_tatqa_retrieval(
        [case], documents, rows, top_k=1, candidate_k=1
    )

    aggregate = report["aggregate"]
    assert aggregate["real_candidate_recall"] == 0.5
    assert aggregate["candidate_state_counts"] == {
        "zero": 0,
        "partial": 1,
        "complete": 0,
    }
    assert aggregate["candidate_any_hit_rate"] == 1.0
    assert aggregate["candidate_all_evidence_rate"] == 0.0
    assert aggregate["multi_evidence"] == {
        "cases": 1,
        "candidate_all_evidence_rate": 0.0,
        "top10_all_evidence_rate": 0.0,
    }
    assert report["per_case"][0]["candidate_missing_reasons"] == {
        "table-2": "parent_unreached"
    }

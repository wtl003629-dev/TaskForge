"""Deterministic retrieval and answer evaluation for TaskForge RAG pipelines.

The evaluator is deliberately independent from model providers and vector
stores.  It consumes stable document/evidence identifiers so the same cases
can compare lexical, dense, hybrid, reranked, and graph-assisted retrievers.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from .domain import StrictModel


class EvalCorpusDocument(StrictModel):
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGEvalCase(StrictModel):
    case_id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    query: str = Field(min_length=1)
    relevant_ids: list[str] = Field(min_length=1)
    category: str = Field(min_length=1)
    answer: str | list[str] | float | int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def relevant_ids_are_unique(self) -> RAGEvalCase:
        if any(not item.strip() for item in self.relevant_ids):
            raise ValueError("relevant_ids must contain non-empty identifiers")
        if len(self.relevant_ids) != len(set(self.relevant_ids)):
            raise ValueError("relevant_ids must not contain duplicates")
        return self


class RAGEvalDataset(StrictModel):
    dataset: str = Field(min_length=1)
    license: str = Field(min_length=1)
    attribution_url: str = Field(min_length=1)
    documents: list[EvalCorpusDocument] = Field(default_factory=list)
    cases: list[RAGEvalCase] = Field(default_factory=list)

    @model_validator(mode="after")
    def identifiers_are_unique(self) -> RAGEvalDataset:
        document_ids = [item.document_id for item in self.documents]
        case_ids = [item.case_id for item in self.cases]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("dataset document identifiers must be unique")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("dataset case identifiers must be unique")
        return self


class RetrievalPrediction(StrictModel):
    case_id: str = Field(min_length=1)
    retrieved_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def retrieved_ids_are_unique(self) -> RetrievalPrediction:
        if len(self.retrieved_ids) != len(set(self.retrieved_ids)):
            raise ValueError("retrieved_ids must not contain duplicates")
        return self


class RetrievalCaseMetrics(StrictModel):
    case_id: str
    category: str
    recall_at_k: dict[str, float]
    reciprocal_rank_at_k: dict[str, float]
    ndcg_at_k: dict[str, float]
    missing_prediction: bool = False


class RetrievalEvaluationSummary(StrictModel):
    total_cases: int = Field(ge=0)
    missing_predictions: int = Field(ge=0)
    recall_at_k: dict[str, float]
    mrr_at_k: dict[str, float]
    ndcg_at_k: dict[str, float]
    by_category_recall_at_k: dict[str, dict[str, float]]


class RetrievalEvaluationReport(StrictModel):
    schema_version: str = "1.0"
    ks: list[int]
    cases: list[RetrievalCaseMetrics]
    summary: RetrievalEvaluationSummary


def _safe_ks(ks: Sequence[int]) -> tuple[int, ...]:
    result = tuple(sorted(set(int(value) for value in ks)))
    if not result or any(value <= 0 or value > 10_000 for value in result):
        raise ValueError("ks must contain values between 1 and 10000")
    return result


def _recall(relevant: set[str], retrieved: Sequence[str], k: int) -> float:
    return len(relevant.intersection(retrieved[:k])) / len(relevant)


def _reciprocal_rank(relevant: set[str], retrieved: Sequence[str], k: int) -> float:
    for index, item in enumerate(retrieved[:k], start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def _ndcg(relevant: set[str], retrieved: Sequence[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(index + 2)
        for index, item in enumerate(retrieved[:k])
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return dcg / idcg


def evaluate_retrieval(
    cases: Sequence[RAGEvalCase],
    predictions: Sequence[RetrievalPrediction],
    *,
    ks: Sequence[int] = (5, 10, 20),
) -> RetrievalEvaluationReport:
    """Compute macro Recall, MRR, and nDCG with missing cases scored as zero."""

    safe_ks = _safe_ks(ks)
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case identifiers must be unique")
    prediction_map: dict[str, RetrievalPrediction] = {}
    for prediction in predictions:
        if prediction.case_id not in set(case_ids):
            raise ValueError(f"prediction references unknown case: {prediction.case_id}")
        if prediction.case_id in prediction_map:
            raise ValueError(f"duplicate prediction for case: {prediction.case_id}")
        prediction_map[prediction.case_id] = prediction

    rows: list[RetrievalCaseMetrics] = []
    category_values: dict[str, dict[str, list[float]]] = {}
    missing = 0
    for case in cases:
        prediction = prediction_map.get(case.case_id)
        if prediction is None:
            missing += 1
            retrieved: list[str] = []
        else:
            retrieved = prediction.retrieved_ids
        relevant = set(case.relevant_ids)
        recall = {str(k): _recall(relevant, retrieved, k) for k in safe_ks}
        reciprocal_rank = {
            str(k): _reciprocal_rank(relevant, retrieved, k) for k in safe_ks
        }
        ndcg = {str(k): _ndcg(relevant, retrieved, k) for k in safe_ks}
        rows.append(
            RetrievalCaseMetrics(
                case_id=case.case_id,
                category=case.category,
                recall_at_k=recall,
                reciprocal_rank_at_k=reciprocal_rank,
                ndcg_at_k=ndcg,
                missing_prediction=prediction is None,
            )
        )
        category = category_values.setdefault(
            case.category,
            {str(k): [] for k in safe_ks},
        )
        for key, value in recall.items():
            category[key].append(value)

    def average(values: Iterable[float]) -> float:
        materialized = list(values)
        return sum(materialized) / len(materialized) if materialized else 0.0

    return RetrievalEvaluationReport(
        ks=list(safe_ks),
        cases=rows,
        summary=RetrievalEvaluationSummary(
            total_cases=len(rows),
            missing_predictions=missing,
            recall_at_k={
                str(k): average(row.recall_at_k[str(k)] for row in rows)
                for k in safe_ks
            },
            mrr_at_k={
                str(k): average(row.reciprocal_rank_at_k[str(k)] for row in rows)
                for k in safe_ks
            },
            ndcg_at_k={
                str(k): average(row.ndcg_at_k[str(k)] for row in rows)
                for k in safe_ks
            },
            by_category_recall_at_k={
                category: {
                    key: average(values)
                    for key, values in per_k.items()
                }
                for category, per_k in sorted(category_values.items())
            },
        ),
    )


def _table_text(rows: Sequence[Sequence[object]]) -> str:
    return "\n".join(
        " | ".join(str(cell).strip() for cell in row)
        for row in rows
    ).strip()


def load_tatqa_dataset(path: str | Path, *, limit: int | None = None) -> RAGEvalDataset:
    """Normalize the official TAT-QA raw JSON into retrievable evidence units."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("TAT-QA payload must be a JSON array")
    documents: list[EvalCorpusDocument] = []
    cases: list[RAGEvalCase] = []
    for context in raw:
        if not isinstance(context, Mapping):
            raise ValueError("TAT-QA context must be an object")
        table = context.get("table")
        paragraphs = context.get("paragraphs")
        questions = context.get("questions")
        if not isinstance(table, Mapping) or not isinstance(paragraphs, list) or not isinstance(questions, list):
            raise ValueError("TAT-QA context is missing table, paragraphs, or questions")
        table_uid = str(table.get("uid", "")).strip()
        table_rows = table.get("table")
        if not table_uid or not isinstance(table_rows, list):
            raise ValueError("TAT-QA table requires uid and rows")
        table_id = f"tatqa:{table_uid}:table"
        documents.append(
            EvalCorpusDocument(
                document_id=table_id,
                text=_table_text(table_rows),
                source_uri=f"tatqa://{table_uid}/table",
                metadata={"kind": "table", "table_uid": table_uid},
            )
        )
        paragraph_ids: dict[str, str] = {}
        for paragraph in paragraphs:
            if not isinstance(paragraph, Mapping):
                raise ValueError("TAT-QA paragraph must be an object")
            order = str(paragraph.get("order", "")).strip()
            uid = str(paragraph.get("uid", "")).strip()
            text = str(paragraph.get("text", "")).strip()
            if not order or not uid or not text:
                raise ValueError("TAT-QA paragraph requires order, uid, and text")
            paragraph_id = f"tatqa:{table_uid}:paragraph:{order}"
            paragraph_ids[order] = paragraph_id
            documents.append(
                EvalCorpusDocument(
                    document_id=paragraph_id,
                    text=text,
                    source_uri=f"tatqa://{table_uid}/paragraph/{order}",
                    metadata={"kind": "paragraph", "uid": uid, "order": order},
                )
            )
        for question in questions:
            if limit is not None and len(cases) >= limit:
                break
            if not isinstance(question, Mapping):
                raise ValueError("TAT-QA question must be an object")
            question_id = str(question.get("uid", "")).strip()
            query = str(question.get("question", "")).strip()
            answer_from = str(question.get("answer_from", "")).strip()
            relevant: list[str] = []
            if "table" in answer_from:
                relevant.append(table_id)
            raw_orders = question.get("rel_paragraphs", [])
            if isinstance(raw_orders, list):
                for raw_order in raw_orders:
                    paragraph_id = paragraph_ids.get(str(raw_order))
                    if paragraph_id and paragraph_id not in relevant:
                        relevant.append(paragraph_id)
            if not relevant and answer_from == "text":
                # Corrupt/underspecified relevance must not silently become an
                # impossible retrieval case.
                continue
            category = "table" if "table" in answer_from else "text"
            answer_type = str(question.get("answer_type", "")).strip()
            if answer_type in {"arithmetic", "count", "multi-span"}:
                category = answer_type
            cases.append(
                RAGEvalCase(
                    case_id=f"tatqa:{question_id}",
                    dataset="TAT-QA",
                    query=query,
                    relevant_ids=relevant,
                    category=category,
                    answer=question.get("answer"),
                    metadata={
                        "answer_from": answer_from,
                        "answer_type": answer_type,
                        "derivation": question.get("derivation", ""),
                        "scale": question.get("scale", ""),
                    },
                )
            )
        if limit is not None and len(cases) >= limit:
            break
    return RAGEvalDataset(
        dataset="TAT-QA",
        license="CC BY 4.0",
        attribution_url="https://github.com/NExTplusplus/TAT-QA",
        documents=documents,
        cases=cases,
    )


def _parse_page_list(value: object) -> list[int]:
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        cleaned = value.strip()
        try:
            parsed = json.loads(cleaned.replace("'", '"'))
        except json.JSONDecodeError as exc:
            raise ValueError("invalid evidence_pages value") from exc
        raw = parsed if isinstance(parsed, list) else []
    else:
        raw = []
    pages = [int(item) for item in raw]
    if any(page < 0 for page in pages):
        raise ValueError("evidence pages must be non-negative")
    return pages


def load_mmlongbench_cases(
    path: str | Path,
    *,
    limit: int | None = None,
) -> RAGEvalDataset:
    """Load MMLongBench-Doc labels; PDF pages are indexed by another adapter."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("MMLongBench-Doc payload must be a JSON array")
    cases: list[RAGEvalCase] = []
    for index, sample in enumerate(raw):
        if limit is not None and len(cases) >= limit:
            break
        if not isinstance(sample, Mapping):
            raise ValueError("MMLongBench-Doc sample must be an object")
        doc_id = str(sample.get("doc_id", "")).strip()
        query = str(sample.get("question", "")).strip()
        pages = _parse_page_list(sample.get("evidence_pages", []))
        if not doc_id or not query or not pages:
            continue
        sources = str(sample.get("evidence_sources", ""))
        category = "cross-page" if len(pages) > 1 else "single-page"
        if "Table" in sources:
            category = "cross-page-table" if len(pages) > 1 else "table"
        cases.append(
            RAGEvalCase(
                case_id=f"mmlongbench:{index}:{doc_id}",
                dataset="MMLongBench-Doc",
                query=query,
                relevant_ids=[
                    f"mmlongbench:{doc_id}:page:{page}" for page in pages
                ],
                category=category,
                answer=sample.get("answer"),
                metadata={
                    "doc_id": doc_id,
                    "doc_type": sample.get("doc_type"),
                    "evidence_pages": pages,
                    "evidence_sources": sources,
                    "answer_format": sample.get("answer_format"),
                },
            )
        )
    return RAGEvalDataset(
        dataset="MMLongBench-Doc",
        license="CC BY-NC 4.0 (research use only)",
        attribution_url="https://github.com/mayubo2333/MMLongBench-Doc",
        cases=cases,
    )


_ANSWER_TOKEN = re.compile(r"[A-Za-z0-9]+|[\u3400-\u9fff]")


def normalize_answer(value: object) -> str:
    text = str(value).casefold().replace(",", "")
    return " ".join(_ANSWER_TOKEN.findall(text))


def answer_exact_match(prediction: object, expected: object) -> float:
    candidates = expected if isinstance(expected, list) else [expected]
    normalized = normalize_answer(prediction)
    return float(any(normalized == normalize_answer(item) for item in candidates))


def answer_token_f1(prediction: object, expected: object) -> float:
    candidates = expected if isinstance(expected, list) else [expected]
    predicted = normalize_answer(prediction).split()
    if not predicted:
        return 0.0
    best = 0.0
    for candidate in candidates:
        gold = normalize_answer(candidate).split()
        if not gold:
            continue
        remaining = list(gold)
        common = 0
        for token in predicted:
            if token in remaining:
                common += 1
                remaining.remove(token)
        if common:
            precision = common / len(predicted)
            recall = common / len(gold)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


__all__ = [
    "EvalCorpusDocument",
    "RAGEvalCase",
    "RAGEvalDataset",
    "RetrievalCaseMetrics",
    "RetrievalEvaluationReport",
    "RetrievalEvaluationSummary",
    "RetrievalPrediction",
    "answer_exact_match",
    "answer_token_f1",
    "evaluate_retrieval",
    "load_mmlongbench_cases",
    "load_tatqa_dataset",
    "normalize_answer",
]

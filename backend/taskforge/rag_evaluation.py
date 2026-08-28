"""Deterministic retrieval and answer evaluation for TaskForge RAG pipelines.

The evaluator is deliberately independent from model providers and vector
stores.  It consumes stable document/evidence identifiers so the same cases
can compare lexical, dense, hybrid, reranked, and graph-assisted retrievers.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from .domain import StrictModel
from .tatqa_table_cleaning import clean_tatqa_table


class EvalCorpusDocument(StrictModel):
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GoldEvidenceUnit(StrictModel):
    """One annotated evidence paragraph with one or more equivalent locations.

    QASPER annotations identify evidence by paragraph text.  The same text can
    legitimately occur at more than one location in a paper, so locations are
    alternatives for a single denominator unit rather than separate relevant
    items.
    """

    unit_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    alternative_paragraph_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def alternative_locations_are_unique(self) -> GoldEvidenceUnit:
        if any(not item.strip() for item in self.alternative_paragraph_ids):
            raise ValueError("alternative paragraph IDs must be non-empty")
        if len(self.alternative_paragraph_ids) != len(
            set(self.alternative_paragraph_ids)
        ):
            raise ValueError("alternative paragraph IDs must not contain duplicates")
        return self


class GoldEvidenceSet(StrictModel):
    """The complete evidence set supplied by one answerable annotator."""

    annotation_id: str = Field(min_length=1)
    units: list[GoldEvidenceUnit] = Field(min_length=1)

    @model_validator(mode="after")
    def evidence_units_are_unique(self) -> GoldEvidenceSet:
        unit_ids = [item.unit_id for item in self.units]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("gold evidence unit IDs must not contain duplicates")
        return self


class QasperGoldLabels(StrictModel):
    """All independently valid evidence annotations for one QASPER question."""

    evidence_sets: list[GoldEvidenceSet] = Field(min_length=1)

    @model_validator(mode="after")
    def annotation_ids_are_unique(self) -> QasperGoldLabels:
        annotation_ids = [item.annotation_id for item in self.evidence_sets]
        if len(annotation_ids) != len(set(annotation_ids)):
            raise ValueError("QASPER annotation IDs must not contain duplicates")
        return self


class RAGEvalCase(StrictModel):
    case_id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    query: str = Field(min_length=1)
    relevant_ids: list[str] = Field(min_length=1)
    category: str = Field(min_length=1)
    answer: str | list[str] | float | int | None = None
    qasper_gold: QasperGoldLabels | None = None
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
    retrieved_parent_ids: list[str] = Field(default_factory=list)
    retrieved_row_ids: list[str] = Field(default_factory=list)
    retrieved_cell_ids: list[str] = Field(default_factory=list)
    retrieved_complete_table_ids: list[str] = Field(default_factory=list)
    retrieved_table_units_by_hit: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def retrieved_ids_are_unique(self) -> RetrievalPrediction:
        if len(self.retrieved_ids) != len(set(self.retrieved_ids)):
            raise ValueError("retrieved_ids must not contain duplicates")
        for field_name in (
            "retrieved_parent_ids",
            "retrieved_row_ids",
            "retrieved_cell_ids",
            "retrieved_complete_table_ids",
        ):
            identifiers = getattr(self, field_name)
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"{field_name} must not contain duplicates")
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


class CitedAnswerPrediction(StrictModel):
    """Host-observed answer, retrieval, presentation, and model citation IDs."""

    case_id: str = Field(min_length=1)
    answer: str = ""
    retrieved_ids: list[str] = Field(default_factory=list)
    presented_evidence_ids: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    parse_error: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def evidence_identifiers_are_coherent(self) -> CitedAnswerPrediction:
        for field_name in (
            "retrieved_ids",
            "presented_evidence_ids",
            "citation_ids",
        ):
            identifiers = getattr(self, field_name)
            if any(not identifier.strip() for identifier in identifiers):
                raise ValueError(f"{field_name} must contain non-empty identifiers")
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"{field_name} must not contain duplicates")
        if not set(self.presented_evidence_ids).issubset(self.retrieved_ids):
            raise ValueError("presented_evidence_ids must be a subset of retrieved_ids")
        return self


class AnswerGroundingCaseMetrics(StrictModel):
    """Deterministic gold-evidence grounding metrics for one short answer."""

    case_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    citation_precision: float = Field(ge=0.0, le=1.0)
    citation_recall: float = Field(ge=0.0, le=1.0)
    retrieval_to_citation_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    exact_match: float = Field(ge=0.0, le=1.0)
    strict_supported_claim: bool
    unsupported_claim: bool
    parse_failed: bool = False
    missing_prediction: bool = False
    valid_citation_ids: list[str] = Field(default_factory=list)
    invalid_citation_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def support_flags_are_complements(self) -> AnswerGroundingCaseMetrics:
        if self.strict_supported_claim == self.unsupported_claim:
            raise ValueError("support flags must be complements")
        return self


class AnswerGroundingEvaluationSummary(StrictModel):
    total_cases: int = Field(ge=0)
    missing_predictions: int = Field(ge=0)
    parse_failures: int = Field(ge=0)
    parse_failure_rate: float = Field(ge=0.0, le=1.0)
    citation_precision: float = Field(ge=0.0, le=1.0)
    citation_recall: float = Field(ge=0.0, le=1.0)
    retrieval_to_citation_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    retrieval_to_citation_eligible_cases: int = Field(ge=0)
    exact_match_accuracy: float = Field(ge=0.0, le=1.0)
    strict_supported_claim_rate: float = Field(ge=0.0, le=1.0)
    strict_unsupported_claim_rate: float = Field(ge=0.0, le=1.0)


class AnswerGroundingEvaluationReport(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    cases: list[AnswerGroundingCaseMetrics]
    summary: AnswerGroundingEvaluationSummary


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


_WEAK_OPERAND_TOKEN = re.compile(
    r"(?:[$€£¥]\s*)?-?\d[\d,]*(?:\.\d+)?%?|[A-Za-z]{2,}|[\u3400-\u9fff]"
)
_WEAK_OPERAND_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "what",
        "which",
        "how",
        "much",
        "many",
        "was",
        "were",
        "are",
        "is",
        "of",
        "in",
        "for",
        "to",
        "from",
        "with",
        "on",
        "by",
        "than",
        "that",
        "this",
        "their",
        "they",
        "does",
        "did",
        "have",
        "has",
        "had",
    }
)


def _normalise_weak_operand_token(token: str) -> str:
    cleaned = token.casefold().replace(",", "").replace(" ", "")
    if cleaned.startswith(("$", "€", "£", "¥")):
        cleaned = cleaned[1:]
    return cleaned


def _weak_operand_terms(case: RAGEvalCase) -> set[str]:
    """Extract answer/derivation terms for a diagnostic, not gold-cell score.

    TAT-QA exposes evidence document IDs but not a canonical cell-coordinate
    annotation.  Numeric operands and longer answer words provide a useful
    weak signal for table/arithmetical failures while remaining explicitly
    separate from strict evidence Recall.
    """

    answer = case.answer
    answer_values = answer if isinstance(answer, list) else [answer]
    raw = " ".join(
        str(value)
        for value in [*answer_values, case.metadata.get("derivation", "")]
        if value is not None
    )
    terms: set[str] = set()
    for token in _WEAK_OPERAND_TOKEN.findall(raw.casefold()):
        cleaned = _normalise_weak_operand_token(token)
        if cleaned.replace(".", "", 1).replace("-", "", 1).isdigit():
            terms.add(cleaned)
        elif len(cleaned) >= 3 and cleaned not in _WEAK_OPERAND_STOPWORDS:
            terms.add(cleaned)
    return terms


def _table_unit_terms(value: object) -> set[str]:
    return {
        _normalise_weak_operand_token(token)
        for token in _WEAK_OPERAND_TOKEN.findall(str(value).casefold())
    }


def _table_unit_gold(
    case: RAGEvalCase,
    document_map: Mapping[str, EvalCorpusDocument],
) -> tuple[set[str], set[str], bool]:
    """Derive weak row/cell labels from answer/derivation overlap.

    The upstream TAT-QA JSON identifies table/paragraph evidence but does not
    publish canonical cell coordinates.  A table unit is therefore eligible
    only when answer/derivation terms match at least one cell; otherwise the
    case is counted as ambiguous and excluded from row/cell macro averages.
    """

    terms = _weak_operand_terms(case)
    if not terms:
        return set(), set(), True
    row_ids: set[str] = set()
    cell_ids: set[str] = set()
    has_table = False
    for evidence_id in case.relevant_ids:
        document = document_map.get(evidence_id)
        if document is None or document.metadata.get("kind") != "table":
            continue
        has_table = True
        raw_rows = document.metadata.get("table_rows")
        if not isinstance(raw_rows, list):
            continue
        rows = [
            list(row) if isinstance(row, (list, tuple)) else [row]
            for row in raw_rows
        ]
        if not rows:
            continue
        width = max(len(row) for row in rows)
        header = [
            str(rows[0][index]).strip() if index < len(rows[0]) else ""
            for index in range(width)
        ]
        for row_index, row in enumerate(rows[1:], start=1):
            row_terms = _table_unit_terms(" ".join(str(value) for value in row))
            matched_row = bool(terms.intersection(row_terms))
            for column_index in range(width):
                value = row[column_index] if column_index < len(row) else ""
                cell_text = " | ".join(
                    (
                        header[column_index],
                        str(value),
                        str(row[0]) if row else "",
                    )
                )
                cell_terms = _table_unit_terms(cell_text)
                if terms.intersection(cell_terms):
                    matched_row = True
                    cell_ids.add(
                        f"{document.document_id}::cell::{row_index}::{column_index}"
                    )
            if matched_row:
                row_ids.add(f"{document.document_id}::row::{row_index}")
    if not has_table:
        return set(), set(), False
    return row_ids, cell_ids, not row_ids and not cell_ids


def _table_unit_catalog(document: EvalCorpusDocument) -> tuple[set[str], set[str]]:
    raw_rows = document.metadata.get("table_rows")
    if document.metadata.get("kind") != "table" or not isinstance(raw_rows, list):
        return set(), set()
    width = max(
        (len(row) for row in raw_rows if isinstance(row, (list, tuple))),
        default=0,
    )
    row_ids = {
        f"{document.document_id}::row::{row_index}"
        for row_index in range(1, len(raw_rows))
    }
    cell_ids = {
        f"{document.document_id}::cell::{row_index}::{column_index}"
        for row_index in range(1, len(raw_rows))
        for column_index in range(width)
    }
    return row_ids, cell_ids


def evaluate_hierarchical_retrieval(
    cases: Sequence[RAGEvalCase],
    predictions: Sequence[RetrievalPrediction],
    documents: Sequence[EvalCorpusDocument],
    *,
    retrieved_texts_by_case: Mapping[str, Sequence[str]] | None = None,
    ks: Sequence[int] = (10, 50),
) -> dict[str, Any]:
    """Return parent and weak operand diagnostics alongside strict Recall.

    ``parent_recall_at_k`` is exact at the parent-document level.  The
    ``weak_operand_recall_at_k`` field is intentionally diagnostic: it checks
    whether answer/derivation terms occur in retrieved text and is *not* a
    substitute for official TAT-QA cell-coordinate labels.
    """

    safe_ks = _safe_ks(ks)
    case_ids = [case.case_id for case in cases]
    known_case_ids = set(case_ids)
    prediction_map: dict[str, RetrievalPrediction] = {}
    for prediction in predictions:
        if prediction.case_id not in known_case_ids:
            raise ValueError(f"prediction references unknown case: {prediction.case_id}")
        if prediction.case_id in prediction_map:
            raise ValueError(f"duplicate prediction for case: {prediction.case_id}")
        prediction_map[prediction.case_id] = prediction
    document_map = {document.document_id: document for document in documents}
    weak_texts = retrieved_texts_by_case or {}

    def average(values: Iterable[float]) -> float:
        materialized = list(values)
        return sum(materialized) / len(materialized) if materialized else 0.0

    parent_values: dict[str, list[float]] = {str(k): [] for k in safe_ks}
    table_values: dict[str, list[float]] = {str(k): [] for k in safe_ks}
    paragraph_values: dict[str, list[float]] = {str(k): [] for k in safe_ks}
    row_values: dict[str, list[float]] = {str(k): [] for k in safe_ks}
    cell_values: dict[str, list[float]] = {str(k): [] for k in safe_ks}
    full_values: dict[str, list[float]] = {str(k): [] for k in safe_ks}
    operand_values: dict[str, list[float]] = {str(k): [] for k in safe_ks}
    category_candidate_values: dict[str, list[float]] = {}
    operand_eligible = 0
    row_eligible = 0
    cell_eligible = 0
    ambiguous_table_units = 0
    for case in cases:
        gold_parents = {
            str(document_map[evidence_id].metadata.get("parent_document_id", evidence_id))
            for evidence_id in case.relevant_ids
            if evidence_id in document_map
        }
        prediction = prediction_map.get(case.case_id)
        retrieved_parents = [] if prediction is None else prediction.retrieved_parent_ids
        retrieved_ids = [] if prediction is None else prediction.retrieved_ids
        retrieved_rows = [] if prediction is None else prediction.retrieved_row_ids
        retrieved_cells = [] if prediction is None else prediction.retrieved_cell_ids
        complete_tables = (
            []
            if prediction is None
            else prediction.retrieved_complete_table_ids
        )
        for k in safe_ks:
            parent_values[str(k)].append(_recall(gold_parents, retrieved_parents, k))
            full_values[str(k)].append(_recall(set(case.relevant_ids), retrieved_ids, k))

        table_gold = {
            evidence_id
            for evidence_id in case.relevant_ids
            if document_map.get(evidence_id, None) is not None
            and document_map[evidence_id].metadata.get("kind") == "table"
        }
        paragraph_gold = {
            evidence_id
            for evidence_id in case.relevant_ids
            if document_map.get(evidence_id, None) is not None
            and document_map[evidence_id].metadata.get("kind") == "paragraph"
        }
        for k in safe_ks:
            if table_gold:
                table_values[str(k)].append(_recall(table_gold, retrieved_ids, k))
            if paragraph_gold:
                paragraph_values[str(k)].append(
                    _recall(paragraph_gold, retrieved_ids, k)
                )

        gold_rows, gold_cells, ambiguous = _table_unit_gold(case, document_map)
        if ambiguous:
            ambiguous_table_units += 1
        if gold_rows:
            row_eligible += 1
        if gold_cells:
            cell_eligible += 1
        for k in safe_ks:
            covered_rows = list(retrieved_rows[:k])
            covered_cells = list(retrieved_cells[:k])
            for table_id in complete_tables:
                if table_id not in retrieved_ids[:k]:
                    continue
                table_document = document_map.get(table_id)
                if table_document is None:
                    continue
                all_rows, all_cells = _table_unit_catalog(table_document)
                covered_rows.extend(sorted(all_rows))
                covered_cells.extend(sorted(all_cells))
            if gold_rows:
                row_values[str(k)].append(_recall(gold_rows, covered_rows, len(covered_rows)))
            if gold_cells:
                cell_values[str(k)].append(
                    _recall(gold_cells, covered_cells, len(covered_cells))
                )

        terms = _weak_operand_terms(case)
        if terms:
            operand_eligible += 1
        texts = list(weak_texts.get(case.case_id, ()))
        for k in safe_ks:
            if not terms:
                operand_values[str(k)].append(0.0)
            else:
                normalized_text = " ".join(
                    text.casefold() for text in texts[:k]
                )
                text_terms = set(_WEAK_OPERAND_TOKEN.findall(normalized_text))
                text_terms = {
                    _normalise_weak_operand_token(token) for token in text_terms
                }
                operand_values[str(k)].append(
                    len(terms.intersection(text_terms)) / len(terms)
                )
        category_candidate_values.setdefault(case.category, []).append(
            _recall(set(case.relevant_ids), retrieved_ids, max(safe_ks))
        )

    def averaged(values: Mapping[str, list[float]]) -> dict[str, float | None]:
        return {
            key: (average(items) if items else None)
            for key, items in values.items()
        }

    return {
        "schema_version": "1.0",
        "parent_recall_at_k": averaged(parent_values),
        "table_recall_at_k": averaged(table_values),
        "paragraph_recall_at_k": averaged(paragraph_values),
        "row_recall_at_k": averaged(row_values),
        "cell_recall_at_k": averaged(cell_values),
        "full_evidence_recall_at_k": averaged(full_values),
        "candidate_recall_at_k_by_category": {
            category: average(values)
            for category, values in sorted(category_candidate_values.items())
        },
        "row_eligible_cases": row_eligible,
        "cell_eligible_cases": cell_eligible,
        "ambiguous_table_unit_cases": ambiguous_table_units,
        "weak_operand_recall_at_k": {
            key: average(values) for key, values in operand_values.items()
        },
        "weak_operand_eligible_cases": operand_eligible,
        "weak_operand_definition": (
            "answer/derivation numeric and content-term overlap in retrieved text; "
            "diagnostic only, not official TAT-QA cell recall"
        ),
    }


def evaluate_answer_grounding(
    cases: Sequence[RAGEvalCase],
    predictions: Sequence[CitedAnswerPrediction],
) -> AnswerGroundingEvaluationReport:
    """Score short answers with gold IDs and host-observed evidence only.

    For each case, ``G`` is the dataset's gold evidence, ``R`` is host-observed
    retrieval, ``P`` is the evidence actually presented to the model, and ``C``
    is the model's citation list.  Only ``V = G & R & P & C`` is valid.  The
    model never decides whether its own answer or citation is supported.
    """

    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case identifiers must be unique")
    known_case_ids = set(case_ids)
    prediction_map: dict[str, CitedAnswerPrediction] = {}
    for prediction in predictions:
        if prediction.case_id not in known_case_ids:
            raise ValueError(f"prediction references unknown case: {prediction.case_id}")
        if prediction.case_id in prediction_map:
            raise ValueError(f"duplicate prediction for case: {prediction.case_id}")
        prediction_map[prediction.case_id] = prediction

    rows: list[AnswerGroundingCaseMetrics] = []
    for case in cases:
        prediction = prediction_map.get(case.case_id)
        missing_prediction = prediction is None
        answer = "" if prediction is None else prediction.answer
        retrieved_ids = [] if prediction is None else prediction.retrieved_ids
        presented_ids = (
            [] if prediction is None else prediction.presented_evidence_ids
        )
        citation_ids = [] if prediction is None else prediction.citation_ids

        gold = set(case.relevant_ids)
        retrieved = set(retrieved_ids)
        presented = set(presented_ids)
        valid = gold.intersection(retrieved, presented, citation_ids)
        valid_ids = [identifier for identifier in citation_ids if identifier in valid]
        invalid_ids = [identifier for identifier in citation_ids if identifier not in valid]
        citation_precision = (
            len(valid_ids) / len(citation_ids) if citation_ids else 0.0
        )
        citation_recall = len(valid) / len(gold)
        retrieved_relevant = gold.intersection(retrieved)
        retrieval_to_citation = (
            len(valid) / len(retrieved_relevant) if retrieved_relevant else None
        )
        exact_match = answer_exact_match(answer, case.answer)
        parse_failed = prediction is not None and prediction.parse_error is not None
        strict_supported = (
            prediction is not None
            and not parse_failed
            and exact_match == 1.0
            and bool(valid)
        )
        rows.append(
            AnswerGroundingCaseMetrics(
                case_id=case.case_id,
                category=case.category,
                citation_precision=citation_precision,
                citation_recall=citation_recall,
                retrieval_to_citation_coverage=retrieval_to_citation,
                exact_match=exact_match,
                strict_supported_claim=strict_supported,
                unsupported_claim=not strict_supported,
                parse_failed=parse_failed,
                missing_prediction=missing_prediction,
                valid_citation_ids=valid_ids,
                invalid_citation_ids=invalid_ids,
            )
        )

    total = len(rows)
    retrieval_to_citation_values = [
        row.retrieval_to_citation_coverage
        for row in rows
        if row.retrieval_to_citation_coverage is not None
    ]

    def average(values: Iterable[float]) -> float:
        materialized = list(values)
        return sum(materialized) / len(materialized) if materialized else 0.0

    supported_count = sum(row.strict_supported_claim for row in rows)
    parse_failures = sum(row.parse_failed for row in rows)
    return AnswerGroundingEvaluationReport(
        cases=rows,
        summary=AnswerGroundingEvaluationSummary(
            total_cases=total,
            missing_predictions=sum(row.missing_prediction for row in rows),
            parse_failures=parse_failures,
            parse_failure_rate=parse_failures / total if total else 0.0,
            citation_precision=average(row.citation_precision for row in rows),
            citation_recall=average(row.citation_recall for row in rows),
            retrieval_to_citation_coverage=(
                average(retrieval_to_citation_values)
                if retrieval_to_citation_values
                else None
            ),
            retrieval_to_citation_eligible_cases=len(retrieval_to_citation_values),
            exact_match_accuracy=average(row.exact_match for row in rows),
            strict_supported_claim_rate=supported_count / total if total else 0.0,
            strict_unsupported_claim_rate=(total - supported_count) / total if total else 0.0,
        ),
    )


def _table_text(rows: Sequence[Sequence[object]]) -> str:
    return "\n".join(
        " | ".join(str(cell).strip() for cell in row)
        for row in rows
    ).strip()


def load_tatqa_dataset(
    path: str | Path,
    *,
    limit: int | None = None,
    table_cleaning: bool = False,
) -> RAGEvalDataset:
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
        table_metadata: dict[str, Any] = {
            "kind": "table",
            "table_uid": table_uid,
            "parent_document_id": f"tatqa:{table_uid}:context",
            "table_rows": table_rows,
            "table_cleaning_enabled": table_cleaning,
        }
        table_text = _table_text(table_rows)
        if table_cleaning:
            paragraph_context = " ".join(
                str(paragraph.get("text", ""))
                for paragraph in paragraphs
                if isinstance(paragraph, Mapping)
            )
            cleaned_table = clean_tatqa_table(
                table_rows,
                context_text=paragraph_context,
            )
            cleaned_rows = [list(row) for row in cleaned_table.rows]
            table_text = _table_text(cleaned_rows)
            table_metadata.update(
                {
                    "table_rows_cleaned": cleaned_rows,
                    "table_cleaning": cleaned_table.metadata(),
                }
            )
        parent_id = f"tatqa:{table_uid}:context"
        table_id = f"tatqa:{table_uid}:table"
        documents.append(
            EvalCorpusDocument(
                document_id=table_id,
                text=table_text,
                source_uri=f"tatqa://{table_uid}/table",
                metadata=table_metadata,
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
                    metadata={
                        "kind": "paragraph",
                        "uid": uid,
                        "order": order,
                        "table_uid": table_uid,
                        "parent_document_id": parent_id,
                    },
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
                        "table_uid": table_uid,
                        "table_id": table_id,
                        "parent_document_id": parent_id,
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


def _multihop_document_id(url: str) -> str:
    return f"multihop:{hashlib.sha256(url.encode('utf-8')).hexdigest()[:24]}"


def load_multihop_rag_dataset(
    queries_path: str | Path,
    corpus_path: str | Path,
    *,
    limit: int | None = None,
) -> RAGEvalDataset:
    """Normalize MultiHop-RAG real news into cross-document retrieval cases.

    Evidence entries reference corpus documents by ``url``; each query is
    grounded in 2-4 full articles.  ``null_query`` rows are unanswerable and
    carry no evidence, so they are intentionally excluded from cases.
    """

    raw_queries = json.loads(Path(queries_path).read_text(encoding="utf-8"))
    raw_corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    if not isinstance(raw_queries, list) or not isinstance(raw_corpus, list):
        raise ValueError("MultiHop-RAG payloads must be JSON arrays")

    documents: list[EvalCorpusDocument] = []
    by_url: dict[str, str] = {}
    for doc in raw_corpus:
        if not isinstance(doc, Mapping):
            raise ValueError("MultiHop-RAG corpus entry must be an object")
        url = str(doc.get("url", "")).strip()
        body = str(doc.get("body", "")).strip()
        if not url or not body:
            raise ValueError("MultiHop-RAG corpus entry requires url and body")
        document_id = _multihop_document_id(url)
        by_url[url] = document_id
        documents.append(
            EvalCorpusDocument(
                document_id=document_id,
                text=body,
                source_uri=url,
                metadata={
                    "title": str(doc.get("title", "")).strip(),
                    "source": str(doc.get("source", "")).strip(),
                    "category": str(doc.get("category", "")).strip(),
                    "author": doc.get("author"),
                    "published_at": str(doc.get("published_at", "")).strip(),
                },
            )
        )

    cases: list[RAGEvalCase] = []
    for index, raw_query in enumerate(raw_queries):
        if limit is not None and len(cases) >= limit:
            break
        if not isinstance(raw_query, Mapping):
            raise ValueError("MultiHop-RAG query must be an object")
        question_type = str(raw_query.get("question_type", "")).strip()
        if question_type == "null_query":
            continue
        query = str(raw_query.get("query", "")).strip()
        if not query:
            raise ValueError("MultiHop-RAG query requires non-empty query text")
        evidence = raw_query.get("evidence_list", [])
        if not isinstance(evidence, list):
            raise ValueError("MultiHop-RAG query evidence_list must be an array")
        relevant: list[str] = []
        for entry in evidence:
            if not isinstance(entry, Mapping):
                raise ValueError("MultiHop-RAG evidence entry must be an object")
            url = str(entry.get("url", "")).strip()
            document_id = by_url.get(url)
            if document_id is None:
                raise ValueError("MultiHop-RAG evidence references an unknown corpus URL")
            if document_id not in relevant:
                relevant.append(document_id)
        if not relevant:
            continue
        cases.append(
            RAGEvalCase(
                case_id=f"multihop:q{index}",
                dataset="MultiHop-RAG",
                query=query,
                relevant_ids=relevant,
                category=question_type,
                answer=raw_query.get("answer"),
                metadata={"source_count": len(relevant)},
            )
        )
    return RAGEvalDataset(
        dataset="MultiHop-RAG",
        license="ODC-BY",
        attribution_url="https://huggingface.co/datasets/yixuantt/MultiHopRAG",
        documents=documents,
        cases=cases,
    )


def _qasper_sentence_spans(text: str) -> list[dict[str, int]]:
    spans: list[dict[str, int]] = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+", text):
        end = match.start()
        if end > start:
            spans.append({"start": start, "end": end})
        start = match.end()
    if start < len(text):
        spans.append({"start": start, "end": len(text)})
    return spans


def _qasper_section_level(section_name: str) -> int:
    match = re.match(r"^\s*(\d+(?:\.\d+)*)\.?\s+", section_name)
    return match.group(1).count(".") + 1 if match else 1


def load_qasper_dataset(path: str | Path, *, limit: int | None = None) -> RAGEvalDataset:
    """Normalize QASPER into paragraph-level general-text evidence.

    Only answerable questions with at least one exact evidence paragraph are
    admitted. Evidence is matched against the paper's full text; no gold
    evidence text is injected into the retrieval corpus.
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("QASPER payload must be an object keyed by paper ID")
    documents: list[EvalCorpusDocument] = []
    cases: list[RAGEvalCase] = []
    for paper_id, paper in raw.items():
        if not isinstance(paper_id, str) or not paper_id.strip():
            raise ValueError("QASPER paper ID must be a non-empty string")
        if not isinstance(paper, Mapping):
            raise ValueError("QASPER paper must be an object")
        stable_paper_id = paper_id.strip()
        parent_id = f"qasper:{stable_paper_id}:paper"
        evidence_to_ids: dict[str, list[str]] = {}
        paper_title = str(paper.get("title", "")).strip()
        abstract = str(paper.get("abstract", "")).strip()
        if abstract:
            abstract_id = f"{parent_id}:abstract"
            documents.append(
                EvalCorpusDocument(
                    document_id=abstract_id,
                    text=abstract,
                    source_uri=f"qasper://{stable_paper_id}/abstract",
                    metadata={
                        "kind": "paragraph",
                        "node_type": "abstract",
                        "section": "abstract",
                        "section_title": "Abstract",
                        "subsection_title": None,
                        "section_id": f"{parent_id}:section:abstract",
                        "parent_id": parent_id,
                        "paper_title": paper_title,
                        "paper_id": stable_paper_id,
                        "source": "qasper",
                        "parent_document_id": parent_id,
                        "paragraph_index": 0,
                        "previous_document_id": None,
                        "next_document_id": None,
                        "char_start": 0,
                        "char_end": len(abstract),
                        "sentence_spans": _qasper_sentence_spans(abstract),
                    },
                )
            )
            evidence_to_ids.setdefault(abstract, []).append(abstract_id)
        full_text = paper.get("full_text")
        if not isinstance(full_text, list):
            raise ValueError("QASPER paper full_text must be an array")
        paragraph_records: list[dict[str, Any]] = []
        char_cursor = 0
        for section_index, section in enumerate(full_text):
            if not isinstance(section, Mapping):
                raise ValueError("QASPER section must be an object")
            section_name = str(section.get("section_name", "")).strip()
            paragraphs = section.get("paragraphs")
            if not isinstance(paragraphs, list):
                raise ValueError("QASPER section paragraphs must be an array")
            for paragraph_index, raw_text in enumerate(paragraphs):
                text = str(raw_text).strip()
                if not text or not re.search(
                    r"[A-Za-z0-9_]|[\u3400-\u4dbf\u4e00-\u9fff]", text
                ):
                    continue
                document_id = (
                    f"{parent_id}:section:{section_index}:paragraph:{paragraph_index}"
                )
                paragraph_records.append(
                    {
                        "document_id": document_id,
                        "text": text,
                        "section_name": section_name,
                        "section_index": section_index,
                        "paragraph_index": paragraph_index,
                        "char_start": char_cursor,
                        "char_end": char_cursor + len(text),
                    }
                )
                char_cursor += len(text) + 2
                evidence_to_ids.setdefault(text, []).append(document_id)
        for record_index, record in enumerate(paragraph_records):
            document_id = str(record["document_id"])
            text = str(record["text"])
            section_name = str(record["section_name"])
            section_index = int(record["section_index"])
            section_level = _qasper_section_level(section_name)
            documents.append(
                EvalCorpusDocument(
                    document_id=document_id,
                    text=text,
                    source_uri=(
                        f"qasper://{stable_paper_id}/section/"
                        f"{section_index}/paragraph/{record['paragraph_index']}"
                    ),
                    metadata={
                        "kind": "paragraph",
                        "node_type": "paragraph",
                        "section": section_name,
                        "section_title": section_name,
                        "subsection_title": (
                            section_name if section_level > 1 else None
                        ),
                        "section_level": section_level,
                        "section_id": f"{parent_id}:section:{section_index}",
                        "parent_id": f"{parent_id}:section:{section_index}",
                        "section_index": section_index,
                        "paragraph_index": int(record["paragraph_index"]),
                        "paper_title": paper_title,
                        "paper_id": stable_paper_id,
                        "source": "qasper",
                        "parent_document_id": parent_id,
                        "previous_document_id": (
                            str(paragraph_records[record_index - 1]["document_id"])
                            if record_index > 0
                            else None
                        ),
                        "next_document_id": (
                            str(paragraph_records[record_index + 1]["document_id"])
                            if record_index + 1 < len(paragraph_records)
                            else None
                        ),
                        "char_start": int(record["char_start"]),
                        "char_end": int(record["char_end"]),
                        "sentence_spans": _qasper_sentence_spans(text),
                    },
                )
            )
        figures_and_tables = paper.get("figures_and_tables", [])
        if figures_and_tables is None:
            figures_and_tables = []
        if not isinstance(figures_and_tables, list):
            raise ValueError("QASPER figures_and_tables must be an array")
        for float_index, raw_float in enumerate(figures_and_tables):
            if not isinstance(raw_float, Mapping):
                raise ValueError("QASPER figure/table entry must be an object")
            caption = str(raw_float.get("caption", "")).strip()
            if not caption:
                continue
            file_name = str(raw_float.get("file", "")).strip()
            lowered = f"{file_name} {caption}".casefold()
            kind = "table" if "table" in lowered else "figure"
            document_id = f"{parent_id}:float:{float_index}"
            documents.append(
                EvalCorpusDocument(
                    document_id=document_id,
                    text=caption,
                    source_uri=(
                        f"qasper://{stable_paper_id}/{kind}/{float_index}"
                    ),
                    metadata={
                        "kind": kind,
                        "node_type": f"{kind}_caption",
                        "section": kind,
                        "section_title": (
                            "Tables" if kind == "table" else "Figures"
                        ),
                        "subsection_title": None,
                        "section_id": f"{parent_id}:section:{kind}s",
                        "parent_id": parent_id,
                        "paper_title": paper_title,
                        "paper_id": stable_paper_id,
                        "source": "qasper",
                        "parent_document_id": parent_id,
                        "float_index": float_index,
                        "float_file": file_name,
                        "previous_document_id": None,
                        "next_document_id": None,
                        "char_start": 0,
                        "char_end": len(caption),
                        "sentence_spans": _qasper_sentence_spans(caption),
                    },
                )
            )
            # QASPER uses this marker in evidence annotations to point at a
            # figure/table caption.  It is an alignment alias only; the marker
            # is never injected into the indexed caption text.
            for evidence_alias in (caption, f"FLOAT SELECTED: {caption}"):
                evidence_to_ids.setdefault(evidence_alias, []).append(document_id)
        qas = paper.get("qas")
        if not isinstance(qas, list):
            raise ValueError("QASPER paper qas must be an array")
        for question in qas:
            if limit is not None and len(cases) >= limit:
                break
            if not isinstance(question, Mapping):
                raise ValueError("QASPER question must be an object")
            query = str(question.get("question", "")).strip()
            question_id = str(question.get("question_id", "")).strip()
            answers = question.get("answers")
            if not query or not question_id or not isinstance(answers, list):
                raise ValueError(
                    "QASPER question requires question, question_id, and answers"
                )
            gold_sets: list[GoldEvidenceSet] = []
            annotations: list[Mapping[str, Any]] = []
            seen_annotation_ids: set[str] = set()
            for answer_index, item in enumerate(answers):
                if not isinstance(item, Mapping):
                    continue
                annotation = item.get("answer")
                if (
                    not isinstance(annotation, Mapping)
                    or annotation.get("unanswerable", False)
                ):
                    continue
                raw_evidence = annotation.get("evidence", [])
                if not isinstance(raw_evidence, list):
                    continue
                evidence_texts = list(
                    dict.fromkeys(
                        text
                        for value in raw_evidence
                        if (text := str(value).strip())
                    )
                )
                # A legal corpus-native annotation must resolve every evidence
                # paragraph exactly against the official full text.  Never
                # silently shorten a partially unresolved annotation because
                # that would make its Recall denominator easier.
                if not evidence_texts or any(
                    text not in evidence_to_ids for text in evidence_texts
                ):
                    continue
                raw_annotation_id = str(
                    item.get("annotation_id") or f"annotation-{answer_index}"
                ).strip()
                annotation_id = raw_annotation_id
                duplicate_suffix = 2
                while annotation_id in seen_annotation_ids:
                    annotation_id = f"{raw_annotation_id}-{duplicate_suffix}"
                    duplicate_suffix += 1
                seen_annotation_ids.add(annotation_id)
                units: list[GoldEvidenceUnit] = []
                for evidence_index, evidence_text in enumerate(evidence_texts):
                    digest = hashlib.sha256(evidence_text.encode("utf-8")).hexdigest()[:16]
                    units.append(
                        GoldEvidenceUnit(
                            unit_id=(
                                f"qasper:{stable_paper_id}:{question_id}:"
                                f"{annotation_id}:evidence:{evidence_index}:{digest}"
                            ),
                            text=evidence_text,
                            alternative_paragraph_ids=list(
                                dict.fromkeys(evidence_to_ids[evidence_text])
                            ),
                        )
                    )
                gold_sets.append(
                    GoldEvidenceSet(annotation_id=annotation_id, units=units)
                )
                annotations.append(annotation)
            if not gold_sets:
                continue
            # Transitional compatibility for generic retrieval components that
            # still require one list of document IDs.  Strict QASPER scoring
            # below and the direct-upload evaluator consume ``qasper_gold`` and
            # never use this first-annotation projection.
            relevant = [
                unit.alternative_paragraph_ids[0]
                for unit in gold_sets[0].units
            ]
            annotation = annotations[0]
            answer: object = annotation.get("free_form_answer", "")
            if not answer:
                spans = annotation.get("extractive_spans", [])
                answer = (
                    spans
                    if isinstance(spans, list) and spans
                    else annotation.get("yes_no")
                )
            cases.append(
                RAGEvalCase(
                    case_id=f"qasper:{stable_paper_id}:{question_id}",
                    dataset="QASPER",
                    query=query,
                    relevant_ids=relevant,
                    category="text",
                    answer=answer,
                    qasper_gold=QasperGoldLabels(evidence_sets=gold_sets),
                    metadata={
                        "paper_id": stable_paper_id,
                        "parent_document_id": parent_id,
                        "question_id": question_id,
                        "evidence_count": len(gold_sets[0].units),
                        "gold_annotation_count": len(gold_sets),
                        "legacy_relevant_ids_projection": True,
                    },
                )
            )
        if limit is not None and len(cases) >= limit:
            break
    return RAGEvalDataset(
        dataset="QASPER",
        license="CC BY 4.0",
        attribution_url="https://huggingface.co/datasets/allenai/qasper",
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
    "AnswerGroundingCaseMetrics",
    "AnswerGroundingEvaluationReport",
    "AnswerGroundingEvaluationSummary",
    "CitedAnswerPrediction",
    "EvalCorpusDocument",
    "GoldEvidenceSet",
    "GoldEvidenceUnit",
    "QasperGoldLabels",
    "RAGEvalCase",
    "RAGEvalDataset",
    "RetrievalCaseMetrics",
    "RetrievalEvaluationReport",
    "RetrievalEvaluationSummary",
    "RetrievalPrediction",
    "answer_exact_match",
    "answer_token_f1",
    "evaluate_answer_grounding",
    "evaluate_hierarchical_retrieval",
    "evaluate_retrieval",
    "load_mmlongbench_cases",
    "load_multihop_rag_dataset",
    "load_qasper_dataset",
    "load_tatqa_dataset",
    "normalize_answer",
]

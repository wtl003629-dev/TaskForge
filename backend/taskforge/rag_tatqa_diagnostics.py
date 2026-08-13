"""Oracle diagnostics and deterministic query plans for TAT-QA retrieval.

The oracle scores are intentionally diagnostic upper bounds.  They never feed
gold labels into the production retriever or into a promotion result.  They
answer a narrower question: is a miss caused by parent routing, section
representation, candidate generation, or only by Top-10 ordering?
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .rag_evaluation import EvalCorpusDocument, RAGEvalCase

_CHUNK_SEP = "::chunk::"
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?(?![A-Za-z])")
_COMPARATORS: tuple[tuple[str, str], ...] = (
    ("at least", "gte"),
    ("at most", "lte"),
    ("greater than", "gt"),
    ("more than", "gt"),
    ("less than", "lt"),
    ("lower than", "lt"),
    ("higher than", "gt"),
    ("decrease", "decrease"),
    ("increase", "increase"),
    ("compare", "compare"),
)


@dataclass(frozen=True)
class TATQARetrievalRow:
    case_id: str
    category: str
    relevant_ids: tuple[str, ...]
    retrieved_ids: tuple[str, ...]
    retrieved_parent_ids: tuple[str, ...] = ()


def _base_id(value: str) -> str:
    return value.split(_CHUNK_SEP, 1)[0]


def _unit_key(
    identifier: str, documents: Mapping[str, EvalCorpusDocument]
) -> tuple[str, str] | None:
    document = documents.get(identifier) or documents.get(_base_id(identifier))
    if document is None:
        return None
    kind = str(document.metadata.get("kind", "unknown"))
    parent = str(document.metadata.get("parent_document_id", document.document_id))
    return parent, kind


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _recall(gold: set[str], retrieved: Sequence[str]) -> float:
    return len(gold.intersection(retrieved)) / len(gold) if gold else 0.0


def _coverage_state(gold: set[str], retrieved: Sequence[str]) -> str:
    hit_count = len(gold.intersection(retrieved))
    if hit_count == 0:
        return "zero"
    if hit_count == len(gold):
        return "complete"
    return "partial"


def _diagnostic_case(
    case: RAGEvalCase,
    row: TATQARetrievalRow,
    documents: Mapping[str, EvalCorpusDocument],
    *,
    top_k: int,
    candidate_k: int,
) -> dict[str, Any]:
    gold = set(case.relevant_ids)
    top = row.retrieved_ids[:top_k]
    candidate = row.retrieved_ids[:candidate_k]
    gold_parents = {
        str(documents[evidence_id].metadata.get("parent_document_id", evidence_id))
        for evidence_id in gold
        if evidence_id in documents
    }
    retrieved_parents = set(row.retrieved_parent_ids)
    if not retrieved_parents:
        for identifier in candidate:
            document = documents.get(identifier) or documents.get(_base_id(identifier))
            if document is not None:
                retrieved_parents.add(
                    str(document.metadata.get("parent_document_id", identifier))
                )
    gold_sections = {
        _unit_key(evidence_id, documents)
        for evidence_id in gold
        if _unit_key(evidence_id, documents) is not None
    }
    retrieved_sections = {
        key
        for identifier in candidate
        if (key := _unit_key(identifier, documents)) is not None
    }
    candidate_set = set(candidate)
    top_set = set(top)
    missing_reasons: dict[str, str] = {}
    for evidence_id in sorted(gold.difference(candidate_set)):
        document = documents.get(evidence_id) or documents.get(_base_id(evidence_id))
        if document is None:
            reason = "unmapped_gold_evidence"
        else:
            parent_id = str(
                document.metadata.get("parent_document_id", document.document_id)
            )
            section = _unit_key(evidence_id, documents)
            if parent_id not in retrieved_parents:
                reason = "parent_unreached"
            elif section not in retrieved_sections:
                reason = "section_kind_unreached"
            else:
                reason = "unit_not_selected"
        missing_reasons[evidence_id] = reason
    parent_access = 1.0 if gold_parents.intersection(retrieved_parents) else 0.0
    section_access = (
        len(gold_sections.intersection(retrieved_sections)) / len(gold_sections)
        if gold_sections
        else 0.0
    )
    # O3 gives a perfect Top-10 reorder to whatever is already in Candidate@K.
    # It is an upper bound, not a prediction of a learned reranker.
    candidate_hits = len(gold.intersection(candidate))
    oracle_top10 = 1.0 if candidate_hits == len(gold) else candidate_hits / len(gold)
    return {
        "case_id": case.case_id,
        "category": case.category,
        "real_recall_at_10": _recall(gold, top),
        "real_candidate_recall": _recall(gold, candidate),
        "top10_state": _coverage_state(gold, top),
        "candidate_state": _coverage_state(gold, candidate),
        "gold_evidence_count": len(gold),
        "top10_hit_count": len(gold.intersection(top_set)),
        "candidate_hit_count": len(gold.intersection(candidate_set)),
        "candidate_missing_evidence_ids": sorted(gold.difference(candidate_set)),
        "candidate_missing_reasons": missing_reasons,
        "gold_parent_to_real_child": parent_access,
        "gold_section_to_real_unit": section_access,
        "oracle_top10_from_candidate": oracle_top10,
        "gold_parent_count": len(gold_parents),
        "gold_section_count": len(gold_sections),
    }


def diagnose_tatqa_retrieval(
    cases: Sequence[RAGEvalCase],
    documents: Sequence[EvalCorpusDocument],
    rows: Sequence[TATQARetrievalRow],
    *,
    top_k: int = 10,
    candidate_k: int = 50,
) -> dict[str, Any]:
    """Return O0/O1/O2/O3 diagnostics for already-produced predictions."""
    if top_k <= 0 or candidate_k < top_k:
        raise ValueError("candidate_k must be at least top_k and both must be positive")
    case_by_id = {case.case_id: case for case in cases}
    row_by_id = {row.case_id: row for row in rows}
    if len(case_by_id) != len(cases) or len(row_by_id) != len(rows):
        raise ValueError("cases and rows must have unique IDs")
    if set(case_by_id) != set(row_by_id):
        raise ValueError("cases and prediction rows must have identical IDs")
    document_map = {document.document_id: document for document in documents}
    per_case = [
        _diagnostic_case(
            case_by_id[row.case_id],
            row,
            document_map,
            top_k=top_k,
            candidate_k=candidate_k,
        )
        for row in rows
    ]

    def aggregate(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            key: _mean([float(row[key]) for row in selected])
            for key in (
                "real_recall_at_10",
                "real_candidate_recall",
                "gold_parent_to_real_child",
                "gold_section_to_real_unit",
                "oracle_top10_from_candidate",
            )
        }
        candidate_states = Counter(str(row["candidate_state"]) for row in selected)
        top10_states = Counter(str(row["top10_state"]) for row in selected)
        missing_reasons = Counter(
            str(reason)
            for row in selected
            for reason in row["candidate_missing_reasons"].values()
        )
        multi_evidence = [
            row for row in selected if int(row["gold_evidence_count"]) > 1
        ]
        count = len(selected)
        metrics.update(
            {
                "candidate_any_hit_rate": (
                    1.0 - candidate_states["zero"] / count if count else 0.0
                ),
                "candidate_all_evidence_rate": (
                    candidate_states["complete"] / count if count else 0.0
                ),
                "top10_any_hit_rate": (
                    1.0 - top10_states["zero"] / count if count else 0.0
                ),
                "top10_all_evidence_rate": (
                    top10_states["complete"] / count if count else 0.0
                ),
                "candidate_state_counts": {
                    state: candidate_states[state]
                    for state in ("zero", "partial", "complete")
                },
                "top10_state_counts": {
                    state: top10_states[state]
                    for state in ("zero", "partial", "complete")
                },
                "candidate_missing_reason_counts": dict(
                    sorted(missing_reasons.items())
                ),
                "multi_evidence": {
                    "cases": len(multi_evidence),
                    "candidate_all_evidence_rate": _mean(
                        [
                            float(row["candidate_state"] == "complete")
                            for row in multi_evidence
                        ]
                    ),
                    "top10_all_evidence_rate": _mean(
                        [
                            float(row["top10_state"] == "complete")
                            for row in multi_evidence
                        ]
                    ),
                },
            }
        )
        return metrics

    by_category: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_case:
        by_category[str(row["category"])].append(row)
    return {
        "schema_version": "1.0",
        "diagnostic": "tatqa_oracle_routing",
        "top_k": top_k,
        "candidate_k": candidate_k,
        "case_count": len(per_case),
        "aggregate": aggregate(per_case),
        "categories": {
            category: {
                "cases": len(values),
                **aggregate(values),
            }
            for category, values in sorted(by_category.items())
        },
        "per_case": per_case,
        "interpretation": {
            "o0_real": "strict retrieved evidence IDs",
            "o1_gold_parent_to_real_child": "parent routing upper bound; gold parent is treated as reachable",
            "o2_gold_section_to_real_unit": "section/table-vs-paragraph accessibility upper bound",
            "o3_oracle_top10_from_candidate": "perfect reordering of the existing Candidate@K set",
        },
    }


def build_tatqa_query_plan_from_text(
    query: str, *, answer_type: str = "", scale: str = ""
) -> dict[str, Any]:
    """Build a deterministic, gold-independent plan from query text."""
    query = query.strip()
    lowered = query.lower()
    answer_type = answer_type.strip().lower()
    if answer_type == "count" or re.search(r"\bhow many\b|\bnumber of\b", lowered):
        operation = "count"
    elif answer_type == "arithmetic" or re.search(
        r"\b(percent|percentage|ratio|difference|change|growth|margin|average)\b",
        lowered,
    ):
        operation = "arithmetic"
    elif answer_type == "multi-span" or re.search(
        r"\b(compare|respectively|both|each|which)\b", lowered
    ):
        operation = "comparison"
    else:
        operation = "lookup"
    comparator = next(
        (name for phrase, name in _COMPARATORS if phrase in lowered), None
    )
    scale = scale.strip().lower() or None
    if scale is None and "%" in query:
        scale = "percent"
    years = [int(value) for value in _YEAR_RE.findall(query)]
    thresholds = [
        value
        for value in _NUMBER_RE.findall(query)
        if not (value.isdigit() and len(value) == 4 and 1900 <= int(value) <= 2099)
    ]
    return {
        "operation": operation,
        "metric_terms": sorted(
            {
                token
                for token in re.findall(r"[a-z][a-z-]+", lowered)
                if token not in {"what", "which", "the", "for", "and", "in", "of", "is"}
            }
        ),
        "years": years,
        "comparator": comparator,
        "thresholds": thresholds,
        "scale": scale,
        "answer_type": answer_type or None,
    }


def build_tatqa_query_plan(case: RAGEvalCase) -> dict[str, Any]:
    """Build a deterministic plan using only the case query metadata."""
    return build_tatqa_query_plan_from_text(
        case.query,
        answer_type=str(case.metadata.get("answer_type", "")),
        scale=str(case.metadata.get("scale", "")),
    )


__all__ = [
    "TATQARetrievalRow",
    "build_tatqa_query_plan",
    "build_tatqa_query_plan_from_text",
    "diagnose_tatqa_retrieval",
]

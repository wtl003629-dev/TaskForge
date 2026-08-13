"""Deterministic failure diagnostics for locked QASPER retrieval runs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .knowledge import tokenise
from .rag_baseline import load_locked_split, select_locked_cases, sha256_file
from .rag_evaluation import load_qasper_dataset


def diagnose_qasper_run(
    run_dir: str | Path,
    *,
    dataset_path: str | Path,
    split_path: str | Path,
    stage: str | None = None,
    candidate_k: int = 50,
    top_k: int = 10,
    representative_limit: int = 10,
) -> dict[str, Any]:
    """Classify candidate misses versus ranking misses without tuning labels."""

    run_root = Path(run_dir)
    metrics = json.loads((run_root / "metrics.json").read_text(encoding="utf-8"))
    predictions = [
        json.loads(line)
        for line in (run_root / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stages = metrics.get("stages")
    if not isinstance(stages, dict) or not stages:
        raise ValueError("run metrics must contain at least one stage")
    chosen_stage = stage or next(iter(stages))
    selected_stage = [row for row in predictions if row.get("stage") == chosen_stage]
    if len(selected_stage) != len(predictions):
        raise ValueError("diagnostics requires a single-stage retrieval run")

    dataset_file = Path(dataset_path)
    dataset = load_qasper_dataset(dataset_file)
    split = load_locked_split(split_path)
    cases = select_locked_cases(
        dataset.cases,
        split,
        dataset_sha256=sha256_file(dataset_file),
    )
    by_case = {case.case_id: case for case in cases}
    by_document = {document.document_id: document for document in dataset.documents}
    if set(by_case) != {str(row.get("case_id")) for row in selected_stage}:
        raise ValueError("prediction case IDs do not match the locked split")

    counts = Counter(
        {
            "covered_top10": 0,
            "top10_ranking_failure": 0,
            "candidate_missing": 0,
            "lexical_mismatch": 0,
            "vocabulary_mismatch": 0,
            "title_abstract_section_dependency": 0,
            "multi_evidence_case": 0,
            "same_section_multiple_evidence": 0,
            "cross_section_evidence": 0,
            "long_evidence_case": 0,
            "evidence_truncated_or_fragmented": 0,
            "evidence_alignment_failure": 0,
        }
    )
    representatives: dict[str, list[str]] = {key: [] for key in counts}
    per_case: list[dict[str, Any]] = []
    for row in sorted(selected_stage, key=lambda value: str(value["case_id"])):
        case_id = str(row["case_id"])
        case = by_case[case_id]
        retrieved = [str(value) for value in row.get("retrieved_ids", [])]
        candidate = set(retrieved[:candidate_k])
        top = set(retrieved[:top_k])
        relevant = set(case.relevant_ids)
        if relevant.intersection(top):
            failure = "covered_top10"
        elif relevant.intersection(candidate):
            failure = "top10_ranking_failure"
        else:
            failure = "candidate_missing"
        counts[failure] += 1
        if len(representatives[failure]) < representative_limit:
            representatives[failure].append(case_id)

        evidence_documents = [
            by_document[item] for item in case.relevant_ids if item in by_document
        ]
        if len(evidence_documents) != len(case.relevant_ids):
            counts["evidence_alignment_failure"] += 1
            if len(representatives["evidence_alignment_failure"]) < representative_limit:
                representatives["evidence_alignment_failure"].append(case_id)
        evidence_text = " ".join(document.text for document in evidence_documents)
        query_terms = set(tokenise(case.query))
        evidence_terms = set(tokenise(evidence_text))
        if query_terms and not query_terms.intersection(evidence_terms):
            counts["lexical_mismatch"] += 1
            counts["vocabulary_mismatch"] += 1
            if len(representatives["lexical_mismatch"]) < representative_limit:
                representatives["lexical_mismatch"].append(case_id)
            if len(representatives["vocabulary_mismatch"]) < representative_limit:
                representatives["vocabulary_mismatch"].append(case_id)
        section_ids = {
            str(document.metadata.get("section_id"))
            for document in evidence_documents
            if document.metadata.get("section_id")
        }
        if len(case.relevant_ids) > 1:
            counts["multi_evidence_case"] += 1
            if len(representatives["multi_evidence_case"]) < representative_limit:
                representatives["multi_evidence_case"].append(case_id)
            if len(section_ids) == 1:
                counts["same_section_multiple_evidence"] += 1
                if len(representatives["same_section_multiple_evidence"]) < representative_limit:
                    representatives["same_section_multiple_evidence"].append(case_id)
            elif len(section_ids) > 1:
                counts["cross_section_evidence"] += 1
                if len(representatives["cross_section_evidence"]) < representative_limit:
                    representatives["cross_section_evidence"].append(case_id)
        if any(len(document.text) > 1500 for document in evidence_documents):
            counts["long_evidence_case"] += 1
            counts["evidence_truncated_or_fragmented"] += 1
            if len(representatives["long_evidence_case"]) < representative_limit:
                representatives["long_evidence_case"].append(case_id)
            if len(representatives["evidence_truncated_or_fragmented"]) < representative_limit:
                representatives["evidence_truncated_or_fragmented"].append(case_id)
        header_terms: set[str] = set()
        for document in evidence_documents:
            for field in ("paper_title", "section_title", "subsection_title"):
                value = document.metadata.get(field)
                if isinstance(value, str):
                    header_terms.update(tokenise(value))
            if document.metadata.get("node_type") == "abstract":
                header_terms.update(tokenise(document.text))
        if query_terms and len(query_terms.intersection(header_terms)) > len(
            query_terms.intersection(evidence_terms)
        ):
            counts["title_abstract_section_dependency"] += 1
            if len(representatives["title_abstract_section_dependency"]) < representative_limit:
                representatives["title_abstract_section_dependency"].append(case_id)
        per_case.append(
            {
                "case_id": case_id,
                "failure": failure,
                "relevant_count": len(relevant),
                "retrieved_candidate_count": len(candidate),
                "retrieved_top10_count": len(top),
            }
        )

    total = len(cases)
    return {
        "schema_version": "1.0",
        "dataset": "QASPER",
        "run_dir": str(run_root),
        "stage": chosen_stage,
        "candidate_k": candidate_k,
        "top_k": top_k,
        "dataset_sha256": sha256_file(dataset_file),
        "locked_split_id": split.split_id,
        "total_cases": total,
        "counts": dict(counts),
        "rates": {key: value / total for key, value in counts.items()},
        "representative_case_ids": representatives,
        "cases": per_case,
    }

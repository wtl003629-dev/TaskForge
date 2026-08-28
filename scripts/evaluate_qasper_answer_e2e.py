"""Evaluate cited QASPER answers from a strict locked retrieval report.

The retrieval input must be a persisted direct-PDF-upload report.  For every
locked case this script presents only the report's Top-K evidence to a live
model, scores the generated short answer against the QASPER annotation, and
checks citations against the same frozen Gold-to-Child content alignment used
by the retrieval evaluator. An optional second live call performs a strict
semantic review against the gold answer and gold evidence.

This is billable and refuses to run without ``--confirm-live-call``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from taskforge.config import Settings  # noqa: E402
from taskforge.domain import AgentProfile, StrictModel, Task  # noqa: E402
from taskforge.openai_provider import OpenAIChatCompletionsProvider  # noqa: E402
from taskforge.providers import (  # noqa: E402
    ModelProvider,
    ProviderError,
    RetryableProviderError,
)
from taskforge.qasper_alignment import (  # noqa: E402
    GoldAlignment,
    alignment_coverage_for_children,
)
from taskforge.rag_answer_eval import (  # noqa: E402
    _generate_answer,
    _normalise_usage,
)
from taskforge.rag_evaluation import (  # noqa: E402
    QasperGoldLabels,
    RAGEvalCase,
    answer_exact_match,
    answer_token_f1,
    load_qasper_dataset,
)


class SemanticJudgement(StrictModel):
    answer_verdict: Literal["correct", "partially_correct", "incorrect"]
    citation_verdict: Literal[
        "fully_supported",
        "partially_supported",
        "unsupported",
        "no_citation",
    ]
    critical_error: bool
    missing_key_points: list[str] = Field(default_factory=list, max_length=8)
    contradictions: list[str] = Field(default_factory=list, max_length=8)
    rationale: str = Field(min_length=1, max_length=1_000)


_JUDGE_INSTRUCTIONS = (
    "You are a strict evaluator of paper-grounded question answering. Use only "
    "the supplied question, acceptable reference answers, gold evidence, candidate answer, "
    "and candidate cited evidence. Do not use outside knowledge. Return exactly "
    "one JSON object with keys answer_verdict, citation_verdict, critical_error, "
    "missing_key_points, contradictions, and rationale. answer_verdict is correct "
    "only when the candidate has the same material meaning as at least one acceptable "
    "reference answer, "
    "allowing harmless paraphrase. Judge answer_verdict independently from "
    "citation_verdict: never downgrade answer_verdict only because citations are "
    "missing or weak. partially_correct means the main direction is "
    "right but an essential detail is missing; incorrect includes contradiction, "
    "an unsupported different answer, or an empty answer. citation_verdict is "
    "fully_supported only when the candidate cited evidence supports every "
    "material claim, partially_supported when it supports only part, unsupported "
    "when it does not support the answer, and no_citation when none was supplied. "
    "critical_error is true for a wrong yes/no direction, wrong entity/number, "
    "fabricated result, or material contradiction. Keep rationale concise and do "
    "not output hidden reasoning or markdown."
)

_QASPER_ANSWER_INSTRUCTIONS = (
    "Answer the research-paper question using ONLY the supplied evidence. Return "
    "exactly one JSON object with exactly two keys: answer and citation_ids. "
    "The JSON value of answer must always be one non-empty string, never an "
    "array or object; join multiple requested items inside that string. "
    "answer must directly match the requested granularity: for yes/no questions "
    "return only Yes or No; for dataset, baseline, architecture, domain, or method "
    "questions list only the requested names and include every name supported by "
    "the evidence; for numeric questions preserve the exact value and unit; for "
    "how/why questions give the named method or cause before at most one concise "
    "explanation. Do not replace a named tool or dataset with a broader related "
    "category, and do not add related facts that the question did not request. "
    "citation_ids must be a non-empty JSON array of the smallest set of unique "
    "evidence_id strings that directly supports the answer. Copy IDs exactly from "
    "the supplied evidence. Each array item must contain only the raw ID beginning "
    "with evidence:, never the literal label evidence_id:. Never cite every candidate "
    "by default, never use outside "
    "knowledge, and never wrap the JSON in markdown. If the evidence truly cannot "
    "answer the question, say Insufficient evidence and cite the most relevant "
    "inspected evidence."
)

_ANSWER_REVISION_INSTRUCTIONS = (
    "Act as a strict evidence-grounded answer editor. The task contains the "
    "original research-paper question and a draft answer with draft citation IDs. "
    "Use ONLY the supplied evidence to check whether the draft answers the exact "
    "question, misses requested items, selects a related but wrong entity, reverses "
    "a yes/no conclusion, or cites passages that do not support it. Return exactly "
    "one JSON object with exactly two keys: answer and citation_ids. Correct the "
    "draft when needed and otherwise preserve it. Keep the answer concise but retain "
    "all material details requested by the question. citation_ids must be a non-empty "
    "array containing the smallest set of unique evidence_id strings that directly "
    "supports the final answer. Copy IDs exactly, never cite every candidate by "
    "default, never use outside knowledge, and never output markdown or commentary."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def qasper_reference_answer(value: object) -> object:
    """Restore QASPER yes/no scalars after Pydantic numeric coercion."""

    if isinstance(value, float) and value in {0.0, 1.0}:
        return "Yes" if value == 1.0 else "No"
    return value


def qasper_answer_references(question: Mapping[str, Any]) -> list[str]:
    """Return every answerable QASPER annotation as an acceptable reference."""

    raw_answers = question.get("answers")
    if not isinstance(raw_answers, list):
        return []
    references: list[str] = []
    for item in raw_answers:
        if not isinstance(item, Mapping):
            continue
        annotation = item.get("answer")
        if not isinstance(annotation, Mapping) or annotation.get("unanswerable", False):
            continue
        free_form = str(annotation.get("free_form_answer") or "").strip()
        spans = annotation.get("extractive_spans")
        yes_no = annotation.get("yes_no")
        if free_form:
            value = free_form
        elif isinstance(spans, list) and spans:
            value = "; ".join(str(span).strip() for span in spans if str(span).strip())
        elif yes_no is True or yes_no == 1:
            value = "Yes"
        elif yes_no is False or yes_no == 0:
            value = "No"
        else:
            continue
        if value and value not in references:
            references.append(value)
    return references


def qasper_gold_evidence_texts(question: Mapping[str, Any]) -> list[str]:
    raw_answers = question.get("answers")
    if not isinstance(raw_answers, list):
        return []
    values: list[str] = []
    for item in raw_answers:
        annotation = item.get("answer") if isinstance(item, Mapping) else None
        if not isinstance(annotation, Mapping) or annotation.get("unanswerable", False):
            continue
        raw_evidence = annotation.get("evidence")
        if not isinstance(raw_evidence, list):
            continue
        for item_text in raw_evidence:
            text = str(item_text).strip()
            if text and text not in values:
                values.append(text)
    return values


def citation_metrics(
    citation_ids: Sequence[str],
    presented_evidence: Sequence[Mapping[str, Any]],
    gold_labels: QasperGoldLabels,
    gold_alignments: Mapping[str, GoldAlignment],
    *,
    minimum_complete_coverage: float = 0.80,
) -> dict[str, Any]:
    """Compute citation validity and strict Gold-content support."""

    presented = {
        str(item.get("evidence_id")): item
        for item in presented_evidence
        if isinstance(item.get("evidence_id"), str)
        and str(item.get("evidence_id")).strip()
    }
    unique_citations = list(dict.fromkeys(str(value) for value in citation_ids))
    valid_ids = [value for value in unique_citations if value in presented]
    cited_child_ids = {
        str(presented[value].get("chunk_id") or "").strip()
        for value in valid_ids
        if str(presented[value].get("chunk_id") or "").strip()
    }
    cited_oracle_units = {
        str(presented[value].get("gold_unit_id") or "").strip()
        for value in valid_ids
        if str(presented[value].get("gold_unit_id") or "").strip()
    }
    scored_sets: list[tuple[float, int, str, set[str]]] = []
    for evidence_set in gold_labels.evidence_sets:
        hit_units: set[str] = set()
        for unit in evidence_set.units:
            if unit.unit_id in cited_oracle_units:
                hit_units.add(unit.unit_id)
                continue
            alignment = gold_alignments.get(unit.unit_id)
            if alignment is None or alignment.status not in {"exact", "fuzzy"}:
                continue
            if (
                alignment_coverage_for_children(alignment, cited_child_ids)
                >= minimum_complete_coverage
            ):
                hit_units.add(unit.unit_id)
        scored_sets.append(
            (
                len(hit_units) / len(evidence_set.units),
                len(evidence_set.units),
                evidence_set.annotation_id,
                hit_units,
            )
        )
    coverage, _, annotation_id, covered_unit_ids = max(
        scored_sets,
        key=lambda item: (item[0], -item[1], item[2]),
    )
    selected_set = next(
        item
        for item in gold_labels.evidence_sets
        if item.annotation_id == annotation_id
    )
    selected_unit_ids = {unit.unit_id for unit in selected_set.units}
    supporting_child_ids: set[str] = set()
    for unit_id in selected_unit_ids:
        alignment = gold_alignments.get(unit_id)
        if alignment is not None:
            supporting_child_ids.update(
                span.child_id for span in alignment.aligned_child_spans
            )
    supported_ids = [
        value
        for value in valid_ids
        if (
            str(presented[value].get("chunk_id") or "") in supporting_child_ids
            or str(presented[value].get("gold_unit_id") or "")
            in selected_unit_ids
        )
    ]
    total = len(unique_citations)
    return {
        "citation_count": total,
        "valid_citation_count": len(valid_ids),
        "gold_supported_citation_count": len(supported_ids),
        "invalid_citation_ids": [
            value for value in unique_citations if value not in presented
        ],
        "citation_validity": len(valid_ids) / total if total else 0.0,
        "gold_content_citation_precision": (
            len(supported_ids) / total if total else 0.0
        ),
        "gold_evidence_unit_coverage": coverage,
        "covered_gold_unit_ids": sorted(covered_unit_ids),
        "selected_gold_annotation_id": annotation_id,
    }


def parse_semantic_judgement(raw: str | None) -> SemanticJudgement:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("semantic judge returned an empty response")
    value = json.loads(raw)
    return SemanticJudgement.model_validate(value)


def _merge_usage(*values: Mapping[str, int] | None) -> dict[str, int]:
    keys = {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "input_cache_hit_tokens",
        "input_cache_miss_tokens",
    }
    return {
        key: sum(int(value.get(key, 0)) for value in values if value is not None)
        for key in sorted(keys)
    }


async def _judge_answer(
    provider: ModelProvider,
    *,
    model: str,
    case: RAGEvalCase,
    reference_answers: Sequence[str],
    candidate_answer: str,
    gold_evidence: Sequence[str],
    cited_evidence: Sequence[str],
) -> tuple[SemanticJudgement | None, dict[str, int] | None, str | None]:
    profile = AgentProfile(
        id="qasper-semantic-judge",
        name="QASPER semantic answer judge",
        instructions=_JUDGE_INSTRUCTIONS,
        model=model,
        allowed_tools=[],
    )
    task = Task(
        tenant_id="tenant-qasper-answer-eval",
        user_id="user-qasper-answer-eval",
        goal="Judge one QASPER answer using only the supplied reference material.",
    )
    context = {
        "assembled": {
            "question": case.query,
            "acceptable_reference_answers": list(reference_answers),
            "gold_evidence": list(gold_evidence),
            "candidate_answer": candidate_answer,
            "candidate_cited_evidence": list(cited_evidence),
        },
        "trajectory": [],
    }
    usage_values: list[dict[str, int] | None] = []
    last_error: str | None = None
    for attempt in range(3):
        try:
            turn = await provider.complete(
                task=task,
                profile=profile,
                context=context,
                tools=[],
            )
            usage_values.append(_normalise_usage(turn.metadata))
            if turn.kind != "final":
                last_error = "judge_non_final_turn"
            else:
                try:
                    judgement = parse_semantic_judgement(turn.final_answer)
                except (ValueError, json.JSONDecodeError) as exc:
                    last_error = f"judge_invalid_json:{type(exc).__name__}"
                else:
                    return judgement, _merge_usage(*usage_values), None
        except RetryableProviderError as exc:
            last_error = type(exc).__name__
        except ProviderError as exc:
            return None, _merge_usage(*usage_values), type(exc).__name__
        if attempt < 2:
            await asyncio.sleep(0.5 * (2**attempt))
    return None, _merge_usage(*usage_values), last_error or "judge_failed"


def _answer_evidence(
    raw_items: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
) -> tuple[list[tuple[str, str]], list[Mapping[str, Any]]]:
    selected = list(raw_items[:top_k])
    values: list[tuple[str, str]] = []
    for item in selected:
        evidence_id = str(item.get("evidence_id") or "").strip()
        if not evidence_id:
            continue
        values.append(
            (
                evidence_id,
                "\n".join(
                    (
                        f"evidence_id: {evidence_id}",
                        f"page: {item.get('page')}",
                        f"text: {str(item.get('text') or '')}",
                    )
                ),
            )
        )
    return values, selected


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty QASPER answer run")

    def mean(field: str) -> float:
        return statistics.fmean(float(row.get(field) or 0.0) for row in rows)

    judged = [row for row in rows if isinstance(row.get("semantic_judgement"), Mapping)]
    correct = [
        row
        for row in judged
        if row["semantic_judgement"].get("answer_verdict") == "correct"
        and not row["semantic_judgement"].get("critical_error")
    ]
    strict = [
        row
        for row in correct
        if row["semantic_judgement"].get("citation_verdict") == "fully_supported"
    ]
    weighted = {
        "correct": 1.0,
        "partially_correct": 0.5,
        "incorrect": 0.0,
    }
    usage = _merge_usage(
        *(row.get("usage") for row in rows if isinstance(row.get("usage"), Mapping))
    )
    hit_rows = [row for row in rows if float(row.get("recall_at_10") or 0.0) > 0.0]
    miss_rows = [row for row in rows if float(row.get("recall_at_10") or 0.0) == 0.0]

    def judged_accuracy(group: Sequence[Mapping[str, Any]]) -> float | None:
        group_judged = [
            row for row in group if isinstance(row.get("semantic_judgement"), Mapping)
        ]
        if not group_judged:
            return None
        return sum(
            row["semantic_judgement"].get("answer_verdict") == "correct"
            and not row["semantic_judgement"].get("critical_error")
            for row in group_judged
        ) / len(group_judged)

    return {
        "total_cases": len(rows),
        "exact_match_accuracy": mean("exact_match"),
        "avg_token_f1": mean("token_f1"),
        "strict_exact_and_gold_citation_accuracy": mean(
            "strict_exact_and_gold_citation"
        ),
        "avg_citation_validity": statistics.fmean(
            float(row["citation_metrics"]["citation_validity"]) for row in rows
        ),
        "avg_gold_content_citation_precision": statistics.fmean(
            float(row["citation_metrics"]["gold_content_citation_precision"])
            for row in rows
        ),
        "avg_gold_evidence_unit_coverage": statistics.fmean(
            float(row["citation_metrics"]["gold_evidence_unit_coverage"])
            for row in rows
        ),
        "semantic_judged_cases": len(judged),
        "semantic_answer_accuracy": len(correct) / len(judged) if judged else None,
        "semantic_strict_grounded_accuracy": (
            len(strict) / len(judged) if judged else None
        ),
        "semantic_weighted_accuracy": (
            statistics.fmean(
                weighted[row["semantic_judgement"]["answer_verdict"]]
                for row in judged
            )
            if judged
            else None
        ),
        "by_retrieval_at_10": {
            "hit_cases": len(hit_rows),
            "hit_semantic_accuracy": judged_accuracy(hit_rows),
            "miss_cases": len(miss_rows),
            "miss_semantic_accuracy": judged_accuracy(miss_rows),
        },
        "failure_counts": {
            "generation_execution_error": sum(bool(row.get("execution_error")) for row in rows),
            "answer_contract_parse_error": sum(bool(row.get("parse_error")) for row in rows),
            "semantic_judge_error": sum(bool(row.get("judge_error")) for row in rows),
        },
        "usage": usage,
    }


def _validate_retrieval_report(
    retrieval_report: Mapping[str, Any],
    *,
    evidence_source: Literal["retrieved", "oracle"],
    benchmark_track: Literal["qasper_strict", "bilingual_paper_smoke"] = "qasper_strict",
) -> None:
    evaluation_type = retrieval_report.get("evaluation_type")
    if benchmark_track == "bilingual_paper_smoke":
        if (
            retrieval_report.get("schema_version") != "2.3"
            or evaluation_type
            != "bilingual_mixed_paper_corpus_cross_language_queries"
            or retrieval_report.get("benchmark_track")
            != "scholarly_paper_fulltext_retrieval"
        ):
            raise ValueError(
                "retrieval report is not the bilingual paper smoke track"
            )
        alignment_gate = retrieval_report.get("alignment_gate")
        if evidence_source == "retrieved" and (
            not isinstance(alignment_gate, Mapping)
            or alignment_gate.get("passed") is not True
        ):
            raise ValueError(
                "retrieval report failed the Gold-to-Child alignment gate; "
                "bilingual retrieved-evidence answer smoke is not allowed"
            )
        return
    if evaluation_type == "qasper_direct_pdf_upload_retrieval":
        raise ValueError(
            "legacy QASPER page-proxy retrieval reports are invalid; rerun the "
            "schema 2 strict content-aligned evaluator"
        )
    if retrieval_report.get("schema_version") not in {"2.0", "2.1", "2.2", "2.3"} or (
        evaluation_type
        not in {
            "qasper_synthetic_pdf_parser_regression",
            "qasper_real_pdf_upload_retrieval",
        }
    ):
        raise ValueError("retrieval report is not a strict QASPER upload report")
    alignment_gate = retrieval_report.get("alignment_gate")
    if evidence_source == "retrieved" and (
        not isinstance(alignment_gate, Mapping)
        or alignment_gate.get("passed") is not True
    ):
        raise ValueError(
            "retrieval report failed the Gold-to-Child alignment gate; "
            "formal retrieved-evidence answer evaluation is not allowed"
        )


async def run(
    *,
    retrieval_report_path: Path,
    dataset_path: Path,
    split_path: Path,
    output_path: Path,
    provider: ModelProvider,
    model: str,
    top_k: int,
    evidence_source: Literal["retrieved", "oracle"],
    max_cases: int | None,
    concurrency: int,
    semantic_judge: bool,
    answer_prompt: Literal["generic", "qasper_v1"],
    answer_revision: Literal["none", "critic_v1"],
    resume: bool,
    provider_name: str = "unspecified",
    benchmark_track: Literal["qasper_strict", "bilingual_paper_smoke"] = "qasper_strict",
) -> dict[str, Any]:
    retrieval_report = json.loads(retrieval_report_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    _validate_retrieval_report(
        retrieval_report,
        evidence_source=evidence_source,
        benchmark_track=benchmark_track,
    )
    split_ids = [str(value) for value in split.get("case_ids", [])]
    if max_cases is not None:
        split_ids = split_ids[:max_cases]
    report_rows = {
        str(row["case_id"]): row
        for row in retrieval_report.get("rows", [])
        if isinstance(row, Mapping) and row.get("case_id")
    }
    missing = [case_id for case_id in split_ids if case_id not in report_rows]
    if missing:
        raise ValueError(f"retrieval report is missing locked cases: {missing[:3]}")

    dataset = load_qasper_dataset(dataset_path)
    raw_dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = {case.case_id: case for case in dataset.cases}
    missing = [case_id for case_id in split_ids if case_id not in cases]
    if missing:
        raise ValueError(f"QASPER dataset is missing locked cases: {missing[:3]}")
    raw_questions: dict[str, Mapping[str, Any]] = {}
    for paper_id, paper in raw_dataset.items():
        qas = paper.get("qas") if isinstance(paper, Mapping) else None
        if not isinstance(qas, list):
            continue
        for question in qas:
            if not isinstance(question, Mapping):
                continue
            question_id = str(question.get("question_id") or "").strip()
            if question_id:
                raw_questions[f"qasper:{paper_id}:{question_id}"] = question
    checkpoint_path = output_path.with_suffix(".predictions.jsonl")
    completed: dict[str, dict[str, Any]] = {}
    if checkpoint_path.exists():
        if not resume:
            raise FileExistsError(
                f"checkpoint already exists; pass --resume or choose another output: {checkpoint_path}"
            )
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                # Resume is also the repair path for transient provider or
                # structured-output failures. Successful cases remain frozen;
                # only failed rows are scheduled again and appended.
                if any(
                    value.get(field)
                    for field in ("execution_error", "parse_error", "judge_error")
                ):
                    continue
                completed[str(value["case_id"])] = value
    else:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate(case_id: str) -> dict[str, Any]:
        async with semaphore:
            case = cases[case_id]
            raw_question = raw_questions[case_id]
            reference_answers = qasper_answer_references(raw_question)
            if not reference_answers:
                reference_answers = [str(qasper_reference_answer(case.answer))]
            retrieval_row = report_rows[case_id]
            raw_evidence = [
                item
                for item in retrieval_row.get("retrieved_evidence", [])
                if isinstance(item, Mapping)
            ]
            paper_id = str(case.metadata["paper_id"])
            if case.qasper_gold is None:
                raise ValueError(f"QASPER case has no strict Gold labels: {case_id}")
            raw_alignments = retrieval_row.get("gold_alignments")
            if not isinstance(raw_alignments, Mapping):
                raise ValueError(
                    f"strict retrieval row has no Gold-to-Child alignment: {case_id}"
                )
            gold_alignments = {
                str(unit_id): GoldAlignment.model_validate(value)
                for unit_id, value in raw_alignments.items()
            }
            gold_units = {
                unit.unit_id: unit
                for evidence_set in case.qasper_gold.evidence_sets
                for unit in evidence_set.units
            }
            if evidence_source == "oracle":
                presented = [
                    {
                        "evidence_id": f"oracle:{unit_id}",
                        "gold_unit_id": unit_id,
                        "text": unit.text,
                    }
                    for unit_id, unit in gold_units.items()
                ]
                answer_evidence, presented = _answer_evidence(
                    presented,
                    top_k=max(top_k, len(presented)),
                )
            else:
                answer_evidence, presented = _answer_evidence(
                    raw_evidence,
                    top_k=top_k,
                )
            try:
                draft = await _generate_answer(
                    provider,
                    case,
                    answer_evidence,
                    model=model,
                    max_evidence_chars=16_000,
                    contract="cited_v1",
                    instructions_override=(
                        _QASPER_ANSWER_INSTRUCTIONS
                        if answer_prompt == "qasper_v1"
                        else None
                    ),
                )
            except Exception as exc:  # retain the case and continue the full run
                draft = None
                generation_exception = f"{type(exc).__name__}: {exc}"
            else:
                generation_exception = None

            generated = draft
            revision_error: str | None = None
            revision_applied = False
            if (
                answer_revision == "critic_v1"
                and draft is not None
                and draft.answer
                and draft.parse_error is None
                and draft.execution_error is None
            ):
                revision_case = case.model_copy(
                    update={
                        "query": (
                            f"ORIGINAL QUESTION:\n{case.query}\n\n"
                            f"DRAFT ANSWER:\n{draft.answer}\n\n"
                            "DRAFT CITATION IDS:\n"
                            + json.dumps(draft.citation_ids, ensure_ascii=False)
                        )
                    }
                )
                try:
                    revised = await _generate_answer(
                        provider,
                        revision_case,
                        answer_evidence,
                        model=model,
                        max_evidence_chars=16_000,
                        contract="cited_v1",
                        instructions_override=_ANSWER_REVISION_INSTRUCTIONS,
                    )
                except Exception as exc:
                    revision_error = f"{type(exc).__name__}: {exc}"
                else:
                    if revised.answer and revised.parse_error is None and revised.execution_error is None:
                        generated = revised
                        revision_applied = True
                    else:
                        revision_error = revised.parse_error or revised.execution_error or "empty_revision"

            answer = generated.answer if generated is not None else ""
            citations = generated.citation_ids if generated is not None else []
            cite_metrics = citation_metrics(
                citations,
                presented,
                case.qasper_gold,
                gold_alignments,
            )
            exact = max(
                (answer_exact_match(answer, reference) for reference in reference_answers),
                default=0.0,
            )
            token_f1 = max(
                (answer_token_f1(answer, reference) for reference in reference_answers),
                default=0.0,
            )
            cited_items = {
                str(item.get("evidence_id")): item
                for item in presented
                if isinstance(item.get("evidence_id"), str)
            }
            cited_text = [
                str(cited_items[value].get("text") or "")
                for value in citations
                if value in cited_items
            ]
            gold_text = [unit.text for unit in gold_units.values()]
            judgement: SemanticJudgement | None = None
            judge_usage: dict[str, int] | None = None
            judge_error: str | None = None
            if semantic_judge and answer and generated is not None and not generated.parse_error:
                judgement, judge_usage, judge_error = await _judge_answer(
                    provider,
                    model=model,
                    case=case,
                    reference_answers=reference_answers,
                    candidate_answer=answer,
                    gold_evidence=gold_text,
                    cited_evidence=cited_text,
                )
            elif semantic_judge:
                judgement = SemanticJudgement(
                    answer_verdict="incorrect",
                    citation_verdict="no_citation" if not citations else "unsupported",
                    critical_error=True,
                    rationale="The answer was empty or failed the structured answer contract.",
                )
            draft_usage = draft.usage if draft is not None else None
            revision_usage = (
                generated.usage if revision_applied and generated is not None else None
            )
            usage = _merge_usage(draft_usage, revision_usage, judge_usage)
            strict_exact = float(
                exact == 1.0
                and cite_metrics["citation_validity"] == 1.0
                and cite_metrics["gold_content_citation_precision"] == 1.0
                and cite_metrics["gold_evidence_unit_coverage"] > 0.0
            )
            if generated is None or not answer or generated.parse_error or generation_exception:
                failure_stage = "answer_reasoning_failure"
            elif cite_metrics["citation_validity"] < 1.0 or cite_metrics[
                "gold_content_citation_precision"
            ] < 1.0:
                failure_stage = "citation_failure"
            elif strict_exact == 1.0:
                failure_stage = "success"
            else:
                failure_stage = "answer_reasoning_failure"
            return {
                "case_id": case_id,
                "paper_id": paper_id,
                "query": case.query,
                "gold_answers": reference_answers,
                "gold_evidence_ids": list(gold_units),
                "recall_at_10": float(retrieval_row["recall_at_k"]["10"]),
                "generated_answer": answer,
                "citation_ids": citations,
                "draft_answer": draft.answer if draft is not None else "",
                "draft_citation_ids": draft.citation_ids if draft is not None else [],
                "revision_applied": revision_applied,
                "revision_error": revision_error,
                "presented_evidence_ids": [value for value, _ in answer_evidence],
                "exact_match": exact,
                "token_f1": token_f1,
                "strict_exact_and_gold_citation": strict_exact,
                "failure_stage": failure_stage,
                "citation_metrics": cite_metrics,
                "semantic_judgement": (
                    judgement.model_dump(mode="json") if judgement is not None else None
                ),
                "parse_error": generated.parse_error if generated is not None else None,
                "execution_error": (
                    generation_exception
                    or (generated.execution_error if generated is not None else None)
                ),
                "judge_error": judge_error,
                "provider_calls": (
                    (draft.provider_calls if draft is not None else 0)
                    + (
                        generated.provider_calls
                        if revision_applied and generated is not None
                        else 0
                    )
                    ) + (1 if judgement is not None and judge_error is None and semantic_judge else 0),
                "usage": usage,
            }

    pending = [case_id for case_id in split_ids if case_id not in completed]
    tasks = [asyncio.create_task(evaluate(case_id)) for case_id in pending]
    for future in asyncio.as_completed(tasks):
        row = await future
        completed[row["case_id"]] = row
        with checkpoint_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()

    rows = [completed[case_id] for case_id in split_ids]
    metrics = _metrics(rows)
    report = {
        "schema_version": "1.0",
        "evaluation_type": (
            "bilingual_paper_cited_answer_e2e_live_smoke"
            if benchmark_track == "bilingual_paper_smoke"
            else "qasper_product_retrieval_cited_answer_e2e_live"
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": (
            "TaskForge bilingual paper mixed smoke split"
            if benchmark_track == "bilingual_paper_smoke"
            else "QASPER v0.3 official dev clean locked split"
        ),
        "benchmark_track": benchmark_track,
        "provider": provider_name,
        "model": model,
        "answer_prompt": answer_prompt,
        "answer_revision": answer_revision,
        "live_model_calls": True,
        "semantic_judge": {
            "enabled": semantic_judge,
            "model": model if semantic_judge else None,
            "independent_from_generator": False if semantic_judge else None,
            "limitation": (
                "The semantic judge uses the same model family as generation and requires human calibration."
                if semantic_judge
                else None
            ),
        },
        "inputs": {
            "retrieval_report": str(retrieval_report_path),
            "retrieval_report_sha256": _sha256(retrieval_report_path),
            "dataset": str(dataset_path),
            "dataset_sha256": _sha256(dataset_path),
            "split": str(split_path),
            "split_sha256": _sha256(split_path),
        },
        "pipeline": [
            (
                "gold_evidence_oracle"
                if evidence_source == "oracle"
                else "persisted_direct_pdf_upload_retrieval"
            ),
            (
                "all_gold_evidence"
                if evidence_source == "oracle"
                else f"top_{top_k}_evidence"
            ),
            "live_cited_answer_generation",
            *( ["live_semantic_judge"] if semantic_judge else [] ),
            "deterministic_answer_and_citation_scoring",
        ],
        "retrieval_metrics": retrieval_report.get("metrics"),
        "evidence_source": evidence_source,
        "metrics": metrics,
        "rows": rows,
        "limitations": [
            *(
                ["Gold Evidence is injected as an Oracle generation upper-bound; this is not an end-to-end retrieval score."]
                if evidence_source == "oracle"
                else ["The retrieval stage is replayed from a hashed product-path report rather than executed in the same process."]
            ),
            "QASPER answers are scored against all distinct answerable annotations available for each locked question.",
            "Exact match and token F1 undercount acceptable free-form paraphrases.",
            *( ["The semantic judge is the same configured model as the answer generator."] if semantic_judge else [] ),
            "This run evaluates one cited answer-generation Agent, not the four-role research orchestration quality.",
        ],
    }
    if output_path.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--retrieval-report",
        type=Path,
        default=PROJECT_ROOT
        / ".taskforge"
        / "eval-runs"
        / "qasper-clean-holdout-real-pdf-strict-bm25-original-v3.json",
    )
    value.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / ".taskforge" / "eval-cache" / "qasper-dev-v0.3.json",
    )
    value.add_argument(
        "--split",
        type=Path,
        default=PROJECT_ROOT
        / "eval"
        / "splits"
        / "qasper-dev-clean-holdout-100-v2.json",
    )
    value.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "eval" / "reports" / "qasper-answer-e2e-live.json",
    )
    value.add_argument("--model", default=None)
    value.add_argument(
        "--provider",
        choices=("deepseek", "bailian"),
        default=None,
        help="Generation provider; defaults to the active TASKFORGE_PROVIDER.",
    )
    value.add_argument(
        "--benchmark-track",
        choices=("qasper-strict", "bilingual-paper-smoke"),
        default="qasper-strict",
        help="Keep the bilingual mixed-paper smoke distinct from official QASPER.",
    )
    value.add_argument(
        "--answer-prompt",
        choices=("generic", "qasper-v1"),
        default="generic",
    )
    value.add_argument(
        "--answer-revision",
        choices=("none", "critic-v1"),
        default="none",
    )
    value.add_argument("--top-k", type=int, default=10)
    value.add_argument(
        "--evidence-source",
        choices=("retrieved", "oracle"),
        default="retrieved",
    )
    value.add_argument("--max-cases", type=int, default=None)
    value.add_argument("--concurrency", type=int, default=2)
    value.add_argument("--no-semantic-judge", action="store_true")
    value.add_argument("--resume", action="store_true")
    value.add_argument("--confirm-live-call", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.confirm_live_call:
        raise SystemExit("refusing billable QASPER answer eval without --confirm-live-call")
    if not 1 <= args.top_k <= 20:
        raise SystemExit("--top-k must be between 1 and 20")
    if args.max_cases is not None and not 1 <= args.max_cases <= 100:
        raise SystemExit("--max-cases must be between 1 and 100")
    if not 1 <= args.concurrency <= 8:
        raise SystemExit("--concurrency must be between 1 and 8")
    settings = Settings(_env_file=PROJECT_ROOT / ".env")
    provider_name = args.provider or settings.provider
    if provider_name == "bailian":
        if settings.bailian_api_key is None:
            raise SystemExit("TASKFORGE_BAILIAN_API_KEY is required")
        api_key = settings.bailian_api_key.get_secret_value()
        model = args.model or settings.bailian_chat_model
        base_url = settings.bailian_base_url
        timeout_seconds = settings.bailian_chat_timeout_seconds
        thinking_mode = None
        trust_env = settings.bailian_chat_trust_env
    elif provider_name == "deepseek":
        if settings.deepseek_api_key is None:
            raise SystemExit("TASKFORGE_DEEPSEEK_API_KEY is required")
        api_key = settings.deepseek_api_key.get_secret_value()
        model = args.model or settings.deepseek_model or "deepseek-v4-flash"
        base_url = settings.deepseek_base_url
        timeout_seconds = 120
        thinking_mode = "disabled"
        trust_env = settings.deepseek_trust_env
    else:
        raise SystemExit(
            "answer evaluation requires --provider deepseek or --provider bailian"
        )

    async def execute() -> dict[str, Any]:
        provider = OpenAIChatCompletionsProvider(
            api_key=api_key,
            enabled=True,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            thinking_mode=thinking_mode,
            json_mode=True,
            trust_env=trust_env,
        )
        try:
            return await run(
                retrieval_report_path=args.retrieval_report.resolve(),
                dataset_path=args.dataset.resolve(),
                split_path=args.split.resolve(),
                output_path=args.output.resolve(),
                provider=provider,
                model=model,
                top_k=args.top_k,
                evidence_source=args.evidence_source,
                max_cases=args.max_cases,
                concurrency=args.concurrency,
                semantic_judge=not args.no_semantic_judge,
                answer_prompt=args.answer_prompt.replace("-", "_"),
                answer_revision=args.answer_revision.replace("-", "_"),
                resume=args.resume,
                provider_name=provider_name,
                benchmark_track=args.benchmark_track.replace("-", "_"),
            )
        finally:
            await provider.aclose()

    report = asyncio.run(execute())
    print(
        json.dumps(
            {"output": str(args.output.resolve()), "metrics": report["metrics"]},
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

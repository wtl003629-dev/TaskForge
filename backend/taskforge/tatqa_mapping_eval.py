"""Diagnostic evaluation for TagOp's heuristic TAT-QA evidence mappings.

The upstream TagOp preprocessing heuristically adds ``facts`` and ``mapping``
fields to TAT-QA.  Those fields are useful for diagnosing whether a retriever
selected the right table rows and cells, but they are not official, manually
verified evidence labels and must not be used as a promotion metric.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tatqa_program_executor import (
    TATQAProgramExecutionError,
    execute_tatqa_derivation,
    tatqa_program_matches_answer,
)
from .tatqa_slot_selector import select_tatqa_table_slots

TAGOP_SOURCE_COMMIT = "870accc41953dcde885aabeb963d94aabdc0fbc3"
TAGOP_TRAIN_SHA256 = (
    "cea22c36f4a3d4b5857bc8756312fa66ddb3b9abebcdd8e536f3fb113993c735"
)
TAGOP_SOURCE_URL = (
    "https://github.com/NExTplusplus/TAT-QA/blob/"
    f"{TAGOP_SOURCE_COMMIT}/dataset_tagop/tatqa_dataset_train.json"
)


class TATQAMappingDiagnosticError(ValueError):
    """Raised when an annotation or prediction artifact is not trustworthy."""


@dataclass(frozen=True)
class _QuestionMapping:
    case_id: str
    table_document_id: str
    coordinates: tuple[tuple[int, int], ...]
    duplicate_coordinate_count: int
    answer_from: str
    question: str
    table: tuple[tuple[str, ...], ...]
    answer_type: str
    answer: Any
    derivation: str
    scale: str


@dataclass(frozen=True)
class _Prediction:
    case_id: str
    category: str
    stage: str
    retrieved_ids: tuple[str, ...]
    retrieved_row_ids: tuple[str, ...]
    retrieved_cell_ids: tuple[str, ...]
    retrieved_complete_table_ids: tuple[str, ...]
    retrieved_table_units_by_hit: tuple[Mapping[str, Any], ...]
    hit_alignment_available: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_string_tuple(value: Any, *, field: str, case_id: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TATQAMappingDiagnosticError(
            f"prediction {case_id!r} field {field!r} must be a list of strings"
        )
    return tuple(value)


def _load_predictions(path: Path, *, stage: str | None) -> list[_Prediction]:
    predictions: list[_Prediction] = []
    seen: set[str] = set()
    observed_stages: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TATQAMappingDiagnosticError(
                f"invalid prediction JSON on line {line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise TATQAMappingDiagnosticError(
                f"prediction line {line_number} must contain a JSON object"
            )
        row_stage = row.get("stage")
        if not isinstance(row_stage, str) or not row_stage:
            raise TATQAMappingDiagnosticError(
                f"prediction line {line_number} has no valid stage"
            )
        observed_stages.add(row_stage)
        if stage is not None and row_stage != stage:
            continue
        case_id = row.get("case_id")
        category = row.get("category")
        if not isinstance(case_id, str) or not case_id.startswith("tatqa:"):
            raise TATQAMappingDiagnosticError(
                f"prediction line {line_number} has no valid TAT-QA case_id"
            )
        if not isinstance(category, str) or not category:
            raise TATQAMappingDiagnosticError(
                f"prediction {case_id!r} has no valid category"
            )
        if case_id in seen:
            raise TATQAMappingDiagnosticError(
                f"duplicate prediction for case {case_id!r}"
            )
        seen.add(case_id)
        raw_hit_units = row.get("retrieved_table_units_by_hit", [])
        if not isinstance(raw_hit_units, list) or any(
            not isinstance(item, dict) for item in raw_hit_units
        ):
            raise TATQAMappingDiagnosticError(
                f"prediction {case_id!r} field 'retrieved_table_units_by_hit' "
                "must be a list of objects"
            )
        hit_ranks: list[int] = []
        for item in raw_hit_units:
            rank = item.get("rank")
            document_id = item.get("document_id")
            if (
                isinstance(rank, bool)
                or not isinstance(rank, int)
                or rank <= 0
                or not isinstance(document_id, str)
                or not document_id
            ):
                raise TATQAMappingDiagnosticError(
                    f"prediction {case_id!r} contains an invalid aligned table hit"
                )
            for identifier_field in ("row_id", "cell_id"):
                identifier = item.get(identifier_field)
                if identifier is not None and not isinstance(identifier, str):
                    raise TATQAMappingDiagnosticError(
                        f"prediction {case_id!r} aligned hit has invalid "
                        f"{identifier_field!r}"
                    )
            if not isinstance(item.get("table_complete", False), bool):
                raise TATQAMappingDiagnosticError(
                    f"prediction {case_id!r} aligned hit has invalid table_complete"
                )
            hit_ranks.append(rank)
        if hit_ranks != sorted(set(hit_ranks)):
            raise TATQAMappingDiagnosticError(
                f"prediction {case_id!r} aligned table hit ranks must be unique and ordered"
            )
        predictions.append(
            _Prediction(
                case_id=case_id,
                category=category,
                stage=row_stage,
                retrieved_ids=_as_string_tuple(
                    row.get("retrieved_ids"), field="retrieved_ids", case_id=case_id
                ),
                retrieved_row_ids=_as_string_tuple(
                    row.get("retrieved_row_ids", []),
                    field="retrieved_row_ids",
                    case_id=case_id,
                ),
                retrieved_cell_ids=_as_string_tuple(
                    row.get("retrieved_cell_ids", []),
                    field="retrieved_cell_ids",
                    case_id=case_id,
                ),
                retrieved_complete_table_ids=_as_string_tuple(
                    row.get("retrieved_complete_table_ids", []),
                    field="retrieved_complete_table_ids",
                    case_id=case_id,
                ),
                retrieved_table_units_by_hit=tuple(raw_hit_units),
                hit_alignment_available="retrieved_table_units_by_hit" in row,
            )
        )
    if not predictions:
        suffix = f" for stage {stage!r}" if stage is not None else ""
        raise TATQAMappingDiagnosticError(f"no predictions found{suffix}")
    if stage is None and len(observed_stages) != 1:
        raise TATQAMappingDiagnosticError(
            "prediction artifact contains multiple stages; select one explicitly"
        )
    return predictions


def _load_mappings(path: Path) -> dict[str, _QuestionMapping]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TATQAMappingDiagnosticError("TagOp annotation root must be a list")
    mappings: dict[str, _QuestionMapping] = {}
    table_uids: set[str] = set()
    for context_index, context in enumerate(raw):
        if not isinstance(context, dict):
            raise TATQAMappingDiagnosticError(
                f"TagOp context {context_index} must be an object"
            )
        table_object = context.get("table")
        questions = context.get("questions")
        if not isinstance(table_object, dict) or not isinstance(questions, list):
            raise TATQAMappingDiagnosticError(
                f"TagOp context {context_index} has invalid table/questions"
            )
        table_uid = table_object.get("uid")
        table = table_object.get("table")
        if not isinstance(table_uid, str) or not table_uid:
            raise TATQAMappingDiagnosticError(
                f"TagOp context {context_index} has no valid table uid"
            )
        if table_uid in table_uids:
            raise TATQAMappingDiagnosticError(f"duplicate table uid {table_uid!r}")
        table_uids.add(table_uid)
        if not isinstance(table, list) or any(not isinstance(row, list) for row in table):
            raise TATQAMappingDiagnosticError(
                f"TagOp table {table_uid!r} must be a list of rows"
            )
        for question in questions:
            if not isinstance(question, dict) or not isinstance(question.get("uid"), str):
                raise TATQAMappingDiagnosticError(
                    f"TagOp table {table_uid!r} contains an invalid question"
                )
            case_id = f"tatqa:{question['uid']}"
            if case_id in mappings:
                raise TATQAMappingDiagnosticError(
                    f"duplicate TagOp question uid {question['uid']!r}"
                )
            mapping = question.get("mapping", {})
            if not isinstance(mapping, dict):
                raise TATQAMappingDiagnosticError(
                    f"TagOp question {question['uid']!r} has invalid mapping"
                )
            raw_coordinates = mapping.get("table", [])
            if not isinstance(raw_coordinates, list):
                raise TATQAMappingDiagnosticError(
                    f"TagOp question {question['uid']!r} table mapping must be a list"
                )
            coordinates: list[tuple[int, int]] = []
            for coordinate in raw_coordinates:
                if (
                    not isinstance(coordinate, list)
                    or len(coordinate) != 2
                    or any(isinstance(value, bool) or not isinstance(value, int) for value in coordinate)
                ):
                    raise TATQAMappingDiagnosticError(
                        f"TagOp question {question['uid']!r} has invalid table coordinate"
                    )
                row_index, column_index = coordinate
                if (
                    row_index < 0
                    or row_index >= len(table)
                    or column_index < 0
                    or column_index >= len(table[row_index])
                ):
                    raise TATQAMappingDiagnosticError(
                        f"TagOp question {question['uid']!r} coordinate is out of bounds"
                    )
                coordinates.append((row_index, column_index))
            unique_coordinates = tuple(dict.fromkeys(coordinates))
            answer_from = question.get("answer_from", "")
            question_text = question.get("question", "")
            if not isinstance(question_text, str) or not question_text.strip():
                raise TATQAMappingDiagnosticError(
                    f"TagOp question {question['uid']!r} has no valid question text"
                )
            mappings[case_id] = _QuestionMapping(
                case_id=case_id,
                table_document_id=f"tatqa:{table_uid}:table",
                coordinates=unique_coordinates,
                duplicate_coordinate_count=len(coordinates) - len(unique_coordinates),
                answer_from=answer_from if isinstance(answer_from, str) else "",
                question=question_text,
                table=tuple(tuple(str(value) for value in row) for row in table),
                answer_type=str(question.get("answer_type", "")),
                answer=question.get("answer"),
                derivation=str(question.get("derivation", "")),
                scale=str(question.get("scale", "")),
            )
    return mappings


def _recall(gold: set[str], retrieved: Sequence[str]) -> float:
    return len(gold.intersection(retrieved)) / len(gold) if gold else 0.0


def _state(value: float) -> str:
    if value == 0.0:
        return "zero"
    if value == 1.0:
        return "complete"
    return "partial"


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate(per_case: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [
        "tagop_heuristic_emitted_row_unit_recall",
        "tagop_heuristic_emitted_cell_unit_recall",
        "tagop_heuristic_context_row_coverage",
        "tagop_heuristic_context_cell_coverage",
        "complete_table_at_document_k",
        "tagop_heuristic_query_slot_cell_recall_at_k",
        "tagop_heuristic_query_slot_row_recall_at_k",
    ]
    if all(bool(row["hit_alignment_available"]) for row in per_case):
        metrics.extend(
            [
                "tagop_heuristic_hit_aligned_row_recall_at_k",
                "tagop_heuristic_hit_aligned_cell_recall_at_k",
                "tagop_heuristic_hit_aligned_context_row_coverage_at_k",
                "tagop_heuristic_hit_aligned_context_cell_coverage_at_k",
                "complete_table_at_evidence_hit_k",
            ]
        )
    aggregate: dict[str, Any] = {
        metric: _mean([float(row[metric]) for row in per_case])
        for metric in metrics
    }
    for prefix in (
        "emitted_row",
        "emitted_cell",
        "context_row",
        "context_cell",
        "query_slot_row",
        "query_slot_cell",
    ):
        counter = Counter(str(row[f"{prefix}_state"]) for row in per_case)
        aggregate[f"{prefix}_state_counts"] = {
            state: counter.get(state, 0) for state in ("zero", "partial", "complete")
        }
    if all(bool(row["hit_alignment_available"]) for row in per_case):
        for prefix in (
            "hit_aligned_row",
            "hit_aligned_cell",
            "hit_aligned_context_row",
            "hit_aligned_context_cell",
        ):
            counter = Counter(str(row[f"{prefix}_state"]) for row in per_case)
            aggregate[f"{prefix}_state_counts"] = {
                state: counter.get(state, 0)
                for state in ("zero", "partial", "complete")
            }
    return aggregate


def evaluate_tagop_mapping_retrieval(
    annotation_path: Path,
    predictions_path: Path,
    *,
    stage: str | None = None,
    document_k: int = 10,
    emitted_unit_k: int = 10,
    evidence_hit_k: int = 10,
    query_slot_k: int = 10,
    expected_annotation_sha256: str | None = TAGOP_TRAIN_SHA256,
) -> dict[str, Any]:
    """Evaluate immutable predictions against TagOp's heuristic table mapping.

    ``context_*`` gives full credit when the complete table document is present
    in the first ``document_k`` retrieved documents.  ``emitted_*`` only checks
    the explicit, de-duplicated row/cell unit lists persisted by the experiment
    runner.  The latter lists are unit order, not evidence-hit rank.
    """

    if (
        document_k <= 0
        or emitted_unit_k <= 0
        or evidence_hit_k <= 0
        or query_slot_k <= 0
    ):
        raise TATQAMappingDiagnosticError("diagnostic cutoffs must be positive")
    annotation_path = annotation_path.resolve()
    predictions_path = predictions_path.resolve()
    annotation_sha256 = _sha256(annotation_path)
    if (
        expected_annotation_sha256 is not None
        and annotation_sha256 != expected_annotation_sha256
    ):
        raise TATQAMappingDiagnosticError(
            "TagOp annotation SHA-256 does not match the pinned artifact"
        )
    mappings = _load_mappings(annotation_path)
    predictions = _load_predictions(predictions_path, stage=stage)
    missing_annotation = [
        prediction.case_id
        for prediction in predictions
        if prediction.case_id not in mappings
    ]
    if missing_annotation:
        raise TATQAMappingDiagnosticError(
            "TagOp annotation does not cover every prediction case: "
            + ", ".join(missing_annotation[:3])
        )

    per_case: list[dict[str, Any]] = []
    no_table_mapping = 0
    duplicate_coordinates = 0
    header_row_cases = 0
    for prediction in predictions:
        mapping = mappings[prediction.case_id]
        if not mapping.coordinates:
            no_table_mapping += 1
            continue
        duplicate_coordinates += mapping.duplicate_coordinate_count
        if any(row_index == 0 for row_index, _ in mapping.coordinates):
            header_row_cases += 1
        gold_rows = {
            f"{mapping.table_document_id}::row::{row_index}"
            for row_index, _ in mapping.coordinates
        }
        gold_cells = {
            f"{mapping.table_document_id}::cell::{row_index}::{column_index}"
            for row_index, column_index in mapping.coordinates
        }
        emitted_rows = prediction.retrieved_row_ids[:emitted_unit_k]
        emitted_cells = prediction.retrieved_cell_ids[:emitted_unit_k]
        row_recall = _recall(gold_rows, emitted_rows)
        cell_recall = _recall(gold_cells, emitted_cells)
        table_rank = None
        if mapping.table_document_id in prediction.retrieved_ids:
            table_rank = prediction.retrieved_ids.index(mapping.table_document_id) + 1
        complete_table = (
            mapping.table_document_id in prediction.retrieved_complete_table_ids
            and table_rank is not None
            and table_rank <= document_k
        )
        context_row_recall = 1.0 if complete_table else row_recall
        context_cell_recall = 1.0 if complete_table else cell_recall
        slot_plan = select_tatqa_table_slots(
            mapping.question,
            [list(row) for row in mapping.table],
            budget=query_slot_k,
        )
        selected_coordinates = {
            (slot.row_index, slot.column_index) for slot in slot_plan.slots
        }
        selected_rows = {row_index for row_index, _ in selected_coordinates}
        gold_coordinates = set(mapping.coordinates)
        gold_row_indices = {row_index for row_index, _ in mapping.coordinates}
        query_slot_cell_recall = len(
            gold_coordinates.intersection(selected_coordinates)
        ) / len(gold_coordinates)
        query_slot_row_recall = len(
            gold_row_indices.intersection(selected_rows)
        ) / len(gold_row_indices)
        case_result: dict[str, Any] = {
                "case_id": prediction.case_id,
                "category": prediction.category,
                "answer_from": mapping.answer_from,
                "gold_coordinate_count": len(gold_cells),
                "gold_row_count": len(gold_rows),
                "contains_header_row_coordinate": any(
                    row_index == 0 for row_index, _ in mapping.coordinates
                ),
                "hit_alignment_available": prediction.hit_alignment_available,
                "complete_table_rank": table_rank,
                "complete_table_at_document_k": float(complete_table),
                "tagop_heuristic_emitted_row_unit_recall": row_recall,
                "tagop_heuristic_emitted_cell_unit_recall": cell_recall,
                "tagop_heuristic_context_row_coverage": context_row_recall,
                "tagop_heuristic_context_cell_coverage": context_cell_recall,
                "query_slot_operator": slot_plan.operator,
                "query_slot_coordinates": [
                    [slot.row_index, slot.column_index] for slot in slot_plan.slots
                ],
                "tagop_heuristic_query_slot_cell_recall_at_k": query_slot_cell_recall,
                "tagop_heuristic_query_slot_row_recall_at_k": query_slot_row_recall,
                "emitted_row_state": _state(row_recall),
                "emitted_cell_state": _state(cell_recall),
                "context_row_state": _state(context_row_recall),
                "context_cell_state": _state(context_cell_recall),
                "query_slot_cell_state": _state(query_slot_cell_recall),
                "query_slot_row_state": _state(query_slot_row_recall),
        }
        if mapping.answer_type in {"arithmetic", "count"}:
            program_executable = False
            program_answer_correct = False
            program_result: str | None = None
            program_error: str | None = None
            try:
                executed = execute_tatqa_derivation(
                    mapping.answer_type,  # type: ignore[arg-type]
                    mapping.derivation,
                    scale=mapping.scale,
                    operator=slot_plan.operator,
                )
                program_executable = True
                program_result = str(executed)
                program_answer_correct = tatqa_program_matches_answer(
                    executed, mapping.answer
                )
            except TATQAProgramExecutionError as exc:
                program_error = str(exc)
            query_slots_cover_program = query_slot_cell_recall == 1.0
            case_result["program_oracle"] = {
                "diagnostic_only": True,
                "uses_gold_derivation": True,
                "executable": program_executable,
                "answer_correct": program_answer_correct,
                "result": program_result,
                "error": program_error,
                "provided_context_full_operand_coverage": complete_table,
                "query_slots_full_operand_coverage": query_slots_cover_program,
                "provided_context_gold_program_success": (
                    complete_table and program_answer_correct
                ),
                "query_slots_gold_program_success": (
                    query_slots_cover_program and program_answer_correct
                ),
            }
        if prediction.hit_alignment_available:
            aligned_hits = [
                hit
                for hit in prediction.retrieved_table_units_by_hit
                if int(hit["rank"]) <= evidence_hit_k
            ]
            hit_rows = [
                str(hit["row_id"])
                for hit in aligned_hits
                if hit.get("row_id") is not None
            ]
            hit_cells = [
                str(hit["cell_id"])
                for hit in aligned_hits
                if hit.get("cell_id") is not None
            ]
            hit_complete_table = any(
                hit["document_id"] == mapping.table_document_id
                and hit.get("table_complete") is True
                for hit in aligned_hits
            )
            hit_row_recall = _recall(gold_rows, hit_rows)
            hit_cell_recall = _recall(gold_cells, hit_cells)
            hit_context_row_recall = 1.0 if hit_complete_table else hit_row_recall
            hit_context_cell_recall = 1.0 if hit_complete_table else hit_cell_recall
            case_result.update(
                {
                    "complete_table_at_evidence_hit_k": float(hit_complete_table),
                    "tagop_heuristic_hit_aligned_row_recall_at_k": hit_row_recall,
                    "tagop_heuristic_hit_aligned_cell_recall_at_k": hit_cell_recall,
                    "tagop_heuristic_hit_aligned_context_row_coverage_at_k": hit_context_row_recall,
                    "tagop_heuristic_hit_aligned_context_cell_coverage_at_k": hit_context_cell_recall,
                    "hit_aligned_row_state": _state(hit_row_recall),
                    "hit_aligned_cell_state": _state(hit_cell_recall),
                    "hit_aligned_context_row_state": _state(hit_context_row_recall),
                    "hit_aligned_context_cell_state": _state(hit_context_cell_recall),
                }
            )
        per_case.append(case_result)

    if not per_case:
        raise TATQAMappingDiagnosticError(
            "none of the prediction cases has a TagOp table mapping"
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_case:
        grouped[str(row["category"])].append(row)
    program_rows = [row for row in per_case if "program_oracle" in row]

    def program_rate(field: str) -> float:
        return _mean(
            [float(bool(row["program_oracle"][field])) for row in program_rows]
        )
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "annotation": {
            "kind": "tagop_heuristic_mapping",
            "source_commit": TAGOP_SOURCE_COMMIT,
            "source_url": TAGOP_SOURCE_URL,
            "path": str(annotation_path),
            "sha256": annotation_sha256,
            "size_bytes": annotation_path.stat().st_size,
            "heuristically_generated": True,
            "manually_verified_gold": False,
        },
        "predictions": {
            "path": str(predictions_path),
            "sha256": _sha256(predictions_path),
            "size_bytes": predictions_path.stat().st_size,
            "stage": predictions[0].stage,
            "case_count": len(predictions),
        },
        "cutoffs": {
            "document_k": document_k,
            "emitted_unit_k": emitted_unit_k,
            "evidence_hit_k": evidence_hit_k,
            "query_slot_k": query_slot_k,
        },
        "coverage": {
            "prediction_cases": len(predictions),
            "mapping_eligible_cases": len(per_case),
            "no_table_mapping_cases": no_table_mapping,
            "header_row_mapping_cases": header_row_cases,
            "deduplicated_repeated_coordinates": duplicate_coordinates,
            "hit_alignment_available_cases": sum(
                prediction.hit_alignment_available for prediction in predictions
            ),
        },
        "aggregate": _aggregate(per_case),
        "by_category": {
            category: {"case_count": len(rows), **_aggregate(rows)}
            for category, rows in sorted(grouped.items())
        },
        "program_oracle": {
            "diagnostic_only": True,
            "uses_gold_derivation": True,
            "eligible_cases": len(program_rows),
            "gold_program_executable_rate": program_rate("executable"),
            "gold_program_answer_accuracy": program_rate("answer_correct"),
            "provided_context_full_operand_coverage": program_rate(
                "provided_context_full_operand_coverage"
            ),
            "query_slots_full_operand_coverage": program_rate(
                "query_slots_full_operand_coverage"
            ),
            "provided_context_gold_program_success": program_rate(
                "provided_context_gold_program_success"
            ),
            "query_slots_gold_program_success": program_rate(
                "query_slots_gold_program_success"
            ),
        },
        "limitations": [
            "TagOp mappings are heuristic preprocessing outputs, not official manually verified TAT-QA evidence labels.",
            "A complete-table hit means the mapped coordinate is available in context, not that the answer generator attended to it.",
            "Emitted row/cell lists are de-duplicated unit order rather than evidence-hit rank, so their cutoff is not document Recall@K.",
            "Hit-aligned metrics are emitted only when every prediction records original evidence-hit ranks.",
            "The current row-unit extractor cannot emit header row 0; header mappings therefore expose an instrumentation/representation gap.",
            "This diagnostic must not be used to tune or inspect the hidden document-disjoint split.",
            "The query slot selector consumes only question text and table contents; TagOp mappings are used only after selection for diagnostic scoring.",
            "Program-oracle results use the gold derivation and only measure executor/operand-coverage upper bounds; they are not end-to-end answer accuracy.",
        ],
        "per_case": per_case,
    }

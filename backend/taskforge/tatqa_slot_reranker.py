"""Trainable query-aware table-cell reranker with immutable feature schema."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .tatqa_slot_selector import (
    TATQA_SLOT_FEATURE_NAMES,
    TATQASlotPlan,
    TATQATableSlot,
    classify_tatqa_operator,
    tatqa_slot_feature_vector,
    tatqa_slot_heuristic_score,
)


class TATQASlotReranker:
    """Pure-Python inference for a category-balanced logistic slot model."""

    def __init__(
        self,
        weights: Sequence[float],
        bias: float,
        feature_mean: Sequence[float],
        feature_scale: Sequence[float],
        *,
        model_id: str,
        training: Mapping[str, Any],
    ) -> None:
        dimension = len(TATQA_SLOT_FEATURE_NAMES)
        if not all(
            len(values) == dimension
            for values in (weights, feature_mean, feature_scale)
        ):
            raise ValueError("slot reranker parameters do not match feature schema")
        values = [*weights, bias, *feature_mean, *feature_scale]
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("slot reranker parameters must be finite")
        if any(float(value) <= 0 for value in feature_scale):
            raise ValueError("slot reranker feature scales must be positive")
        self.weights = tuple(float(value) for value in weights)
        self.bias = float(bias)
        self.feature_mean = tuple(float(value) for value in feature_mean)
        self.feature_scale = tuple(float(value) for value in feature_scale)
        self.model_id = str(model_id).strip()
        self.training = dict(training)
        if not self.model_id:
            raise ValueError("slot reranker model_id must not be empty")

    def score_coordinate(
        self,
        question: str,
        table: list[list[str]],
        row_index: int,
        column_index: int,
    ) -> float:
        features = tatqa_slot_feature_vector(
            question, table, row_index, column_index
        )
        return self.bias + sum(
            weight * ((value - mean) / scale)
            for weight, value, mean, scale in zip(
                self.weights,
                features,
                self.feature_mean,
                self.feature_scale,
                strict=True,
            )
        )

    def select(
        self,
        question: str,
        table: list[list[str]],
        *,
        budget: int = 10,
        fusion: str = "equal_rrf",
        rrf_k: int = 60,
        heuristic_weight: float = 1.0,
        learned_weight: float = 1.0,
    ) -> TATQASlotPlan:
        if budget <= 0 or rrf_k <= 0:
            raise ValueError("slot and RRF budgets must be positive")
        if (
            not math.isfinite(heuristic_weight)
            or not math.isfinite(learned_weight)
            or heuristic_weight <= 0
            or learned_weight <= 0
        ):
            raise ValueError("slot RRF weights must be finite and positive")
        coordinates = [
            (row_index, column_index)
            for row_index, row in enumerate(table)
            for column_index, value in enumerate(row)
            if str(value).strip()
        ]
        learned = sorted(
            coordinates,
            key=lambda coordinate: (
                -self.score_coordinate(question, table, *coordinate),
                coordinate[0],
                coordinate[1],
            ),
        )
        heuristic = sorted(
            coordinates,
            key=lambda coordinate: (
                -tatqa_slot_heuristic_score(question, table, *coordinate),
                coordinate[0],
                coordinate[1],
            ),
        )
        if fusion == "learned_only":
            ranked = learned
        elif fusion in {"equal_rrf", "weighted_rrf"}:
            if fusion == "equal_rrf":
                heuristic_weight = learned_weight = 1.0
            learned_rank = {coordinate: rank for rank, coordinate in enumerate(learned, 1)}
            heuristic_rank = {
                coordinate: rank for rank, coordinate in enumerate(heuristic, 1)
            }
            ranked = sorted(
                coordinates,
                key=lambda coordinate: (
                    -(
                        learned_weight / (rrf_k + learned_rank[coordinate])
                        + heuristic_weight / (rrf_k + heuristic_rank[coordinate])
                    ),
                    coordinate[0],
                    coordinate[1],
                ),
            )
        else:
            raise ValueError(
                "slot fusion must be learned_only, equal_rrf, or weighted_rrf"
            )
        width = max((len(row) for row in table), default=0)
        header = [str(value) for value in table[0]] + [""] * (width - len(table[0]))
        slots = []
        for row_index, column_index in ranked[:budget]:
            row = [str(value) for value in table[row_index]] + [""] * (
                width - len(table[row_index])
            )
            slots.append(
                TATQATableSlot(
                    row_index=row_index,
                    column_index=column_index,
                    value=row[column_index],
                    row_label=row[0],
                    column_header=header[column_index],
                    score=self.score_coordinate(
                        question, table, row_index, column_index
                    ),
                    signals=("learned_slot_reranker", fusion),
                )
            )
        return TATQASlotPlan(
            operator=classify_tatqa_operator(question),
            query_terms=(),
            years=(),
            slots=tuple(slots),
        )

    def model_dump(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "model_id": self.model_id,
            "kind": "category_balanced_logistic_slot_reranker",
            "feature_names": list(TATQA_SLOT_FEATURE_NAMES),
            "weights": list(self.weights),
            "bias": self.bias,
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "training": self.training,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> TATQASlotReranker:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("slot reranker artifact must be an object")
        if payload.get("feature_names") != list(TATQA_SLOT_FEATURE_NAMES):
            raise ValueError("slot reranker feature schema is incompatible")
        training = payload.get("training")
        if not isinstance(training, dict):
            raise ValueError("slot reranker training provenance is missing")
        return cls(
            payload["weights"],
            payload["bias"],
            payload["feature_mean"],
            payload["feature_scale"],
            model_id=payload["model_id"],
            training=training,
        )

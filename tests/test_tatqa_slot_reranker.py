from __future__ import annotations

from pathlib import Path

import pytest

from taskforge.tatqa_slot_reranker import TATQASlotReranker
from taskforge.tatqa_slot_selector import TATQA_SLOT_FEATURE_NAMES


def _model() -> TATQASlotReranker:
    weights = [0.0] * len(TATQA_SLOT_FEATURE_NAMES)
    weights[0] = 3.0
    weights[3] = 2.0
    return TATQASlotReranker(
        weights,
        0.0,
        [0.0] * len(weights),
        [1.0] * len(weights),
        model_id="fixture",
        training={"split_id": "fit"},
    )


def test_slot_reranker_selects_and_round_trips(tmp_path: Path) -> None:
    table = [["", "2021", "2020"], ["Revenue", "120", "100"], ["Cost", "80", "70"]]
    model = _model()
    plan = model.select("What was revenue in 2021?", table, budget=2)
    assert (plan.slots[0].row_index, plan.slots[0].column_index) == (1, 1)
    path = tmp_path / "slot-model.json"
    model.save(path)
    assert TATQASlotReranker.load(path).model_dump() == model.model_dump()


def test_slot_reranker_rejects_schema_and_fusion(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fusion"):
        _model().select("question", [["x"]], fusion="unknown")
    with pytest.raises(ValueError, match="weights"):
        _model().select(
            "question",
            [["x"]],
            fusion="weighted_rrf",
            heuristic_weight=0,
        )
    path = tmp_path / "bad.json"
    path.write_text('{"feature_names": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        TATQASlotReranker.load(path)

from __future__ import annotations

import pytest

from taskforge.tatqa_slot_selector import (
    classify_tatqa_operator,
    render_tatqa_slot_context,
    select_tatqa_table_slots,
)


def test_selector_uses_metric_and_year_without_labels_or_answers() -> None:
    table = [
        ["", "2021", "2020"],
        ["Revenue", "120", "100"],
        ["Operating costs", "80", "70"],
    ]

    plan = select_tatqa_table_slots(
        "What was the difference in revenue between 2021 and 2020?",
        table,
        budget=2,
    )

    assert plan.operator == "subtract"
    assert {(slot.row_index, slot.column_index) for slot in plan.slots} == {
        (1, 1),
        (1, 2),
    }
    assert all("row_label_overlap" in slot.signals for slot in plan.slots)
    assert all("query_year_header_match" in slot.signals for slot in plan.slots)


def test_selector_is_deterministic_and_budgeted() -> None:
    table = [["", "2021"], ["Revenue", "10"]]
    first = select_tatqa_table_slots("What was revenue in 2021?", table, budget=2)
    second = select_tatqa_table_slots("What was revenue in 2021?", table, budget=2)

    assert first == second
    assert len(first.slots) == 2
    assert classify_tatqa_operator("How many entries exceeded 10?") == "count"
    assert classify_tatqa_operator("What was the percentage change?") == (
        "percentage_change"
    )
    rendered = render_tatqa_slot_context(first)
    assert "operator: lookup" in rendered
    assert "row_index=" in rendered


def test_selector_validates_its_public_contract() -> None:
    with pytest.raises(ValueError, match="question"):
        select_tatqa_table_slots("", [["x"]])
    with pytest.raises(ValueError, match="budget"):
        select_tatqa_table_slots("question", [["x"]], budget=0)
    with pytest.raises(ValueError, match="table"):
        select_tatqa_table_slots("question", [])

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.create_eval_split import (
    _parse_category_minimums,
    _select_parent_disjoint_cases,
    parser,
)


def test_all_eligible_contract_is_parent_grouped() -> None:
    args = parser().parse_args(
        [
            "--input",
            "input.json",
            "--output",
            "output.json",
            "--split-id",
            "fit",
            "--seed",
            "1",
            "--all-eligible",
            "--group-by-parent",
        ]
    )
    assert args.all_eligible is True
    assert args.group_by_parent is True


def test_qasper_adapter_can_be_selected_explicitly() -> None:
    args = parser().parse_args(
        [
            "--input",
            "input.json",
            "--output",
            "output.json",
            "--split-id",
            "qasper-tuning",
            "--seed",
            "1",
            "--dataset-adapter",
            "qasper",
        ]
    )
    assert args.dataset_adapter == "qasper"


def _case(case_id: str, parent: str) -> SimpleNamespace:
    return SimpleNamespace(case_id=case_id, metadata={"parent_document_id": parent})


def test_parent_disjoint_selection_keeps_whole_documents() -> None:
    cases = [
        _case("a-1", "a"),
        _case("a-2", "a"),
        _case("b-1", "b"),
        _case("c-1", "c"),
        _case("c-2", "c"),
        _case("c-3", "c"),
    ]

    selected = _select_parent_disjoint_cases(cases, limit=5, seed=7)

    by_parent: dict[str, set[str]] = {}
    for item in selected:
        by_parent.setdefault(item.metadata["parent_document_id"], set()).add(item.case_id)
    assert len(selected) == 5
    assert {"a-1", "a-2"} <= {case_id for ids in by_parent.values() for case_id in ids}
    assert all(
        ids == {case.case_id for case in cases if case.metadata["parent_document_id"] == parent}
        for parent, ids in by_parent.items()
    )


def test_parent_disjoint_selection_requires_exact_group_sum() -> None:
    cases = [_case("a-1", "a"), _case("a-2", "a"), _case("b-1", "b")]

    with pytest.raises(ValueError, match="cannot select"):
        _select_parent_disjoint_cases(cases, limit=4, seed=7)


def test_parent_disjoint_selection_requires_parent_metadata() -> None:
    with pytest.raises(ValueError, match="parent_document_id"):
        _select_parent_disjoint_cases([_case("a-1", "")], limit=1, seed=7)


def test_parent_disjoint_selection_can_enforce_category_minimums() -> None:
    cases = [
        _case("a-1", "a"),
        _case("a-2", "a"),
        _case("b-1", "b"),
        _case("b-2", "b"),
        _case("c-1", "c"),
        _case("c-2", "c"),
    ]
    for item, category in zip(cases, ["text", "text", "table", "table", "count", "count"], strict=True):
        item.category = category

    selected = _select_parent_disjoint_cases(
        cases,
        limit=4,
        seed=7,
        category_minimums={"text": 2, "table": 2},
    )

    assert {item.category for item in selected} == {"text", "table"}
    assert {item.metadata["parent_document_id"] for item in selected} == {"a", "b"}


def test_category_minimum_parser_is_strict() -> None:
    assert _parse_category_minimums(["table=20", "count=8"]) == {
        "table": 20,
        "count": 8,
    }
    with pytest.raises(ValueError, match="CATEGORY=COUNT"):
        _parse_category_minimums(["table"])

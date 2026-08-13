from __future__ import annotations

from decimal import Decimal

import pytest

from taskforge.tatqa_program_executor import (
    TATQAProgramExecutionError,
    execute_tatqa_derivation,
    tatqa_program_matches_answer,
)


@pytest.mark.parametrize(
    ("derivation", "scale", "expected"),
    [
        ("4,487-2,201", "million", Decimal("2286")),
        ("(72,130 - 41,165)/41,165", "percent", Decimal("75.2216688935")),
        ("14%-12%", "percent", Decimal("2.00")),
        ("37-35", "percent", Decimal("2")),
        ("(273+274)/2", "million", Decimal("273.5")),
    ],
)
def test_arithmetic_executor_is_constrained_and_scale_aware(
    derivation: str,
    scale: str,
    expected: Decimal,
) -> None:
    result = execute_tatqa_derivation("arithmetic", derivation, scale=scale)
    assert abs(result - expected) < Decimal("0.000000001")


def test_count_executor_counts_explicit_items() -> None:
    assert execute_tatqa_derivation("count", "2019##2018##2017") == Decimal(3)
    assert execute_tatqa_derivation("count", "Technology") == Decimal(1)


def test_percent_average_does_not_get_ratio_scaling() -> None:
    assert execute_tatqa_derivation(
        "arithmetic",
        "(41.5 + 39.8 + 41.0) / 3",
        scale="percent",
        operator="average",
    ) == Decimal("40.76666666666666666666666667")


def test_executor_rejects_calls_and_matches_rounded_answers() -> None:
    with pytest.raises(TATQAProgramExecutionError, match="invalid|unsupported"):
        execute_tatqa_derivation("arithmetic", "sum([1, 2])")
    assert tatqa_program_matches_answer(Decimal("75.225312"), 75.23)
    assert not tatqa_program_matches_answer(Decimal("75.225312"), 75.1)

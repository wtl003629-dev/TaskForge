"""A constrained, side-effect-free executor for TAT-QA gold-program oracles."""

from __future__ import annotations

import ast
import re
from decimal import Decimal, DivisionByZero, InvalidOperation
from typing import Any, Literal

TATQAExecutableAnswerType = Literal["arithmetic", "count"]

_PERCENT_LITERAL = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)%")
_CURRENCY = re.compile(r"[$£€¥]")


class TATQAProgramExecutionError(ValueError):
    """Raised when a derivation is outside the constrained arithmetic grammar."""


def _evaluate_node(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        try:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        except (DivisionByZero, InvalidOperation) as exc:
            raise TATQAProgramExecutionError("invalid arithmetic operation") from exc
    raise TATQAProgramExecutionError(
        f"unsupported derivation syntax: {type(node).__name__}"
    )


def execute_tatqa_derivation(
    answer_type: TATQAExecutableAnswerType,
    derivation: str,
    *,
    scale: str = "",
    operator: str | None = None,
) -> Decimal:
    """Execute only arithmetic expressions or explicit count item lists."""

    cleaned = derivation.strip()
    if not cleaned:
        raise TATQAProgramExecutionError("derivation must not be empty")
    if answer_type == "count":
        items = [item.strip() for item in cleaned.split("##") if item.strip()]
        if not items:
            raise TATQAProgramExecutionError("count derivation has no items")
        return Decimal(len(items))
    if answer_type != "arithmetic":
        raise TATQAProgramExecutionError(
            f"unsupported answer type for execution: {answer_type}"
        )

    had_percent_literal = bool(_PERCENT_LITERAL.search(cleaned))
    normalized = cleaned.replace("−", "-").replace(",", "")
    normalized = _CURRENCY.sub("", normalized)
    normalized = _PERCENT_LITERAL.sub(r"(\1/100)", normalized)
    try:
        parsed = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise TATQAProgramExecutionError("invalid arithmetic derivation") from exc
    result = _evaluate_node(parsed)
    explicit_times_100 = bool(re.search(r"\*\s*100(?:\.0+)?\b", normalized))
    inferred_ratio = operator is None and "/" in normalized
    declared_ratio = operator == "percentage_change" and "/" in normalized
    if scale.casefold() == "percent" and (
        had_percent_literal
        or ((inferred_ratio or declared_ratio) and not explicit_times_100)
    ):
        result *= Decimal(100)
    return result


def tatqa_program_matches_answer(result: Decimal, answer: Any) -> bool:
    """Compare an executed number to TAT-QA's released rounded answer."""

    if isinstance(answer, bool) or isinstance(answer, list | dict) or answer is None:
        return False
    try:
        expected = Decimal(str(answer).replace(",", "").strip())
    except InvalidOperation:
        return False
    tolerance = max(Decimal("0.011"), abs(expected) * Decimal("0.000000001"))
    return abs(result - expected) <= tolerance

"""Dependency-free implementation of the released TAT-QA answer metric.

The upstream evaluator uses DROP-style token bags, numeric/scale
normalisation, and one-to-one alignment for multi-span answers.  This module
keeps the same semantics without adding pandas, numpy, or scipy to TaskForge's
runtime dependencies.  It is deliberately separate from the generic token
F1 scorer so reports cannot silently mix the two metrics.
"""

from __future__ import annotations

import re
import string
from collections.abc import Sequence
from typing import Any

_NUMBER_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)|([+-]?\.\d+)")
_SCALE_RE = re.compile(r"\b(hundred|thousand|million|billion|percent)\b", re.I)
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u3400-\u9fff]")
_PUNCTUATION = set(string.punctuation)
_NUM_EXCLUDED = set("'\"\\$€£¥%(),[]")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def tatqa_scale_to_num(scale: object) -> float:
    value = str(scale or "").casefold()
    if "hundred" in value:
        return 100.0
    if "thousand" in value:
        return 1_000.0
    if "million" in value:
        return 1_000_000.0
    if "billion" in value:
        return 1_000_000_000.0
    if "percent" in value:
        return 0.01
    return 1.0


def _clean_number_text(value: object) -> str:
    return "".join(character for character in str(value) if character not in _NUM_EXCLUDED)


def _extract_number(value: object) -> int | float | None:
    match = _NUMBER_RE.search(_clean_number_text(value))
    if match is None:
        return None
    token = match.group(0)
    try:
        return float(token) if "." in token else int(token)
    except ValueError:
        return None


def _is_number(value: object) -> bool:
    words = _clean_number_text(value).split()
    if not words:
        return False
    try:
        number = float(words[0])
    except ValueError:
        return False
    if number != number:  # NaN
        return False
    return len(words) < 2 or tatqa_scale_to_num(words[1]) == 1.0


def _to_number(value: object) -> float | None:
    number = _extract_number(value)
    if number is None:
        return None
    text = str(value)
    multiplier = tatqa_scale_to_num(_SCALE_RE.search(text).group(1)) if _SCALE_RE.search(text) else 1.0
    if re.search(r"\([\d.\s,]+\)", text.strip()):
        multiplier *= -1.0
    if re.search(r"[\d.\s]+%", text.strip()):
        multiplier *= 0.01
    return round(float(number) * multiplier, 4)


def tatqa_normalize_answer(value: object) -> str:
    """Match the normalization in the released ``tatqa_utils.py``."""

    tokens = str(value).casefold().split()
    normalized: list[str] = []
    for token in tokens:
        token = token.strip()
        if _is_number(token):
            number = _to_number(token)
            token = str(number) if number is not None else token
        elif not _is_number(token):
            token = "".join(character for character in token if character not in _PUNCTUATION)
        token = _ARTICLES_RE.sub(" ", token)
        token = " ".join(token.split())
        if token:
            normalized.append(token)
    return " ".join(normalized).strip()


def _answer_to_bags(answer: object) -> tuple[list[str], list[set[str]]]:
    raw_spans = list(answer) if isinstance(answer, (list, tuple)) else [answer]
    normalized = [tatqa_normalize_answer(span) for span in raw_spans]
    return normalized, [set(span.split()) for span in normalized]


def _pair_f1(predicted: set[str], gold: set[str]) -> float:
    intersection = len(predicted.intersection(gold))
    precision = 1.0 if not predicted else intersection / len(predicted)
    recall = 1.0 if not gold else intersection / len(gold)
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _maximum_alignment_sum(predicted: Sequence[set[str]], gold: Sequence[set[str]]) -> float:
    """Return the maximum one-to-one sum, with zero for unmatched bags."""

    if not predicted or not gold:
        return 0.0
    if len(predicted) > len(gold):
        predicted, gold = gold, predicted
    scores = [[_pair_f1(item, other) for other in gold] for item in predicted]
    memo: dict[tuple[int, int], float] = {}

    def visit(index: int, used: int) -> float:
        if index == len(predicted):
            return 0.0
        key = (index, used)
        if key in memo:
            return memo[key]
        best = 0.0
        for column, score in enumerate(scores[index]):
            bit = 1 << column
            if not used & bit:
                best = max(best, score + visit(index + 1, used | bit))
        memo[key] = best
        return best

    return visit(0, 0)


def _format_answer_strings(values: Sequence[object], scale: object) -> list[str]:
    rendered: list[str] = []
    for value in sorted((str(item) for item in values)):
        if _is_number(value):
            number = _to_number(value)
            if number is not None:
                if "%" in value:
                    value = f"{number:.4f}"
                else:
                    value = f"{round(number, 2) * tatqa_scale_to_num(scale):.4f}"
        elif str(scale or ""):
            value = f"{value} {scale}"
        rendered.append(value)
    return [" ".join(rendered)]


def _infer_prediction_scale(value: object) -> tuple[str, str]:
    text = str(value).strip()
    if "%" in text:
        return "percent", text.replace("%", " ").strip()
    match = _SCALE_RE.search(text)
    if match is None:
        return "", text
    scale = match.group(1).casefold()
    return scale, re.sub(rf"\b{re.escape(match.group(1))}\b", " ", text, flags=re.I).strip()


def tatqa_answer_metrics(
    prediction: object,
    gold_answer: object,
    *,
    answer_type: str = "",
    gold_scale: str = "",
    prediction_scale: str | None = None,
) -> dict[str, Any]:
    """Score one prediction using the released TAT-QA EM/F1 semantics."""

    inferred_scale, cleaned_prediction = _infer_prediction_scale(prediction)
    pred_scale = inferred_scale if prediction_scale is None else prediction_scale
    if isinstance(gold_answer, list) and answer_type not in {"span", "multi-span"}:
        gold_values = [gold_answer[0]] if gold_answer else []
    elif answer_type == "count" and gold_answer is not None:
        gold_values = [str(int(float(gold_answer)))]
    elif isinstance(gold_answer, (list, tuple)):
        gold_values = list(gold_answer)
    else:
        gold_values = [gold_answer]
    if not cleaned_prediction.strip() or not gold_values:
        return {
            "exact_match": 0.0,
            "f1": 0.0,
            "scale_match": float(pred_scale == gold_scale),
            "prediction_scale": pred_scale,
            "gold_scale": gold_scale,
        }
    gold_strings = _format_answer_strings(gold_values, gold_scale)
    pred_values = prediction if isinstance(prediction, (list, tuple)) else [cleaned_prediction]
    prediction_strings = _format_answer_strings(pred_values, pred_scale)
    if not pred_scale and "%" not in str(prediction) and _is_number(cleaned_prediction):
        prediction_strings.append(f"{float(_to_number(cleaned_prediction) or 0):.4f}")
    exact = 0.0
    best_f1 = 0.0
    for predicted_string in prediction_strings:
        predicted_norm, predicted_bags = _answer_to_bags(predicted_string)
        for gold_string in gold_strings:
            gold_norm, gold_bags = _answer_to_bags(gold_string)
            if set(predicted_norm) == set(gold_norm) and len(predicted_norm) == len(gold_norm):
                exact = max(exact, 1.0)
            denominator = max(len(predicted_bags), len(gold_bags))
            aligned = _maximum_alignment_sum(predicted_bags, gold_bags)
            best_f1 = max(best_f1, aligned / denominator if denominator else 0.0)
    if answer_type in {"arithmetic", "count"}:
        best_f1 = exact
    return {
        "exact_match": exact,
        "f1": round(best_f1, 2),
        "scale_match": float(pred_scale == gold_scale),
        "prediction_scale": pred_scale,
        "gold_scale": gold_scale,
    }


__all__ = ["tatqa_answer_metrics", "tatqa_normalize_answer", "tatqa_scale_to_num"]

"""Strict, paired promotion gates for immutable RAG answer-eval artifacts.

The answer evaluator deliberately owns execution while this module owns only
comparison.  A run is accepted only after its manifest hashes have been
verified and the fields needed by the promotion gate are present.  Historical
runs that omit latency, evidence labels, execution errors, fallback disclosure,
or an answer contract therefore cannot accidentally pass a modern gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ComparisonType = Literal["retriever", "agentic"]
_ARTIFACT_NAMES = ("metrics.json", "predictions.jsonl")
_HEX = frozenset("0123456789abcdef")


class RAGAnswerGateError(RuntimeError):
    """Raised when an answer-eval artifact cannot be trusted or compared."""


@dataclass(frozen=True)
class BootstrapConfig:
    repetitions: int = 10_000
    confidence: float = 0.95
    seed: int = 20_260_809

    def __post_init__(self) -> None:
        if self.repetitions < 100:
            raise ValueError("bootstrap repetitions must be at least 100")
        if not 0.5 < self.confidence < 1.0:
            raise ValueError("bootstrap confidence must be between 0.5 and 1")


@dataclass(frozen=True)
class GateThresholds:
    min_token_f1_delta: float = 0.03
    min_evidence_recall_delta: float = 0.03
    max_category_degradation: float = 0.03
    min_category_cases: int = 10
    max_p95_ratio: float = 2.0
    require_nonnegative_ci_lower: bool = True

    def __post_init__(self) -> None:
        numeric = (
            self.min_token_f1_delta,
            self.min_evidence_recall_delta,
            self.max_category_degradation,
            self.max_p95_ratio,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("gate thresholds must be finite")
        if self.max_category_degradation < 0:
            raise ValueError("max_category_degradation must be non-negative")
        if self.min_category_cases < 1:
            raise ValueError("min_category_cases must be positive")
        if self.max_p95_ratio <= 0:
            raise ValueError("max_p95_ratio must be positive")

    @classmethod
    def for_comparison(cls, comparison_type: ComparisonType) -> GateThresholds:
        if comparison_type == "agentic":
            return cls(max_p95_ratio=2.5)
        return cls()

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_token_f1_delta": self.min_token_f1_delta,
            "min_evidence_recall_delta": self.min_evidence_recall_delta,
            "max_category_degradation": self.max_category_degradation,
            "min_category_cases": self.min_category_cases,
            "max_p95_ratio": self.max_p95_ratio,
            "require_nonnegative_ci_lower": self.require_nonnegative_ci_lower,
        }


@dataclass(frozen=True)
class AnswerPrediction:
    case_id: str
    category: str
    token_f1: float
    evidence_recall: float
    latency_ms: float
    execution_error: bool
    fallback_used: bool


@dataclass(frozen=True)
class AnswerEvalRun:
    label: str
    path: Path
    run_id: str
    dataset_sha256: str
    model: str
    retriever: str
    mode: str
    answer_contract: Any
    budget_signature: Mapping[str, Any]
    predictions: tuple[AnswerPrediction, ...]
    artifact_sha256: Mapping[str, str]

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(row.case_id for row in self.predictions)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RAGAnswerGateError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _json_loads(raw: str, label: str) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=_strict_object)
    except RAGAnswerGateError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RAGAnswerGateError(f"{label} is not valid UTF-8 JSON") from exc


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise RAGAnswerGateError(f"cannot read {label}: {path}") from exc


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    raw = _read_bytes(path, label)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RAGAnswerGateError(f"{label} is not valid UTF-8") from exc
    value = _json_loads(text, label)
    if not isinstance(value, dict):
        raise RAGAnswerGateError(f"{label} must contain one JSON object")
    return value


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RAGAnswerGateError("comparison metadata is not canonical JSON") from exc


def _required_string(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RAGAnswerGateError(f"{label}.{key} must be a non-empty string")
    return value.strip()


def _bounded_score(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RAGAnswerGateError(f"{label} must be a number")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RAGAnswerGateError(f"{label} must be between 0 and 1")
    return score


def _nonnegative_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RAGAnswerGateError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise RAGAnswerGateError(f"{label} must be finite and non-negative")
    return number


def _artifact_hashes(
    run_dir: Path, manifest: Mapping[str, Any]
) -> dict[str, str]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RAGAnswerGateError("manifest.artifacts must be an object")
    verified: dict[str, str] = {}
    for name in _ARTIFACT_NAMES:
        descriptor = artifacts.get(name)
        if not isinstance(descriptor, Mapping):
            raise RAGAnswerGateError(f"manifest.artifacts.{name} is required")
        expected = descriptor.get("sha256")
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in _HEX for character in expected.lower())
        ):
            raise RAGAnswerGateError(f"manifest artifact {name} has invalid sha256")
        raw = _read_bytes(run_dir / name, name)
        actual = _sha256(raw)
        if actual != expected.lower():
            raise RAGAnswerGateError(f"artifact sha256 mismatch: {name}")
        expected_size = descriptor.get("size_bytes")
        if expected_size is not None and expected_size != len(raw):
            raise RAGAnswerGateError(f"artifact size mismatch: {name}")
        verified[name] = actual
    return verified


def _prediction_rows(path: Path) -> list[dict[str, Any]]:
    raw = _read_bytes(path, "predictions.jsonl")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RAGAnswerGateError("predictions.jsonl is not valid UTF-8") from exc
    stripped = text.strip()
    if not stripped:
        raise RAGAnswerGateError("predictions.jsonl must not be empty")
    try:
        value = _json_loads(stripped, "predictions.jsonl")
    except RAGAnswerGateError:
        rows: list[Any] = []
        for index, line in enumerate(text.splitlines(), start=1):
            if line.strip():
                rows.append(_json_loads(line, f"predictions.jsonl line {index}"))
    else:
        rows = value if isinstance(value, list) else [value]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RAGAnswerGateError("predictions.jsonl must contain prediction objects")
    return rows


def _dataset_hash(manifest: Mapping[str, Any]) -> str:
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        raise RAGAnswerGateError("manifest.dataset must be an object")
    value = dataset.get("normalized_sha256")
    if value is None:
        value = dataset.get("sha256")
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value.lower())
    ):
        raise RAGAnswerGateError("manifest dataset requires a 64-character sha256")
    return value.lower()


def _effective_config(manifest: Mapping[str, Any]) -> dict[str, Any]:
    config = manifest.get("config")
    if not isinstance(config, Mapping) or not isinstance(config.get("effective"), dict):
        raise RAGAnswerGateError("manifest.config.effective must be an object")
    return dict(config["effective"])


def _answer_contract(
    manifest: Mapping[str, Any], metrics: Mapping[str, Any], effective: Mapping[str, Any]
) -> Any:
    candidates = (
        manifest.get("answer_contract"),
        effective.get("answer_contract"),
        metrics.get("answer_contract"),
    )
    for value in candidates:
        if value is not None:
            if value == "" or value == {} or value == []:
                raise RAGAnswerGateError("answer_contract must not be empty")
            _canonical(value)
            return value
    raise RAGAnswerGateError("new answer-eval runs must declare answer_contract")


_BUDGET_KEYS = frozenset(
    {
        "agent_max_steps",
        "candidate_k",
        "chunk_max_chars",
        "chunk_overlap_chars",
        "evidence_top_k",
        "global_evidence_top_k",
        "max_cases",
        "max_evidence_chars",
        "max_expanded_hits",
        "max_queries",
        "max_search_calls",
        "max_tool_calls",
        "neighbor_window",
        "search_limit",
        "top_k",
    }
)


def _derived_budget_signature(effective: Mapping[str, Any]) -> dict[str, Any]:
    signature: dict[str, Any] = {}

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value):
                visit(value[key], (*path, str(key)))
            return
        if not path:
            return
        key = path[-1]
        if (
            key in _BUDGET_KEYS
            or "budget" in key
            or key.endswith(("_limit", "_max_steps", "_top_k"))
        ):
            signature[".".join(path)] = value

    visit(effective, ())
    if not signature:
        raise RAGAnswerGateError("effective config contains no comparable budgets")
    _canonical(signature)
    return signature


def _budget_signature(
    manifest: Mapping[str, Any], effective: Mapping[str, Any]
) -> Mapping[str, Any]:
    explicit = manifest.get("budgets", effective.get("budgets"))
    if explicit is not None:
        if not isinstance(explicit, Mapping) or not explicit:
            raise RAGAnswerGateError("budgets must be a non-empty object")
        signature = dict(explicit)
        _canonical(signature)
        return signature
    return _derived_budget_signature(effective)


def _consistent_value(
    metrics: Mapping[str, Any], effective: Mapping[str, Any], key: str
) -> str:
    values = [value for value in (metrics.get(key), effective.get(key)) if value is not None]
    if not values:
        raise RAGAnswerGateError(f"answer-eval run does not declare {key}")
    cleaned = [str(value).strip() for value in values]
    if any(not value for value in cleaned) or len(set(cleaned)) != 1:
        raise RAGAnswerGateError(f"metrics/config disagree on {key}")
    return cleaned[0]


def _evidence_recall(row: Mapping[str, Any], label: str) -> float:
    # New answer-eval artifacts expose the actual Top-10 retrieval metric
    # explicitly.  Keep the legacy field as a compatibility fallback for
    # historical runs, whose ``evidence_recall`` was candidate-list recall.
    if "retrieval_top10_recall" in row:
        return _bounded_score(row["retrieval_top10_recall"], f"{label}.retrieval_top10_recall")
    if "evidence_recall" in row:
        return _bounded_score(row["evidence_recall"], f"{label}.evidence_recall")
    relevant = row.get("relevant_ids")
    retrieved = row.get("retrieved_ids")
    if (
        not isinstance(relevant, list)
        or not relevant
        or any(not isinstance(value, str) or not value for value in relevant)
    ):
        raise RAGAnswerGateError(f"{label}.relevant_ids must be a non-empty string list")
    if not isinstance(retrieved, list) or any(
        not isinstance(value, str) or not value for value in retrieved
    ):
        raise RAGAnswerGateError(f"{label}.retrieved_ids must be a string list")
    relevant_set = set(relevant)
    return len(relevant_set.intersection(retrieved)) / len(relevant_set)


def _latency(row: Mapping[str, Any], label: str) -> float:
    candidates: list[Any] = [row.get("total_latency_ms")]
    latency = row.get("latency_ms")
    if isinstance(latency, Mapping):
        candidates.extend(
            latency.get(key) for key in ("total", "end_to_end", "e2e")
        )
    else:
        candidates.append(latency)
    for value in candidates:
        if value is not None:
            return _nonnegative_number(value, f"{label}.latency_ms.total")
    raise RAGAnswerGateError(f"{label} must disclose end-to-end latency")


def _execution_error(row: Mapping[str, Any], label: str) -> bool:
    if "execution_error" not in row:
        raise RAGAnswerGateError(f"{label} must disclose execution_error")
    value = row["execution_error"]
    if value is None or value is False or value == "" or value == {}:
        return False
    if value is True or isinstance(value, (str, Mapping)):
        return True
    raise RAGAnswerGateError(f"{label}.execution_error has an invalid type")


def _fallback_used(row: Mapping[str, Any], label: str) -> bool:
    value = row.get("fallback_used")
    if not isinstance(value, bool):
        raise RAGAnswerGateError(f"{label}.fallback_used must be a boolean")
    return value


def _normalize_predictions(rows: Sequence[Mapping[str, Any]]) -> tuple[AnswerPrediction, ...]:
    predictions: list[AnswerPrediction] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        label = f"prediction[{index}]"
        case_id = _required_string(row, "case_id", label)
        if case_id in seen:
            raise RAGAnswerGateError(f"duplicate prediction case_id: {case_id}")
        seen.add(case_id)
        predictions.append(
            AnswerPrediction(
                case_id=case_id,
                category=_required_string(row, "category", label),
                token_f1=_bounded_score(row.get("token_f1"), f"{label}.token_f1"),
                evidence_recall=_evidence_recall(row, label),
                latency_ms=_latency(row, label),
                execution_error=_execution_error(row, label),
                fallback_used=_fallback_used(row, label),
            )
        )
    if not predictions:
        raise RAGAnswerGateError("answer-eval run contains no predictions")
    return tuple(predictions)


def _validate_declared_case_order(
    manifest: Mapping[str, Any], predictions: Sequence[AnswerPrediction]
) -> None:
    sample = manifest.get("sample")
    if not isinstance(sample, Mapping):
        raise RAGAnswerGateError("new answer-eval runs must declare manifest.sample")
    case_ids = sample.get("case_ids")
    if not isinstance(case_ids, list) or any(
        not isinstance(case_id, str) or not case_id for case_id in case_ids
    ):
        raise RAGAnswerGateError("manifest.sample.case_ids must be an ordered string list")
    observed = [prediction.case_id for prediction in predictions]
    if case_ids != observed:
        raise RAGAnswerGateError(
            "manifest.sample.case_ids does not match prediction order"
        )
    selected = sample.get("selected_cases")
    if selected is not None and selected != len(predictions):
        raise RAGAnswerGateError("manifest.sample.selected_cases does not match predictions")


def _validate_prediction_contract_fields(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str,
    retriever: str,
    mode: str,
    answer_contract: Any,
) -> None:
    expected = {
        "model": model,
        "retriever": retriever,
        "mode": mode,
        "answer_contract": answer_contract,
    }
    for index, row in enumerate(rows):
        for key, value in expected.items():
            if key not in row:
                raise RAGAnswerGateError(f"prediction[{index}] must declare {key}")
            if _canonical(row[key]) != _canonical(value):
                raise RAGAnswerGateError(
                    f"prediction[{index}].{key} disagrees with the run contract"
                )


def load_answer_eval_run(path: str | Path, *, label: str | None = None) -> AnswerEvalRun:
    """Load and cryptographically verify one immutable answer-eval run."""

    run_dir = Path(path).resolve()
    if not run_dir.is_dir():
        raise RAGAnswerGateError(f"answer-eval run directory does not exist: {run_dir}")
    manifest = _read_json_object(run_dir / "manifest.json", "manifest.json")
    artifact_hashes = _artifact_hashes(run_dir, manifest)
    metrics = _read_json_object(run_dir / "metrics.json", "metrics.json")
    prediction_rows = _prediction_rows(run_dir / "predictions.jsonl")
    predictions = _normalize_predictions(prediction_rows)
    _validate_declared_case_order(manifest, predictions)
    total_cases = metrics.get("total_cases")
    if total_cases != len(predictions):
        raise RAGAnswerGateError("metrics.total_cases does not match predictions")
    reported_f1 = metrics.get("avg_token_f1")
    if reported_f1 is not None:
        expected_f1 = sum(row.token_f1 for row in predictions) / len(predictions)
        if not math.isclose(
            _bounded_score(reported_f1, "metrics.avg_token_f1"),
            expected_f1,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RAGAnswerGateError("metrics.avg_token_f1 does not match predictions")
    effective = _effective_config(manifest)
    model = _consistent_value(metrics, effective, "model")
    retriever = _consistent_value(metrics, effective, "retriever")
    mode = _consistent_value(metrics, effective, "mode")
    answer_contract = _answer_contract(manifest, metrics, effective)
    _validate_prediction_contract_fields(
        prediction_rows,
        model=model,
        retriever=retriever,
        mode=mode,
        answer_contract=answer_contract,
    )
    run_label = label.strip() if isinstance(label, str) else run_dir.name
    if not run_label:
        raise RAGAnswerGateError("run label must not be blank")
    return AnswerEvalRun(
        label=run_label,
        path=run_dir,
        run_id=_required_string(manifest, "run_id", "manifest"),
        dataset_sha256=_dataset_hash(manifest),
        model=model,
        retriever=retriever,
        mode=mode,
        answer_contract=answer_contract,
        budget_signature=_budget_signature(manifest, effective),
        predictions=predictions,
        artifact_sha256=artifact_hashes,
    )


def _compatibility_errors(
    baseline: AnswerEvalRun,
    candidate: AnswerEvalRun,
    comparison_type: ComparisonType,
) -> list[str]:
    errors: list[str] = []
    if baseline.dataset_sha256 != candidate.dataset_sha256:
        errors.append("dataset_sha256_mismatch")
    if baseline.case_ids != candidate.case_ids:
        errors.append("ordered_case_ids_mismatch")
    if baseline.model != candidate.model:
        errors.append("model_mismatch")
    if _canonical(baseline.answer_contract) != _canonical(candidate.answer_contract):
        errors.append("answer_contract_mismatch")
    if _canonical(baseline.budget_signature) != _canonical(candidate.budget_signature):
        errors.append("budget_invariants_mismatch")
    if comparison_type == "retriever":
        if baseline.mode != candidate.mode:
            errors.append("retriever_comparison_requires_same_mode")
        if baseline.retriever == candidate.retriever:
            errors.append("retriever_comparison_requires_different_retrievers")
    elif comparison_type == "agentic":
        if baseline.retriever != candidate.retriever:
            errors.append("agentic_comparison_requires_same_retriever")
        if baseline.mode != "naive" or candidate.mode != "agentic":
            errors.append("agentic_comparison_requires_naive_to_agentic")
    else:
        raise ValueError(f"unsupported comparison type: {comparison_type}")
    if baseline.case_ids == candidate.case_ids:
        for baseline_row, candidate_row in zip(
            baseline.predictions, candidate.predictions, strict=True
        ):
            if baseline_row.category != candidate_row.category:
                errors.append(f"category_mismatch:{baseline_row.case_id}")
                break
    return errors


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise RAGAnswerGateError("cannot average an empty metric")
    return sum(values) / len(values)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_seed(seed: int, metric: str, values: Sequence[float]) -> int:
    material = f"{seed}\0{metric}\0{_canonical(list(values))}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _paired_bootstrap(
    deltas: Sequence[float], *, metric: str, config: BootstrapConfig
) -> dict[str, float]:
    if not deltas:
        raise RAGAnswerGateError("paired bootstrap requires at least one case")
    rng = random.Random(_bootstrap_seed(config.seed, metric, deltas))
    count = len(deltas)
    means = [
        sum(deltas[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(config.repetitions)
    ]
    tail = (1.0 - config.confidence) / 2.0
    return {
        "mean_delta": _mean(deltas),
        "lower": _percentile(means, tail),
        "upper": _percentile(means, 1.0 - tail),
        "confidence": config.confidence,
        "repetitions": config.repetitions,
        "seed": config.seed,
    }


def _p95(values: Sequence[float]) -> float:
    if not values:
        raise RAGAnswerGateError("p95 latency requires at least one case")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _summary(run: AnswerEvalRun) -> dict[str, Any]:
    rows = run.predictions
    return {
        "token_f1": _mean([row.token_f1 for row in rows]),
        "evidence_recall": _mean([row.evidence_recall for row in rows]),
        "p95_latency_ms": _p95([row.latency_ms for row in rows]),
        "execution_errors": sum(row.execution_error for row in rows),
        "fallback_cases": sum(row.fallback_used for row in rows),
    }


def _category_deltas(
    baseline: AnswerEvalRun,
    candidate: AnswerEvalRun,
    *,
    min_cases: int,
    max_degradation: float,
) -> tuple[list[dict[str, Any]], bool]:
    groups: dict[str, list[tuple[AnswerPrediction, AnswerPrediction]]] = defaultdict(list)
    for baseline_row, candidate_row in zip(
        baseline.predictions, candidate.predictions, strict=True
    ):
        groups[baseline_row.category].append((baseline_row, candidate_row))
    output: list[dict[str, Any]] = []
    passed = True
    for category in sorted(groups):
        pairs = groups[category]
        token_delta = _mean(
            [candidate_row.token_f1 - baseline_row.token_f1 for baseline_row, candidate_row in pairs]
        )
        recall_delta = _mean(
            [
                candidate_row.evidence_recall - baseline_row.evidence_recall
                for baseline_row, candidate_row in pairs
            ]
        )
        gated = len(pairs) >= min_cases
        category_passed = not gated or (
            token_delta >= -max_degradation and recall_delta >= -max_degradation
        )
        passed = passed and category_passed
        output.append(
            {
                "category": category,
                "cases": len(pairs),
                "gated": gated,
                "token_f1_delta": token_delta,
                "evidence_recall_delta": recall_delta,
                "passed": category_passed,
            }
        )
    return output, passed


def _gate_delta(
    interval: Mapping[str, float],
    *,
    minimum: float,
    require_nonnegative_ci_lower: bool,
) -> dict[str, Any]:
    observed_passed = interval["mean_delta"] >= minimum
    ci_passed = not require_nonnegative_ci_lower or interval["lower"] >= 0.0
    return {
        "passed": observed_passed and ci_passed,
        "minimum_delta": minimum,
        "observed_delta": interval["mean_delta"],
        "ci_lower": interval["lower"],
        "ci_lower_required": 0.0 if require_nonnegative_ci_lower else None,
    }


def _compare_candidate(
    baseline: AnswerEvalRun,
    candidate: AnswerEvalRun,
    *,
    comparison_type: ComparisonType,
    thresholds: GateThresholds,
    bootstrap: BootstrapConfig,
) -> dict[str, Any]:
    errors = _compatibility_errors(baseline, candidate, comparison_type)
    identity = {
        "label": candidate.label,
        "path": str(candidate.path),
        "run_id": candidate.run_id,
        "artifact_sha256": dict(candidate.artifact_sha256),
    }
    if errors:
        return {
            **identity,
            "compatible": False,
            "compatibility_errors": errors,
            "passed": False,
        }
    token_deltas = [
        candidate_row.token_f1 - baseline_row.token_f1
        for baseline_row, candidate_row in zip(
            baseline.predictions, candidate.predictions, strict=True
        )
    ]
    recall_deltas = [
        candidate_row.evidence_recall - baseline_row.evidence_recall
        for baseline_row, candidate_row in zip(
            baseline.predictions, candidate.predictions, strict=True
        )
    ]
    token_interval = _paired_bootstrap(
        token_deltas, metric="token_f1", config=bootstrap
    )
    recall_interval = _paired_bootstrap(
        recall_deltas, metric="evidence_recall", config=bootstrap
    )
    baseline_summary = _summary(baseline)
    candidate_summary = _summary(candidate)
    baseline_p95 = baseline_summary["p95_latency_ms"]
    candidate_p95 = candidate_summary["p95_latency_ms"]
    ratio = candidate_p95 / baseline_p95 if baseline_p95 > 0 else None
    latency_passed = ratio is not None and ratio <= thresholds.max_p95_ratio
    categories, categories_passed = _category_deltas(
        baseline,
        candidate,
        min_cases=thresholds.min_category_cases,
        max_degradation=thresholds.max_category_degradation,
    )
    token_gate = _gate_delta(
        token_interval,
        minimum=thresholds.min_token_f1_delta,
        require_nonnegative_ci_lower=thresholds.require_nonnegative_ci_lower,
    )
    recall_gate = _gate_delta(
        recall_interval,
        minimum=thresholds.min_evidence_recall_delta,
        require_nonnegative_ci_lower=thresholds.require_nonnegative_ci_lower,
    )
    execution_passed = (
        baseline_summary["execution_errors"] == 0
        and candidate_summary["execution_errors"] == 0
    )
    fallback_passed = (
        baseline_summary["fallback_cases"] == 0
        and candidate_summary["fallback_cases"] == 0
    )
    gates = {
        "token_f1_delta": token_gate,
        "evidence_recall_delta": recall_gate,
        "category_degradation": {
            "passed": categories_passed,
            "maximum_allowed": thresholds.max_category_degradation,
            "minimum_gated_cases": thresholds.min_category_cases,
        },
        "p95_latency_ratio": {
            "passed": latency_passed,
            "observed": ratio,
            "maximum_allowed": thresholds.max_p95_ratio,
        },
        "execution_errors": {
            "passed": execution_passed,
            "baseline": baseline_summary["execution_errors"],
            "candidate": candidate_summary["execution_errors"],
            "maximum_allowed": 0,
        },
        "fallback": {
            "passed": fallback_passed,
            "baseline": baseline_summary["fallback_cases"],
            "candidate": candidate_summary["fallback_cases"],
            "maximum_allowed": 0,
        },
    }
    return {
        **identity,
        "compatible": True,
        "compatibility_errors": [],
        "sample_size": len(baseline.predictions),
        "summary": {"baseline": baseline_summary, "candidate": candidate_summary},
        "paired_bootstrap": {
            "token_f1": token_interval,
            "evidence_recall": recall_interval,
        },
        "categories": categories,
        "gates": gates,
        "passed": all(gate["passed"] for gate in gates.values()),
    }


def compare_answer_eval_runs(
    baseline: AnswerEvalRun,
    candidates: Sequence[AnswerEvalRun],
    *,
    comparison_type: ComparisonType,
    thresholds: GateThresholds | None = None,
    bootstrap: BootstrapConfig | None = None,
) -> dict[str, Any]:
    """Compare one baseline with one or more paired candidate runs."""

    if comparison_type not in {"retriever", "agentic"}:
        raise ValueError("comparison_type must be retriever or agentic")
    if not candidates:
        raise ValueError("at least one candidate run is required")
    labels = [candidate.label for candidate in candidates]
    if len(labels) != len(set(labels)):
        raise ValueError("candidate labels must be unique")
    effective_thresholds = thresholds or GateThresholds.for_comparison(comparison_type)
    effective_bootstrap = bootstrap or BootstrapConfig()
    comparisons = [
        _compare_candidate(
            baseline,
            candidate,
            comparison_type=comparison_type,
            thresholds=effective_thresholds,
            bootstrap=effective_bootstrap,
        )
        for candidate in candidates
    ]
    return {
        "schema_version": "1.0",
        "comparison_type": comparison_type,
        "gate_implementation": {
            "module": "taskforge.rag_answer_gate",
            "sha256": _sha256(Path(__file__).resolve().read_bytes()),
        },
        "baseline": {
            "label": baseline.label,
            "path": str(baseline.path),
            "run_id": baseline.run_id,
            "artifact_sha256": dict(baseline.artifact_sha256),
        },
        "thresholds": effective_thresholds.as_dict(),
        "bootstrap": {
            "repetitions": effective_bootstrap.repetitions,
            "confidence": effective_bootstrap.confidence,
            "seed": effective_bootstrap.seed,
        },
        "candidates": comparisons,
        "passed": all(candidate["passed"] for candidate in comparisons),
    }


__all__ = [
    "AnswerEvalRun",
    "AnswerPrediction",
    "BootstrapConfig",
    "ComparisonType",
    "GateThresholds",
    "RAGAnswerGateError",
    "compare_answer_eval_runs",
    "load_answer_eval_run",
]

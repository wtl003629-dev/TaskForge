"""Absolute gates for the immutable DeepSeek TAT-QA online baseline."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .rag_answer_gate import (
    RAGAnswerGateError,
    _prediction_rows,
    _read_bytes,
    _read_json_object,
    _sha256,
    load_answer_eval_run,
)
from .rag_experiment import _PROBE_PREFIX

OnlineGateProfile = Literal["canary20", "full100", "repeat30"]


@dataclass(frozen=True)
class OnlineGateThresholds:
    expected_cases: int
    gate_quality: bool
    min_provider_success_rate: float = 0.98
    min_answer_f1: float = 0.60
    min_count_f1: float = 0.55
    min_multi_span_f1: float = 0.55
    min_citation_recall: float = 0.80
    max_unsupported_answer_rate: float = 0.05
    max_p95_latency_ms: float = 15_000.0

    @classmethod
    def for_profile(cls, profile: OnlineGateProfile) -> OnlineGateThresholds:
        if profile == "canary20":
            return cls(expected_cases=20, gate_quality=False)
        if profile == "full100":
            return cls(expected_cases=100, gate_quality=True)
        if profile == "repeat30":
            return cls(expected_cases=30, gate_quality=False)
        raise ValueError(f"unsupported online gate profile: {profile}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "expected_cases": self.expected_cases,
            "gate_quality": self.gate_quality,
            "min_provider_success_rate": self.min_provider_success_rate,
            "min_answer_f1": self.min_answer_f1,
            "min_count_f1": self.min_count_f1,
            "min_multi_span_f1": self.min_multi_span_f1,
            "min_citation_recall": self.min_citation_recall,
            "max_unsupported_answer_rate": self.max_unsupported_answer_rate,
            "max_p95_latency_ms": self.max_p95_latency_ms,
        }


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RAGAnswerGateError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise RAGAnswerGateError(f"{label} must be finite")
    return number


def _score(value: Any, label: str) -> float:
    score = _number(value, label)
    if not 0.0 <= score <= 1.0:
        raise RAGAnswerGateError(f"{label} must be between 0 and 1")
    return score


def _p95(values: Sequence[float]) -> float:
    if not values:
        raise RAGAnswerGateError("online gate requires latency for every case")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _verified_extra_artifact(
    run_dir: Path,
    manifest: Mapping[str, Any],
    name: str,
) -> bytes:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RAGAnswerGateError("manifest.artifacts must be an object")
    descriptor = artifacts.get(name)
    if not isinstance(descriptor, Mapping):
        raise RAGAnswerGateError(f"manifest.artifacts.{name} is required")
    expected_hash = descriptor.get("sha256")
    expected_size = descriptor.get("size_bytes")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise RAGAnswerGateError(f"manifest.artifacts.{name}.sha256 is invalid")
    raw = _read_bytes(run_dir / name, name)
    if _sha256(raw) != expected_hash:
        raise RAGAnswerGateError(f"artifact sha256 mismatch: {name}")
    if expected_size != len(raw):
        raise RAGAnswerGateError(f"artifact size mismatch: {name}")
    return raw


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RAGAnswerGateError(f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RAGAnswerGateError(f"{label} must be an array")
    return value


def _gate(
    *,
    observed: Any,
    passed: bool,
    requirement: Any,
    gated: bool = True,
) -> dict[str, Any]:
    return {
        "gated": gated,
        "passed": bool(passed) if gated else True,
        "observed": observed,
        "requirement": requirement,
    }


def evaluate_online_answer_run(
    path: str | Path,
    *,
    profile: OnlineGateProfile,
    thresholds: OnlineGateThresholds | None = None,
) -> dict[str, Any]:
    """Validate one run and apply structural, safety, cost, and quality gates."""

    run_dir = Path(path).resolve()
    effective = thresholds or OnlineGateThresholds.for_profile(profile)
    loaded = load_answer_eval_run(run_dir, label=profile)
    manifest = _read_json_object(run_dir / "manifest.json", "manifest.json")
    metrics = _read_json_object(run_dir / "metrics.json", "metrics.json")
    rows = _prediction_rows(run_dir / "predictions.jsonl")
    failures_raw = _verified_extra_artifact(run_dir, manifest, "failures.jsonl")
    _ = failures_raw  # zero bytes is a valid no-failure artifact
    _verified_extra_artifact(run_dir, manifest, "costs.jsonl")
    cost_rows = _prediction_rows(run_dir / "costs.jsonl")

    provider = _mapping(manifest.get("provider"), "manifest.provider")
    config = _mapping(
        _mapping(manifest.get("config"), "manifest.config").get("effective"),
        "manifest.config.effective",
    )
    budgets = _mapping(manifest.get("budgets"), "manifest.budgets")
    tools = _mapping(manifest.get("tools"), "manifest.tools")
    online_safety = _mapping(metrics.get("online_safety"), "metrics.online_safety")
    model_usage = _mapping(metrics.get("model_usage"), "metrics.model_usage")

    latencies: list[float] = []
    provider_attempted = 0
    provider_success = 0
    execution_errors = 0
    parse_errors = 0
    fallback_cases = 0
    invalid_ids = 0
    acl_violations = 0
    unsupported = 0
    calculator_failures = 0
    citation_recalls: list[float] = []
    usage_reported = 0
    cost_reported = 0
    category_f1: dict[str, list[float]] = defaultdict(list)
    case_ids: list[str] = []
    for index, row in enumerate(rows):
        label = f"prediction[{index}]"
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise RAGAnswerGateError(f"{label}.case_id is invalid")
        case_ids.append(case_id)
        category = row.get("category")
        if not isinstance(category, str) or not category:
            raise RAGAnswerGateError(f"{label}.category is invalid")
        category_f1[category].append(_score(row.get("token_f1"), f"{label}.token_f1"))
        latency = _mapping(row.get("latency_ms"), f"{label}.latency_ms")
        latencies.append(_number(latency.get("total"), f"{label}.latency_ms.total"))
        attempted = row.get("provider_call_attempted") is True
        calls = row.get("model_calls")
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
            raise RAGAnswerGateError(f"{label}.model_calls is invalid")
        provider_attempted += attempted and calls >= 1
        error = row.get("execution_error")
        has_execution_error = error is not None and error is not False and error != ""
        execution_errors += has_execution_error
        provider_success += attempted and calls >= 1 and not has_execution_error
        parse_error = row.get("parse_error")
        parse_errors += (
            parse_error is not None and parse_error is not False and parse_error != ""
        )
        fallback = row.get("fallback_used")
        if not isinstance(fallback, bool):
            raise RAGAnswerGateError(f"{label}.fallback_used must be boolean")
        fallback_cases += fallback
        invalid = _list(row.get("invalid_evidence_ids"), f"{label}.invalid_evidence_ids")
        invalid_ids += len(invalid)
        retrieved = _list(row.get("retrieved_ids"), f"{label}.retrieved_ids")
        acl_violations += any(
            isinstance(identifier, str) and identifier.startswith(_PROBE_PREFIX)
            for identifier in retrieved
        )
        unsupported_value = row.get("unsupported_answer")
        if not isinstance(unsupported_value, bool):
            raise RAGAnswerGateError(f"{label}.unsupported_answer must be boolean")
        unsupported += unsupported_value
        raw_calculator_failures = row.get("calculator_failure_count")
        if (
            isinstance(raw_calculator_failures, bool)
            or not isinstance(raw_calculator_failures, int)
            or raw_calculator_failures < 0
        ):
            raise RAGAnswerGateError(
                f"{label}.calculator_failure_count must be a non-negative integer"
            )
        calculator_failures += raw_calculator_failures
        grounding = _mapping(row.get("grounding"), f"{label}.grounding")
        citation_recalls.append(
            _score(grounding.get("citation_recall"), f"{label}.grounding.citation_recall")
        )
        if isinstance(row.get("model_usage"), Mapping):
            usage_reported += 1
        estimated = _mapping(row.get("estimated_cost"), f"{label}.estimated_cost")
        if estimated.get("status") == "estimated_from_reported_usage" and isinstance(
            estimated.get("amount"), (int, float)
        ):
            cost_reported += 1

    if [row.get("case_id") for row in cost_rows] != case_ids:
        raise RAGAnswerGateError("costs.jsonl case order does not match predictions")
    total = len(rows)
    provider_rate = provider_attempted / total if total else 0.0
    success_rate = provider_success / provider_attempted if provider_attempted else 0.0
    answer_f1 = _score(metrics.get("avg_token_f1"), "metrics.avg_token_f1")
    citation_recall = sum(citation_recalls) / total if total else 0.0
    unsupported_rate = unsupported / total if total else 1.0
    p95_latency = _p95(latencies)

    def category_mean(name: str) -> float | None:
        values = category_f1.get(name, [])
        return sum(values) / len(values) if values else None

    count_f1 = category_mean("count")
    multi_span_f1 = category_mean("multi-span")
    quality_gated = effective.gate_quality
    price_table = _mapping(provider.get("price_table"), "manifest.provider.price_table")
    gates = {
        "sample_size": _gate(
            observed=total,
            passed=total == effective.expected_cases,
            requirement=effective.expected_cases,
        ),
        "strict_contract": _gate(
            observed=loaded.answer_contract,
            passed=loaded.answer_contract == "online_cited_v1",
            requirement="online_cited_v1",
        ),
        "real_provider": _gate(
            observed={
                "execution_mode": provider.get("execution_mode"),
                "adapter_module": provider.get("adapter_module"),
                "adapter_class": provider.get("adapter_class"),
            },
            passed=(
                provider.get("execution_mode") == "live"
                and provider.get("adapter_module") == "taskforge.openai_provider"
                and provider.get("adapter_class")
                in {"OpenAIChatCompletionsProvider", "OpenAIResponsesProvider"}
            ),
            requirement="live TaskForge network adapter",
        ),
        "model_and_provider_modes": _gate(
            observed={
                "model": loaded.model,
                "thinking_mode": config.get("thinking_mode"),
                "json_mode": config.get("json_mode"),
            },
            passed=(
                loaded.model == "deepseek-v4-flash"
                and config.get("thinking_mode") == "disabled"
                and config.get("json_mode") is True
            ),
            requirement={
                "model": "deepseek-v4-flash",
                "thinking_mode": "disabled",
                "json_mode": True,
            },
        ),
        "retrieval_budget": _gate(
            observed={
                "retriever": loaded.retriever,
                "top_k": budgets.get("top_k"),
                "candidate_k": budgets.get("candidate_k"),
                "evidence_top_k": budgets.get("evidence_top_k"),
            },
            passed=(
                loaded.retriever
                in {"tatqa_frozen_bm25", "tatqa_frozen_pair_rerank"}
                and budgets.get("top_k") == [1, 5, 10]
                and budgets.get("candidate_k") == 50
                and budgets.get("evidence_top_k") == 10
            ),
            requirement="frozen Top-10 / Candidate@50",
        ),
        "calculator_schema": _gate(
            observed=tools.get("names"),
            passed=tools.get("names") == ["calculator"],
            requirement=["calculator"],
        ),
        "calculator_tool_failures": _gate(
            observed=calculator_failures,
            passed=calculator_failures == 0,
            requirement=0,
        ),
        "provider_call_rate": _gate(
            observed=provider_rate,
            passed=provider_rate == 1.0,
            requirement=1.0,
        ),
        "provider_success_rate": _gate(
            observed=success_rate,
            passed=success_rate >= effective.min_provider_success_rate,
            requirement={"minimum": effective.min_provider_success_rate},
        ),
        "execution_errors": _gate(
            observed=execution_errors,
            passed=execution_errors <= math.floor(
                total * (1.0 - effective.min_provider_success_rate)
            ),
            requirement={"maximum_rate": 1.0 - effective.min_provider_success_rate},
        ),
        "parse_errors": _gate(observed=parse_errors, passed=parse_errors == 0, requirement=0),
        "fallback": _gate(observed=fallback_cases, passed=fallback_cases == 0, requirement=0),
        "invalid_evidence_ids": _gate(
            observed=invalid_ids, passed=invalid_ids == 0, requirement=0
        ),
        "acl_tenant_violations": _gate(
            observed=acl_violations, passed=acl_violations == 0, requirement=0
        ),
        "usage_and_cost_trace": _gate(
            observed={
                "usage_reported_cases": usage_reported,
                "cost_reported_cases": cost_reported,
                "currency": price_table.get("currency"),
                "price_model": price_table.get("model"),
                "metrics_status": model_usage.get("estimated_cost_status"),
            },
            passed=(
                usage_reported == total
                and cost_reported == total
                and price_table.get("currency") == "USD"
                and price_table.get("model") == "deepseek-v4-flash"
                and model_usage.get("estimated_cost_status") == "complete"
            ),
            requirement="all cases have provider usage and versioned USD cost",
        ),
        "p95_latency_ms": _gate(
            observed=p95_latency,
            passed=p95_latency <= effective.max_p95_latency_ms,
            requirement={"maximum": effective.max_p95_latency_ms},
        ),
        "answer_f1": _gate(
            observed=answer_f1,
            passed=answer_f1 >= effective.min_answer_f1,
            requirement={"minimum": effective.min_answer_f1},
            gated=quality_gated,
        ),
        "count_f1": _gate(
            observed=count_f1,
            passed=count_f1 is not None and count_f1 >= effective.min_count_f1,
            requirement={"minimum": effective.min_count_f1},
            gated=quality_gated,
        ),
        "multi_span_f1": _gate(
            observed=multi_span_f1,
            passed=(
                multi_span_f1 is not None
                and multi_span_f1 >= effective.min_multi_span_f1
            ),
            requirement={"minimum": effective.min_multi_span_f1},
            gated=quality_gated,
        ),
        "citation_recall": _gate(
            observed=citation_recall,
            passed=citation_recall >= effective.min_citation_recall,
            requirement={"minimum": effective.min_citation_recall},
            gated=quality_gated,
        ),
        "unsupported_answer_rate": _gate(
            observed=unsupported_rate,
            passed=unsupported_rate <= effective.max_unsupported_answer_rate,
            requirement={"maximum": effective.max_unsupported_answer_rate},
            gated=quality_gated,
        ),
    }
    safety_reported_acl = online_safety.get("acl_tenant_violation_cases")
    if safety_reported_acl != acl_violations:
        raise RAGAnswerGateError("metrics ACL count does not match predictions")
    passed = all(gate["passed"] for gate in gates.values())
    return {
        "schema_version": "1.0",
        "profile": profile,
        "run": {
            "path": str(run_dir),
            "run_id": loaded.run_id,
            "artifact_sha256": dict(loaded.artifact_sha256),
        },
        "thresholds": effective.as_dict(),
        "gate_implementation": {
            "module": "taskforge.rag_online_gate",
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "gates": gates,
        "passed": passed,
    }


__all__ = [
    "OnlineGateProfile",
    "OnlineGateThresholds",
    "evaluate_online_answer_run",
]

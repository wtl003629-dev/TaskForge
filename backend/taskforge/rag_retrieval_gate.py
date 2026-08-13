"""Paired promotion gates for immutable retrieval experiment artifacts.

This module deliberately compares the prediction rows instead of trusting the
aggregate numbers written by a runner.  A retrieval change can therefore be
promoted only when the dataset/split, case order, filters, chunking and budget
are comparable, and when the complete case-level distribution does not regress.
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
from typing import Any

_ARTIFACT_NAMES = ("metrics.json", "predictions.jsonl")
_HEX = frozenset("0123456789abcdef")
_PROBE_PREFIX = "__taskforge_filter_probe__"


class RAGRetrievalGateError(RuntimeError):
    """Raised when an artifact is invalid or cannot be compared fairly."""


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
    min_recall_delta: float = 0.0
    max_recall_drop: float = 0.01
    max_candidate_drop: float = 0.01
    min_candidate_recall_at_10: float | None = None
    min_candidate_recall_at_candidate_k: float | None = None
    max_category_degradation: float = 0.03
    min_category_cases: int = 10
    max_p95_ratio: float = 1.2
    max_p95_ms: float | None = None
    require_nonnegative_ci_lower: bool = True

    def __post_init__(self) -> None:
        values = (
            self.min_recall_delta,
            self.max_recall_drop,
            self.max_candidate_drop,
            self.max_category_degradation,
            self.max_p95_ratio,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("gate thresholds must be finite")
        for name, value in (
            ("min_candidate_recall_at_10", self.min_candidate_recall_at_10),
            (
                "min_candidate_recall_at_candidate_k",
                self.min_candidate_recall_at_candidate_k,
            ),
        ):
            if value is not None and (
                not math.isfinite(value) or value < 0 or value > 1
            ):
                raise ValueError(f"{name} must be between 0 and 1")
        if self.max_recall_drop < 0 or self.max_candidate_drop < 0:
            raise ValueError("recall drops must be non-negative")
        if self.max_category_degradation < 0:
            raise ValueError("max_category_degradation must be non-negative")
        if self.min_category_cases < 1:
            raise ValueError("min_category_cases must be positive")
        if self.max_p95_ratio <= 0:
            raise ValueError("max_p95_ratio must be positive")
        if self.max_p95_ms is not None and (
            not math.isfinite(self.max_p95_ms) or self.max_p95_ms <= 0
        ):
            raise ValueError("max_p95_ms must be positive when provided")

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_recall_delta": self.min_recall_delta,
            "max_recall_drop": self.max_recall_drop,
            "max_candidate_drop": self.max_candidate_drop,
            "min_candidate_recall_at_10": self.min_candidate_recall_at_10,
            "min_candidate_recall_at_candidate_k": self.min_candidate_recall_at_candidate_k,
            "max_category_degradation": self.max_category_degradation,
            "min_category_cases": self.min_category_cases,
            "max_p95_ratio": self.max_p95_ratio,
            "max_p95_ms": self.max_p95_ms,
            "require_nonnegative_ci_lower": self.require_nonnegative_ci_lower,
        }


@dataclass(frozen=True)
class RetrievalPrediction:
    case_id: str
    category: str
    relevant_ids: tuple[str, ...]
    retrieved_ids: tuple[str, ...]
    raw_candidate_counts: Mapping[str, int]
    profile_name: str

    def score(self, k: int) -> float:
        if not self.relevant_ids:
            return 0.0
        hits = len(set(self.relevant_ids).intersection(self.retrieved_ids[:k]))
        return hits / len(set(self.relevant_ids))


@dataclass(frozen=True)
class RetrievalRun:
    label: str
    path: Path
    run_id: str
    dataset_sha256: str
    locked_split_sha256: str
    case_ids: tuple[str, ...]
    top_k: tuple[int, ...]
    candidate_k: int
    filters: Any
    retrieval_scope: str
    chunking: Any
    stage: str
    p95_ms: float
    predictions: tuple[RetrievalPrediction, ...]
    artifact_sha256: Mapping[str, str]
    stage_descriptor: Mapping[str, Any]


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RAGRetrievalGateError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _json_loads(raw: bytes, label: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object)
    except RAGRetrievalGateError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RAGRetrievalGateError(f"{label} is not valid UTF-8 JSON") from exc


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = _json_loads(path.read_bytes(), label)
    except OSError as exc:
        raise RAGRetrievalGateError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise RAGRetrievalGateError(f"{label} must contain one JSON object")
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
        raise RAGRetrievalGateError("comparison metadata is not canonical JSON") from exc


def _required_string(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RAGRetrievalGateError(f"{label}.{key} must be a non-empty string")
    return value.strip()


def _locked_split_identity(dataset: Mapping[str, Any]) -> str:
    """Return the immutable evaluation-set identity used by the gate.

    Synthetic PDF runs do not have a separately sampled split manifest: the
    suite itself plus the manifest's ordered case IDs is the locked set.  Keep
    that exception explicit instead of weakening validation for arbitrary
    artifacts.
    """
    raw_split = dataset.get("locked_split_sha256")
    if isinstance(raw_split, str) and raw_split.strip():
        return raw_split.strip()
    adapter = dataset.get("adapter")
    if adapter == "taskforge_synthetic_pdf_real_pypdf":
        suite_sha = dataset.get("suite_sha256")
        if isinstance(suite_sha, str) and suite_sha.strip():
            return f"suite:{suite_sha.strip()}"
    raise RAGRetrievalGateError(
        "manifest.dataset.locked_split_sha256 must be a non-empty string"
    )


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RAGRetrievalGateError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0) or result < 0:
        raise RAGRetrievalGateError(f"{label} must be finite and non-negative")
    return result


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise RAGRetrievalGateError(f"{label} must be a non-empty list")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise RAGRetrievalGateError(f"{label}[{index}] must be a non-empty string")
        result.append(item)
    if len(set(result)) != len(result):
        raise RAGRetrievalGateError(f"{label} contains duplicate values")
    return tuple(result)


def _verify_artifacts(run_dir: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RAGRetrievalGateError("manifest.artifacts must be an object")
    result: dict[str, str] = {}
    for name in _ARTIFACT_NAMES:
        descriptor = artifacts.get(name)
        if not isinstance(descriptor, Mapping):
            raise RAGRetrievalGateError(f"manifest.artifacts.{name} is required")
        expected = descriptor.get("sha256")
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in _HEX for character in expected.lower())
        ):
            raise RAGRetrievalGateError(f"manifest artifact {name} has invalid sha256")
        try:
            raw = (run_dir / name).read_bytes()
        except OSError as exc:
            raise RAGRetrievalGateError(f"cannot read artifact {name}") from exc
        actual = _sha256(raw)
        if actual != expected.lower():
            raise RAGRetrievalGateError(f"artifact sha256 mismatch: {name}")
        size = descriptor.get("size_bytes")
        if size is not None and size != len(raw):
            raise RAGRetrievalGateError(f"artifact size mismatch: {name}")
        result[name] = actual
    return result


def _stage_p95(stage_metrics: Mapping[str, Any], label: str) -> float:
    latency = stage_metrics.get("latency")
    if not isinstance(latency, Mapping):
        raise RAGRetrievalGateError(f"{label}.latency is required")
    raw = latency.get("p95", latency.get("p95_ms"))
    return _number(raw, f"{label}.latency.p95", positive=True)


def _row_predictions(raw: bytes, stage: str) -> list[RetrievalPrediction]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RAGRetrievalGateError("predictions.jsonl is not valid UTF-8") from exc
    rows: list[RetrievalPrediction] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise RAGRetrievalGateError(f"predictions.jsonl line {line_no} is blank")
        try:
            value = json.loads(line, object_pairs_hook=_strict_object)
        except RAGRetrievalGateError:
            raise
        except json.JSONDecodeError as exc:
            raise RAGRetrievalGateError(
                f"predictions.jsonl line {line_no} is invalid JSON"
            ) from exc
        if not isinstance(value, Mapping):
            raise RAGRetrievalGateError(f"predictions.jsonl line {line_no} is not an object")
        if value.get("stage") != stage:
            continue
        case_id = value.get("case_id")
        category = value.get("category")
        if not isinstance(case_id, str) or not case_id:
            raise RAGRetrievalGateError(f"line {line_no}.case_id is required")
        if not isinstance(category, str) or not category:
            raise RAGRetrievalGateError(f"line {line_no}.category is required")
        relevant_ids = _string_tuple(value.get("relevant_ids"), f"line {line_no}.relevant_ids")
        retrieved_raw = value.get("retrieved_ids")
        if not isinstance(retrieved_raw, list):
            raise RAGRetrievalGateError(f"line {line_no}.retrieved_ids must be a list")
        retrieved_ids: list[str] = []
        for index, item in enumerate(retrieved_raw):
            if not isinstance(item, str) or not item:
                raise RAGRetrievalGateError(
                    f"line {line_no}.retrieved_ids[{index}] must be a non-empty string"
                )
            if item in retrieved_ids:
                raise RAGRetrievalGateError(
                    f"line {line_no}.retrieved_ids contains duplicate {item!r}"
                )
            retrieved_ids.append(item)
        if any(item.startswith(_PROBE_PREFIX) for item in retrieved_ids):
            raise RAGRetrievalGateError(
                f"line {line_no} contains an inaccessible filter probe"
            )
        raw_counts = value.get("raw_candidate_counts")
        if not isinstance(raw_counts, Mapping) or not raw_counts:
            raise RAGRetrievalGateError(
                f"line {line_no}.raw_candidate_counts is required"
            )
        parsed_counts: dict[str, int] = {}
        for branch, count in raw_counts.items():
            if (
                not isinstance(branch, str)
                or not branch
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
            ):
                raise RAGRetrievalGateError(
                    f"line {line_no}.raw_candidate_counts must map names to non-negative integers"
                )
            parsed_counts[branch] = count
        profile = value.get("retrieval_profile")
        if not isinstance(profile, Mapping):
            raise RAGRetrievalGateError(
                f"line {line_no}.retrieval_profile is required"
            )
        profile_name = profile.get("name")
        if not isinstance(profile_name, str) or not profile_name.strip():
            raise RAGRetrievalGateError(
                f"line {line_no}.retrieval_profile.name is required"
            )
        rows.append(
            RetrievalPrediction(
                case_id=case_id,
                category=category,
                relevant_ids=relevant_ids,
                retrieved_ids=tuple(retrieved_ids),
                raw_candidate_counts=parsed_counts,
                profile_name=profile_name.strip(),
            )
        )
    if not rows:
        raise RAGRetrievalGateError(f"predictions.jsonl contains no stage {stage!r}")
    return rows


def _effective_metadata(
    manifest: Mapping[str, Any], metrics: Mapping[str, Any], stage: str
) -> tuple[Any, str, Any, tuple[int, ...], int, Mapping[str, Any]]:
    config = manifest.get("config")
    effective = config.get("effective") if isinstance(config, Mapping) else None
    if not isinstance(effective, Mapping):
        raise RAGRetrievalGateError("manifest.config.effective is required")
    retrieval = effective.get("retrieval")
    if not isinstance(retrieval, Mapping):
        raise RAGRetrievalGateError("manifest.config.effective.retrieval is required")
    top_k_raw = retrieval.get("top_k", metrics.get("top_k"))
    if not isinstance(top_k_raw, list) or not top_k_raw:
        raise RAGRetrievalGateError("retrieval.top_k is required")
    top_k = tuple(int(item) for item in top_k_raw)
    if any(item <= 0 for item in top_k):
        raise RAGRetrievalGateError("retrieval.top_k must contain positive integers")
    candidate_k_raw = retrieval.get("candidate_k", metrics.get("candidate_k"))
    if isinstance(candidate_k_raw, bool) or not isinstance(candidate_k_raw, int):
        raise RAGRetrievalGateError("retrieval.candidate_k is required")
    if candidate_k_raw <= 0:
        raise RAGRetrievalGateError("retrieval.candidate_k must be positive")
    filters = effective.get("filters")
    if not isinstance(filters, Mapping):
        raise RAGRetrievalGateError("manifest.config.effective.filters is required")
    dataset_config = effective.get("dataset")
    if dataset_config is None:
        retrieval_scope = "global_discovery"
    elif not isinstance(dataset_config, Mapping):
        raise RAGRetrievalGateError("manifest.config.effective.dataset must be an object")
    else:
        raw_scope = (
            dataset_config.get("qasper_context_mode", "global_discovery")
            if dataset_config.get("kind") == "qasper_locked"
            else dataset_config.get("tatqa_context_mode", "global_discovery")
        )
        if raw_scope not in {
            "global_discovery",
            "provided_hybrid_context",
            "provided_document_context",
        }:
            raise RAGRetrievalGateError(
                "manifest.config.effective.dataset context mode is invalid"
            )
        retrieval_scope = str(raw_scope)
    chunking = metrics.get("chunking")
    if not isinstance(chunking, Mapping):
        chunking = {
            key: retrieval.get(key)
            for key in (
                "chunking",
                "table_aware_chunking",
                "chunk_max_chars",
                "chunk_overlap_chars",
            )
        }
    stages = metrics.get("stages")
    if not isinstance(stages, Mapping) or not isinstance(stages.get(stage), Mapping):
        raise RAGRetrievalGateError(f"metrics.stages.{stage} is required")
    return filters, retrieval_scope, chunking, top_k, candidate_k_raw, stages[stage]


def load_retrieval_run(
    run_dir: str | Path, *, label: str, stage: str, profile_name: str | None = None
) -> RetrievalRun:
    """Load, hash-verify and structurally validate one retrieval run."""
    if not label.strip():
        raise RAGRetrievalGateError("run label must not be empty")
    if profile_name is not None and not profile_name.strip():
        raise RAGRetrievalGateError("profile_name must not be empty")
    path = Path(run_dir).resolve()
    if not path.is_dir():
        raise RAGRetrievalGateError(f"run directory does not exist: {path}")
    manifest = _read_json(path / "manifest.json", "manifest.json")
    metrics = _read_json(path / "metrics.json", "metrics.json")
    artifact_hashes = _verify_artifacts(path, manifest)
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        raise RAGRetrievalGateError("manifest.dataset is required")
    dataset_sha = _required_string(dataset, "normalized_sha256", "manifest.dataset")
    split_sha = _locked_split_identity(dataset)
    sample = manifest.get("sample")
    if not isinstance(sample, Mapping):
        raise RAGRetrievalGateError("manifest.sample is required")
    case_ids = _string_tuple(sample.get("case_ids"), "manifest.sample.case_ids")
    metrics_case_ids = _string_tuple(metrics.get("case_ids"), "metrics.case_ids")
    if metrics_case_ids != case_ids:
        raise RAGRetrievalGateError("manifest and metrics case order differ")
    (
        filters,
        retrieval_scope,
        chunking,
        top_k,
        candidate_k,
        stage_metrics,
    ) = _effective_metadata(manifest, metrics, stage)
    rows = _row_predictions((path / "predictions.jsonl").read_bytes(), stage)
    if tuple(row.case_id for row in rows) != case_ids:
        raise RAGRetrievalGateError(
            f"stage {stage!r} predictions do not match manifest case order"
        )
    all_rows = rows
    for row in all_rows:
        oversized = {
            branch: count
            for branch, count in row.raw_candidate_counts.items()
            if branch != "fused" and count > candidate_k
        }
        if oversized:
            raise RAGRetrievalGateError(
                f"stage {stage!r} exceeds candidate_k in raw branches: {oversized}"
            )
    reported_raw = stage_metrics.get("raw_candidate_counts")
    if not isinstance(reported_raw, Mapping) or not reported_raw:
        raise RAGRetrievalGateError(
            f"metrics.stages.{stage}.raw_candidate_counts is required"
        )
    actual_raw: dict[str, list[int]] = defaultdict(list)
    for row in all_rows:
        for branch, count in row.raw_candidate_counts.items():
            actual_raw[branch].append(count)
    for branch, values in actual_raw.items():
        reported = reported_raw.get(branch)
        if not isinstance(reported, Mapping):
            raise RAGRetrievalGateError(
                f"metrics.stages.{stage}.raw_candidate_counts omits {branch!r}"
            )
        for key, expected in (
            ("min", min(values)),
            ("max", max(values)),
            ("mean", sum(values) / len(values)),
        ):
            actual = reported.get(key)
            if (
                isinstance(actual, bool)
                or not isinstance(actual, (int, float))
                or not math.isfinite(float(actual))
                or abs(float(actual) - float(expected)) > 1e-9
            ):
                raise RAGRetrievalGateError(
                    f"metrics.stages.{stage}.raw_candidate_counts.{branch}.{key} disagrees"
                )
    if profile_name is not None:
        rows = [row for row in all_rows if row.profile_name == profile_name.strip()]
        if not rows:
            raise RAGRetrievalGateError(
                f"stage {stage!r} contains no retrieval profile {profile_name!r}"
            )
        case_ids = tuple(row.case_id for row in rows)
    p95_ms = _stage_p95(stage_metrics, f"metrics.stages.{stage}")
    stage_descriptor = manifest.get("ablation", {}).get("stages", {}).get(stage, {})
    if not isinstance(stage_descriptor, Mapping):
        stage_descriptor = {}
    return RetrievalRun(
        label=label.strip(),
        path=path,
        run_id=_required_string(manifest, "run_id", "manifest"),
        dataset_sha256=dataset_sha,
        locked_split_sha256=split_sha,
        case_ids=case_ids,
        top_k=top_k,
        candidate_k=candidate_k,
        filters=filters,
        retrieval_scope=retrieval_scope,
        chunking=chunking,
        stage=stage,
        p95_ms=p95_ms,
        predictions=tuple(rows),
        artifact_sha256=artifact_hashes,
        stage_descriptor=stage_descriptor,
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_bootstrap(
    deltas: Sequence[float], config: BootstrapConfig
) -> tuple[float, float]:
    if not deltas:
        raise RAGRetrievalGateError("paired bootstrap requires at least one case")
    rng = random.Random(config.seed)
    n = len(deltas)
    means = [
        _mean([deltas[rng.randrange(n)] for _ in range(n)])
        for _ in range(config.repetitions)
    ]
    alpha = (1.0 - config.confidence) / 2.0
    return _percentile(means, alpha), _percentile(means, 1.0 - alpha)


def _metric_rows(
    baseline: RetrievalRun, candidate: RetrievalRun
) -> tuple[list[float], list[float], list[float]]:
    baseline_by_id = {row.case_id: row for row in baseline.predictions}
    candidate_by_id = {row.case_id: row for row in candidate.predictions}
    recall_deltas: list[float] = []
    candidate_deltas: list[float] = []
    ndcg_deltas: list[float] = []
    for case_id in baseline.case_ids:
        left = baseline_by_id[case_id]
        right = candidate_by_id[case_id]
        recall_deltas.append(right.score(10) - left.score(10))
        candidate_deltas.append(
            right.score(candidate.candidate_k) - left.score(baseline.candidate_k)
        )
        # nDCG@10 is kept as a diagnostic; it is not a hard promotion gate.
        relevant = set(left.relevant_ids)
        if relevant != set(right.relevant_ids):
            raise RAGRetrievalGateError(f"relevant IDs differ for case {case_id}")
        def dcg(row: RetrievalPrediction) -> float:
            return sum(
                (1.0 / math.log2(index + 2))
                for index, item in enumerate(row.retrieved_ids[:10])
                if item in relevant
            )
        ideal = sum(1.0 / math.log2(index + 2) for index in range(min(10, len(relevant))))
        ndcg_deltas.append((dcg(right) / ideal if ideal else 0.0) - (dcg(left) / ideal if ideal else 0.0))
    return recall_deltas, candidate_deltas, ndcg_deltas


def _category_deltas(
    baseline: RetrievalRun, candidate: RetrievalRun
) -> dict[str, dict[str, float | int]]:
    left: dict[str, list[float]] = defaultdict(list)
    right: dict[str, list[float]] = defaultdict(list)
    baseline_by_id = {row.case_id: row for row in baseline.predictions}
    candidate_by_id = {row.case_id: row for row in candidate.predictions}
    for case_id in baseline.case_ids:
        left[baseline_by_id[case_id].category].append(baseline_by_id[case_id].score(10))
        right[candidate_by_id[case_id].category].append(candidate_by_id[case_id].score(10))
    result: dict[str, dict[str, float | int]] = {}
    for category in sorted(set(left) | set(right)):
        if len(left[category]) != len(right[category]):
            raise RAGRetrievalGateError(f"category membership differs: {category}")
        result[category] = {
            "cases": len(left[category]),
            "baseline_recall_at_10": _mean(left[category]),
            "candidate_recall_at_10": _mean(right[category]),
            "delta": _mean(right[category]) - _mean(left[category]),
        }
    return result


def _raw_candidate_summary(
    run: RetrievalRun,
) -> dict[str, dict[str, float | int]]:
    values: dict[str, list[int]] = defaultdict(list)
    for row in run.predictions:
        for branch, count in row.raw_candidate_counts.items():
            values[branch].append(count)
    return {
        branch: {
            "min": min(counts),
            "max": max(counts),
            "mean": _mean(counts),
        }
        for branch, counts in sorted(values.items())
    }


def _common_metadata(baseline: RetrievalRun, candidate: RetrievalRun) -> list[str]:
    errors: list[str] = []
    if baseline.dataset_sha256 != candidate.dataset_sha256:
        errors.append("dataset normalized hash differs")
    if baseline.locked_split_sha256 != candidate.locked_split_sha256:
        errors.append("locked split hash differs")
    if baseline.case_ids != candidate.case_ids:
        errors.append("case IDs/order differs")
    if baseline.top_k != candidate.top_k:
        errors.append("top_k differs")
    if baseline.candidate_k != candidate.candidate_k:
        errors.append("candidate_k differs")
    if _canonical(baseline.filters) != _canonical(candidate.filters):
        errors.append("filters differ")
    if baseline.retrieval_scope != candidate.retrieval_scope:
        errors.append("retrieval scope differs")
    if _canonical(baseline.chunking) != _canonical(candidate.chunking):
        errors.append("chunking differs")
    if tuple(row.profile_name for row in baseline.predictions) != tuple(
        row.profile_name for row in candidate.predictions
    ):
        errors.append("retrieval profile assignment differs")
    return errors


def compare_retrieval_runs(
    baseline: RetrievalRun,
    candidates: Sequence[RetrievalRun],
    *,
    thresholds: GateThresholds | None = None,
    bootstrap: BootstrapConfig | None = None,
    require_identical_retrieval: bool = False,
) -> dict[str, Any]:
    """Compare candidates and return a JSON-serializable gate report."""
    if not candidates:
        raise RAGRetrievalGateError("at least one candidate is required")
    thresholds = thresholds or GateThresholds()
    bootstrap = bootstrap or BootstrapConfig()
    if baseline.p95_ms <= 0:
        raise RAGRetrievalGateError("baseline p95 must be positive")
    baseline_probe_free = all(
        not any(item.startswith(_PROBE_PREFIX) for item in row.retrieved_ids)
        for row in baseline.predictions
    )
    if not baseline_probe_free:
        raise RAGRetrievalGateError("baseline contains an inaccessible filter probe")
    output_candidates: list[dict[str, Any]] = []
    overall_passed = True
    for candidate in candidates:
        metadata_errors = _common_metadata(baseline, candidate)
        if candidate.p95_ms <= 0:
            metadata_errors.append("candidate p95 must be positive")
        recall_deltas, candidate_deltas, ndcg_deltas = _metric_rows(baseline, candidate)
        recall_ci = paired_bootstrap(recall_deltas, bootstrap)
        candidate_ci = paired_bootstrap(candidate_deltas, bootstrap)
        categories = _category_deltas(baseline, candidate)
        category_failures = [
            category
            for category, value in categories.items()
            if value["cases"] >= thresholds.min_category_cases
            and float(value["delta"]) < -thresholds.max_category_degradation
        ]
        recall_delta = _mean(recall_deltas)
        candidate_delta = _mean(candidate_deltas)
        candidate_recall_at_10 = _mean(
            [row.score(10) for row in candidate.predictions]
        )
        candidate_recall_at_candidate_k = _mean(
            [row.score(candidate.candidate_k) for row in candidate.predictions]
        )
        p95_ratio = candidate.p95_ms / baseline.p95_ms
        recall_floor = max(
            -thresholds.max_recall_drop,
            thresholds.min_recall_delta,
        )
        # A profile may have both a relative budget (protects against
        # regressions when the baseline is fast) and an absolute cap (keeps
        # the user-facing SLA bounded).  Supplying both must enforce both;
        # otherwise an absolute cap could silently disable the ratio guard.
        latency_passed = p95_ratio <= thresholds.max_p95_ratio + 1e-12
        if thresholds.max_p95_ms is not None:
            latency_passed = latency_passed and (
                candidate.p95_ms <= thresholds.max_p95_ms + 1e-12
            )
        checks = {
            "metadata": not metadata_errors,
            "recall_at_10": recall_delta >= recall_floor - 1e-12,
            "candidate_recall_at_candidate_k": candidate_delta
            >= -thresholds.max_candidate_drop - 1e-12,
            "candidate_recall_floor": (
                thresholds.min_candidate_recall_at_10 is None
                or candidate_recall_at_10
                >= thresholds.min_candidate_recall_at_10 - 1e-12
            ),
            "candidate_candidate_recall_floor": (
                thresholds.min_candidate_recall_at_candidate_k is None
                or candidate_recall_at_candidate_k
                >= thresholds.min_candidate_recall_at_candidate_k - 1e-12
            ),
            "category_non_regression": not category_failures,
            "latency": latency_passed,
            "security": True,
            "bootstrap_ci": (
                recall_ci[0] >= -1e-12
                if thresholds.require_nonnegative_ci_lower
                else True
            ),
        }
        if require_identical_retrieval:
            checks["identical_retrieval"] = all(
                left.retrieved_ids == right.retrieved_ids
                for left, right in zip(baseline.predictions, candidate.predictions)
            )
        passed = all(checks.values())
        overall_passed = overall_passed and passed
        output_candidates.append(
            {
                "label": candidate.label,
                "run_id": candidate.run_id,
                "stage": candidate.stage,
                "passed": passed,
                "checks": checks,
                "metadata_errors": metadata_errors,
                "category_failures": category_failures,
                "metrics": {
                    "baseline_recall_at_10": _mean([row.score(10) for row in baseline.predictions]),
                    "candidate_recall_at_10": candidate_recall_at_10,
                    "recall_delta": recall_delta,
                    "recall_ci_95": list(recall_ci),
                    "baseline_candidate_recall": _mean([row.score(baseline.candidate_k) for row in baseline.predictions]),
                    "candidate_candidate_recall": candidate_recall_at_candidate_k,
                    "candidate_recall_delta": candidate_delta,
                    "candidate_recall_ci_95": list(candidate_ci),
                    "ndcg_delta": _mean(ndcg_deltas),
                    "baseline_p95_ms": baseline.p95_ms,
                    "candidate_p95_ms": candidate.p95_ms,
                    "p95_ratio": p95_ratio,
                    "categories": categories,
                    "baseline_raw_candidate_counts": _raw_candidate_summary(baseline),
                    "candidate_raw_candidate_counts": _raw_candidate_summary(candidate),
                },
                "artifact_sha256": dict(candidate.artifact_sha256),
            }
        )
    return {
        "schema_version": "1.0",
        "gate": "retrieval_promotion",
        "passed": overall_passed,
        "baseline": {
            "label": baseline.label,
            "run_id": baseline.run_id,
            "stage": baseline.stage,
            "artifact_sha256": dict(baseline.artifact_sha256),
        },
        "thresholds": thresholds.as_dict(),
        "bootstrap": {
            "repetitions": bootstrap.repetitions,
            "confidence": bootstrap.confidence,
            "seed": bootstrap.seed,
        },
        "require_identical_retrieval": require_identical_retrieval,
        "candidates": output_candidates,
    }


__all__ = [
    "BootstrapConfig",
    "GateThresholds",
    "RAGRetrievalGateError",
    "RetrievalPrediction",
    "RetrievalRun",
    "compare_retrieval_runs",
    "load_retrieval_run",
    "paired_bootstrap",
]

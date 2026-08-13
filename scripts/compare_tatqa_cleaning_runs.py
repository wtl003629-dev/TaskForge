"""Compare paired TAT-QA cleaning control/candidate retrieval runs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _verify_run(path: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    directory = path.resolve(strict=True)
    manifest = _json(directory / "manifest.json")
    metrics = _json(directory / "metrics.json")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"{directory}: manifest.artifacts is required")
    for name in ("metrics.json", "predictions.jsonl"):
        record = artifacts.get(name)
        artifact = directory / name
        if not isinstance(record, Mapping) or not artifact.is_file():
            raise ValueError(f"{directory}: artifact {name} is missing")
        if record.get("sha256") != _sha256(artifact):
            raise ValueError(f"{directory}: artifact {name} hash differs")
        if record.get("size_bytes") != artifact.stat().st_size:
            raise ValueError(f"{directory}: artifact {name} size differs")
    return manifest, metrics


def _effective(manifest: Mapping[str, Any]) -> dict[str, Any]:
    config = manifest.get("config")
    effective = config.get("effective") if isinstance(config, Mapping) else None
    if not isinstance(effective, Mapping):
        raise ValueError("manifest.config.effective is required")
    return copy.deepcopy(dict(effective))


def _required_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is required")
    return value


def _predictions(path: Path, stage: str) -> list[Mapping[str, Any]]:
    output: list[Mapping[str, Any]] = []
    for line_no, line in enumerate(
        (path / "predictions.jsonl").read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"predictions line {line_no} is not an object")
        if value.get("stage") == stage:
            output.append(value)
    if not output:
        raise ValueError(f"predictions contain no stage {stage!r}")
    return output


def _stage_summary(metrics: Mapping[str, Any], stage: str) -> dict[str, Any]:
    stages = _required_mapping(metrics.get("stages"), "metrics.stages")
    stage_metrics = _required_mapping(stages.get(stage), f"metrics.stages.{stage}")
    retrieval = _required_mapping(stage_metrics.get("retrieval"), "stage.retrieval")
    summary = _required_mapping(retrieval.get("summary"), "stage.retrieval.summary")
    hierarchical = _required_mapping(stage_metrics.get("hierarchical"), "hierarchical")
    latency = _required_mapping(stage_metrics.get("latency"), "latency")
    by_category = _required_mapping(
        summary.get("by_category_recall_at_k"), "by_category_recall_at_k"
    )
    return {
        "candidate_recall_at_50": float(stage_metrics["candidate_recall"]),
        "recall_at_10": float(_required_mapping(summary["recall_at_k"], "recall")["10"]),
        "ndcg_at_10": float(_required_mapping(summary["ndcg_at_k"], "ndcg")["10"]),
        "by_category_recall_at_10": {
            str(category): float(_required_mapping(values, "category metrics")["10"])
            for category, values in by_category.items()
        },
        "row_recall_at_10": float(
            _required_mapping(hierarchical["row_recall_at_k"], "row recall")["10"]
        ),
        "cell_recall_at_10": float(
            _required_mapping(hierarchical["cell_recall_at_k"], "cell recall")["10"]
        ),
        "weak_operand_recall_at_10": float(
            _required_mapping(
                hierarchical["weak_operand_recall_at_k"], "weak operand recall"
            )["10"]
        ),
        "missing_predictions": int(summary["missing_predictions"]),
        "p95_ms": float(latency["p95"]),
    }


def _pair(
    control_path: Path,
    candidate_path: Path,
    *,
    stage: str,
) -> dict[str, Any]:
    control_manifest, control_metrics = _verify_run(control_path)
    candidate_manifest, candidate_metrics = _verify_run(candidate_path)
    control_effective = _effective(control_manifest)
    candidate_effective = _effective(candidate_manifest)
    control_dataset_config = _required_mapping(
        control_effective.get("dataset"), "control dataset config"
    )
    candidate_dataset_config = _required_mapping(
        candidate_effective.get("dataset"), "candidate dataset config"
    )
    if control_dataset_config.get("tatqa_table_cleaning") is not False:
        raise ValueError("control run must have tatqa_table_cleaning=false")
    if candidate_dataset_config.get("tatqa_table_cleaning") is not True:
        raise ValueError("candidate run must have tatqa_table_cleaning=true")
    comparable_control = copy.deepcopy(control_effective)
    comparable_candidate = copy.deepcopy(candidate_effective)
    comparable_control["dataset"].pop("tatqa_table_cleaning", None)
    comparable_candidate["dataset"].pop("tatqa_table_cleaning", None)
    if comparable_control != comparable_candidate:
        raise ValueError("paired configs differ beyond tatqa_table_cleaning")
    control_dataset = _required_mapping(
        control_manifest.get("dataset"), "control manifest.dataset"
    )
    candidate_dataset = _required_mapping(
        candidate_manifest.get("dataset"), "candidate manifest.dataset"
    )
    for key in ("input_sha256", "locked_split_sha256", "selected_cases"):
        if control_dataset.get(key) != candidate_dataset.get(key):
            raise ValueError(f"paired dataset invariant differs: {key}")
    if control_manifest.get("sample") != candidate_manifest.get("sample"):
        raise ValueError("paired case IDs/order differ")
    control_code = _required_mapping(control_manifest.get("code"), "control code")
    candidate_code = _required_mapping(candidate_manifest.get("code"), "candidate code")
    if control_code.get("sha256") != candidate_code.get("sha256"):
        raise ValueError("paired code hashes differ")

    control = _stage_summary(control_metrics, stage)
    candidate = _stage_summary(candidate_metrics, stage)
    categories = sorted(control["by_category_recall_at_10"])
    if categories != sorted(candidate["by_category_recall_at_10"]):
        raise ValueError("paired category sets differ")
    deltas = {
        "candidate_recall_at_50": (
            candidate["candidate_recall_at_50"] - control["candidate_recall_at_50"]
        ),
        "recall_at_10": candidate["recall_at_10"] - control["recall_at_10"],
        "ndcg_at_10": candidate["ndcg_at_10"] - control["ndcg_at_10"],
        "row_recall_at_10": (
            candidate["row_recall_at_10"] - control["row_recall_at_10"]
        ),
        "cell_recall_at_10": (
            candidate["cell_recall_at_10"] - control["cell_recall_at_10"]
        ),
        "weak_operand_recall_at_10": (
            candidate["weak_operand_recall_at_10"]
            - control["weak_operand_recall_at_10"]
        ),
        "by_category_recall_at_10": {
            category: (
                candidate["by_category_recall_at_10"][category]
                - control["by_category_recall_at_10"][category]
            )
            for category in categories
        },
        "p95_ratio": candidate["p95_ms"] / control["p95_ms"],
    }
    control_predictions = _predictions(control_path, stage)
    candidate_predictions = _predictions(candidate_path, stage)
    if [row["case_id"] for row in control_predictions] != [
        row["case_id"] for row in candidate_predictions
    ]:
        raise ValueError("paired prediction case order differs")
    changed_rankings = sum(
        left.get("retrieved_ids") != right.get("retrieved_ids")
        for left, right in zip(control_predictions, candidate_predictions, strict=True)
    )
    checks = {
        "zero_missing_predictions": (
            control["missing_predictions"] == candidate["missing_predictions"] == 0
        ),
        "recall_non_regression": deltas["recall_at_10"] >= -0.01,
        "candidate_non_regression": deltas["candidate_recall_at_50"] >= -0.01,
        "category_non_regression": min(
            deltas["by_category_recall_at_10"].values(), default=0.0
        )
        >= -0.03,
        "row_non_regression": deltas["row_recall_at_10"] >= -0.03,
        "cell_non_regression": deltas["cell_recall_at_10"] >= -0.03,
        "latency_ratio": deltas["p95_ratio"] <= 1.2,
        "measurable_quality_gain": any(
            (
                deltas["recall_at_10"] >= 0.005,
                deltas["candidate_recall_at_50"] >= 0.005,
                deltas["row_recall_at_10"] >= 0.01,
                deltas["cell_recall_at_10"] >= 0.01,
            )
        ),
    }
    non_regression_checks = {
        key: value for key, value in checks.items() if key != "measurable_quality_gain"
    }
    return {
        "context_mode": control_dataset_config.get("tatqa_context_mode"),
        "control": {
            "path": control_path.resolve().as_posix(),
            "run_id": control_manifest.get("run_id"),
            "metrics": control,
        },
        "candidate": {
            "path": candidate_path.resolve().as_posix(),
            "run_id": candidate_manifest.get("run_id"),
            "metrics": candidate,
        },
        "deltas": deltas,
        "retrieved_id_rankings_changed": changed_rankings,
        "checks": checks,
        "safe_to_keep_optional": all(non_regression_checks.values()),
        "promotion_eligible": all(checks.values()),
    }


def compare_runs(
    *,
    global_control: Path,
    global_candidate: Path,
    provided_control: Path,
    provided_candidate: Path,
    cleaning_audit: Path,
    stage: str,
) -> dict[str, Any]:
    audit = _json(cleaning_audit.resolve(strict=True))
    pairs = [
        _pair(global_control, global_candidate, stage=stage),
        _pair(provided_control, provided_candidate, stage=stage),
    ]
    return {
        "schema_version": "1.0",
        "experiment": "tatqa_coordinate_preserving_table_cleaning",
        "stage": stage,
        "cleaning_audit": {
            "path": cleaning_audit.resolve().as_posix(),
            "sha256": _sha256(cleaning_audit.resolve()),
            "source_sha256": _required_mapping(audit.get("source"), "audit source").get(
                "sha256"
            ),
        },
        "thresholds": {
            "max_recall_drop": 0.01,
            "max_candidate_drop": 0.01,
            "max_category_drop": 0.03,
            "max_row_or_cell_drop": 0.03,
            "max_p95_ratio": 1.2,
            "measurable_gain": (
                "Recall@10 or Candidate@50 >= +0.005, or row/cell Recall@10 >= +0.01"
            ),
        },
        "pairs": pairs,
        "safe_to_keep_optional": all(pair["safe_to_keep_optional"] for pair in pairs),
        "promotion_eligible": all(pair["promotion_eligible"] for pair in pairs),
        "decision": (
            "promote_default"
            if all(pair["promotion_eligible"] for pair in pairs)
            else "keep_opt_in_not_default"
        ),
        "limitations": [
            "No hidden split was opened.",
            "Latency is one warm offline run per arm and is not a statistical benchmark.",
            "Weak operand recall is diagnostic, not official TAT-QA cell recall.",
        ],
    }


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--global-control", type=Path, required=True)
    value.add_argument("--global-candidate", type=Path, required=True)
    value.add_argument("--provided-control", type=Path, required=True)
    value.add_argument("--provided-candidate", type=Path, required=True)
    value.add_argument("--cleaning-audit", type=Path, required=True)
    value.add_argument("--stage", required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    report = compare_runs(
        global_control=args.global_control,
        global_candidate=args.global_candidate,
        provided_control=args.provided_control,
        provided_candidate=args.provided_candidate,
        cleaning_audit=args.cleaning_audit,
        stage=args.stage,
    )
    _write_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["promotion_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


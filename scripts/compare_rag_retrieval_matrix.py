"""Run the retrieval promotion gate across an explicit scenario matrix."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.rag_retrieval_gate import (  # noqa: E402
    BootstrapConfig,
    GateThresholds,
    RAGRetrievalGateError,
    compare_retrieval_runs,
    load_retrieval_run,
)


class JSONArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def parser() -> argparse.ArgumentParser:
    value = JSONArgumentParser(description=__doc__)
    value.add_argument(
        "--matrix",
        type=Path,
        required=True,
        help="JSON matrix with baseline/candidate entries for each scenario.",
    )
    value.add_argument("--output", type=Path)
    value.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    value.add_argument("--bootstrap-seed", type=int, default=20_260_809)
    return value


def _read_matrix(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RAGRetrievalGateError(f"matrix is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != "1.0":
        raise RAGRetrievalGateError("matrix.schema_version must be '1.0'")
    scenarios = value.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise RAGRetrievalGateError("matrix.scenarios must be a non-empty list")
    return value


def _path(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RAGRetrievalGateError(f"{label} must be a non-empty path")
    candidate = Path(value)
    return (base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def _entry(base: Path, value: Any, label: str) -> tuple[str, Path, str]:
    if not isinstance(value, dict):
        raise RAGRetrievalGateError(f"{label} must be an object")
    raw_label = value.get("label")
    raw_stage = value.get("stage")
    if not isinstance(raw_label, str) or not raw_label.strip():
        raise RAGRetrievalGateError(f"{label}.label must be non-empty")
    if not isinstance(raw_stage, str) or not raw_stage.strip():
        raise RAGRetrievalGateError(f"{label}.stage must be non-empty")
    return raw_label.strip(), _path(base, value.get("path"), f"{label}.path"), raw_stage.strip()


def _thresholds(value: Any) -> GateThresholds:
    if value is None:
        return GateThresholds()
    if not isinstance(value, dict):
        raise RAGRetrievalGateError("scenario.policy must be an object")
    allowed = {
        "profile",
        "min_recall_delta",
        "max_recall_drop",
        "max_candidate_drop",
        "min_candidate_recall_at_10",
        "min_candidate_recall_at_candidate_k",
        "max_category_degradation",
        "min_category_cases",
        "max_p95_ratio",
        "max_p95_ms",
        "require_nonnegative_ci_lower",
    }
    unknown = set(value).difference(allowed)
    if unknown:
        raise RAGRetrievalGateError(
            "scenario.policy contains unknown fields: " + ", ".join(sorted(unknown))
        )
    return GateThresholds(**value)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        matrix_path = args.matrix.resolve()
        matrix = _read_matrix(matrix_path)
        base = matrix_path.parent
        bootstrap = BootstrapConfig(
            repetitions=args.bootstrap_repetitions,
            seed=args.bootstrap_seed,
        )
        reports: list[dict[str, Any]] = []
        names: set[str] = set()
        for index, scenario in enumerate(matrix["scenarios"]):
            if not isinstance(scenario, dict):
                raise RAGRetrievalGateError(f"scenarios[{index}] must be an object")
            raw_name = scenario.get("name")
            name = raw_name.strip() if isinstance(raw_name, str) else ""
            if not name or name in names:
                raise RAGRetrievalGateError(
                    f"scenarios[{index}].name must be unique and non-empty"
                )
            names.add(name)
            baseline_label, baseline_path, baseline_stage = _entry(
                base, scenario.get("baseline"), f"scenarios[{index}].baseline"
            )
            candidate_label, candidate_path, candidate_stage = _entry(
                base, scenario.get("candidate"), f"scenarios[{index}].candidate"
            )
            raw_policy = scenario.get("policy") or {}
            if not isinstance(raw_policy, dict):
                raise RAGRetrievalGateError(
                    f"scenarios[{index}].policy must be an object"
                )
            policy = dict(raw_policy)
            raw_profile = policy.pop("profile", None)
            if raw_profile is not None and (
                not isinstance(raw_profile, str) or not raw_profile.strip()
            ):
                raise RAGRetrievalGateError(
                    f"scenarios[{index}].policy.profile must be a non-empty string"
                )
            raw_identical = policy.pop("require_identical_retrieval", False)
            if not isinstance(raw_identical, bool):
                raise RAGRetrievalGateError(
                    f"scenarios[{index}].policy.require_identical_retrieval must be boolean"
                )
            require_identical = raw_identical
            baseline = load_retrieval_run(
                baseline_path,
                label=baseline_label,
                stage=baseline_stage,
                profile_name=raw_profile,
            )
            candidate = load_retrieval_run(
                candidate_path,
                label=candidate_label,
                stage=candidate_stage,
                profile_name=raw_profile,
            )
            report = compare_retrieval_runs(
                baseline,
                [candidate],
                thresholds=_thresholds(policy),
                bootstrap=bootstrap,
                require_identical_retrieval=require_identical,
            )
            reports.append({"name": name, **report})
        passed = all(report["passed"] for report in reports)
        output = {
            "schema_version": "1.0",
            "gate": "retrieval_promotion_matrix",
            "passed": passed,
            "scenario_count": len(reports),
            "scenarios": reports,
        }
        payload = _json_bytes(output)
        if args.output is not None:
            _atomic_write(args.output, payload)
        sys.stdout.write(payload.decode("utf-8"))
        return 0 if passed else 1
    except (RAGRetrievalGateError, ValueError, OSError) as exc:
        payload = _json_bytes(
            {"schema_version": "1.0", "gate": "retrieval_promotion_matrix", "error": str(exc)}
        )
        sys.stdout.write(payload.decode("utf-8"))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

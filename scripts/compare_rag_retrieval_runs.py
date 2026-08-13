"""Compare immutable retrieval runs against a paired promotion gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

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
        "--stage",
        default=None,
        help="Use one stage name for both runs (shorthand for the two stage options).",
    )
    value.add_argument("--baseline-stage", default=None)
    value.add_argument("--candidate-stage", default=None)
    value.add_argument("--baseline", required=True, help="label=run-directory")
    value.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="label=run-directory; repeat for multiple candidates",
    )
    value.add_argument("--output", type=Path)
    value.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    value.add_argument("--bootstrap-seed", type=int, default=20_260_809)
    value.add_argument("--min-recall-delta", type=float, default=0.0)
    value.add_argument("--max-recall-drop", type=float, default=0.01)
    value.add_argument("--max-candidate-drop", type=float, default=0.01)
    value.add_argument("--max-category-degradation", type=float, default=0.03)
    value.add_argument("--min-category-cases", type=int, default=10)
    value.add_argument("--max-p95-ratio", type=float, default=1.2)
    value.add_argument(
        "--max-p95-ms",
        type=float,
        default=None,
        help="Add an absolute profile-local p95 limit alongside the ratio guard.",
    )
    value.add_argument(
        "--allow-negative-ci-lower",
        action="store_true",
        help="do not require the paired 95%% CI lower bound to be non-negative",
    )
    value.add_argument(
        "--require-identical-retrieval",
        action="store_true",
        help="require case-level retrieved ID sequences to be byte-identical",
    )
    return value


def _labelled_path(value: str, option: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise ValueError(f"{option} must use label=run-directory")
    return label.strip(), Path(raw_path.strip())


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
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
        baseline_stage = args.baseline_stage or args.stage
        candidate_stage = args.candidate_stage or args.stage
        if not baseline_stage or not candidate_stage:
            raise ValueError(
                "provide --stage or both --baseline-stage and --candidate-stage"
            )
        baseline_label, baseline_path = _labelled_path(args.baseline, "--baseline")
        candidate_specs = [_labelled_path(item, "--candidate") for item in args.candidate]
        thresholds = GateThresholds(
            min_recall_delta=args.min_recall_delta,
            max_recall_drop=args.max_recall_drop,
            max_candidate_drop=args.max_candidate_drop,
            max_category_degradation=args.max_category_degradation,
            min_category_cases=args.min_category_cases,
            max_p95_ratio=args.max_p95_ratio,
            max_p95_ms=args.max_p95_ms,
            require_nonnegative_ci_lower=not args.allow_negative_ci_lower,
        )
        bootstrap = BootstrapConfig(
            repetitions=args.bootstrap_repetitions,
            seed=args.bootstrap_seed,
        )
        baseline = load_retrieval_run(
            baseline_path, label=baseline_label, stage=baseline_stage
        )
        candidates = [
            load_retrieval_run(path, label=label, stage=candidate_stage)
            for label, path in candidate_specs
        ]
        report = compare_retrieval_runs(
            baseline,
            candidates,
            thresholds=thresholds,
            bootstrap=bootstrap,
            require_identical_retrieval=args.require_identical_retrieval,
        )
        payload = _json_bytes(report)
        if args.output is not None:
            _atomic_write(args.output, payload)
        sys.stdout.write(payload.decode("utf-8"))
        return 0 if report["passed"] else 1
    except (RAGRetrievalGateError, ValueError, OSError) as exc:
        error = {"schema_version": "1.0", "gate": "retrieval_promotion", "error": str(exc)}
        payload = _json_bytes(error)
        sys.stdout.write(payload.decode("utf-8"))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

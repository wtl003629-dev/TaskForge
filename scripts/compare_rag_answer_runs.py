"""Compare immutable RAG answer-eval runs against a paired promotion gate."""

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

from taskforge.rag_answer_gate import (
    BootstrapConfig,
    GateThresholds,
    RAGAnswerGateError,
    compare_answer_eval_runs,
    load_answer_eval_run,
)


class JSONArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def parser() -> argparse.ArgumentParser:
    value = JSONArgumentParser(description=__doc__)
    value.add_argument("--comparison", choices=("retriever", "agentic"), required=True)
    value.add_argument("--baseline", required=True, help="Baseline as label=run-directory.")
    value.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate as label=run-directory; repeat for multiple runs.",
    )
    value.add_argument("--output", type=Path)
    value.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    value.add_argument("--bootstrap-seed", type=int, default=20_260_809)
    value.add_argument("--min-token-f1-delta", type=float, default=0.03)
    value.add_argument("--min-evidence-recall-delta", type=float, default=0.03)
    value.add_argument("--max-category-degradation", type=float, default=0.03)
    value.add_argument("--min-category-cases", type=int, default=10)
    value.add_argument("--max-p95-ratio", type=float, default=None)
    value.add_argument(
        "--allow-negative-ci-lower",
        action="store_true",
        help="Do not require the paired 95%% CI lower bound to be non-negative.",
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
        baseline_label, baseline_path = _labelled_path(args.baseline, "--baseline")
        candidate_specs = [
            _labelled_path(value, "--candidate") for value in args.candidate
        ]
        default_ratio = 2.5 if args.comparison == "agentic" else 2.0
        thresholds = GateThresholds(
            min_token_f1_delta=args.min_token_f1_delta,
            min_evidence_recall_delta=args.min_evidence_recall_delta,
            max_category_degradation=args.max_category_degradation,
            min_category_cases=args.min_category_cases,
            max_p95_ratio=(
                args.max_p95_ratio if args.max_p95_ratio is not None else default_ratio
            ),
            require_nonnegative_ci_lower=not args.allow_negative_ci_lower,
        )
        bootstrap = BootstrapConfig(
            repetitions=args.bootstrap_repetitions,
            seed=args.bootstrap_seed,
        )
        baseline = load_answer_eval_run(baseline_path, label=baseline_label)
        candidates = [
            load_answer_eval_run(path, label=label) for label, path in candidate_specs
        ]
        report = compare_answer_eval_runs(
            baseline,
            candidates,
            comparison_type=args.comparison,
            thresholds=thresholds,
            bootstrap=bootstrap,
        )
        exit_code = 0 if report["passed"] else 1
    except (OSError, RAGAnswerGateError, ValueError) as exc:
        report = {
            "schema_version": "1.0",
            "passed": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        exit_code = 2
    output = args.output if "args" in locals() else None
    payload = _json_bytes(report)
    if output is not None:
        try:
            _atomic_write(output, payload)
        except OSError as exc:
            payload = _json_bytes(
                {
                    "schema_version": "1.0",
                    "passed": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
            exit_code = 2
    # stdout is the sole reporting channel; even validation failures are one
    # complete JSON document and never a prose traceback.
    if argv is None:
        sys.stdout.buffer.write(payload)
    else:
        print(payload.decode("utf-8"), end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

"""Run TaskForge's deterministic offline evaluation suite."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.evaluation import (
    EvaluationRunner,
    OfflineRuntimeFactory,
    load_cases,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic TaskForge trajectory evaluations."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=REPOSITORY_ROOT / "eval" / "cases.json",
        help="Path to the JSON case suite.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path; the report is always printed to stdout.",
    )
    return parser.parse_args()


async def run() -> int:
    args = parse_args()
    cases = load_cases(args.cases)
    report = await EvaluationRunner(OfflineRuntimeFactory()).run(cases)
    if args.output:
        report.write_json(args.output)
    print(report.to_json())
    return 0 if report.summary.failed_cases == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))


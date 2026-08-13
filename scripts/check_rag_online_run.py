"""Apply absolute safety, cost, latency, and quality gates to one online run."""

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

from taskforge.rag_answer_gate import RAGAnswerGateError  # noqa: E402
from taskforge.rag_online_gate import evaluate_online_answer_run  # noqa: E402


class JSONArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def parser() -> argparse.ArgumentParser:
    value = JSONArgumentParser(description=__doc__)
    value.add_argument("--run", type=Path, required=True)
    value.add_argument(
        "--profile", choices=("canary20", "full100", "repeat30"), required=True
    )
    value.add_argument("--output", type=Path)
    return value


def _payload(value: object) -> bytes:
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
        report = evaluate_online_answer_run(args.run, profile=args.profile)
        exit_code = 0 if report["passed"] else 1
    except (OSError, RAGAnswerGateError, ValueError) as exc:
        report = {
            "schema_version": "1.0",
            "passed": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        exit_code = 2
    output = args.output if "args" in locals() else None
    payload = _payload(report)
    if output is not None:
        try:
            _atomic_write(output, payload)
        except OSError as exc:
            payload = _payload(
                {
                    "schema_version": "1.0",
                    "passed": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
            exit_code = 2
    if argv is None:
        sys.stdout.buffer.write(payload)
    else:
        print(payload.decode("utf-8"), end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

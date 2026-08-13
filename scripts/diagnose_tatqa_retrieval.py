"""Run non-promotable O0/O1/O2/O3 diagnostics on a TAT-QA artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.rag_evaluation import load_tatqa_dataset  # noqa: E402
from taskforge.rag_retrieval_gate import (  # noqa: E402
    RAGRetrievalGateError,
    load_retrieval_run,
)
from taskforge.rag_tatqa_diagnostics import (  # noqa: E402
    TATQARetrievalRow,
    build_tatqa_query_plan,
    diagnose_tatqa_retrieval,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--run", type=Path, required=True)
    value.add_argument("--stage", required=True)
    value.add_argument("--dataset", type=Path, required=True)
    value.add_argument("--output", type=Path)
    value.add_argument("--top-k", type=int, default=10)
    value.add_argument("--candidate-k", type=int, default=None)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        run = load_retrieval_run(args.run, label="diagnostic", stage=args.stage)
        dataset = load_tatqa_dataset(args.dataset)
        case_map = {case.case_id: case for case in dataset.cases}
        if any(case_id not in case_map for case_id in run.case_ids):
            raise RAGRetrievalGateError("dataset does not contain the run's ordered cases")
        cases = [case_map[case_id] for case_id in run.case_ids]
        raw_rows = []
        raw_path = run.path / "predictions.jsonl"
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("stage") == args.stage:
                raw_rows.append(
                    TATQARetrievalRow(
                        case_id=row["case_id"],
                        category=row["category"],
                        relevant_ids=tuple(row["relevant_ids"]),
                        retrieved_ids=tuple(row["retrieved_ids"]),
                        retrieved_parent_ids=tuple(row.get("retrieved_parent_ids", [])),
                    )
                )
        report = diagnose_tatqa_retrieval(
            cases,
            dataset.documents,
            raw_rows,
            top_k=args.top_k,
            candidate_k=args.candidate_k or run.candidate_k,
        )
        report["run_id"] = run.run_id
        report["stage"] = args.stage
        report["query_plans"] = {
            case.case_id: build_tatqa_query_plan(case) for case in cases
        }
        payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
        sys.stdout.write(payload)
        return 0
    except (RAGRetrievalGateError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        sys.stdout.write(json.dumps({"error": str(exc)}, ensure_ascii=False) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

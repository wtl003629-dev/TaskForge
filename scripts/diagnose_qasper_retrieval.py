from __future__ import annotations

import argparse
import json
from pathlib import Path

from taskforge.qasper_diagnostics import diagnose_qasper_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose a locked QASPER retrieval run")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--stage", default=None)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose_qasper_run(
        args.run,
        dataset_path=args.dataset,
        split_path=args.split,
        stage=args.stage,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: result[key] for key in ("stage", "total_cases", "counts")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


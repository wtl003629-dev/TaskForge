"""Run frozen multilingual paper retrieval scenarios through paired chunkers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = (
    "neuclir-csl-zh-zh-10-v1",
    "neuclir-csl-en-zh-10-v1",
    "neuclir-csl-mixed-query-zh-10-v1",
    "bilingual-multipaper-mixed-corpus-15-v1",
)


def _case_count(split_path: Path) -> int:
    return len(json.loads(split_path.read_text(encoding="utf-8"))["case_ids"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument(
        "--reranker-profile",
        choices=("current", "multilingual"),
        default="current",
    )
    parser.add_argument(
        "--chunking-modes",
        nargs="+",
        choices=("flat", "parent_child"),
        default=("flat", "parent_child"),
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    dataset = (
        PROJECT_ROOT
        / ".taskforge"
        / "eval-cache"
        / "multilingual-paper"
        / f"{args.scenario}.json"
    )
    split = PROJECT_ROOT / "eval" / "splits" / f"{args.scenario}.json"
    cache = (
        PROJECT_ROOT
        / ".taskforge"
        / "eval-cache"
        / "multilingual-paper-bailian-v1.sqlite3"
    )
    if args.reranker_profile == "multilingual":
        reranker_backend = "fastembed"
        reranker_model = "jinaai/jina-reranker-v2-base-multilingual"
    else:
        reranker_backend = "fastembed_ensemble"
        reranker_model = (
            "jinaai/jina-reranker-v1-tiny-en,"
            "Xenova/ms-marco-MiniLM-L-6-v2"
        )

    for mode in args.chunking_modes:
        run_id = f"{args.scenario}-bailian-{args.reranker_profile}-{mode}-v1"
        output = PROJECT_ROOT / "eval" / "reports" / f"{run_id}.json"
        state = PROJECT_ROOT / ".taskforge" / "eval-runs" / run_id
        if args.skip_existing and output.is_file():
            print(json.dumps({"run_id": run_id, "status": "skipped"}))
            continue
        if state.exists() and any(state.iterdir()):
            raise SystemExit(f"refusing to reuse non-empty evaluation state: {state}")
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_qasper_direct_upload.py"),
            "--dataset",
            str(dataset),
            "--split",
            str(split),
            "--limit",
            str(_case_count(split)),
            "--rag-profile",
            "current",
            "--rag-ablation",
            "a",
            "--backend",
            "bailian",
            "--embedding-cache",
            str(cache),
            "--semantic-device",
            "cpu",
            "--reranker-backend",
            reranker_backend,
            "--reranker-model",
            reranker_model,
            "--no-graph",
            "--candidate-k",
            "50",
            "--agent-visible-k",
            "8",
            "--pdf-parser-backend",
            "native",
            "--pdf-chunking-mode",
            mode,
            "--operator-budget",
            "0",
            "--output",
            str(output),
            "--state-dir",
            str(state),
        ]
        if mode == "flat":
            command.extend(
                [
                    "--pdf-flat-chunk-chars",
                    "2000",
                    "--pdf-flat-overlap-chars",
                    "0",
                ]
            )
        else:
            command.extend(
                [
                    "--pdf-parent-target-tokens",
                    "2000",
                    "--pdf-parent-max-tokens",
                    "3000",
                    "--pdf-child-target-tokens",
                    "400",
                    "--pdf-child-max-tokens",
                    "500",
                    "--pdf-child-overlap-tokens",
                    "60",
                ]
            )
        print(json.dumps({"run_id": run_id, "status": "started"}))
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()

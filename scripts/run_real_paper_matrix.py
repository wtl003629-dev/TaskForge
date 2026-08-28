"""Run paired Flat/Parent-Child evaluations on frozen real-paper cohorts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = {
    "chinese-paper-fulltext-15-v1": (
        PROJECT_ROOT / ".taskforge" / "eval-cache" / "chinese-paper-fulltext-15-v1.json",
        PROJECT_ROOT / "eval" / "splits" / "chinese-paper-fulltext-15-v1.json",
        PROJECT_ROOT / ".taskforge" / "eval-cache" / "chinese-paper-fulltext-real-pdfs-v1.json",
    ),
    "chinese-paper-fulltext-en-query-15-v1": (
        PROJECT_ROOT / ".taskforge" / "eval-cache" / "paper-scenarios" / "chinese-paper-fulltext-en-query-15-v1.json",
        PROJECT_ROOT / "eval" / "splits" / "chinese-paper-fulltext-en-query-15-v1.json",
        PROJECT_ROOT / ".taskforge" / "eval-cache" / "paper-scenarios" / "chinese-paper-fulltext-en-query-15-v1-real-pdfs.json",
    ),
    "bilingual-paper-mixed-15-v1": (
        PROJECT_ROOT / ".taskforge" / "eval-cache" / "paper-scenarios" / "bilingual-paper-mixed-15-v1.json",
        PROJECT_ROOT / "eval" / "splits" / "bilingual-paper-mixed-15-v1.json",
        PROJECT_ROOT / ".taskforge" / "eval-cache" / "paper-scenarios" / "bilingual-paper-mixed-15-v1-real-pdfs.json",
    ),
}


def _count(split: Path) -> int:
    return len(json.loads(split.read_text(encoding="utf-8"))["case_ids"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), required=True)
    parser.add_argument("--reranker-profile", choices=("current", "multilingual"), default="current")
    parser.add_argument(
        "--include-dual-route",
        action="store_true",
        help="Also run the opt-in hybrid Flat-primary + Child-auxiliary route.",
    )
    parser.add_argument(
        "--dual-route-rag-ablation",
        choices=("c", "d", "e"),
        default="c",
        help="Optimized ablation for the dual-route run (default: c).",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    dataset, split, manifest = SCENARIOS[args.scenario]
    cache = PROJECT_ROOT / ".taskforge" / "eval-cache" / "paper-bailian-v1.sqlite3"
    if args.reranker_profile == "multilingual":
        # Keep the English model available for English-only queries, but put
        # the multilingual checkpoint on the explicit CJK/cross-lingual lane.
        reranker_backend = "fastembed_ensemble"
        reranker_model = "jinaai/jina-reranker-v1-tiny-en,Xenova/ms-marco-MiniLM-L-6-v2"
        multilingual_reranker_args = [
            "--multilingual-reranker-backend", "fastembed",
            "--multilingual-reranker-model", "jinaai/jina-reranker-v2-base-multilingual",
        ]
    else:
        reranker_backend = "fastembed_ensemble"
        reranker_model = "jinaai/jina-reranker-v1-tiny-en,Xenova/ms-marco-MiniLM-L-6-v2"
        multilingual_reranker_args = []
    modes = [("flat", "current", "a"), ("parent_child", "current", "a")]
    if args.include_dual_route:
        modes.append(("hybrid", "optimized", args.dual_route_rag_ablation))
    for mode, rag_profile, rag_ablation in modes:
        run_label = "dual" if mode == "hybrid" else mode
        run_id = f"{args.scenario}-bailian-{args.reranker_profile}-{run_label}-v1"
        output = PROJECT_ROOT / "eval" / "reports" / f"{run_id}.json"
        state = PROJECT_ROOT / ".taskforge" / "eval-runs" / run_id
        log = PROJECT_ROOT / ".taskforge" / "eval-runs" / f"{run_id}.log"
        if args.skip_existing and output.is_file():
            print(json.dumps({"run_id": run_id, "status": "skipped"}), flush=True)
            continue
        if state.exists() and any(state.iterdir()):
            raise SystemExit(f"refusing to reuse non-empty evaluation state: {state}")
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_qasper_direct_upload.py"),
            "--dataset", str(dataset),
            "--split", str(split),
            "--pdf-manifest", str(manifest),
            "--limit", str(_count(split)),
            "--rag-profile", rag_profile,
            "--rag-ablation", rag_ablation,
            "--backend", "bailian",
            "--embedding-cache", str(cache),
            "--semantic-device", "cpu",
            "--reranker-backend", reranker_backend,
            "--reranker-model", reranker_model,
            *multilingual_reranker_args,
            "--no-graph",
            "--candidate-k", "50",
            "--agent-visible-k", "8",
            "--pdf-parser-backend", "native",
            "--pdf-chunking-mode", mode,
            "--operator-budget", "0",
            "--output", str(output),
            "--state-dir", str(state),
        ]
        if mode == "hybrid":
            command.extend(["--dual-route", "--dual-route-rerank-candidate-k", "10"])
        if mode == "flat":
            command.extend(["--pdf-flat-chunk-chars", "2000", "--pdf-flat-overlap-chars", "0"])
        else:
            command.extend([
                "--pdf-parent-target-tokens", "2000",
                "--pdf-parent-max-tokens", "3000",
                "--pdf-child-target-tokens", "400",
                "--pdf-child-max-tokens", "500",
                "--pdf-child-overlap-tokens", "60",
            ])
        print(json.dumps({"run_id": run_id, "status": "started", "log": str(log)}), flush=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            tail = log.read_text(encoding="utf-8", errors="replace")[-4000:]
            print(json.dumps({"run_id": run_id, "status": "failed", "returncode": completed.returncode, "tail": tail}, ensure_ascii=True), flush=True)
            raise SystemExit(completed.returncode)
        print(json.dumps({"run_id": run_id, "status": "complete", "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()

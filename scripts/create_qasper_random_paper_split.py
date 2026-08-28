"""Create a reproducible QASPER split containing exactly N whole papers."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from taskforge.rag_baseline import sha256_file  # noqa: E402
from taskforge.rag_evaluation import load_qasper_dataset  # noqa: E402


def create(*, source: Path, output: Path, papers: int, seed: int) -> dict[str, object]:
    if papers <= 0:
        raise ValueError("papers must be positive")
    if output.exists():
        raise FileExistsError(f"split already exists: {output}")
    dataset = load_qasper_dataset(source)
    by_paper: dict[str, list[object]] = defaultdict(list)
    for case in dataset.cases:
        by_paper[str(case.metadata["paper_id"])].append(case)
    paper_ids = sorted(by_paper)
    if len(paper_ids) < papers:
        raise ValueError(f"only {len(paper_ids)} papers have answerable cases")
    selected_papers = sorted(random.Random(seed).sample(paper_ids, papers))
    selected_cases = sorted(
        (case for paper_id in selected_papers for case in by_paper[paper_id]),
        key=lambda case: case.case_id,
    )
    category_counts = Counter(case.category for case in selected_cases)
    manifest = {
        "schema_version": "1.0",
        "split_id": "qasper-dev-random-papers-30-v1",
        "dataset": dataset.dataset,
        "source_split": "official_dev",
        "source_sha256": sha256_file(source),
        "selection": {
            "strategy": "uniform_random_whole_papers_then_all_answerable_cases",
            "seed": seed,
            "requested_papers": papers,
            "selected_papers": len(selected_papers),
            "selected_cases": len(selected_cases),
            "answerable_only": True,
            "paper_ids": selected_papers,
        },
        "case_ids": [case.case_id for case in selected_cases],
        "category_counts": dict(sorted(category_counts.items())),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / ".taskforge" / "eval-cache" / "qasper-dev-v0.3.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "eval" / "splits" / "qasper-dev-random-papers-30-v1.json",
    )
    parser.add_argument("--papers", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    print(json.dumps(create(source=args.source, output=args.output, papers=args.papers, seed=args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

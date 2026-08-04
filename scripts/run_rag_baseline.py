"""Run TaskForge's deterministic TAT-QA lexical retrieval baseline."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.rag_baseline import (  # noqa: E402
    RAGBaselineConfig,
    load_baseline_config,
    run_rag_baseline,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run the deterministic offline BM25 TAT-QA baseline."
    )
    value.add_argument(
        "--input",
        type=Path,
        default=REPOSITORY_ROOT / ".taskforge" / "eval-cache" / "tatqa_dataset_dev.json",
        help="Path to the official TAT-QA JSON split.",
    )
    value.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "eval" / "rag_baseline_config.json",
        help="Versioned baseline configuration.",
    )
    value.add_argument(
        "--output",
        type=Path,
        help="Exact output run directory (must not already exist).",
    )
    value.add_argument("--limit", type=int, help="Override the sampled case count.")
    value.add_argument("--seed", type=int, help="Override the deterministic seed.")
    value.add_argument(
        "--unlocked-sample",
        action="store_true",
        help="Allow exploratory --limit/--seed sampling instead of the versioned locked split.",
    )
    return value


def _default_output(now: datetime) -> Path:
    stamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return REPOSITORY_ROOT / ".taskforge" / "eval-runs" / f"tatqa-bm25-{stamp}"


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = load_baseline_config(args.config)
        overrides: dict[str, object] = {}
        if args.limit is not None:
            overrides["limit"] = args.limit
        if args.seed is not None:
            overrides["seed"] = args.seed
        if overrides:
            if config.locked_split is not None and not args.unlocked_sample:
                raise ValueError(
                    "--limit/--seed require --unlocked-sample when a locked split is configured"
                )
            sampling = config.sampling.model_copy(update=overrides)
            # model_copy does not revalidate updates in Pydantic v2.
            config = RAGBaselineConfig.model_validate(
                {
                    **config.model_dump(mode="json"),
                    "sampling": sampling.model_dump(mode="json"),
                    "locked_split": None,
                }
            )
        now = datetime.now(timezone.utc)
        output = args.output or _default_output(now)
        result = run_rag_baseline(
            input_path=args.input,
            output_dir=output,
            config=config,
            repository_root=REPOSITORY_ROOT,
            created_at=now,
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
        print(f"rag baseline failed: {exc}", file=sys.stderr)
        return 2

    summary = result.report.summary
    print(f"run_id={result.manifest['run_id']}")
    print(f"output={result.output_dir.resolve()}")
    print(f"cases={summary.total_cases}")
    for k, recall in summary.recall_at_k.items():
        print(
            f"k={k} recall={recall:.6f} "
            f"mrr={summary.mrr_at_k[k]:.6f} ndcg={summary.ndcg_at_k[k]:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

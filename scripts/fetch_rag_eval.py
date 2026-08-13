from __future__ import annotations

import argparse
import json
from pathlib import Path

from taskforge.eval_datasets import (
    DatasetDownloadError,
    download_dataset_source,
    load_dataset_catalog,
)

ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Fetch pinned TaskForge RAG evaluation labels safely."
    )
    value.add_argument(
        "--catalog",
        type=Path,
        default=ROOT / "eval" / "rag_datasets.json",
    )
    value.add_argument("--dataset", action="append", default=[])
    value.add_argument("--all-automated", action="store_true")
    value.add_argument("--list", action="store_true")
    value.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / ".taskforge" / "eval-cache",
    )
    value.add_argument("--accept-noncommercial", action="store_true")
    value.add_argument(
        "--trust-environment-proxy",
        action="store_true",
        help=(
            "Use explicitly configured HTTP(S) proxy environment variables. "
            "Redirect hosts remain allowlisted and artifact hashes remain mandatory."
        ),
    )
    return value


def main() -> int:
    args = parser().parse_args()
    catalog = load_dataset_catalog(args.catalog)
    if args.list:
        for source in catalog.sources:
            mode = "automated" if source.automated else "manual"
            use = "commercial" if source.commercial_use else "non-commercial"
            print(f"{source.dataset_id:24} {mode:10} {use:15} {source.license}")
        return 0
    selected = list(args.dataset)
    if args.all_automated:
        selected.extend(
            source.dataset_id for source in catalog.sources if source.automated
        )
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise SystemExit("select --dataset ID, --all-automated, or --list")
    receipts = []
    for dataset_id in selected:
        try:
            source = catalog.get(dataset_id)
        except KeyError as exc:
            raise SystemExit(f"unknown dataset: {dataset_id}") from exc
        try:
            receipts.extend(
                download_dataset_source(
                    source,
                    output_dir=args.output_dir,
                    accept_noncommercial=args.accept_noncommercial,
                    trust_env=args.trust_environment_proxy,
                )
            )
        except DatasetDownloadError as exc:
            raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            [receipt.model_dump(mode="json") for receipt in receipts],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

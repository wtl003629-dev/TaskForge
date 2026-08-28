"""Create a deterministic evaluation split while excluding earlier locked IDs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.rag_baseline import (  # noqa: E402
    LockedSplitManifest,
    load_locked_split,
    select_stratified_cases,
    sha256_file,
)
from taskforge.rag_evaluation import (  # noqa: E402
    load_qasper_dataset,
    load_tatqa_dataset,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--input", type=Path, required=True)
    value.add_argument(
        "--dataset-adapter",
        choices=("tatqa", "qasper"),
        default="tatqa",
        help="Normalize the source with the matching locked evaluation adapter.",
    )
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--split-id", required=True)
    value.add_argument("--limit", type=int, default=100)
    value.add_argument(
        "--all-eligible",
        action="store_true",
        help=(
            "Write the complete deterministic complement after exclusions. "
            "Requires --group-by-parent and ignores --limit."
        ),
    )
    value.add_argument("--seed", type=int, required=True)
    value.add_argument(
        "--source-split",
        default="official_dev",
        help="Provenance label for the source artifact, for example official_train.",
    )
    value.add_argument("--exclude", type=Path, action="append", default=[])
    value.add_argument(
        "--include-from",
        type=Path,
        help="Restrict selection to the case IDs in an existing locked split.",
    )
    value.add_argument(
        "--group-by-parent",
        action="store_true",
        help="Select whole parent documents/contexts so cases from one report cannot straddle splits.",
    )
    value.add_argument(
        "--category-minimum",
        action="append",
        default=[],
        metavar="CATEGORY=COUNT",
        help="Minimum cases per category for a parent-disjoint split; repeatable.",
    )
    return value


def _parse_category_minimums(values: Sequence[str]) -> dict[str, int]:
    minimums: dict[str, int] = {}
    for raw in values:
        category, separator, count = raw.partition("=")
        category = category.strip()
        if not separator or not category:
            raise ValueError("--category-minimum must use CATEGORY=COUNT")
        try:
            parsed = int(count)
        except ValueError as exc:
            raise ValueError("--category-minimum count must be an integer") from exc
        if parsed < 1:
            raise ValueError("--category-minimum count must be positive")
        if category in minimums:
            raise ValueError(f"duplicate category minimum: {category}")
        minimums[category] = parsed
    return minimums


def _qasper_parent_from_case_id(case_id: str) -> str | None:
    """Recover the paper scope from a stable QASPER case identifier."""

    parts = str(case_id).split(":", 2)
    if len(parts) != 3 or parts[0] != "qasper" or not parts[1].strip():
        return None
    return f"qasper:{parts[1].strip()}:paper"


def _category_counts(groups: Sequence[tuple[str, Sequence[object]]]) -> Counter[str]:
    return Counter(
        getattr(case, "category", "")
        for _, cases in groups
        for case in cases
    )


def _select_parent_disjoint_cases(
    cases: Sequence[object],
    *,
    limit: int,
    seed: int,
    category_minimums: Mapping[str, int] | None = None,
) -> list[object]:
    """Select an exact-size, deterministic subset of whole parent contexts."""

    grouped: dict[str, list[object]] = defaultdict(list)
    for case in cases:
        metadata = getattr(case, "metadata", {})
        parent = str(metadata.get("parent_document_id", "")).strip()
        if not parent:
            raise ValueError("group-by-parent requires parent_document_id on every case")
        grouped[parent].append(case)
    groups = sorted(grouped.items(), key=lambda item: item[0])
    minimums = dict(category_minimums or {})

    def flatten(selected_groups: Sequence[tuple[str, Sequence[object]]]) -> list[object]:
        selected: list[object] = []
        for _, group in selected_groups:
            selected.extend(group)
        return selected

    if minimums:
        unknown = set(minimums) - {
            getattr(case, "category", "")
            for _, group in groups
            for case in group
        }
        if unknown:
            raise ValueError(
                "category minimum has no matching cases: "
                + ", ".join(sorted(unknown))
            )
        size_counts = Counter(len(group) for _, group in groups)
        uniform_options = [
            (count, size)
            for size, count in size_counts.items()
            if limit % size == 0 and count >= limit // size
        ]
        if not uniform_options:
            raise ValueError(
                "category minimum selection requires enough equal-sized parent groups"
            )
        _, group_size = max(uniform_options)
        groups = [item for item in groups if len(item[1]) == group_size]
        if limit % group_size:
            raise ValueError(
                "category minimum selection requires limit divisible by parent group size"
            )
        group_count = limit // group_size
        if group_count > len(groups):
            raise ValueError("not enough parent groups for the requested split")
        rng = random.Random(seed)
        best: tuple[int, list[tuple[str, Sequence[object]]], Counter[str]] | None = None
        # Sampling is deterministic for a fixed seed.  It is appropriate here
        # because the TAT-QA train corpus has thousands of nearly uniform
        # six-question parent groups; no gold answers or retrieval scores enter
        # the objective.
        for _ in range(100_000):
            candidate = rng.sample(groups, group_count)
            counts = _category_counts(candidate)
            deficit = sum(
                max(0, minimum - counts.get(category, 0)) ** 2
                for category, minimum in minimums.items()
            )
            if best is None or deficit < best[0]:
                best = (deficit, candidate, counts)
            if deficit == 0:
                return flatten(candidate)
        assert best is not None
        unmet = {
            category: minimum - best[2].get(category, 0)
            for category, minimum in minimums.items()
            if best[2].get(category, 0) < minimum
        }
        raise ValueError(
            "could not satisfy category minimums; unmet="
            + json.dumps(unmet, sort_keys=True)
        )

    random.Random(seed).shuffle(groups)
    # Subset-sum over group sizes finds an exact limit without selecting a
    # partial report.  The predecessor tuple is bounded by ``limit``.
    reachable: dict[int, tuple[int, ...]] = {0: ()}
    for index, (_, group) in enumerate(groups):
        size = len(group)
        for total in sorted(tuple(reachable), reverse=True):
            next_total = total + size
            if next_total <= limit and next_total not in reachable:
                reachable[next_total] = reachable[total] + (index,)
        if limit in reachable:
            break
    indexes = reachable.get(limit)
    if indexes is None:
        raise ValueError("cannot select the requested limit using whole parent contexts")
    return flatten([groups[index] for index in indexes])


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        category_minimums = _parse_category_minimums(args.category_minimum)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if category_minimums and not args.group_by_parent:
        raise SystemExit("--category-minimum requires --group-by-parent")
    if args.all_eligible and not args.group_by_parent:
        raise SystemExit("--all-eligible requires --group-by-parent")
    if args.all_eligible and category_minimums:
        raise SystemExit("--all-eligible cannot use --category-minimum")
    source = args.input.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise SystemExit(f"input does not exist: {source}")
    try:
        output.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise SystemExit("output must stay inside the repository") from exc
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    dataset = (
        load_qasper_dataset(source)
        if args.dataset_adapter == "qasper"
        else load_tatqa_dataset(source)
    )
    included: set[str] | None = None
    parent_split_id: str | None = None
    if args.include_from is not None:
        parent = load_locked_split(args.include_from)
        if parent.dataset != dataset.dataset:
            raise SystemExit("include-from split uses a different dataset")
        if parent.source_sha256 != sha256_file(source):
            raise SystemExit("include-from split uses a different source artifact")
        included = set(parent.case_ids)
        parent_split_id = parent.split_id
    excluded: set[str] = set()
    excluded_parents: set[str] = set()
    excluded_split_ids: list[str] = []
    for path in args.exclude:
        manifest = load_locked_split(path)
        excluded.update(manifest.case_ids)
        excluded_split_ids.append(manifest.split_id)
        if args.group_by_parent:
            if args.dataset_adapter == "qasper":
                excluded_parents.update(
                    parent
                    for case_id in manifest.case_ids
                    if (parent := _qasper_parent_from_case_id(case_id))
                )
            excluded_parents.update(
                str(case.metadata.get("parent_document_id", "")).strip()
                for case in dataset.cases
                if case.case_id in manifest.case_ids
                and str(case.metadata.get("parent_document_id", "")).strip()
            )
    eligible = [
        case
        for case in dataset.cases
        if case.case_id not in excluded
        and (
            not args.group_by_parent
            or str(case.metadata.get("parent_document_id", "")).strip()
            not in excluded_parents
        )
        and (included is None or case.case_id in included)
    ]
    if args.all_eligible:
        selected = sorted(eligible, key=lambda case: case.case_id)
    elif args.group_by_parent:
        try:
            selected = _select_parent_disjoint_cases(
                eligible,
                limit=args.limit,
                seed=args.seed,
                category_minimums=category_minimums,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        selected = select_stratified_cases(
            eligible,
            limit=args.limit,
            seed=args.seed,
        )
    if not args.all_eligible and len(selected) != args.limit:
        raise SystemExit("not enough eligible cases for the requested split")
    manifest = LockedSplitManifest(
        split_id=args.split_id,
        dataset=dataset.dataset,
        source_split=args.source_split,
        source_sha256=sha256_file(source),
        selection={
            "strategy": (
                "parent_document_disjoint_complement"
                if args.all_eligible
                else "parent_document_disjoint_subset"
                if args.group_by_parent
                else
                "category_round_robin_within_locked"
                if included is not None
                else "category_round_robin_excluding_locked"
            ),
            "limit": len(selected) if args.all_eligible else args.limit,
            "seed": args.seed,
            "excluded_split_ids": excluded_split_ids,
            "parent_split_id": parent_split_id,
            "group_by_parent": args.group_by_parent,
            "all_eligible": args.all_eligible,
            "category_minimums": category_minimums,
            "dataset_adapter": args.dataset_adapter,
        },
        case_ids=[case.case_id for case in selected],
        category_counts=dict(sorted(Counter(case.category for case in selected).items())),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"output={output}")
    print(f"cases={len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

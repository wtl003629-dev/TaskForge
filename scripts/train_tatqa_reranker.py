"""Train the lightweight TAT-QA domain reranker from official train data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.rag_baseline import load_locked_split, sha256_file
from taskforge.rag_evaluation import load_tatqa_dataset
from taskforge.tatqa_reranker import TATQADomainReranker


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _examples(
    dataset_path: Path,
    case_ids: set[str],
    *,
    negatives_per_positive: int,
    seed: int,
):
    dataset = load_tatqa_dataset(dataset_path)
    documents = {document.document_id: document for document in dataset.documents}
    by_parent: dict[str, list[str]] = {}
    for document in dataset.documents:
        parent = str(document.metadata.get("parent_document_id", document.document_id))
        by_parent.setdefault(parent, []).append(document.document_id)
    rng = random.Random(seed)
    examples: list[tuple[str, str, int]] = []
    selected_cases = [case for case in dataset.cases if case.case_id in case_ids]
    if len(selected_cases) != len(case_ids):
        raise ValueError("training split references cases missing from the dataset")
    for case in selected_cases:
        positives = [documents[item] for item in case.relevant_ids if item in documents]
        if not positives:
            continue
        parent = str(case.metadata.get("parent_document_id", ""))
        same_parent = [
            documents[item]
            for item in by_parent.get(parent, [])
            if item not in set(case.relevant_ids)
        ]
        excluded_ids = set(case.relevant_ids).union(
            item.document_id for item in same_parent
        )
        global_candidates = [
            document for document in dataset.documents
            if document.document_id not in excluded_ids
        ]
        for positive in positives:
            examples.append((case.query, positive.text, 1))
            local_pool = list(same_parent)
            rng.shuffle(local_pool)
            global_count = min(
                len(global_candidates),
                max(8, negatives_per_positive * 2),
            )
            pool = [
                *local_pool,
                *rng.sample(global_candidates, global_count),
            ]
            for negative in pool[:negatives_per_positive]:
                examples.append((case.query, negative.text, 0))
    return dataset, examples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=Path,
        default=REPOSITORY_ROOT / ".taskforge" / "eval-cache" / "tatqa_dataset_train.json",
    )
    parser.add_argument(
        "--train-split",
        type=Path,
        required=True,
        help="Document-disjoint training partition manifest; full train is forbidden.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT
        / ".taskforge"
        / "eval-cache"
        / "tatqa_domain_reranker.json",
    )
    parser.add_argument("--negatives-per-positive", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args(argv)
    if args.negatives_per_positive < 1:
        raise ValueError("--negatives-per-positive must be positive")
    split = load_locked_split(args.train_split)
    if split.dataset != "TAT-QA":
        raise ValueError("--train-split must describe TAT-QA")
    if split.source_sha256 != sha256_file(args.train):
        raise ValueError("--train-split source hash does not match --train")
    if split.selection.get("strategy") != "parent_document_disjoint_complement":
        raise ValueError(
            "--train-split must be a parent-document-disjoint complement"
        )
    dataset, examples = _examples(
        args.train,
        set(split.case_ids),
        negatives_per_positive=args.negatives_per_positive,
        seed=args.seed,
    )
    model = TATQADomainReranker.fit(examples, epochs=args.epochs)
    model.save(args.output)
    payload = {
        "artifact": str(args.output.resolve()),
        "artifact_sha256": _sha256(args.output),
        "dataset": "TAT-QA",
        "dataset_path": str(args.train.resolve()),
        "dataset_sha256": _sha256(args.train),
        "train_split_id": split.split_id,
        "train_split_path": str(args.train_split.resolve()),
        "train_split_sha256": _sha256(args.train_split),
        "corpus_documents": len(dataset.documents),
        "training_cases": len(split.case_ids),
        "examples": len(examples),
        "positive_examples": sum(label for _, _, label in examples),
        "negative_examples": sum(1 - label for _, _, label in examples),
        "seed": args.seed,
        "epochs": args.epochs,
        "model_id": model.model_id,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

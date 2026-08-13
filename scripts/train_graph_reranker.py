"""Train the lightweight graph-feature ranker from a fit-complement run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.graph_reranker import (  # noqa: E402
    sha256_model,
    train_pairwise_graph_reranker,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--fit-run", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument(
        "--stage",
        default="graph_feature_rerank",
        help="Prediction stage whose graph features supply the training rows.",
    )
    value.add_argument("--epochs", type=int, default=35)
    value.add_argument("--learning-rate", type=float, default=0.04)
    value.add_argument("--l2", type=float, default=0.002)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    run_dir = (REPOSITORY_ROOT / args.fit_run).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [row for row in rows if row.get("stage") == args.stage]
    if not selected:
        raise SystemExit(f"no rows found for stage {args.stage!r}")
    base_rows = {
        str(row["case_id"]): row
        for row in rows
        if row.get("stage") == "qdrant_qasper_dense_rerank"
    }
    # Join immutable base-stage score arrays so training can use raw dense and
    # cross-encoder evidence without reading labels from another split.
    for row in selected:
        base = base_rows.get(str(row["case_id"]))
        graph = row.get("graph")
        if not isinstance(base, dict) or not isinstance(graph, dict):
            continue
        feature_map = graph.get("features")
        if not isinstance(feature_map, dict):
            continue
        ids = [str(value) for value in base.get("retrieved_ids", [])]
        raw_scores = list(base.get("scores", []))
        raw_base_scores = list(base.get("base_scores", []))
        raw_reranker_scores = list(base.get("reranker_scores", []))
        for index, chunk_id in enumerate(ids):
            feature = feature_map.get(chunk_id)
            if not isinstance(feature, dict):
                continue
            if index < len(raw_scores):
                feature["raw_score"] = raw_scores[index]
            if index < len(raw_base_scores):
                feature["raw_base_score"] = raw_base_scores[index]
            if index < len(raw_reranker_scores):
                feature["reranker_score"] = raw_reranker_scores[index]
    model = train_pairwise_graph_reranker(
        selected,
        fit_run_id=str(manifest["run_id"]),
        dataset_sha256=str(manifest["dataset"]["normalized_sha256"]),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    output = (REPOSITORY_ROOT / args.output).resolve()
    try:
        output.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise SystemExit("--output must stay inside the repository") from exc
    model.save(output)
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "sha256": sha256_model(output),
                "fit_run_id": model.fit_run_id,
                "dataset_sha256": model.dataset_sha256,
                "training_cases": model.training_cases,
                "positive_pairs": model.positive_pairs,
                "feature_names": list(model.feature_names),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

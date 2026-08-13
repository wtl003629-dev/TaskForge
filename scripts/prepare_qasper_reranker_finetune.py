"""Convert the audited QASPER candidate report to FlagEmbedding reranker JSON.

The report is produced by ``build_qasper_reranker_training.py``.  This adapter
keeps the paper-disjoint split and turns document IDs into their authoritative
paragraph text; gold evidence is used only as labels, never as retrieval input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.rag_evaluation import load_qasper_dataset  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(report_path: Path, dataset_path: Path, output: Path, *, max_negatives: int = 50) -> dict[str, Any]:
    if max_negatives < 1:
        raise ValueError("max_negatives must be positive")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = report.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("reranker report has no records")
    dataset = load_qasper_dataset(dataset_path)
    by_id = {document.document_id: document for document in dataset.documents}
    examples: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for record in records:
        query = str(record.get("query") or "").strip()
        positive_ids = [str(value) for value in record.get("positives", [])]
        negative_ids = [
            str(value)
            for value in (*record.get("hard_negatives", []), *record.get("negatives", []))
        ]
        positives = [by_id[item].text for item in positive_ids if item in by_id and by_id[item].text.strip()]
        negatives = [
            by_id[item].text
            for item in dict.fromkeys(negative_ids)
            if item in by_id and by_id[item].text.strip()
        ][:max_negatives]
        if not query or not positives or not negatives:
            dropped.append(
                {
                    "case_id": record.get("case_id"),
                    "missing_positive": not bool(positives),
                    "missing_negative": not bool(negatives),
                }
            )
            continue
        examples.append(
            {
                "query": query,
                "pos": positives,
                "neg": negatives,
                "metadata": {
                    "case_id": record.get("case_id"),
                    "paper_id": record.get("paper_id"),
                    "query_type": record.get("query_type", "general"),
                    "first_positive_rank": record.get("first_positive_rank"),
                },
            }
        )
    result = {
        "schema_version": "1.0",
        "format": "flagembedding_reranker_grouped_json",
        "dataset": "QASPER",
        "source_report": str(report_path),
        "source_report_sha256": _sha256(report_path),
        "source_dataset": str(dataset_path),
        "source_dataset_sha256": _sha256(dataset_path),
        "split_id": report.get("split_id"),
        "examples": len(examples),
        "dropped": dropped,
        "records": examples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # FlagEmbedding consumes a JSON array of grouped examples.  Keep audit
    # provenance in a sidecar rather than adding a wrapper object that its
    # Hugging Face dataset loader would interpret as a single training row.
    output.write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".manifest.json").write_text(
        json.dumps({key: value for key, value in result.items() if key != "records"}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-negatives", type=int, default=50)
    args = parser.parse_args()
    result = build(args.report, args.dataset, args.output, max_negatives=args.max_negatives)
    print(json.dumps({key: result[key] for key in ("split_id", "examples", "dropped")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

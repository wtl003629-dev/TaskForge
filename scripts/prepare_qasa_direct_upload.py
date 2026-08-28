"""Prepare a frozen ScholarQABench QASA cohort for direct-upload retrieval.

QASA provides one full scientific paper as paragraph ``ctxs`` plus integer
indices in ``gold_ctxs``.  This command converts a deterministic document-
diverse subset into the QASPER-shaped internal fixture already consumed by
TaskForge's direct PDF upload retrieval evaluator.  It does not alter the
source annotations or inject gold text into retrieval queries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / ".taskforge"
    / "eval-cache"
    / "ScholarQABench"
    / "data"
    / "single_paper_tasks"
    / "qasa_test.jsonl"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paper_id(contexts: list[Mapping[str, Any]]) -> str:
    identifiers = {
        str(item.get("id", "")).split("_all_", 1)[0].strip()
        for item in contexts
    }
    identifiers.discard("")
    if len(identifiers) != 1:
        raise ValueError("QASA row must contain contexts from exactly one paper")
    return identifiers.pop()


def _question_id(question: str) -> str:
    return hashlib.sha1(question.encode("utf-8")).hexdigest()


def prepare(
    source: Path,
    dataset_output: Path,
    split_output: Path,
    *,
    limit: int,
) -> dict[str, Any]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    selected: list[tuple[dict[str, Any], str]] = []
    seen_papers: set[str] = set()
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"QASA line {line_number} must be an object")
            contexts = raw.get("ctxs")
            gold_contexts = raw.get("gold_ctxs")
            question = str(raw.get("input", "")).strip()
            if (
                not isinstance(contexts, list)
                or not contexts
                or any(not isinstance(item, Mapping) for item in contexts)
                or not isinstance(gold_contexts, list)
                or not gold_contexts
                or not question
            ):
                raise ValueError(f"QASA line {line_number} is incomplete")
            paper_id = _paper_id(contexts)
            if paper_id in seen_papers:
                continue
            gold_indices = [int(index) for index in gold_contexts]
            if any(index < 0 or index >= len(contexts) for index in gold_indices):
                raise ValueError(f"QASA line {line_number} has an invalid gold index")
            if any(not str(item.get("text", "")).strip() for item in contexts):
                raise ValueError(f"QASA line {line_number} has an empty context")
            seen_papers.add(paper_id)
            selected.append((raw, paper_id))
            if len(selected) >= limit:
                break
    if len(selected) != limit:
        raise ValueError(f"QASA source only yielded {len(selected)} unique papers")

    fixture: dict[str, Any] = {}
    case_ids: list[str] = []
    selected_rows: list[dict[str, Any]] = []
    for raw, paper_id in selected:
        contexts = raw["ctxs"]
        question = str(raw["input"]).strip()
        question_id = _question_id(question)
        title = str(contexts[0].get("title", "")).strip() or paper_id
        gold_indices = list(dict.fromkeys(int(index) for index in raw["gold_ctxs"]))
        paragraphs = [str(item["text"]).strip() for item in contexts]
        evidence = [paragraphs[index] for index in gold_indices]
        fixture[paper_id] = {
            "title": title,
            "abstract": "",
            "full_text": [
                {
                    "section_name": "Paper body",
                    "paragraphs": paragraphs,
                }
            ],
            "figures_and_tables": [],
            "qas": [
                {
                    "question": question,
                    "question_id": question_id,
                    "answers": [
                        {
                            "annotation_id": "qasa-gold",
                            "answer": {
                                "unanswerable": False,
                                "free_form_answer": str(raw.get("answer", "")).strip(),
                                "extractive_spans": [],
                                "yes_no": None,
                                "evidence": evidence,
                            },
                        }
                    ],
                }
            ],
        }
        case_id = f"qasper:{paper_id}:{question_id}"
        case_ids.append(case_id)
        selected_rows.append(
            {
                "case_id": case_id,
                "qasa_paper_id": paper_id,
                "paper_title": title,
                "context_count": len(paragraphs),
                "gold_context_indices": gold_indices,
            }
        )

    source_sha256 = _sha256(source)
    split = {
        "schema_version": "1.0",
        "split_id": f"scholarqabench-qasa-document-diverse-{limit}-v1",
        "dataset": "ScholarQABench QASA",
        "source": str(source),
        "source_sha256": source_sha256,
        "selection_policy": "first question from each unique paper in source order",
        "synthetic_pdf_layout": "compact_scientific_paper_v1",
        "case_ids": case_ids,
        "selected_rows": selected_rows,
        "report_metadata": {
            "evaluation_type": "qasa_synthetic_pdf_upload_retrieval",
            "benchmark_track": "scientific_paper_fulltext_retrieval",
            "dataset": "ScholarQABench QASA official test, document-diverse frozen subset",
            "license": "ODC-BY (aggregate ScholarQABench data; constituent terms apply)",
            "synthetic_layout_limitation": (
                "QASA text and labels are official ScholarQABench data; the PDF "
                "layout is generated locally, so this isolates full-text retrieval "
                "but is not a real-PDF parser benchmark."
            ),
        },
    }
    dataset_output.parent.mkdir(parents=True, exist_ok=True)
    split_output.parent.mkdir(parents=True, exist_ok=True)
    dataset_output.write_text(
        json.dumps(fixture, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    split_output.write_text(
        json.dumps(split, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "source_sha256": source_sha256,
        "papers": len(fixture),
        "cases": len(case_ids),
        "dataset_output": str(dataset_output),
        "split_output": str(split_output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=(
            PROJECT_ROOT
            / ".taskforge"
            / "eval-cache"
            / "qasa-paper-retrieval-20-v1.json"
        ),
    )
    parser.add_argument(
        "--split-output",
        type=Path,
        default=PROJECT_ROOT / "eval" / "splits" / "qasa-paper-retrieval-20-v1.json",
    )
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    report = prepare(
        args.source,
        args.dataset_output,
        args.split_output,
        limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()

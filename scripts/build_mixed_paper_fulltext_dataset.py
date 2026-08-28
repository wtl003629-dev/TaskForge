"""Build a mixed Chinese/English full-text paper corpus.

The repository contains two complementary local datasets:

* QASPER train + validation: 1,169 English papers with full text and
  paragraph-grounded questions;
* ``chinese-ai-oa-jos-v2``: 1,000 quality-gated Chinese AI papers with
  extracted full text from public Journal of Software article PDFs.

This script combines only full-text records.  It intentionally does not carry
the 100,000 Chinese NeuCLIR-CSL title/abstract records from
``bilingual-paper-corpus-v1`` into this corpus, because mixing abstract-only
and full-text documents would make retrieval and chunking comparisons harder
to interpret.  QASPER queries and qrels are copied as an English-only
evaluation subset; no Chinese relevance labels are inferred.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENGLISH_DIR = PROJECT_ROOT / ".taskforge" / "datasets" / "bilingual-paper-corpus-v1"
DEFAULT_CHINESE_DIR = PROJECT_ROOT / ".taskforge" / "datasets" / "chinese-ai-oa-jos-v2"
DEFAULT_OUTPUT = PROJECT_ROOT / ".taskforge" / "datasets" / "mixed-paper-fulltext-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl_gz(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            yield value


def _write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=False))
            stream.write("\n")
            count += 1
    return count


def _full_text(value: Mapping[str, Any], *, source: str) -> str:
    text = str(value.get("text") or "").strip()
    if not text:
        raise ValueError(f"{source} record has empty text: {value.get('document_id')!r}")
    return text


def _load_documents(
    english_path: Path,
    chinese_path: Path,
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str], dict[str, int]]:
    documents: list[dict[str, Any]] = []
    by_language: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    text_hashes: Counter[str] = Counter()
    seen_ids: set[str] = set()

    for row in _read_jsonl_gz(english_path):
        if row.get("language") != "en" or row.get("document_type") != "full_text":
            continue
        document_id = str(row.get("document_id") or "").strip()
        if not document_id:
            raise ValueError("English record has no document_id")
        if document_id in seen_ids:
            raise ValueError(f"duplicate document_id: {document_id}")
        text = _full_text(row, source="QASPER")
        normalized = {
            "document_id": document_id,
            "paper_id": str(row.get("paper_id") or document_id).strip(),
            "language": "en",
            "title": str(row.get("title") or "").strip(),
            "abstract": str(row.get("abstract") or "").strip(),
            "text": text,
            "document_type": "full_text",
            "source_dataset": "QASPER",
            "source_split": row.get("source_split"),
            "license": str(row.get("license") or "CC BY 4.0"),
        }
        documents.append(normalized)
        seen_ids.add(document_id)
        by_language["en"] += 1
        by_source["QASPER"] += 1
        text_hashes[hashlib.sha256(text.encode("utf-8")).hexdigest()] += 1

    for row in _read_jsonl_gz(chinese_path):
        if row.get("language") != "zh":
            continue
        document_id = str(row.get("document_id") or "").strip()
        if not document_id:
            raise ValueError("Chinese record has no document_id")
        if document_id in seen_ids:
            raise ValueError(f"duplicate document_id: {document_id}")
        text = _full_text(row, source="chinese-ai-oa-jos-v2")
        normalized = {
            "document_id": document_id,
            "paper_id": str(row.get("paper_id") or document_id).strip(),
            "language": "zh",
            "title": str(row.get("title") or "").strip(),
            "abstract": str(row.get("abstract") or "").strip(),
            "text": text,
            "document_type": "full_text",
            "source_dataset": "chinese-ai-oa-jos-v2",
            "source_split": None,
            "license": str(
                row.get("license") or "website-download; redistribution not assumed"
            ),
            # Preserve provenance and the quality gate for downstream audits.
            "source": row.get("source"),
            "source_url": row.get("source_url"),
            "pdf_url": row.get("pdf_url"),
            "quality_bucket": row.get("quality_bucket"),
            "quality_score": row.get("quality_score"),
            "metadata": row.get("metadata") or {},
        }
        documents.append(normalized)
        seen_ids.add(document_id)
        by_language["zh"] += 1
        by_source["chinese-ai-oa-jos-v2"] += 1
        text_hashes[hashlib.sha256(text.encode("utf-8")).hexdigest()] += 1

    if not documents:
        raise ValueError("no full-text documents found")
    duplicate_text_records = sum(count - 1 for count in text_hashes.values() if count > 1)
    return documents, by_language, by_source, {
        "unique_texts": len(text_hashes),
        "duplicate_text_records": duplicate_text_records,
    }


def _copy_qasper_eval(
    source_dir: Path,
    output_dir: Path,
    valid_document_ids: set[str],
) -> tuple[int, int]:
    """Copy only QASPER queries/qrels whose documents are in this corpus."""

    query_rows: list[dict[str, Any]] = []
    for row in _read_jsonl_gz(source_dir / "queries.jsonl.gz"):
        document_id = str(row.get("document_id") or "")
        if row.get("source_dataset") == "QASPER" and document_id in valid_document_ids:
            query_rows.append(row)

    query_ids = {str(row.get("query_id")) for row in query_rows}
    qrel_rows: list[dict[str, Any]] = []
    for row in _read_jsonl_gz(source_dir / "qrels.jsonl.gz"):
        if (
            row.get("source_dataset") == "QASPER"
            and str(row.get("query_id")) in query_ids
            and str(row.get("document_id")) in valid_document_ids
        ):
            qrel_rows.append(row)

    _write_jsonl_gz(output_dir / "queries.jsonl.gz", query_rows)
    _write_jsonl_gz(output_dir / "qrels.jsonl.gz", qrel_rows)
    return len(query_rows), len(qrel_rows)


def build_dataset(
    *,
    english_dir: Path,
    chinese_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    english_corpus = english_dir / "corpus.jsonl.gz"
    chinese_corpus = chinese_dir / "corpus.jsonl.gz"
    for path in (english_corpus, chinese_corpus):
        if not path.exists():
            raise FileNotFoundError(path)

    documents, by_language, by_source, text_stats = _load_documents(
        english_corpus, chinese_corpus
    )
    documents.sort(key=lambda row: (row["language"], row["source_dataset"], row["document_id"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.jsonl.gz"
    papers_path = output_dir / "papers.jsonl.gz"
    _write_jsonl_gz(corpus_path, documents)
    papers = [{key: value for key, value in row.items() if key != "text"} for row in documents]
    _write_jsonl_gz(papers_path, papers)

    valid_document_ids = {str(row["document_id"]) for row in documents}
    query_count, qrel_count = _copy_qasper_eval(
        english_dir, output_dir, valid_document_ids
    )

    source_manifest: dict[str, Any] = {}
    for label, directory, corpus_file in (
        ("QASPER", english_dir, english_corpus),
        ("chinese-ai-oa-jos-v2", chinese_dir, chinese_corpus),
    ):
        source_manifest[label] = {
            "dataset": label,
            "path": str(directory),
            "corpus_file": str(corpus_file),
            "corpus_sha256": _sha256(corpus_file),
            "documents": by_source[label],
        }

    manifest = {
        "schema_version": "1.0",
        "dataset_id": "mixed-paper-fulltext-v1",
        "built_at": datetime.now(UTC).isoformat(),
        "purpose": "mixed Chinese/English full-text retrieval corpus",
        "counts": {
            "documents": len(documents),
            "documents_by_language": dict(sorted(by_language.items())),
            "documents_by_source": dict(sorted(by_source.items())),
            "documents_by_type": {"full_text": len(documents)},
            "queries": query_count,
            "qrels": qrel_count,
        },
        "text_deduplication": text_stats,
        "sources": source_manifest,
        "evaluation": {
            "queries_and_qrels": "QASPER English only",
            "chinese_qrels": 0,
            "note": "No Chinese relevance labels were inferred for the JOS full-text papers.",
        },
        "files": {
            "corpus": {
                "path": corpus_path.name,
                "records": len(documents),
                "sha256": _sha256(corpus_path),
            },
            "papers": {
                "path": papers_path.name,
                "records": len(papers),
                "sha256": _sha256(papers_path),
            },
            "queries": {
                "path": "queries.jsonl.gz",
                "records": query_count,
                "sha256": _sha256(output_dir / "queries.jsonl.gz"),
            },
            "qrels": {
                "path": "qrels.jsonl.gz",
                "records": qrel_count,
                "sha256": _sha256(output_dir / "qrels.jsonl.gz"),
            },
        },
        "limitations": [
            "The English portion is QASPER full text and is licensed CC BY 4.0.",
            "The Chinese portion is extracted from public Journal of Software PDF pages; redistribution rights are not inferred.",
            "QASPER queries/qrels evaluate only English documents; Chinese retrieval needs a separately judged query set.",
            "The corpus contains source-language text and does not translate or align papers across languages.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--english-dir", type=Path, default=DEFAULT_ENGLISH_DIR)
    parser.add_argument("--chinese-dir", type=Path, default=DEFAULT_CHINESE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = build_dataset(
        english_dir=args.english_dir,
        chinese_dir=args.chinese_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

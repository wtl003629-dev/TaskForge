"""Build a reproducible Chinese/English paper retrieval corpus.

The repository already contains two complementary, license-pinned sources:

* QASPER train + validation: 1,169 English scientific papers with full text
  and paragraph-grounded questions;
* NeuCLIR CSL: 395k Chinese scientific-paper records with title/abstract
  metadata, plus 110 Chinese/English search topics and qrels.

This script combines them into compressed JSONL files without translating or
modifying source text.  The default build keeps all judged Chinese documents
and a deterministic 100k-document Chinese sample, while retaining every
English QASPER paper.  Pass ``--chinese-limit 0`` to include the full CSL
corpus.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QASPER_TRAIN = PROJECT_ROOT / ".taskforge" / "eval-cache" / "qasper-train-v0.3.json"
DEFAULT_QASPER_VALIDATION = PROJECT_ROOT / ".taskforge" / "eval-cache" / "qasper-validation-v0.3.json"
DEFAULT_CSL = PROJECT_ROOT / ".taskforge" / "eval-cache" / "neuclir-csl" / "data" / "csl.jsonl.gz"
DEFAULT_ENG_QUERIES = PROJECT_ROOT / ".taskforge" / "eval-cache" / "neuclir-tech" / "data" / "eng.tsv"
DEFAULT_ZHO_QUERIES = PROJECT_ROOT / ".taskforge" / "eval-cache" / "neuclir-tech" / "data" / "zho.tsv"
DEFAULT_QRELS = PROJECT_ROOT / ".taskforge" / "eval-cache" / "neuclir-tech" / "data" / "qrels.gains.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / ".taskforge" / "datasets" / "bilingual-paper-corpus-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl_sha256(path: Path) -> tuple[str, int]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        count = sum(1 for _ in stream)
    return _sha256(path), count


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_queries(path: Path) -> dict[int, str]:
    values: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        topic, query = line.split("\t", 1)
        values[int(topic)] = query.strip()
    return values


def _load_qrels(path: Path) -> dict[int, list[tuple[str, int]]]:
    values: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        topic, _unused, document_id, relevance = line.split()
        values[int(topic)].append((document_id, int(relevance)))
    return dict(values)


def _qasper_text(paper: Mapping[str, Any]) -> str:
    parts = [
        f"TITLE: {str(paper.get('title') or '').strip()}",
        f"ABSTRACT: {str(paper.get('abstract') or '').strip()}",
    ]
    for section in paper.get("full_text", []):
        heading = str(section.get("section_name") or "").strip()
        paragraphs = [
            str(value).strip()
            for value in section.get("paragraphs", [])
            if str(value).strip()
        ]
        if heading:
            parts.append(f"SECTION: {heading}")
        parts.extend(paragraphs)
    return "\n\n".join(value for value in parts if value)


def _qasper_documents(
    paths: Iterable[tuple[str, Path]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    documents: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    seen: set[str] = set()
    for split, path in paths:
        data = _load_json(path)
        for paper_id, paper in data.items():
            document_id = f"qasper:{paper_id}"
            if document_id in seen:
                raise ValueError(f"duplicate QASPER paper: {paper_id}")
            seen.add(document_id)
            documents.append(
                {
                    "document_id": document_id,
                    "paper_id": str(paper_id),
                    "language": "en",
                    "title": str(paper.get("title") or "").strip(),
                    "abstract": str(paper.get("abstract") or "").strip(),
                    "text": _qasper_text(paper),
                    "document_type": "full_text",
                    "source_dataset": "QASPER",
                    "source_split": split,
                    "license": "CC BY 4.0",
                }
            )
            for question in paper.get("qas", []):
                query_id = f"qasper:{split}:{paper_id}:{question['question_id']}"
                evidence: list[str] = []
                answers: list[dict[str, Any]] = []
                for annotation in question.get("answers", []):
                    answer = dict(annotation.get("answer") or {})
                    evidence.extend(
                        str(value).strip()
                        for value in answer.get("evidence", [])
                        if str(value).strip()
                    )
                    answers.append(
                        {
                            "annotation_id": annotation.get("annotation_id"),
                            "unanswerable": bool(answer.get("unanswerable", False)),
                            "free_form_answer": str(answer.get("free_form_answer") or ""),
                            "extractive_spans": list(answer.get("extractive_spans") or []),
                        }
                    )
                evidence = list(dict.fromkeys(evidence))
                query = str(question.get("question") or "").strip()
                if not query:
                    continue
                queries.append(
                    {
                        "query_id": query_id,
                        "query": query,
                        "language": "en",
                        "paper_id": str(paper_id),
                        "source_dataset": "QASPER",
                        "source_split": split,
                        "document_id": document_id,
                        "relevant_document_ids": [document_id],
                        "evidence_texts": evidence,
                        "answers": answers,
                        "answerable": any(not item["unanswerable"] for item in answers),
                    }
                )
                qrels.append(
                    {
                        "query_id": query_id,
                        "document_id": document_id,
                        "relevance": 1,
                        "source_dataset": "QASPER",
                    }
                )
    return documents, queries, qrels


def _csl_document(row: Mapping[str, Any]) -> dict[str, Any] | None:
    document_id = str(row.get("doc_id") or "").strip()
    title = str(row.get("title") or "").strip()
    abstract = str(row.get("abstract") or "").strip()
    if not document_id or not title and not abstract:
        return None
    keywords = [str(value).strip() for value in row.get("keywords", []) if str(value).strip()]
    text_parts = [f"题目：{title}", f"摘要：{abstract}"]
    if keywords:
        text_parts.append(f"关键词：{'、'.join(keywords)}")
    return {
        "document_id": document_id,
        "paper_id": document_id,
        "language": "zh",
        "title": title,
        "abstract": abstract,
        "text": "\n".join(text_parts),
        "keywords": keywords,
        "category": str(row.get("category") or "").strip(),
        "category_en": str(row.get("category_eng") or "").strip(),
        "discipline": str(row.get("discipline") or "").strip(),
        "discipline_en": str(row.get("discipline_eng") or "").strip(),
        "document_type": "abstract",
        "source_dataset": "NeuCLIR-CSL",
        "license": "Apache 2.0",
    }


def _select_csl_documents(
    path: Path,
    *,
    limit: int,
    required_ids: set[str],
    seed: int,
) -> tuple[list[dict[str, Any]], int, int]:
    """Select all judged CSL docs plus a deterministic hash sample."""

    if limit < 0:
        raise ValueError("chinese_limit must be non-negative")
    selected_required: dict[str, dict[str, Any]] = {}
    # A max-heap represented by negative hash values keeps memory bounded for
    # the optional large sample while making selection independent of stream
    # order.  ``random`` is used only to derive a documented seed namespace.
    sample_size = 0 if limit == 0 else max(0, limit - len(required_ids))
    sample_heap: list[tuple[int, str, dict[str, Any]]] = []
    seed_bytes = str(random.Random(seed).getrandbits(128)).encode("ascii")
    total = 0
    valid = 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            total += 1
            row = _csl_document(json.loads(line))
            if row is None:
                continue
            valid += 1
            document_id = str(row["document_id"])
            if document_id in required_ids:
                selected_required[document_id] = row
                continue
            if limit == 0:
                selected_required[document_id] = row
                continue
            if sample_size <= 0:
                continue
            digest = hashlib.sha256(seed_bytes + document_id.encode("utf-8")).digest()
            score = int.from_bytes(digest[:16], "big")
            entry = (-score, document_id, row)
            if len(sample_heap) < sample_size:
                heapq.heappush(sample_heap, entry)
            elif entry > sample_heap[0]:
                heapq.heapreplace(sample_heap, entry)
    selected = {**selected_required, **{document_id: row for _score, document_id, row in sample_heap}}
    if limit and len(selected) < limit and len(selected_required) < limit:
        raise ValueError(
            f"CSL contains only {len(selected)} selectable documents, below requested {limit}"
        )
    return [selected[key] for key in sorted(selected)], total, valid


def _neuclir_queries(
    *,
    english_path: Path,
    chinese_path: Path,
    qrels_path: Path,
    selected_chinese_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    english = _load_queries(english_path)
    chinese = _load_queries(chinese_path)
    qrels = _load_qrels(qrels_path)
    queries: list[dict[str, Any]] = []
    output_qrels: list[dict[str, Any]] = []
    for topic_id in sorted(set(english) & set(chinese) & set(qrels)):
        rows = qrels[topic_id]
        positive = [doc_id for doc_id, relevance in rows if relevance > 0 and doc_id in selected_chinese_ids]
        if not positive:
            continue
        query_id = f"neuclir:{topic_id}"
        queries.append(
            {
                "query_id": query_id,
                "query": chinese[topic_id],
                "english_query": english[topic_id],
                "language": "zh",
                "source_dataset": "NeuCLIR-Tech/CSL",
                "topic_id": topic_id,
                "relevant_document_ids": positive,
            }
        )
        for document_id, relevance in rows:
            if document_id in selected_chinese_ids:
                output_qrels.append(
                    {
                        "query_id": query_id,
                        "document_id": document_id,
                        "relevance": relevance,
                        "source_dataset": "NeuCLIR-Tech/CSL",
                    }
                )
    return queries, output_qrels


def _write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[str, int]:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    return _jsonl_sha256(path)


def build(
    *,
    qasper_train: Path,
    qasper_validation: Path,
    csl: Path,
    english_queries: Path,
    chinese_queries: Path,
    qrels: Path,
    output_dir: Path,
    chinese_limit: int,
    seed: int,
) -> dict[str, Any]:
    required_ids = {
        document_id
        for rows in _load_qrels(qrels).values()
        for document_id, relevance in rows
        if relevance > 0
    }
    chinese_docs, csl_total, csl_valid = _select_csl_documents(
        csl,
        limit=chinese_limit,
        required_ids=required_ids,
        seed=seed,
    )
    selected_chinese_ids = {str(row["document_id"]) for row in chinese_docs}
    english_docs, english_queries_rows, english_qrels_rows = _qasper_documents(
        (("train", qasper_train), ("validation", qasper_validation))
    )
    chinese_queries_rows, chinese_qrels_rows = _neuclir_queries(
        english_path=english_queries,
        chinese_path=chinese_queries,
        qrels_path=qrels,
        selected_chinese_ids=selected_chinese_ids,
    )
    documents = [*english_docs, *chinese_docs]
    queries = [*english_queries_rows, *chinese_queries_rows]
    qrels_rows = [*english_qrels_rows, *chinese_qrels_rows]

    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
    corpus_path = output_dir / "corpus.jsonl.gz"
    queries_path = output_dir / "queries.jsonl.gz"
    qrels_path_out = output_dir / "qrels.jsonl.gz"
    corpus_hash, corpus_count = _write_jsonl_gz(corpus_path, documents)
    queries_hash, queries_count = _write_jsonl_gz(queries_path, queries)
    qrels_hash, qrels_count = _write_jsonl_gz(qrels_path_out, qrels_rows)

    language_counts = Counter(str(row["language"]) for row in documents)
    document_type_counts = Counter(str(row["document_type"]) for row in documents)
    query_language_counts = Counter(str(row["language"]) for row in queries)
    manifest = {
        "schema_version": "1.0",
        "dataset_id": "bilingual-paper-corpus-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "sampling": {
            "seed": seed,
            "chinese_limit": chinese_limit,
            "chinese_selection": (
                "all positive NeuCLIR qrels plus deterministic hash sample"
                if chinese_limit
                else "all valid CSL records"
            ),
        },
        "sources": [
            {
                "dataset": "QASPER",
                "language": "en",
                "documents": len(english_docs),
                "document_type": "full_text",
                "license": "CC BY 4.0",
                "source_files": [
                    {"path": str(qasper_train), "sha256": _sha256(qasper_train)},
                    {"path": str(qasper_validation), "sha256": _sha256(qasper_validation)},
                ],
            },
            {
                "dataset": "NeuCLIR-CSL",
                "language": "zh",
                "documents": len(chinese_docs),
                "available_records": csl_valid,
                "scanned_records": csl_total,
                "document_type": "abstract",
                "license": "Apache 2.0",
                "source_file": {"path": str(csl), "sha256": _sha256(csl)},
            },
            {
                "dataset": "NeuCLIR-Tech",
                "language": "zh/en",
                "queries": len(chinese_queries_rows),
                "source_files": [
                    {"path": str(english_queries), "sha256": _sha256(english_queries)},
                    {"path": str(chinese_queries), "sha256": _sha256(chinese_queries)},
                    {"path": str(qrels), "sha256": _sha256(qrels)},
                ],
            },
        ],
        "counts": {
            "documents": corpus_count,
            "documents_by_language": dict(language_counts),
            "documents_by_type": dict(document_type_counts),
            "queries": queries_count,
            "queries_by_language": dict(query_language_counts),
            "qrels": qrels_count,
        },
        "files": {
            "corpus": {"path": corpus_path.name, "records": corpus_count, "sha256": corpus_hash},
            "queries": {"path": queries_path.name, "records": queries_count, "sha256": queries_hash},
            "qrels": {"path": qrels_path_out.name, "records": qrels_count, "sha256": qrels_hash},
        },
        "limitations": [
            "English QASPER records contain full paper text; Chinese CSL records contain title/abstract metadata rather than full PDFs.",
            "NeuCLIR relevance is document-level graded judgment; QASPER relevance additionally includes evidence text and answer annotations.",
            "The generated JSONL files are a local derived dataset; retain source attribution and obey each source's terms before redistribution.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# Bilingual Paper Corpus v1\n\n"
        "This local corpus combines English QASPER full text with Chinese "
        "NeuCLIR-CSL paper abstracts. See `manifest.json` for checksums, "
        "sampling, provenance, and limitations.\n\n"
        "`queries.jsonl.gz` contains English QASPER evidence questions and "
        "Chinese NeuCLIR topics; `qrels.jsonl.gz` contains normalized document "
        "relevance rows.\n",
        encoding="utf-8",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qasper-train", type=Path, default=DEFAULT_QASPER_TRAIN)
    parser.add_argument("--qasper-validation", type=Path, default=DEFAULT_QASPER_VALIDATION)
    parser.add_argument("--csl", type=Path, default=DEFAULT_CSL)
    parser.add_argument("--english-queries", type=Path, default=DEFAULT_ENG_QUERIES)
    parser.add_argument("--chinese-queries", type=Path, default=DEFAULT_ZHO_QUERIES)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--chinese-limit",
        type=int,
        default=100_000,
        help="Chinese CSL records to include; 0 includes all valid records.",
    )
    parser.add_argument("--seed", type=int, default=20260827)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    required = [
        args.qasper_train,
        args.qasper_validation,
        args.csl,
        args.english_queries,
        args.chinese_queries,
        args.qrels,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing input files: {missing}")
    try:
        manifest = build(
            qasper_train=args.qasper_train,
            qasper_validation=args.qasper_validation,
            csl=args.csl,
            english_queries=args.english_queries,
            chinese_queries=args.chinese_queries,
            qrels=args.qrels,
            output_dir=args.output_dir,
            chinese_limit=args.chinese_limit,
            seed=args.seed,
        )
    except (FileExistsError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

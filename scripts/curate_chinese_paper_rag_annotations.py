"""Deterministically curate the silver Chinese paper RAG annotations.

This command does not call an LLM.  It preserves the generated questions and
answers, but re-checks every evidence reference against the source chunk,
repairs an unambiguous chunk index, derives an exact local quote when the
generated quote is not recoverable, and writes a review queue for semantic
issues that still require a human (or a later model pass).

The original ``chinese-paper-rag-30-v1`` directory is never modified.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_chinese_paper_rag_annotations as builder  # noqa: E402

DEFAULT_SOURCE = PROJECT_ROOT / ".taskforge" / "datasets" / "chinese-ai-oa-jos-v2" / "corpus.jsonl.gz"
DEFAULT_INPUT = PROJECT_ROOT / "eval" / "queries" / "chinese-paper-rag-30-v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "eval" / "queries" / "chinese-paper-rag-30-v2-precision"
GENERIC_CONTRIBUTION_RE = re.compile(r"^(?:本文|该文|论文)的?(?:主要)?贡献(?:有哪些|是什么)[？?。.!！]*$")
WHITESPACE_RE = re.compile(r"\s+")

# These are editorial specificity repairs, not new answers.  They retain the
# generated answer/evidence while replacing a repeated template with terms
# already present in that answer.  Keeping this small map explicit makes the
# change auditable and avoids leaking a paper title into every query.
QUESTION_REWRITES: dict[tuple[str, str], str] = {
    ("jos:5090", "contribution"): "围绕ACIO约束和HKD-tree，论文的具体贡献是什么？",
    ("jos:7078", "contribution"): "细粒度视频实体链接模型的核心贡献包括哪些方面？",
    ("jos:7330", "contribution"): "LLM赋能的Datalog代码翻译技术及增量分析框架有哪些贡献？",
    ("jos:7365", "contribution"): "MACR如何支持对话属性情感理解，其主要贡献是什么？",
    ("jos:7419", "contribution"): "多视角推荐方法在物品关系图和对比学习方面有哪些贡献？",
    ("jos:7492", "contribution"): "BPNL损失函数与TSR方法如何改进ECG多标签学习？",
    ("jos:7588", "contribution"): "符号执行如何增强LLM生成代码缺陷检测和供应链安全？",
}


def _normalise(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).replace("\xa0", " ")
    return WHITESPACE_RE.sub(" ", text).strip()


def _compact(value: object) -> str:
    return builder._compact_for_match(value)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    builder._write_jsonl(path, rows)


def _write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    builder._write_jsonl_gz(path, rows)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _find_quote(quote: str, chunks: list[str]) -> tuple[int | None, str]:
    """Return the unique source chunk containing quote, if one exists."""

    if not quote:
        return None, "missing"
    exact = [
        index
        for index, chunk in enumerate(chunks)
        if quote in _normalise(chunk)
    ]
    if len(exact) == 1:
        return exact[0], "model_exact"
    compact = _compact(quote)
    compact_matches = [
        index
        for index, chunk in enumerate(chunks)
        if compact and compact in _compact(chunk)
    ]
    if len(compact_matches) == 1:
        return compact_matches[0], "model_compact"
    return None, "ambiguous" if exact or compact_matches else "missing"


def _clip_quote(quote: str, source_chunk: str) -> str:
    """Keep quotes within the annotation contract while retaining provenance."""

    quote = _normalise(quote)
    if len(quote) <= 180:
        return quote
    # Prefer a complete sentence from the source.  If the sentence itself is
    # long, a prefix is still an exact contiguous source span.
    sentence = builder._derive_quote(quote, source_chunk)
    if sentence and len(sentence) <= 180:
        return sentence
    return _normalise(source_chunk)[:180]


def _refine_item(item: Mapping[str, Any], chunks: list[str]) -> tuple[dict[str, Any], list[str]]:
    flags: list[str] = []
    clean_evidence: list[dict[str, Any]] = []
    evidence = item.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("evidence is required")
    for raw_ref in evidence[:3]:
        if not isinstance(raw_ref, Mapping):
            raise ValueError("evidence entries must be objects")
        try:
            requested_index = int(raw_ref.get("chunk_index"))
        except (TypeError, ValueError) as exc:
            raise ValueError("chunk_index must be an integer") from exc
        if not 0 <= requested_index < len(chunks):
            raise ValueError(f"chunk_index out of range: {requested_index}")
        requested_quote = _normalise(raw_ref.get("quote"))
        matched_index, origin = _find_quote(requested_quote, chunks)
        index = matched_index if matched_index is not None else requested_index
        if matched_index is not None and index != requested_index:
            flags.append("evidence_chunk_index_corrected")
            origin = "chunk_index_corrected"
        source_chunk = _normalise(chunks[index])
        quote = requested_quote
        if matched_index is None:
            # The quote cannot be proven from the generated payload.  Derive
            # a contiguous quote from the answer's target chunk and flag it
            # for review instead of silently presenting it as model evidence.
            quote = builder._derive_quote(_normalise(item.get("answer")), source_chunk)
            flags.append("evidence_quote_derived")
            origin = "derived_local"
        quote = _clip_quote(quote, source_chunk)
        quote_verified = bool(
            quote
            and (
                quote in source_chunk
                or _compact(quote) in _compact(source_chunk)
            )
        )
        if not quote_verified:
            flags.append("evidence_quote_unverified")
        clean_evidence.append(
            {
                "chunk_index": index,
                "quote": quote,
                "quote_verified": quote_verified,
                "quote_origin": origin,
            }
        )
    refined = {
        "question_type": _normalise(item.get("question_type")).casefold(),
        "question": _normalise(item.get("question")),
        "answer": _normalise(item.get("answer")),
        "evidence": clean_evidence,
        "difficulty": _normalise(item.get("difficulty") or "medium").casefold(),
    }
    if not refined["question"] or not refined["answer"]:
        raise ValueError("question and answer are required")
    return refined, sorted(set(flags))


def _make_query_and_qrels(
    paper: Mapping[str, Any],
    item: Mapping[str, Any],
    question_index: int,
    item_flags: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query, qrels = builder._annotation_row(
        paper,
        item=item,
        question_index=question_index,
        chunks=[],
    )
    query["annotation_status"] = "silver_curated"
    query["annotation_method"] = "deterministic_source_binding_and_review_queue"
    query["curation_flags"] = item_flags
    query["evidence_quote_origin"] = [
        str(ref.get("quote_origin", "unknown")) for ref in item["evidence"]
    ]
    query["evidence_quote_verified"] = [
        bool(ref.get("quote_verified", False)) for ref in item["evidence"]
    ]
    if item.get("question_origin"):
        query["question_origin"] = item["question_origin"]
        query["original_question"] = item.get("original_question", "")
    return query, qrels


def curate(*, source: Path, input_dir: Path, output: Path) -> dict[str, Any]:
    source_rows = {
        str(row.get("paper_id") or row.get("document_id")): row
        for row in builder._read_jsonl_gz(source)
    }
    annotation_rows = _load_jsonl(input_dir / "annotations.jsonl")
    if not annotation_rows:
        raise RuntimeError(f"no annotations found in {input_dir}")

    output.mkdir(parents=True, exist_ok=True)
    refined_annotations: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    chunk_qrels: list[dict[str, Any]] = []
    paper_qrels: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    model_quote_verified = 0
    curated_quote_verified = 0
    evidence_quotes = 0
    derived_quotes = 0
    question_rewrites: list[dict[str, Any]] = []

    for annotation in sorted(annotation_rows, key=lambda row: str(row.get("paper_id", ""))):
        paper_id = str(annotation.get("paper_id") or annotation.get("document_id"))
        if paper_id not in source_rows:
            raise RuntimeError(f"annotation paper is missing from source corpus: {paper_id}")
        paper = source_rows[paper_id]
        chunks = builder._chunk_text(str(paper.get("text", "")))
        selected_ids.add(paper_id)
        document_id = str(paper.get("document_id") or paper_id)
        for index, chunk in enumerate(chunks):
            chunk_rows.append(
                {
                    "chunk_id": f"{document_id}::chunk-{index:04d}",
                    "document_id": document_id,
                    "paper_id": paper_id,
                    "title": str(paper.get("title", "")),
                    "language": "zh",
                    "text": chunk,
                    "chunk_index": index,
                    "source_dataset": "chinese-ai-oa-jos-v2",
                }
            )

        refined_items: list[dict[str, Any]] = []
        for question_index, raw_item in enumerate(annotation.get("items", []), start=1):
            refined_item, flags = _refine_item(raw_item, chunks)
            rewrite_key = (paper_id, refined_item["question_type"])
            replacement = QUESTION_REWRITES.get(rewrite_key)
            if replacement and GENERIC_CONTRIBUTION_RE.match(refined_item["question"]):
                original_question = refined_item["question"]
                refined_item["question"] = replacement
                refined_item["question_origin"] = "editorial_specificity_rewrite"
                refined_item["original_question"] = original_question
                question_rewrites.append(
                    {
                        "query_id": f"zhpaper-rag30:{paper_id}:q{question_index}",
                        "paper_id": paper_id,
                        "question_type": refined_item["question_type"],
                        "original_question": original_question,
                        "question": replacement,
                        "reason": "replace repeated generic contribution template with answer-grounded method terms",
                    }
                )
            refined_item["curation_flags"] = flags
            refined_items.append(refined_item)
            for ref in refined_item["evidence"]:
                evidence_quotes += 1
                if bool(raw_item.get("evidence", [{}])[0].get("quote_verified", False)):
                    # This is only a diagnostic count for the original model
                    # validation; the curated value is calculated below.
                    pass
                if ref.get("quote_origin") == "derived_local":
                    derived_quotes += 1
                if ref.get("quote_verified"):
                    curated_quote_verified += 1
            for raw_ref in raw_item.get("evidence", []):
                if isinstance(raw_ref, Mapping) and raw_ref.get("quote_verified"):
                    model_quote_verified += 1
            query, qrels = _make_query_and_qrels(paper, refined_item, question_index, flags)
            query_rows.append(query)
            chunk_qrels.extend(qrels)
            paper_qrels.append(
                {
                    "query_id": query["query_id"],
                    "document_id": document_id,
                    "relevance": 1,
                    "paper_id": paper_id,
                    "source_dataset": "chinese-ai-oa-jos-v2",
                }
            )
            if flags:
                review_queue.append(
                    {
                        "query_id": query["query_id"],
                        "paper_id": paper_id,
                        "title": str(paper.get("title", "")),
                        "question_type": refined_item["question_type"],
                        "question": refined_item["question"],
                        "answer": refined_item["answer"],
                        "flags": flags,
                    }
                )
        refined_annotations.append(
            {
                "paper_id": paper_id,
                "document_id": document_id,
                "title": str(paper.get("title", "")),
                "source_dataset": "chinese-ai-oa-jos-v2",
                "annotation_status": "silver_curated",
                "annotation_method": "deterministic_source_binding_and_review_queue",
                "chunk_count": len(chunks),
                "items": refined_items,
            }
        )

    # Detect duplicate query text after source binding.  Do not rewrite it
    # automatically: adding a title to a query would change its difficulty.
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in query_rows:
        by_question[_normalise(query["query"]).casefold()].append(query)
    duplicate_groups = [group for group in by_question.values() if len(group) > 1]
    duplicate_instances = 0
    for group in duplicate_groups:
        duplicate_instances += len(group) - 1
        query_ids = [row["query_id"] for row in group]
        for query in group:
            flag = "duplicate_question_text"
            query.setdefault("curation_flags", []).append(flag)
            query["curation_flags"] = sorted(set(query["curation_flags"]))
            review_queue.append(
                {
                    "query_id": query["query_id"],
                    "paper_id": query["paper_id"],
                    "title": "",
                    "question_type": query["question_type"],
                    "question": query["query"],
                    "answer": query["answer"],
                    "flags": [flag],
                    "duplicate_group_query_ids": query_ids,
                }
            )
    for query in query_rows:
        if query["question_type"] == "contribution" and GENERIC_CONTRIBUTION_RE.match(query["query"]):
            query.setdefault("curation_flags", []).append("generic_contribution_question")
            query["curation_flags"] = sorted(set(query["curation_flags"]))
            review_queue.append(
                {
                    "query_id": query["query_id"],
                    "paper_id": query["paper_id"],
                    "title": "",
                    "question_type": query["question_type"],
                    "question": query["query"],
                    "answer": query["answer"],
                    "flags": ["generic_contribution_question"],
                }
            )

    # De-duplicate review rows while retaining all flags and duplicate-group IDs.
    review_by_id: dict[str, dict[str, Any]] = {}
    for row in review_queue:
        current = review_by_id.setdefault(row["query_id"], dict(row))
        current["flags"] = sorted(set(current.get("flags", []) + row.get("flags", [])))
        if row.get("duplicate_group_query_ids"):
            current["duplicate_group_query_ids"] = row["duplicate_group_query_ids"]
    review_queue = sorted(review_by_id.values(), key=lambda row: row["query_id"])

    query_rows.sort(key=lambda row: row["query_id"])
    chunk_qrels.sort(key=lambda row: (row["query_id"], row["document_id"]))
    paper_qrels.sort(key=lambda row: row["query_id"])
    chunk_rows.sort(key=lambda row: row["chunk_id"])
    refined_annotations.sort(key=lambda row: row["paper_id"])
    _write_jsonl(output / "annotations.jsonl", refined_annotations)
    _write_jsonl(output / "queries.jsonl", query_rows)
    _write_jsonl(output / "qrels.jsonl", chunk_qrels)
    _write_jsonl(output / "paper_qrels.jsonl", paper_qrels)
    _write_jsonl(output / "review_queue.jsonl", review_queue)
    _write_jsonl(output / "question_rewrites.jsonl", question_rewrites)
    _write_jsonl_gz(output / "chunks.jsonl.gz", chunk_rows)

    manifest = {
        "schema_version": "zhpaper_rag_annotation.v1",
        "dataset_id": "chinese-paper-rag-30-v2-precision",
        "source_dataset": "chinese-ai-oa-jos-v2",
        "source_annotations": str(input_dir / "annotations.jsonl"),
        "source_path": str(source),
        "annotated_papers": len(selected_ids),
        "queries": len(query_rows),
        "chunk_qrels": len(chunk_qrels),
        "paper_qrels": len(paper_qrels),
        "chunks": len(chunk_rows),
        "evidence_quotes": evidence_quotes,
        "model_evidence_quotes_verified": model_quote_verified,
        "curated_evidence_quotes_verified": curated_quote_verified,
        "derived_evidence_quotes": derived_quotes,
        "question_rewrites": len(question_rewrites),
        "duplicate_question_groups": len(duplicate_groups),
        "duplicate_question_instances": duplicate_instances,
        "review_queue_queries": len(review_queue),
        "annotation_status": "silver_curated",
        "annotation_method": "deterministic source binding, quote provenance repair, and answer-grounded question specificity repair; no external API calls",
        "human_reviewed": False,
        "external_api_used": False,
        "limitations": [
            "Question and answer semantics remain inherited from the automatic silver annotations.",
            "Seven repeated generic contribution questions were editorially rewritten using method terms already present in their answers; originals are retained in question_rewrites.jsonl.",
            "Derived quotes are exact spans from the target chunk but require a human to confirm that they fully support the answer.",
            "Page coordinates are not available in this corpus export.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = curate(source=args.source, input_dir=args.input, output=args.output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

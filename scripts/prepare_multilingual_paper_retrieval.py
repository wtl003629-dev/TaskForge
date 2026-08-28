"""Prepare frozen Chinese, cross-lingual, and bilingual paper RAG fixtures."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TECH_ROOT = PROJECT_ROOT / ".taskforge" / "eval-cache" / "neuclir-tech"
DEFAULT_CSL = (
    PROJECT_ROOT
    / ".taskforge"
    / "eval-cache"
    / "neuclir-csl"
    / "data"
    / "csl.jsonl.gz"
)
DEFAULT_QASA = (
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


def _load_queries(path: Path) -> dict[int, str]:
    values: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        identifier, query = line.split("\t", 1)
        values[int(identifier)] = query.strip()
    return values


def _load_qrels(path: Path) -> dict[int, list[tuple[str, int]]]:
    values: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        topic, _unused, document_id, relevance = line.split()
        values[int(topic)].append((document_id, int(relevance)))
    return values


def _selected_document_ids(
    topic_ids: Iterable[int],
    qrels: Mapping[int, list[tuple[str, int]]],
    *,
    positives_per_topic: int,
    negatives_per_topic: int,
) -> tuple[list[str], dict[int, list[str]]]:
    selected: list[str] = []
    relevant: dict[int, list[str]] = {}
    for topic_id in topic_ids:
        rows = qrels[topic_id]
        positives = [docid for docid, score in rows if score > 0]
        negatives = [docid for docid, score in rows if score <= 0]
        if len(positives) < positives_per_topic:
            raise ValueError(f"topic {topic_id} lacks enough positive judgments")
        if len(negatives) < negatives_per_topic:
            raise ValueError(f"topic {topic_id} lacks enough negative judgments")
        relevant[topic_id] = positives[:positives_per_topic]
        selected.extend(positives[:positives_per_topic])
        selected.extend(negatives[:negatives_per_topic])
    return list(dict.fromkeys(selected)), relevant


def _load_csl_documents(path: Path, required_ids: set[str]) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            document_id = str(row.get("doc_id", "")).strip()
            if document_id in required_ids:
                documents[document_id] = row
                if len(documents) == len(required_ids):
                    break
    missing = sorted(required_ids - documents.keys())
    if missing:
        raise ValueError(f"CSL corpus is missing judged documents: {missing[:5]}")
    return documents


def _csl_text(row: Mapping[str, Any]) -> str:
    title = str(row.get("title", "")).strip()
    abstract = str(row.get("abstract", "")).strip()
    keywords = "、".join(str(item).strip() for item in row.get("keywords", []))
    return f"题目：{title}\n摘要：{abstract}\n关键词：{keywords}".strip()


def _answer(question: str, evidence: list[str], *, prefix: str) -> dict[str, Any]:
    question_id = hashlib.sha1(question.encode("utf-8")).hexdigest()
    return {
        "question": question,
        "question_id": question_id,
        "answers": [
            {
                "annotation_id": f"{prefix}-{index}",
                "answer": {
                    "unanswerable": False,
                    "free_form_answer": "relevant scientific paper",
                    "extractive_spans": [],
                    "yes_no": None,
                    "evidence": [text],
                },
            }
            for index, text in enumerate(evidence)
        ],
    }


def _qasa_rows(path: Path, *, limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_papers: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            contexts = row.get("ctxs", [])
            if not contexts or not row.get("gold_ctxs"):
                continue
            paper_id = str(contexts[0].get("id", "")).split("_all_", 1)[0]
            if not paper_id or paper_id in seen_papers:
                continue
            seen_papers.add(paper_id)
            selected.append(row)
            if len(selected) == limit:
                break
    if len(selected) != limit:
        raise ValueError("QASA did not yield enough unique papers")
    return selected


def _fixture(
    *,
    scenario_id: str,
    title: str,
    sections: list[dict[str, Any]],
    qas: list[dict[str, Any]],
    source_hashes: dict[str, str],
    query_language: str,
    corpus_language: str,
    output_root: Path,
) -> dict[str, Any]:
    paper_id = f"multilingual-{scenario_id}"
    dataset = {
        paper_id: {
            "title": title,
            "abstract": "",
            "full_text": sections,
            "figures_and_tables": [],
            "qas": qas,
        }
    }
    case_ids = [f"qasper:{paper_id}:{item['question_id']}" for item in qas]
    dataset_path = output_root / f"{scenario_id}.json"
    split_path = PROJECT_ROOT / "eval" / "splits" / f"{scenario_id}.json"
    split = {
        "schema_version": "1.0",
        "split_id": f"{scenario_id}-v1",
        "dataset": "NeuCLIRTech/CSL and ScholarQABench QASA",
        "source_sha256": source_hashes,
        "synthetic_pdf_layout": "compact_scientific_paper_v1",
        "case_ids": case_ids,
        "scenario": {
            "query_language": query_language,
            "corpus_language": corpus_language,
            "multi_paper": True,
            "candidate_paper_sections": len(sections),
        },
        "metric_interpretation": (
            "NeuCLIR relevance documents are encoded as alternative legal Gold "
            "annotations, so Recall@K is any-relevant Success@K rather than the "
            "official graded NeuCLIR nDCG@20. QASA cases retain strict Gold "
            "paragraph coverage semantics."
        ),
        "report_metadata": {
            "evaluation_type": "multilingual_multi_paper_pdf_upload_retrieval",
            "benchmark_track": scenario_id,
            "dataset": f"Frozen {scenario_id} scientific-paper retrieval scenario",
            "license": (
                "NeuCLIRTech CC BY 4.0; CSL Apache 2.0; "
                "ScholarQABench aggregate ODC-BY"
            ),
            "synthetic_layout_limitation": (
                "Queries and relevance labels come from public benchmarks, while "
                "the multi-paper PDF layout is generated locally for controlled "
                "text retrieval and is not an original-PDF parser benchmark."
            ),
        },
    }
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    split_path.write_text(
        json.dumps(split, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "scenario_id": scenario_id,
        "dataset": str(dataset_path),
        "split": str(split_path),
        "cases": len(qas),
        "sections": len(sections),
    }


def prepare(
    *,
    tech_root: Path,
    csl_path: Path,
    qasa_path: Path,
    output_root: Path,
    topic_count: int,
) -> list[dict[str, Any]]:
    eng_path = tech_root / "data" / "eng.tsv"
    zho_path = tech_root / "data" / "zho.tsv"
    qrels_path = tech_root / "data" / "qrels.gains.txt"
    eng_queries = _load_queries(eng_path)
    zho_queries = _load_queries(zho_path)
    qrels = _load_qrels(qrels_path)
    common_topics = [
        topic_id
        for topic_id in sorted(set(eng_queries).intersection(zho_queries, qrels))
        if sum(score > 0 for _docid, score in qrels[topic_id]) >= 3
    ][:topic_count]
    selected_ids, relevant_ids = _selected_document_ids(
        common_topics,
        qrels,
        positives_per_topic=3,
        negatives_per_topic=7,
    )
    documents = _load_csl_documents(csl_path, set(selected_ids))
    text_by_id = {docid: _csl_text(documents[docid]) for docid in selected_ids}
    csl_sections = [
        {
            "section_name": (
                f"中文论文 {index + 1}: {documents[docid]['title']} [{docid}]"
            ),
            "paragraphs": [text_by_id[docid]],
        }
        for index, docid in enumerate(selected_ids)
    ]
    source_hashes = {
        "neuclir_eng": _sha256(eng_path),
        "neuclir_zho": _sha256(zho_path),
        "neuclir_qrels": _sha256(qrels_path),
        "csl_corpus": _sha256(csl_path),
        "qasa": _sha256(qasa_path),
    }

    def csl_qas(queries: Mapping[int, str], prefix: str) -> list[dict[str, Any]]:
        return [
            _answer(
                queries[topic_id],
                [text_by_id[docid] for docid in relevant_ids[topic_id]],
                prefix=f"{prefix}-{topic_id}",
            )
            for topic_id in common_topics
        ]

    scenarios = [
        _fixture(
            scenario_id="neuclir-csl-zh-zh-10-v1",
            title="中文科研论文多文档检索基准",
            sections=csl_sections,
            qas=csl_qas(zho_queries, "zh"),
            source_hashes=source_hashes,
            query_language="Chinese",
            corpus_language="Chinese",
            output_root=output_root,
        ),
        _fixture(
            scenario_id="neuclir-csl-en-zh-10-v1",
            title="English queries over Chinese scientific papers",
            sections=csl_sections,
            qas=csl_qas(eng_queries, "en"),
            source_hashes=source_hashes,
            query_language="English",
            corpus_language="Chinese",
            output_root=output_root,
        ),
        _fixture(
            scenario_id="neuclir-csl-mixed-query-zh-10-v1",
            title="中英混合查询检索中文科研论文",
            sections=csl_sections,
            qas=csl_qas(
                {
                    topic_id: (
                        f"{zho_queries[topic_id]} English topic: "
                        f"{eng_queries[topic_id].split(' I am looking', 1)[0]}"
                    )
                    for topic_id in common_topics
                },
                "mixed-query",
            ),
            source_hashes=source_hashes,
            query_language="Chinese-English code-switch",
            corpus_language="Chinese",
            output_root=output_root,
        ),
    ]

    mixed_topics = common_topics[:5]
    mixed_selected_ids, mixed_relevant = _selected_document_ids(
        mixed_topics,
        qrels,
        positives_per_topic=3,
        negatives_per_topic=7,
    )
    mixed_sections = [
        section
        for section, docid in zip(csl_sections, selected_ids, strict=True)
        if docid in set(mixed_selected_ids)
    ]
    mixed_qas: list[dict[str, Any]] = []
    for topic_id in mixed_topics:
        evidence = [text_by_id[docid] for docid in mixed_relevant[topic_id]]
        mixed_qas.append(_answer(zho_queries[topic_id], evidence, prefix=f"mix-zh-{topic_id}"))
        mixed_qas.append(_answer(eng_queries[topic_id], evidence, prefix=f"mix-en-{topic_id}"))

    for row in _qasa_rows(qasa_path, limit=5):
        contexts = row["ctxs"]
        paper_title = str(contexts[0].get("title", "English scientific paper"))
        paragraphs = [str(item["text"]).strip() for item in contexts]
        mixed_sections.append(
            {
                "section_name": f"English paper: {paper_title}",
                "paragraphs": paragraphs,
            }
        )
        gold = [paragraphs[int(index)] for index in row["gold_ctxs"]]
        question = str(row["input"]).strip()
        question_id = hashlib.sha1(question.encode("utf-8")).hexdigest()
        mixed_qas.append(
            {
                "question": question,
                "question_id": question_id,
                "answers": [
                    {
                        "annotation_id": "qasa-gold",
                        "answer": {
                            "unanswerable": False,
                            "free_form_answer": str(row.get("answer", "")),
                            "extractive_spans": [],
                            "yes_no": None,
                            "evidence": gold,
                        },
                    }
                ],
            }
        )
    scenarios.append(
        _fixture(
            scenario_id="bilingual-multipaper-mixed-corpus-15-v1",
            title="中英文混合科研论文多文档检索基准",
            sections=mixed_sections,
            qas=mixed_qas,
            source_hashes=source_hashes,
            query_language="Chinese and English",
            corpus_language="Chinese and English",
            output_root=output_root,
        )
    )
    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tech-root", type=Path, default=DEFAULT_TECH_ROOT)
    parser.add_argument("--csl", type=Path, default=DEFAULT_CSL)
    parser.add_argument("--qasa", type=Path, default=DEFAULT_QASA)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / ".taskforge" / "eval-cache" / "multilingual-paper",
    )
    parser.add_argument("--topic-count", type=int, default=10)
    args = parser.parse_args()
    results = prepare(
        tech_root=args.tech_root,
        csl_path=args.csl,
        qasa_path=args.qasa,
        output_root=args.output_root,
        topic_count=args.topic_count,
    )
    print(json.dumps(results, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

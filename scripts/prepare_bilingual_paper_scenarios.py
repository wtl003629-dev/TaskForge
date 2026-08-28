"""Build paper-only Chinese/cross-language/mixed-corpus RAG fixtures.

The Chinese papers are real publisher PDFs already frozen by
``prepare_chinese_paper_fulltext.py``.  The English papers come from the
QASPER dev cohort whose original PDFs are checksum-pinned in the existing
manifest.  Query translations are authored and frozen here; evidence text is
never translated or injected, so retrieval is evaluated against the original
paper language.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZH_DATASET = PROJECT_ROOT / ".taskforge" / "eval-cache" / "chinese-paper-fulltext-15-v1.json"
ZH_MANIFEST = PROJECT_ROOT / ".taskforge" / "eval-cache" / "chinese-paper-fulltext-real-pdfs-v1.json"
QASPER_DATASET = PROJECT_ROOT / ".taskforge" / "eval-cache" / "qasper-dev-v0.3.json"
QASPER_SPLIT = PROJECT_ROOT / "eval" / "splits" / "qasper-dev-clean-holdout-100-v2.json"
QASPER_MANIFEST = PROJECT_ROOT / ".taskforge" / "eval-cache" / "qasper-clean-holdout-real-pdfs-v3.json"
OUTPUT_ROOT = PROJECT_ROOT / ".taskforge" / "eval-cache" / "paper-scenarios"
SPLIT_ROOT = PROJECT_ROOT / "eval" / "splits"


ZH_TO_EN: dict[str, str] = {
    "综述从哪两个方面对检索增强生成技术路线进行分类？": "Which two aspects does the survey use to classify retrieval-augmented generation routes?",
    "RAG技术融合了哪两个过程，从而能够动态查询外部知识库？": "Which two processes does RAG combine to dynamically query an external knowledge base?",
    "论文指出RAG更适合哪些类型的应用任务？": "What types of application tasks does the paper say RAG is better suited for?",
    "NALSpatial方法包含哪两个核心阶段？": "What are the two core stages of the NALSpatial method?",
    "在自然语言理解阶段，NALSpatial如何确定查询类型？": "How does NALSpatial determine the query type during natural-language understanding?",
    "论文报告的NALSpatial平均响应时间、可翻译性和翻译精度分别是多少？": "What average response time, translatability, and translation accuracy does NALSpatial report?",
    "论文将Issue中的代码补充信息整理成多少种类型？": "Into how many types does the paper organize supplementary information from Issues?",
    "该方法从Issue中检索信息后，如何用于补充性代码注释生成？": "After retrieving information from Issues, how is it used to generate supplementary code comments?",
    "实验中GPT-4o生成注释对人工补充性注释的覆盖率提升到了多少？": "To what coverage did GPT-4o comments of human supplementary comments improve in the experiment?",
    "SmartGen-AADL方法的核心流程包括哪三个阶段？": "What are the three core stages of the SmartGen-AADL method?",
    "SmartGen-AADL中的RAG模块主要解决什么问题？": "What problem does the RAG module in SmartGen-AADL mainly address?",
    "与仅依赖简单提示工程相比，该方法在组件错误率和FBERT上有什么变化？": "Compared with simple prompt engineering, how did the method change component error rate and FBERT?",
    "TWE-NMF方法针对Mashup服务短文本的哪些问题进行改进？": "Which short-text problems in Mashup services does TWE-NMF address?",
    "TWE-NMF如何自动确定主题数量？": "How does TWE-NMF automatically determine the number of topics?",
    "与现有服务聚类方法相比，TWE-NMF实验结果的总体趋势是什么？": "Compared with existing service-clustering methods, what is the overall trend of the TWE-NMF results?",
}

QASPER_TO_ZH: dict[str, str] = {
    "What type of system does the baseline classification use?": "基线分类使用了哪种系统？",
    "How much is classification performance improved in experiments for low data regime and class-imbalance problems?": "在低数据和类别不平衡问题上，实验中的分类性能提升了多少？",
    "How many different characters were in dataset?": "数据集中包含多少个不同的角色？",
    "How is the quality of singing voice measured?": "歌声质量是如何衡量的？",
    "How big are improvements of small-scale unbalanced datasets when sentence representation is enhanced with topic information?": "在小规模不平衡数据集上加入主题信息增强句子表示后，性能提升了多少？",
    "To what baseline models is proposed model compared?": "提出的方法与哪些基线模型进行了比较？",
}


def _qid(query: str) -> str:
    return hashlib.sha1(query.encode("utf-8")).hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rewrite_qas(paper: dict[str, Any], translation: dict[str, str], *, limit: int | None = None) -> dict[str, Any]:
    result = copy.deepcopy(paper)
    selected: list[dict[str, Any]] = []
    for question in result.get("qas", []):
        original = str(question.get("question", "")).strip()
        translated = translation.get(original)
        if not translated:
            continue
        item = copy.deepcopy(question)
        item["question"] = translated
        item["question_id"] = _qid(translated)
        selected.append(item)
        if limit is not None and len(selected) >= limit:
            break
    if not selected:
        raise ValueError(f"no translated questions matched paper {result.get('title')!r}")
    result["qas"] = selected
    return result


def _cases(fixture: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    case_ids: list[str] = []
    rows: list[dict[str, Any]] = []
    for paper_id, paper in fixture.items():
        for question in paper.get("qas", []):
            query = str(question["question"]).strip()
            question_id = str(question["question_id"]).strip()
            case_id = f"qasper:{paper_id}:{question_id}"
            case_ids.append(case_id)
            rows.append(
                {
                    "case_id": case_id,
                    "paper_id": paper_id,
                    "query": query,
                    "query_language": "zh" if any("\u3400" <= char <= "\u9fff" for char in query) else "en",
                }
            )
    return case_ids, rows


def _manifest_for(paper_ids: set[str]) -> dict[str, Any]:
    zh = _load(ZH_MANIFEST)
    qasper = _load(QASPER_MANIFEST)
    rows_by_id = {str(row["paper_id"]): row for row in zh["papers"]}
    rows_by_id.update({str(row["paper_id"]): row for row in qasper["papers"]})
    missing = sorted(paper_ids - rows_by_id.keys())
    if missing:
        raise ValueError(f"real PDF manifest lacks paper IDs: {missing[:5]}")
    normalized_rows: list[dict[str, Any]] = []
    zh_ids = {str(row["paper_id"]) for row in zh["papers"]}
    for paper_id in sorted(paper_ids):
        row = copy.deepcopy(rows_by_id[paper_id])
        source_root = ZH_MANIFEST.parent if paper_id in zh_ids else QASPER_MANIFEST.parent
        row["path"] = str((source_root / str(row["path"])).resolve())
        normalized_rows.append(row)
    return {
        "schema_version": "1.0",
        "dataset": "TaskForge Paper RAG",
        "cohort_id": "paper-scenarios-real-pdfs-v1",
        "source": "JOS official PDFs plus the existing checksum-pinned QASPER PDF cohort",
        "papers": normalized_rows,
    }


def _write_scenario(name: str, fixture: dict[str, Any], *, metadata: dict[str, Any]) -> dict[str, Any]:
    dataset_path = OUTPUT_ROOT / f"{name}.json"
    split_path = SPLIT_ROOT / f"{name}.json"
    manifest_path = OUTPUT_ROOT / f"{name}-real-pdfs.json"
    case_ids, rows = _cases(fixture)
    paper_ids = set(fixture)
    split = {
        "schema_version": "1.0",
        "split_id": name,
        "dataset": "TaskForge Paper RAG",
        "source": metadata["source"],
        "case_ids": case_ids,
        "selected_rows": rows,
        "pdf_manifest": str(manifest_path),
        "synthetic_pdf_layout": None,
        "report_metadata": metadata,
    }
    manifest = _manifest_for(paper_ids)
    for path in (dataset_path, split_path, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    split_path.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"scenario": name, "papers": len(paper_ids), "cases": len(case_ids), "dataset": str(dataset_path), "split": str(split_path), "manifest": str(manifest_path)}


def prepare() -> list[dict[str, Any]]:
    zh_source = _load(ZH_DATASET)
    zh_en = {paper_id: _rewrite_qas(paper, ZH_TO_EN) for paper_id, paper in zh_source.items()}
    outputs = [
        _write_scenario(
            "chinese-paper-fulltext-en-query-15-v1",
            zh_en,
            metadata={
                "source": "Five real JOS Chinese full-paper PDFs with frozen English translations of manually authored questions",
                "evaluation_type": "cross_language_query_to_chinese_fulltext",
                "benchmark_track": "scholarly_paper_fulltext_retrieval",
                "annotation_status": "manual_auditable_smoke_cohort",
                "metric_interpretation": "Recall/MRR/nDCG are evidence-paragraph retrieval over parser-native gold spans.",
            },
        )
    ]

    qasper_raw = _load(QASPER_DATASET)
    qasper_split = _load(QASPER_SPLIT)
    qasper_paper_ids: list[str] = []
    qasper_case_ids_by_paper: dict[str, list[str]] = {}
    for case_id in qasper_split["case_ids"]:
        parts = str(case_id).split(":")
        if len(parts) < 3:
            continue
        paper_id = parts[1]
        if paper_id not in qasper_case_ids_by_paper:
            qasper_paper_ids.append(paper_id)
            qasper_case_ids_by_paper[paper_id] = []
        qasper_case_ids_by_paper[paper_id].append(parts[2])
        if len(qasper_paper_ids) >= 5 and all(len(qasper_case_ids_by_paper[item]) >= 1 for item in qasper_paper_ids[:5]):
            break
    qasper_paper_ids = qasper_paper_ids[:5]
    mixed: dict[str, Any] = {}
    for paper_id, paper in zh_source.items():
        mixed[paper_id] = _rewrite_qas(paper, ZH_TO_EN, limit=2)
    for paper_id in qasper_paper_ids:
        paper = qasper_raw[paper_id]
        allowed = set(qasper_case_ids_by_paper[paper_id][:1])
        selected = [item for item in paper.get("qas", []) if str(item.get("question_id")) in allowed]
        if not selected:
            raise ValueError(f"no locked QASPER question found for {paper_id}")
        translated = copy.deepcopy(paper)
        translated["qas"] = []
        for item in selected:
            original = str(item["question"]).strip()
            if original not in QASPER_TO_ZH:
                raise ValueError(f"missing frozen Chinese translation for QASPER query: {original}")
            row = copy.deepcopy(item)
            row["question"] = QASPER_TO_ZH[original]
            row["question_id"] = _qid(row["question"])
            translated["qas"].append(row)
        mixed[paper_id] = translated
    outputs.append(
        _write_scenario(
            "bilingual-paper-mixed-15-v1",
            mixed,
            metadata={
                "source": "Five real Chinese JOS PDFs plus five checksum-pinned real QASPER PDFs",
                "evaluation_type": "bilingual_mixed_paper_corpus_cross_language_queries",
                "benchmark_track": "scholarly_paper_fulltext_retrieval",
                "annotation_status": "Chinese labels manually authored; English labels from QASPER with frozen Chinese query translations",
                "metric_interpretation": "Recall/MRR/nDCG are evidence-paragraph retrieval over original-language paper text; this is not a public bilingual benchmark.",
            },
        )
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(prepare(), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()

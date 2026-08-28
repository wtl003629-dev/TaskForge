"""Freeze a real Chinese full-paper PDF cohort for direct-upload RAG evaluation.

The fixture uses the same native PDF parser as the runtime to derive evidence
paragraphs.  This keeps the gold spans corpus-native while the evaluator still
uploads and parses the original PDFs through the production ingestion path.
The questions are a small, auditable, manually authored smoke cohort rather
than a claim of a public benchmark annotation.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.pdf_parsing.native_parser import NativePDFParser  # noqa: E402  # isort: skip


DEFAULT_PDF_ROOT = (
    PROJECT_ROOT / ".taskforge" / "eval-cache" / "chinese-papers-jos-v1"
)
DEFAULT_DATASET = (
    PROJECT_ROOT / ".taskforge" / "eval-cache" / "chinese-paper-fulltext-15-v1.json"
)
DEFAULT_SPLIT = PROJECT_ROOT / "eval" / "splits" / "chinese-paper-fulltext-15-v1.json"
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / ".taskforge"
    / "eval-cache"
    / "chinese-paper-fulltext-real-pdfs-v1.json"
)


@dataclass(frozen=True)
class PaperSpec:
    paper_id: str
    filename: str
    title: str
    source_url: str
    questions: tuple[tuple[str, str, tuple[str, ...], str], ...]


PAPERS: tuple[PaperSpec, ...] = (
    PaperSpec(
        "jos-7684",
        "7684.pdf",
        "面向大语言模型生成能力提升的检索增强生成研究进展",
        "https://www.jos.org.cn//jos/article/pdf/7684",
        (
            (
                "综述从哪两个方面对检索增强生成技术路线进行分类？",
                "检索增强和增强生成。",
                ("检索增强和增强生成两个方面",),
                "cn_rag_taxonomy",
            ),
            (
                "RAG技术融合了哪两个过程，从而能够动态查询外部知识库？",
                "信息检索与自然语言生成。",
                ("RAG技术融合了信息检索",),
                "cn_rag_components",
            ),
            (
                "论文指出RAG更适合哪些类型的应用任务？",
                "知识密集、答案可追溯性强且交互复杂的任务，例如科研文献问答。",
                ("RAG更适合用于知识密集",),
                "cn_rag_application",
            ),
        ),
    ),
    PaperSpec(
        "jos-7514",
        "7514.pdf",
        "基于大语言模型的空间数据库自然语言查询转换方法",
        "https://www.jos.org.cn//jos/article/pdf/7514",
        (
            (
                "NALSpatial方法包含哪两个核心阶段？",
                "自然语言理解和可执行语言生成。",
                ("该方法有两个核心阶段",),
                "cn_nalspatial_stages",
            ),
            (
                "在自然语言理解阶段，NALSpatial如何确定查询类型？",
                "提取关键查询实体，并基于大语言模型构建空间数据查询语料库来确定查询类型。",
                ("提取关键查询实体", "构建空间数据查询语料库"),
                "cn_nalspatial_understanding",
            ),
            (
                "论文报告的NALSpatial平均响应时间、可翻译性和翻译精度分别是多少？",
                "平均响应时间约2.5秒，可翻译性95%，翻译精度92%。",
                ("NALSpatial的平均响应时间约为2.5",),
                "cn_nalspatial_results",
            ),
        ),
    ),
    PaperSpec(
        "jos-7369",
        "7369.pdf",
        "基于Issue检索增强大语言模型的补充性代码注释生成",
        "https://www.jos.org.cn//jos/article/pdf/7369",
        (
            (
                "论文将Issue中的代码补充信息整理成多少种类型？",
                "5种类型。",
                ("整理分类为5种类型",),
                "cn_issue_types",
            ),
            (
                "该方法从Issue中检索信息后，如何用于补充性代码注释生成？",
                "先检索包含潜在补充信息类型的语句，再依据这些语句生成补充性代码注释。",
                ("检索出包含潜在类型补充信息的语句",),
                "cn_issue_pipeline",
            ),
            (
                "实验中GPT-4o生成注释对人工补充性注释的覆盖率提升到了多少？",
                "88.4%。",
                ("覆盖率从35.8%提升至88.4%",),
                "cn_issue_result",
            ),
        ),
    ),
    PaperSpec(
        "jos-7595",
        "7595.pdf",
        "SmartGen-AADL：多智能体系统需求分析与AADL模型生成",
        "https://www.jos.org.cn//jos/article/pdf/7595",
        (
            (
                "SmartGen-AADL方法的核心流程包括哪三个阶段？",
                "结构化需求识别与标准化提取、子问题细化分析、以及融合结构引导和RAG的构件生成。",
                ("方法核心包括3个阶段", "方法核心包括三个阶段"),
                "cn_smartgen_stages",
            ),
            (
                "SmartGen-AADL中的RAG模块主要解决什么问题？",
                "通过检索相似组件和历史系统文档，为生成模型补充领域语义与上下文，减少结构和接口错误。",
                ("RAG模块", "检索增强生成机制"),
                "cn_smartgen_rag",
            ),
            (
                "与仅依赖简单提示工程相比，该方法在组件错误率和FBERT上有什么变化？",
                "组件错误率平均降低34.37%，FBERT平均提升6.21%。",
                ("组件错误率上平均降低34.37%", "FBERT语义相似度上平均提升6.21%"),
                "cn_smartgen_result",
            ),
        ),
    ),
    PaperSpec(
        "jos-6508",
        "6508.pdf",
        "基于TWE-NMF主题模型的Mashup服务聚类方法",
        "https://www.jos.org.cn//jos/article/pdf/6508",
        (
            (
                "TWE-NMF方法针对Mashup服务短文本的哪些问题进行改进？",
                "针对描述文档短、特征稀疏和信息量少导致的主题建模困难进行改进。",
                ("描述文档通常比较简短", "特征稀疏", "信息量少"),
                "cn_twenmf_problem",
            ),
            (
                "TWE-NMF如何自动确定主题数量？",
                "将狄利克雷过程混合模型与NMF求解主题特征相结合，由DPMM自动估计主题数量。",
                ("自动估计主题的数量", "DPMM模型"),
                "cn_twenmf_topics",
            ),
            (
                "与现有服务聚类方法相比，TWE-NMF实验结果的总体趋势是什么？",
                "在精确率、召回率、F-measure、纯度和熵等指标上均有明显改善。",
                ("提高聚类的精度",),
                "cn_twenmf_result",
            ),
        ),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pick_blocks(blocks: list[Any], terms: tuple[str, ...]) -> list[Any]:
    normalized_terms = tuple("".join(term.split()) for term in terms)
    matches = []
    for block in blocks:
        text = str(getattr(block, "text", "") or "").strip()
        normalized = "".join(text.split())
        if text and all(term in normalized for term in normalized_terms):
            matches.append(block)
    if matches:
        return matches[:2]
    # A fallback accepts a block containing any one distinctive phrase. It is
    # intentionally explicit and fails loudly if the paper version changes.
    for block in blocks:
        text = str(getattr(block, "text", "") or "").strip()
        normalized = "".join(text.split())
        if text and any(term in normalized for term in normalized_terms):
            return [block]
    raise ValueError(f"could not locate evidence terms: {terms!r}")


def _paper_documents(paper: PaperSpec, parsed: Any) -> tuple[list[dict[str, Any]], dict[int, int]]:
    documents: list[dict[str, Any]] = []
    block_to_doc: dict[int, int] = {}
    current_section = "Paper body"
    for block_index, block in enumerate(parsed.blocks):
        text = str(getattr(block, "text", "") or "").strip()
        if not text:
            continue
        if getattr(block, "is_heading", False) or getattr(block, "block_type", "") == "title":
            current_section = text[:180]
        document_index = len(documents)
        documents.append(
            {
                "text": text,
                "section": current_section,
                "page": int(getattr(block, "page", 0) or 0),
                "block_type": str(getattr(block, "block_type", "paragraph") or "paragraph"),
            }
        )
        block_to_doc[block_index] = document_index
    if not documents:
        raise ValueError(f"paper {paper.paper_id} produced no text blocks")
    return documents, block_to_doc


async def prepare(
    *,
    pdf_root: Path,
    dataset_output: Path,
    split_output: Path,
    manifest_output: Path,
) -> dict[str, Any]:
    parser = NativePDFParser()
    fixture: dict[str, Any] = {}
    case_ids: list[str] = []
    selected_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    acquired_at = datetime.now(UTC).isoformat()

    for paper in PAPERS:
        path = (pdf_root / paper.filename).resolve(strict=True)
        parsed = await parser.parse(path, source_uri=paper.source_url)
        documents, block_to_doc = _paper_documents(paper, parsed)
        sections: dict[str, list[str]] = {}
        for item in documents:
            sections.setdefault(item["section"], []).append(item["text"])
        qas: list[dict[str, Any]] = []
        for question, answer, terms, slug in paper.questions:
            evidence_blocks = _pick_blocks(list(parsed.blocks), terms)
            evidence = [str(getattr(block, "text", "") or "").strip() for block in evidence_blocks]
            question_id = hashlib.sha1(question.encode("utf-8")).hexdigest()
            qas.append(
                {
                    "question": question,
                    "question_id": question_id,
                    "answers": [
                        {
                            "annotation_id": f"manual-{slug}",
                            "answer": {
                                "unanswerable": False,
                                "free_form_answer": answer,
                                "extractive_spans": [],
                                "yes_no": None,
                                "evidence": evidence,
                            },
                        }
                    ],
                }
            )
            case_id = f"qasper:{paper.paper_id}:{question_id}"
            case_ids.append(case_id)
            selected_rows.append(
                {
                    "case_id": case_id,
                    "paper_id": paper.paper_id,
                    "question": question,
                    "evidence_block_count": len(evidence),
                    "evidence_pages": sorted(
                        {
                            int(getattr(block, "page", 0) or 0)
                            for block in evidence_blocks
                        }
                    ),
                    "evidence_preview": [item[:160] for item in evidence],
                }
            )
        fixture[paper.paper_id] = {
            "title": paper.title,
            "abstract": "",
            "full_text": [
                {
                    "section_name": section,
                    "paragraphs": values,
                }
                for section, values in sections.items()
            ],
            "figures_and_tables": [],
            "qas": qas,
        }
        manifest_rows.append(
            {
                "paper_id": paper.paper_id,
                # The manifest lives one directory above the PDF cohort.
                "path": f"chinese-papers-jos-v1/{paper.filename}",
                "sha256": _sha256(path),
                "source_url": paper.source_url,
                "acquired_at": acquired_at,
                "page_count": int(parsed.page_count),
                "title": paper.title,
            }
        )

    source_files = [
        {
            "paper_id": paper.paper_id,
            "path": str((pdf_root / paper.filename).resolve()),
            "sha256": row["sha256"],
        }
        for paper, row in zip(PAPERS, manifest_rows, strict=True)
    ]
    split = {
        "schema_version": "1.0",
        "split_id": "taskforge-chinese-paper-fulltext-15-v1",
        "dataset": "TaskForge Paper RAG",
        "source": "Five open-access Journal of Software full-paper PDFs",
        "source_files": source_files,
        "selection_policy": "Three manually authored evidence-grounded questions per paper",
        "case_ids": case_ids,
        "selected_rows": selected_rows,
        "pdf_manifest": str(manifest_output),
        "synthetic_pdf_layout": None,
        "report_metadata": {
            "evaluation_type": "chinese_real_pdf_fulltext_retrieval",
            "benchmark_track": "scholarly_paper_fulltext_retrieval",
            "dataset": "TaskForge Paper RAG Chinese full-text smoke cohort",
            "license": "Each source PDF follows its publisher terms; use for evaluation only",
            "annotation_status": "manual_auditable_smoke_cohort",
            "limitation": "This is not a public benchmark annotation; labels are frozen and evidence is parser-native.",
        },
    }
    manifest = {
        "schema_version": "1.0",
        "dataset": "TaskForge Paper RAG",
        "cohort_id": "chinese-paper-fulltext-real-pdfs-v1",
        "source": "Journal of Software (JOS) official PDF endpoints",
        "papers": manifest_rows,
    }
    for output in (dataset_output, split_output, manifest_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    dataset_output.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    split_output.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "papers": len(PAPERS),
        "cases": len(case_ids),
        "dataset_output": str(dataset_output),
        "split_output": str(split_output),
        "manifest_output": str(manifest_output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-root", type=Path, default=DEFAULT_PDF_ROOT)
    parser.add_argument("--dataset-output", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split-output", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    report = asyncio.run(
        prepare(
            pdf_root=args.pdf_root,
            dataset_output=args.dataset_output,
            split_output=args.split_output,
            manifest_output=args.manifest_output,
        )
    )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()

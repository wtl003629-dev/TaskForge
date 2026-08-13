"""Build the pinned 30-task paper-research functional E2E benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAPERS = [
    {
        "paper_key": "attention",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "year": 2017,
        "arxiv_id": "1706.03762",
        "abstract": "The Transformer replaces recurrence with multi-head self-attention and reports strong machine translation quality with more parallelizable training.",
    },
    {
        "paper_key": "bert",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "authors": ["Jacob Devlin", "Ming-Wei Chang"],
        "year": 2018,
        "arxiv_id": "1810.04805",
        "abstract": "BERT pre-trains deep bidirectional Transformer representations using masked language modeling and next sentence prediction before task fine-tuning.",
    },
    {
        "paper_key": "gpt3",
        "title": "Language Models are Few-Shot Learners",
        "authors": ["Tom B. Brown", "Benjamin Mann"],
        "year": 2020,
        "arxiv_id": "2005.14165",
        "abstract": "GPT-3 studies in-context zero-shot, one-shot, and few-shot learning at large scale without gradient updates for each downstream task.",
    },
    {
        "paper_key": "dpr",
        "title": "Dense Passage Retrieval for Open-Domain Question Answering",
        "authors": ["Vladimir Karpukhin", "Barlas Oguz"],
        "year": 2020,
        "arxiv_id": "2004.04906",
        "abstract": "DPR learns separate dense encoders for questions and passages and improves top-k passage retrieval for open-domain question answering.",
    },
    {
        "paper_key": "rag",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": ["Patrick Lewis", "Ethan Perez"],
        "year": 2020,
        "arxiv_id": "2005.11401",
        "abstract": "RAG combines a pretrained sequence-to-sequence generator with a non-parametric dense passage index and marginalizes retrieved documents during generation.",
    },
    {
        "paper_key": "cot",
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "authors": ["Jason Wei", "Xuezhi Wang"],
        "year": 2022,
        "arxiv_id": "2201.11903",
        "abstract": "Chain-of-thought prompting supplies intermediate reasoning demonstrations and improves arithmetic, commonsense, and symbolic reasoning in sufficiently large language models.",
    },
    {
        "paper_key": "react",
        "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
        "authors": ["Shunyu Yao", "Jeffrey Zhao"],
        "year": 2022,
        "arxiv_id": "2210.03629",
        "abstract": "ReAct interleaves verbal reasoning traces with actions and observations, allowing language models to update plans while interacting with external environments.",
    },
    {
        "paper_key": "toolformer",
        "title": "Toolformer: Language Models Can Teach Themselves to Use Tools",
        "authors": ["Timo Schick", "Jane Dwivedi-Yu"],
        "year": 2023,
        "arxiv_id": "2302.04761",
        "abstract": "Toolformer self-supervises API-call demonstrations so a language model can decide which tool to call, when to call it, and how to use returned results.",
    },
    {
        "paper_key": "selfrag",
        "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        "authors": ["Akari Asai", "Zexuan Zhong"],
        "year": 2023,
        "arxiv_id": "2310.11511",
        "abstract": "Self-RAG trains a language model to retrieve on demand and emit reflection tokens that critique relevance, support, and generation quality.",
    },
    {
        "paper_key": "crag",
        "title": "Corrective Retrieval Augmented Generation",
        "authors": ["Shi-Qi Yan", "Jia-Chen Gu"],
        "year": 2024,
        "arxiv_id": "2401.15884",
        "abstract": "Corrective RAG evaluates retrieved documents, triggers corrective web retrieval when confidence is low, and refines evidence before generation.",
    },
]


def build_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for index, paper in enumerate(PAPERS, start=1):
        cases.append(
            {
                "case_id": f"single-{index:02d}",
                "task_type": "single_paper",
                "query": f"Explain the main method and limitation of {paper['title']}",
                "user_intent": "Explain the selected paper's method and evidence-backed limitation.",
                "papers": [paper],
                "expected_selected_paper_count": 1,
            }
        )
    for index in range(10):
        papers = [PAPERS[index], PAPERS[(index + 1) % len(PAPERS)]]
        cases.append(
            {
                "case_id": f"comparison-{index + 1:02d}",
                "task_type": "comparison",
                "query": f"Compare {papers[0]['title']} with {papers[1]['title']}",
                "user_intent": "Compare the methods, retrieval or reasoning mechanism, and stated limitations across the selected papers.",
                "papers": papers,
                "expected_selected_paper_count": 2,
            }
        )
    for index in range(10):
        papers = [
            PAPERS[index],
            PAPERS[(index + 3) % len(PAPERS)],
            PAPERS[(index + 6) % len(PAPERS)],
        ]
        cases.append(
            {
                "case_id": f"survey-{index + 1:02d}",
                "task_type": "survey",
                "query": f"Survey a research thread connecting {papers[0]['title']}, {papers[1]['title']}, and {papers[2]['title']}",
                "user_intent": "Synthesize the selected papers into a short evidence-grounded survey with differences and open limitations.",
                "papers": papers,
                "expected_selected_paper_count": 3,
            }
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "eval" / "paper-research-e2e-cases-30.json",
    )
    args = parser.parse_args()
    payload = {
        "schema_version": "1.0",
        "evaluation_type": "paper_research_deterministic_functional_e2e",
        "provenance": (
            "TaskForge-authored integration fixtures using public paper metadata; "
            "not a human relevance or answer-quality benchmark."
        ),
        "counts": {"single_paper": 10, "comparison": 10, "survey": 10},
        "cases": build_cases(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "case_count": len(payload["cases"])}, indent=2))


if __name__ == "__main__":
    main()

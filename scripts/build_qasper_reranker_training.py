"""Build paper-disjoint QASPER reranker data from the current candidate pool.

The script deliberately uses only the question and the uploaded-paper text to
produce candidates. Gold evidence is used for labels after retrieval; it is
never injected into the candidate search text. The output is suitable for
cross-encoder/listwise training and includes enough diagnostics to audit
candidate misses separately from ranking misses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.hybrid_knowledge import knowledge_to_hybrid_chunk  # noqa: E402
from taskforge.hybrid_retrieval import BM25Index, HybridSearchRequest  # noqa: E402
from taskforge.knowledge import AccessContext, KnowledgeChunk, tokenise  # noqa: E402
from taskforge.rag_baseline import sha256_file  # noqa: E402
from taskforge.rag_evaluation import (  # noqa: E402
    EvalCorpusDocument,
    RAGEvalCase,
    load_qasper_dataset,
)

_TYPE_TERMS = {
    "method": {"method", "approach", "architecture", "model", "component", "mechanism"},
    "dataset": {"dataset", "corpus", "benchmark", "data", "collection"},
    "baseline": {"baseline", "compare", "comparison", "compared", "state-of-the-art", "prior"},
    "metric": {"metric", "score", "accuracy", "f1", "precision", "recall", "performance", "result"},
    "numeric": {"how", "much", "percent", "percentage", "number", "value", "rate", "increase", "decrease"},
    "limitation": {"limitation", "error", "failure", "future", "limitation", "weakness"},
}


def _paper_id(case: RAGEvalCase) -> str:
    value = str(case.metadata.get("paper_id") or "").strip()
    if not value:
        raise ValueError(f"case {case.case_id} is missing paper_id")
    return value


def _case_type(query: str) -> str:
    lowered = set(tokenise(query.casefold()))
    scores = {
        name: len(lowered.intersection(terms)) for name, terms in _TYPE_TERMS.items()
    }
    numeric = bool(re.search(r"\b(?:19|20)\d{2}\b|\d+(?:\.\d+)?%?", query))
    if numeric:
        scores["numeric"] += 1
    best = max(scores, key=lambda name: (scores[name], name))
    return best if scores[best] else "general"


def _ranked_candidates(
    query: str,
    documents: list[EvalCorpusDocument],
    *,
    top_k: int,
) -> list[tuple[str, float, int]]:
    chunks = [
        KnowledgeChunk(
            chunk_id=document.document_id,
            tenant_id="qasper-reranker-build",
            text=document.text,
            source_uri=document.source_uri,
            document_id=str(document.metadata.get("parent_id") or document.document_id),
            acl=frozenset({"tenant"}),
            metadata=document.metadata,
        )
        for document in documents
    ]
    index = BM25Index(knowledge_to_hybrid_chunk(chunk) for chunk in chunks)
    principal = AccessContext(tenant_id="qasper-reranker-build", user_id="builder")
    response = index.search(
        HybridSearchRequest(
            query=query,
            tenant_id=principal.tenant_id,
            acl_principals=principal.acl_tokens,
            allowed_chunk_ids=frozenset(document.document_id for document in documents),
            top_k=top_k,
            candidate_k=top_k,
            max_expanded_hits=top_k,
        )
    )
    return [
        (hit.chunk.chunk_id, float(hit.score), rank)
        for rank, hit in enumerate(response.hits, start=1)
    ]


def _hash_ids(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def build(
    dataset_path: Path,
    split_path: Path,
    output: Path,
    *,
    candidate_k: int = 50,
    hard_negative_k: int = 10,
    regular_negative_start: int = 11,
    regular_negative_k: int = 40,
) -> dict[str, Any]:
    if candidate_k < regular_negative_start:
        raise ValueError("candidate_k must include the regular-negative start rank")
    if hard_negative_k < 1 or regular_negative_k < 1:
        raise ValueError("negative sample budgets must be positive")
    dataset = load_qasper_dataset(dataset_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    case_ids = [str(value) for value in split.get("case_ids", [])]
    if not case_ids:
        raise ValueError("split manifest has no case_ids")
    by_case = {case.case_id: case for case in dataset.cases}
    missing = [case_id for case_id in case_ids if case_id not in by_case]
    if missing:
        raise ValueError(f"split contains unknown QASPER cases: {missing[:3]}")
    cases = [by_case[case_id] for case_id in case_ids]
    selected_papers = {_paper_id(case) for case in cases}
    by_paper: dict[str, list[EvalCorpusDocument]] = defaultdict(list)
    for document in dataset.documents:
        paper_id = str(document.metadata.get("paper_id") or "").strip()
        if paper_id in selected_papers:
            by_paper[paper_id].append(document)

    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    counts = Counter()
    for case in cases:
        paper_id = _paper_id(case)
        documents = by_paper.get(paper_id, [])
        if not documents:
            raise ValueError(f"paper {paper_id} has no documents")
        candidates = _ranked_candidates(case.query, documents, top_k=candidate_k)
        gold = set(case.relevant_ids)
        candidate_ids = [item[0] for item in candidates]
        gold_ranks = [rank for document_id, _, rank in candidates if document_id in gold]
        first_gold_rank = min(gold_ranks) if gold_ranks else None
        if first_gold_rank is None:
            failure = "candidate_missing"
        elif first_gold_rank <= 10:
            failure = "covered_top10"
        else:
            failure = "top10_ranking_failure"
        counts[failure] += 1
        hard_negatives = [
            document_id
            for document_id, _, rank in candidates
            if 1 <= rank <= hard_negative_k and document_id not in gold
        ]
        regular_negatives = [
            document_id
            for document_id, _, rank in candidates
            if regular_negative_start <= rank <= candidate_k and document_id not in gold
        ][:regular_negative_k]
        if not hard_negatives:
            counts["no_hard_negative"] += 1
        if not regular_negatives:
            counts["no_regular_negative"] += 1
        records.append(
            {
                "case_id": case.case_id,
                "paper_id": paper_id,
                "query": case.query,
                "query_type": _case_type(case.query),
                "positives": sorted(gold),
                "hard_negatives": hard_negatives,
                "negatives": regular_negatives,
                "candidate_ids": candidate_ids,
                "candidate_scores": [score for _, score, _ in candidates],
                "first_positive_rank": first_gold_rank,
            }
        )
        section_counts = Counter(
            str(document.metadata.get("section_title") or document.metadata.get("section") or "unknown")
            for document in documents
            if document.document_id in {item[0] for item in candidates[:10]}
        )
        diagnostics.append(
            {
                "case_id": case.case_id,
                "paper_id": paper_id,
                "failure": failure,
                "query_type": _case_type(case.query),
                "gold_count": len(gold),
                "gold_in_top10": len(gold.intersection(candidate_ids[:10])),
                "gold_in_top50": len(gold.intersection(candidate_ids[:candidate_k])),
                "first_positive_rank": first_gold_rank,
                "top10_sections": dict(section_counts),
            }
        )

    paper_ids = sorted(selected_papers)
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "dataset": "QASPER",
        "source_dataset": str(dataset_path),
        "source_sha256": sha256_file(dataset_path),
        "split": str(split_path),
        "split_id": str(split.get("split_id") or ""),
        "split_case_ids_sha256": _hash_ids(case_ids),
        "paper_ids_sha256": _hash_ids(paper_ids),
        "cases": len(records),
        "papers": len(paper_ids),
        "candidate_k": candidate_k,
        "hard_negative_ranks": [1, hard_negative_k],
        "regular_negative_ranks": [regular_negative_start, candidate_k],
        "records": records,
        "diagnostics": {
            "counts": dict(counts),
            "rates": {key: value / len(records) for key, value in counts.items()},
            "by_query_type": {
                query_type: {
                    "cases": sum(1 for item in diagnostics if item["query_type"] == query_type),
                    "top10_covered": sum(
                        1
                        for item in diagnostics
                        if item["query_type"] == query_type and item["gold_in_top10"] > 0
                    ),
                    "top50_covered": sum(
                        1
                        for item in diagnostics
                        if item["query_type"] == query_type and item["gold_in_top50"] > 0
                    ),
                }
                for query_type in sorted({item["query_type"] for item in diagnostics})
            },
            "cases": diagnostics,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--hard-negative-k", type=int, default=10)
    parser.add_argument("--regular-negative-start", type=int, default=11)
    parser.add_argument("--regular-negative-k", type=int, default=40)
    args = parser.parse_args()
    result = build(
        args.dataset,
        args.split,
        args.output,
        candidate_k=args.candidate_k,
        hard_negative_k=args.hard_negative_k,
        regular_negative_start=args.regular_negative_start,
        regular_negative_k=args.regular_negative_k,
    )
    print(
        json.dumps(
            {
                "split_id": result["split_id"],
                "cases": result["cases"],
                "papers": result["papers"],
                "diagnostic_counts": result["diagnostics"]["counts"],
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

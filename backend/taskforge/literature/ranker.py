"""Explainable low-cost paper ranking before optional LLM screening."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from ..research_protocol import LiteratureRequest, PaperCard
from .query_planner import english_academic_bridge


def _short_description(paper: PaperCard) -> str:
    abstract = " ".join(paper.abstract.split())
    if abstract:
        sentence = re.split(r"(?<=[.!?。！？])\s+", abstract, maxsplit=1)[0]
        return sentence[:497].rstrip() + ("..." if len(sentence) > 497 else "")
    return f"该论文围绕“{paper.canonical_title[:430]}”展开研究。"


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[\w-]+", value.casefold(), re.UNICODE) if len(token) > 1}


def rank_papers(
    request: LiteratureRequest,
    papers: Sequence[PaperCard],
    *,
    dense_scores: Mapping[str, float] | None = None,
) -> list[PaperCard]:
    bridged = [
        value
        for text in [request.query, *request.research_questions]
        if (value := english_academic_bridge(text))
    ]
    query_tokens = _tokens(
        " ".join([request.query, *request.research_questions, *bridged])
    )
    required = {value.casefold() for value in request.required_terms}
    excluded = {value.casefold() for value in request.excluded_terms}
    now_year = datetime.now(UTC).year
    ranked: list[PaperCard] = []

    for paper in papers:
        body = f"{paper.canonical_title} {paper.abstract} {paper.venue or ''} {' '.join(paper.authors)}"
        lowered = body.casefold()
        if excluded and any(term in lowered for term in excluded):
            continue
        if request.year_from is not None and paper.year is not None and paper.year < request.year_from:
            continue
        if request.year_to is not None and paper.year is not None and paper.year > request.year_to:
            continue
        if request.venues and paper.venue and not any(
            venue.casefold() in paper.venue.casefold() for venue in request.venues
        ):
            continue
        if request.authors and paper.authors and not any(
            wanted.casefold() in author.casefold()
            for wanted in request.authors
            for author in paper.authors
        ):
            continue

        body_tokens = _tokens(body)
        overlap = len(query_tokens & body_tokens)
        lexical = overlap / max(1, len(query_tokens))
        dense = max(
            0.0,
            min(1.0, float((dense_scores or {}).get(paper.paper_id, 0.0))),
        )
        provider_rank = max(
            (
                1.0 / math.log2(rank + 1.0)
                for rank in paper.provider_ranks.values()
                if rank > 0
            ),
            default=0.0,
        )
        provider_count = len(
            {leg.split(":", 1)[0] for leg in paper.provider_ranks}
        )
        cross_source = min(1.0, max(0, provider_count - 1) / 2.0)
        query_coverage = min(1.0, len(paper.matched_queries) / 3.0)
        recency = (
            max(0.0, min(1.0, 1.0 - max(0, now_year - paper.year) / 20.0))
            if paper.year is not None
            else 0.0
        )
        citations = math.log1p(paper.citation_count or 0) / math.log1p(10_000)
        venue = 1.0 if paper.venue else 0.0
        if dense_scores is None:
            score = (
                0.38 * lexical
                + 0.32 * provider_rank
                + 0.10 * query_coverage
                + 0.08 * cross_source
                + 0.05 * recency
                + 0.05 * min(1.0, citations)
                + 0.02 * venue
            )
        else:
            score = (
                0.25 * lexical
                + 0.28 * dense
                + 0.25 * provider_rank
                + 0.08 * query_coverage
                + 0.06 * cross_source
                + 0.03 * recency
                + 0.03 * min(1.0, citations)
                + 0.02 * venue
            )
        matched = sorted(term for term in required if term in lowered)
        if required:
            score *= 0.5 + 0.5 * len(matched) / len(required)
        reason_parts = []
        if overlap:
            reason_parts.append(f"matches {overlap} query terms")
        if matched:
            reason_parts.append("covers required terms: " + ", ".join(matched[:5]))
        if paper.verification_status == "cross_source_verified":
            reason_parts.append("verified by multiple scholarly sources")
        if provider_rank:
            reason_parts.append("ranked by scholarly provider retrieval")
        if paper.year:
            reason_parts.append(f"published in {paper.year}")
        ranked.append(
            paper.model_copy(
                update={
                    "relevance_score": min(1.0, max(0.0, score)),
                    "relevance_reason": "; ".join(reason_parts) or "provider-ranked candidate",
                    "matched_requirements": matched,
                    "short_description": _short_description(paper),
                }
            )
        )

    ranked.sort(key=lambda item: (-item.relevance_score, item.canonical_title.casefold(), item.paper_id))
    return ranked[: request.result_limit]


__all__ = ["rank_papers"]

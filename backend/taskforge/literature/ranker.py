"""Explainable low-cost paper ranking before optional LLM screening."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from ..research_protocol import LiteratureRequest, PaperCard
from .query_planner import english_academic_bridge

_CJK_BIGRAM_STOPWORDS = frozenset(
    {
        "一种",
        "中的",
        "以及",
        "关于",
        "其中",
        "哪些",
        "主要",
        "什么",
        "如何",
        "何提",
        "对于",
        "应用",
        "影响",
        "提高",
        "方法",
        "是否",
        "有效",
        "研究",
        "结果",
        "进行",
        "问题",
        "中文",
        "文文",
        "论文",
        "语言",
    }
)
_NON_PAPER_PUBLICATION_TYPES = {
    "book",
    "component",
    "dataset",
    "editorial",
    "erratum",
    "grant",
    "letter",
    "news",
    "paratext",
    "report",
    "standard",
}
_TRUSTED_CROSSREF_PUBLISHER_MARKERS = (
    "acm",
    "academy",
    "american chemical society",
    "american library association",
    "american medical association",
    "american physical society",
    "association for computational linguistics",
    "association for computing machinery",
    "bmj",
    "cambridge university press",
    "de gruyter",
    "elsevier",
    "emerald publishing",
    "frontiers media",
    "ieee",
    "institute of electrical and electronics engineers",
    "iop publishing",
    "karger",
    "lippincott",
    "mdpi",
    "nature portfolio",
    "oxford university press",
    "public library of science",
    "royal society",
    "sage publications",
    "science china press",
    "springer",
    "taylor & francis",
    "thieme",
    "university press",
    "wiley",
    "wolters kluwer",
    "world scientific",
    "中国科学院",
    "中国科学出版",
    "大学出版社",
    "教育部",
    "科学出版社",
)


def _trusted_crossref_only_record(paper: PaperCard) -> bool:
    publisher = (paper.publisher or "").casefold()
    return any(marker in publisher for marker in _TRUSTED_CROSSREF_PUBLISHER_MARKERS)


def _short_description(paper: PaperCard) -> str:
    abstract = " ".join(paper.abstract.split())
    if abstract:
        sentence = re.split(r"(?<=[.!?。！？])\s+", abstract, maxsplit=1)[0]
        return sentence[:497].rstrip() + ("..." if len(sentence) > 497 else "")
    return f"该论文围绕“{paper.canonical_title[:430]}”展开研究。"


def _tokens(value: str) -> set[str]:
    lowered = value.casefold()
    word_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", lowered)
        if len(token) > 1
    }
    # A regex ``\w+`` treats an entire Chinese sentence as one token, so even
    # closely related Chinese queries and titles otherwise have zero lexical
    # overlap. Character bigrams give us a deterministic, dependency-free
    # relevance signal without pretending to perform full word segmentation.
    chinese_bigrams = {
        segment[index : index + 2]
        for segment in re.findall(r"[\u3400-\u9fff]+", lowered)
        for index in range(len(segment) - 1)
        if segment[index : index + 2] not in _CJK_BIGRAM_STOPWORDS
    }
    return word_tokens | chinese_bigrams


def _contains_cjk(value: str) -> bool:
    return re.search(r"[\u3400-\u9fff]", value) is not None


def _is_chinese_paper(paper: PaperCard) -> bool:
    language = (paper.language or "").casefold()
    return language == "zh" or language.startswith("zh-") or _contains_cjk(
        paper.canonical_title
    )


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
        publication_type = (paper.publication_type or "").casefold()
        if publication_type in _NON_PAPER_PUBLICATION_TYPES:
            continue
        provider_names = {
            leg.split(":", 1)[0] for leg in paper.provider_ranks
        }
        if (
            paper.verification_status == "metadata_partial"
            and provider_names == {"crossref"}
        ):
            # Crossref is a DOI metadata registry, not a peer-review or content
            # quality authority. Sparse records seen nowhere else are too weak
            # to present as papers in the default discovery experience.
            continue
        if (
            paper.verification_status == "provider_verified"
            and provider_names == {"crossref"}
            and not _trusted_crossref_only_record(paper)
        ):
            # A DOI and the label ``journal-article`` can be self-supplied by
            # a publisher. For Crossref-only records, require a conservative
            # institutional publisher signal; otherwise wait for another
            # scholarly index to corroborate the item.
            continue
        body = f"{paper.canonical_title} {paper.abstract} {paper.venue or ''} {' '.join(paper.authors)}"
        lowered = body.casefold()
        if excluded and any(term in lowered for term in excluded):
            continue
        if (
            request.year_from is not None
            and paper.year is not None
            and paper.year < request.year_from
        ):
            continue
        if (
            request.year_to is not None
            and paper.year is not None
            and paper.year > request.year_to
        ):
            continue
        if (
            request.venues
            and paper.venue
            and not any(
                venue.casefold() in paper.venue.casefold() for venue in request.venues
            )
        ):
            continue
        if (
            request.authors
            and paper.authors
            and not any(
                wanted.casefold() in author.casefold()
                for wanted in request.authors
                for author in paper.authors
            )
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
        if (
            _contains_cjk(request.query)
            and _is_chinese_paper(paper)
        ):
            # Native-language lexical APIs can return a high provider rank for
            # generic Chinese phrasing. Require at least a small amount of
            # topic overlap before trusting that rank at full strength, while
            # still retaining cross-lingual semantic candidates for later
            # quota selection.
            native_topic_confidence = min(1.0, overlap / 3.0)
            provider_rank *= 0.4 + 0.6 * native_topic_confidence
        provider_count = len({leg.split(":", 1)[0] for leg in paper.provider_ranks})
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
        if paper.verification_status == "metadata_partial":
            score *= 0.8
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
        elif paper.verification_status == "metadata_partial":
            reason_parts.append("single-source metadata requires review")
        if publication_type:
            reason_parts.append(f"classified as {publication_type}")
        if provider_rank:
            reason_parts.append("ranked by scholarly provider retrieval")
        if paper.year:
            reason_parts.append(f"published in {paper.year}")
        ranked.append(
            paper.model_copy(
                update={
                    "relevance_score": min(1.0, max(0.0, score)),
                    "relevance_reason": "; ".join(reason_parts)
                    or "provider-ranked candidate",
                    "matched_requirements": matched,
                    "short_description": _short_description(paper),
                }
            )
        )

    ranked.sort(
        key=lambda item: (
            -item.relevance_score,
            item.canonical_title.casefold(),
            item.paper_id,
        )
    )
    selected = ranked[: request.result_limit]
    chinese_target_ratio = (
        0.5
        if request.language_preference == "chinese_first"
        else 0.3
        if request.language_preference == "balanced" and _contains_cjk(request.query)
        else 0.0
    )
    if chinese_target_ratio and ranked:
        target = max(1, math.ceil(request.result_limit * chinese_target_ratio))
        current = sum(_is_chinese_paper(item) for item in selected)
        floor = max(0.08, ranked[0].relevance_score * 0.5)
        chinese_candidates = [
            item
            for item in ranked[request.result_limit :]
            if _is_chinese_paper(item) and item.relevance_score >= floor
        ]
        while current < target and chinese_candidates and selected:
            replacement = chinese_candidates.pop(0)
            replace_at = next(
                (
                    index
                    for index in range(len(selected) - 1, -1, -1)
                    if not _is_chinese_paper(selected[index])
                ),
                None,
            )
            if replace_at is None:
                break
            selected[replace_at] = replacement
            current += 1
        selected.sort(
            key=lambda item: (
                -item.relevance_score,
                item.canonical_title.casefold(),
                item.paper_id,
            )
        )
    return selected


__all__ = ["rank_papers"]

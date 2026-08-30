"""Cross-provider scholarly record consolidation."""

from __future__ import annotations

from collections.abc import Iterable
from difflib import SequenceMatcher

from ..research_protocol import PaperCard
from .models import ProviderPaper
from .normalizer import (
    canonical_paper_id,
    normalise_arxiv_id,
    normalise_doi,
    normalise_title,
)


def _strong_keys(paper: ProviderPaper) -> set[str]:
    keys: set[str] = set()
    doi = normalise_doi(paper.doi)
    arxiv_id = normalise_arxiv_id(paper.arxiv_id)
    if doi:
        keys.add(f"doi:{doi}")
    if arxiv_id:
        keys.add(f"arxiv:{arxiv_id}")
    if paper.semantic_scholar_id:
        keys.add(f"s2:{paper.semantic_scholar_id.casefold()}")
    if paper.openalex_id:
        keys.add(f"openalex:{paper.openalex_id.casefold().rsplit('/', 1)[-1]}")
    return keys


def _same_work(left: ProviderPaper, right: ProviderPaper) -> bool:
    if _strong_keys(left) & _strong_keys(right):
        return True
    left_title = normalise_title(left.title)
    right_title = normalise_title(right.title)
    if not left_title or not right_title:
        return False
    similarity = SequenceMatcher(None, left_title, right_title).ratio()
    if similarity < 0.94:
        return False
    if left.year is not None and right.year is not None and abs(left.year - right.year) > 1:
        return False
    if left.authors and right.authors:
        left_author = normalise_title(left.authors[0])
        right_author = normalise_title(right.authors[0])
        if left_author and right_author and left_author != right_author:
            return False
    return True


def _best_text(values: Iterable[str | None]) -> str:
    candidates = [value.strip() for value in values if value and value.strip()]
    return max(candidates, key=len, default="")


def _best_publication_type(values: Iterable[str | None]) -> str | None:
    priority = {
        "journal-article": 7,
        "proceedings-article": 6,
        "article": 5,
        "review": 5,
        "preprint": 4,
        "posted-content": 4,
        "dissertation": 3,
        "book-chapter": 2,
    }
    normalized = {
        value.strip().casefold()
        for value in values
        if value and value.strip()
    }
    return max(
        normalized,
        key=lambda value: (priority.get(value, 0), value),
        default=None,
    )


def _unique(values: Iterable[str | None]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))


def merge_provider_papers(papers: Iterable[ProviderPaper]) -> list[PaperCard]:
    groups: list[list[ProviderPaper]] = []
    for paper in papers:
        group = next((item for item in groups if _same_work(item[0], paper)), None)
        if group is None:
            groups.append([paper])
        else:
            group.append(paper)

    cards: list[PaperCard] = []
    for group in groups:
        doi = next((normalise_doi(item.doi) for item in group if normalise_doi(item.doi)), None)
        arxiv_id = next(
            (
                normalise_arxiv_id(item.arxiv_id)
                for item in group
                if normalise_arxiv_id(item.arxiv_id)
            ),
            None,
        )
        semantic_scholar_id = next(
            (item.semantic_scholar_id for item in group if item.semantic_scholar_id),
            None,
        )
        openalex_id = next((item.openalex_id for item in group if item.openalex_id), None)
        authors = max((item.authors for item in group), key=len, default=[])
        year = next((item.year for item in group if item.year is not None), None)
        title = _best_text(item.title for item in group)
        provider_count = len({item.provider for item in group})
        provider_ranks: dict[str, int] = {}
        for item in group:
            leg = f"{item.provider}:{item.query_id or 'direct'}"
            provider_ranks[leg] = min(
                provider_ranks.get(leg, item.provider_rank),
                item.provider_rank,
            )
        languages = [
            item.language.casefold()
            for item in group
            if item.language and item.language.strip()
        ]
        # OpenAlex's language metadata is authoritative for native-language
        # retrieval.  A corroborating provider may label the same work ``en``
        # (especially when only an English title/abstract is indexed), so keep
        # ``zh`` stable regardless of provider result order.
        language = next(
            (value for value in languages if value == "zh" or value.startswith("zh-")),
            languages[0] if languages else None,
        )
        cards.append(
            PaperCard(
                paper_id=canonical_paper_id(
                    title=title,
                    authors=authors,
                    year=year,
                    doi=doi,
                    arxiv_id=arxiv_id,
                    semantic_scholar_id=semantic_scholar_id,
                    openalex_id=openalex_id,
                ),
                canonical_title=title,
                authors=authors,
                abstract=_best_text(item.abstract for item in group),
                short_description="",
                year=year,
                language=language,
                publication_type=_best_publication_type(
                    item.publication_type for item in group
                ),
                venue=_best_text(item.venue for item in group) or None,
                publisher=_best_text(item.publisher for item in group) or None,
                doi=doi,
                arxiv_id=arxiv_id,
                semantic_scholar_id=semantic_scholar_id,
                openalex_id=openalex_id,
                source_urls=_unique(item.source_url for item in group),
                pdf_url=next((item.pdf_url for item in group if item.pdf_url), None),
                citation_count=max(
                    (item.citation_count for item in group if item.citation_count is not None),
                    default=None,
                ),
                references=_unique(value for item in group for value in item.references),
                cited_by=_unique(value for item in group for value in item.cited_by),
                matched_queries=_unique(item.query_id for item in group),
                provider_ranks=provider_ranks,
                verification_status=(
                    "cross_source_verified"
                    if provider_count > 1
                    else "provider_verified"
                    if any(item.provider == "arxiv" for item in group)
                    or (
                        any(item.publication_type for item in group)
                        and bool(authors)
                        and bool(doi or _best_text(item.venue for item in group))
                    )
                    else "metadata_partial"
                ),
                full_text_status=(
                    "available"
                    if any(item.pdf_url for item in group)
                    else "abstract_only"
                ),
            )
        )
    return cards


__all__ = ["merge_provider_papers"]

"""Semantic Scholar Graph API adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

from ...research_protocol import SearchQuery
from ..models import ProviderPaper
from .base import ResilientHTTPProvider

_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_FIELDS = (
    "paperId,title,abstract,year,venue,authors,externalIds,url,openAccessPdf,"
    "citationCount,referenceCount,publicationTypes"
)
_NON_PAPER_TYPES = {"book", "editorial", "lettersandcomments", "news"}


def _publication_type(value: object) -> tuple[str | None, bool]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None, True
    types = [str(item).strip() for item in value if str(item).strip()]
    if not types:
        return None, True
    normalized = {item.casefold().replace("_", "") for item in types}
    if normalized <= _NON_PAPER_TYPES:
        return None, False
    if "conference" in normalized:
        return "proceedings-article", True
    if "journalarticle" in normalized:
        return "journal-article", True
    if "review" in normalized:
        return "review", True
    if "booksection" in normalized:
        return "book-chapter", True
    if normalized & {"casereport", "clinicaltrial", "metaanalysis", "study"}:
        return "article", True
    return sorted(normalized)[0], True


def _paper(value: Mapping[str, Any], *, query_id: str | None, rank: int) -> ProviderPaper | None:
    title = str(value.get("title") or "").strip()
    paper_id = str(value.get("paperId") or "").strip()
    publication_type, is_paper = _publication_type(value.get("publicationTypes"))
    if not title or not paper_id or not is_paper:
        return None
    external = value.get("externalIds")
    external_ids = external if isinstance(external, Mapping) else {}
    access = value.get("openAccessPdf")
    open_access = access if isinstance(access, Mapping) else {}
    raw_authors = value.get("authors")
    authors = [
        str(author.get("name") or "").strip()
        for author in raw_authors
        if isinstance(author, Mapping) and str(author.get("name") or "").strip()
    ] if isinstance(raw_authors, Sequence) and not isinstance(raw_authors, str) else []
    return ProviderPaper(
        provider="semantic_scholar",
        provider_id=paper_id,
        title=title,
        authors=authors,
        abstract=str(value.get("abstract") or "").strip(),
        year=value.get("year") if isinstance(value.get("year"), int) else None,
        publication_type=publication_type,
        venue=str(value.get("venue") or "").strip() or None,
        doi=str(external_ids.get("DOI") or "").strip() or None,
        arxiv_id=str(external_ids.get("ArXiv") or "").strip() or None,
        semantic_scholar_id=paper_id,
        source_url=str(value.get("url") or "").strip() or None,
        pdf_url=str(open_access.get("url") or "").strip() or None,
        citation_count=(
            value.get("citationCount")
            if isinstance(value.get("citationCount"), int)
            else None
        ),
        query_id=query_id,
        provider_rank=rank,
    )


class SemanticScholarProvider(ResilientHTTPProvider):
    name = "semantic_scholar"

    async def search(self, query: SearchQuery, limit: int) -> list[ProviderPaper]:
        params: dict[str, object] = {
            "query": query.text,
            "limit": min(max(1, int(limit)), 100),
            "fields": _FIELDS,
        }
        year_from = query.provider_filters.get("year_from")
        year_to = query.provider_filters.get("year_to")
        if year_from or year_to:
            params["year"] = f"{year_from or ''}-{year_to or ''}"
        payload = await self._get_json(f"{_BASE_URL}/paper/search", params=params)
        data = payload.get("data") if isinstance(payload, Mapping) else None
        if not isinstance(data, Sequence) or isinstance(data, str):
            return []
        results: list[ProviderPaper] = []
        for rank, item in enumerate(data, start=1):
            if isinstance(item, Mapping):
                parsed = _paper(item, query_id=query.query_id, rank=rank)
                if parsed is not None:
                    results.append(parsed)
        return results

    async def get_paper(self, paper_id: str) -> ProviderPaper | None:
        payload = await self._get_json(
            f"{_BASE_URL}/paper/{quote(paper_id, safe='')}",
            params={"fields": _FIELDS},
        )
        return _paper(payload, query_id=None, rank=1) if isinstance(payload, Mapping) else None

    async def _relations(self, paper_id: str, relation: str, limit: int) -> list[ProviderPaper]:
        payload = await self._get_json(
            f"{_BASE_URL}/paper/{quote(paper_id, safe='')}/{relation}",
            params={"limit": min(max(1, int(limit)), 100), "fields": _FIELDS},
        )
        data = payload.get("data") if isinstance(payload, Mapping) else None
        if not isinstance(data, Sequence) or isinstance(data, str):
            return []
        key = "citedPaper" if relation == "references" else "citingPaper"
        results: list[ProviderPaper] = []
        for rank, item in enumerate(data, start=1):
            nested = item.get(key) if isinstance(item, Mapping) else None
            if isinstance(nested, Mapping):
                parsed = _paper(nested, query_id=None, rank=rank)
                if parsed is not None:
                    results.append(parsed)
        return results

    async def references(self, paper_id: str, limit: int) -> list[ProviderPaper]:
        return await self._relations(paper_id, "references", limit)

    async def citations(self, paper_id: str, limit: int) -> list[ProviderPaper]:
        return await self._relations(paper_id, "citations", limit)


__all__ = ["SemanticScholarProvider"]

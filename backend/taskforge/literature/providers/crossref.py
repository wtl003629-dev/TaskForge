"""Crossref Works API adapter for DOI metadata discovery and verification."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

from ...research_protocol import SearchQuery
from ..models import ProviderPaper
from ..normalizer import arxiv_id_from_doi
from .base import ResilientHTTPProvider

_BASE_URL = "https://api.crossref.org"
_TAG = re.compile(r"<[^>]+>")
_SCHOLARLY_TYPES = {
    "book-chapter",
    "dissertation",
    "journal-article",
    "peer-review",
    "posted-content",
    "proceedings-article",
}


def _first(value: object) -> str | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            text = str(item or "").strip()
            if text:
                return text
    text = str(value or "").strip()
    return text or None


def _year(value: object) -> int | None:
    if not isinstance(value, Mapping):
        return None
    parts = value.get("date-parts")
    if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)) or not parts:
        return None
    first = parts[0]
    if not isinstance(first, Sequence) or isinstance(first, (str, bytes)) or not first:
        return None
    return int(first[0]) if isinstance(first[0], int) else None


def _paper(
    value: Mapping[str, Any],
    *,
    query_id: str | None,
    rank: int,
) -> ProviderPaper | None:
    doi = str(value.get("DOI") or "").strip().lower()
    title = _first(value.get("title"))
    publication_type = str(value.get("type") or "").strip().casefold() or None
    if not doi or not title or (
        publication_type is not None and publication_type not in _SCHOLARLY_TYPES
    ):
        return None
    authors: list[str] = []
    raw_authors = value.get("author")
    if isinstance(raw_authors, Sequence) and not isinstance(raw_authors, (str, bytes)):
        for raw in raw_authors:
            if not isinstance(raw, Mapping):
                continue
            name = " ".join(
                part
                for part in (
                    str(raw.get("given") or "").strip(),
                    str(raw.get("family") or "").strip(),
                )
                if part
            )
            if name:
                authors.append(name)
    abstract = _TAG.sub(" ", str(value.get("abstract") or ""))
    abstract = " ".join(abstract.split())
    links = value.get("link")
    pdf_url: str | None = None
    if isinstance(links, Sequence) and not isinstance(links, (str, bytes)):
        for link in links:
            if not isinstance(link, Mapping):
                continue
            content_type = str(link.get("content-type") or "").casefold()
            url = str(link.get("URL") or "").strip()
            if url.startswith("https://") and "pdf" in content_type:
                pdf_url = url
                break
    references: list[str] = []
    raw_references = value.get("reference")
    if isinstance(raw_references, Sequence) and not isinstance(
        raw_references, (str, bytes)
    ):
        references = [
            str(item.get("DOI") or "").strip().lower()
            for item in raw_references[:500]
            if isinstance(item, Mapping) and str(item.get("DOI") or "").strip()
        ]
    return ProviderPaper(
        provider="crossref",
        provider_id=doi,
        title=title,
        authors=authors,
        abstract=abstract,
        year=(
            _year(value.get("published-print"))
            or _year(value.get("published-online"))
            or _year(value.get("issued"))
        ),
        publication_type=publication_type,
        venue=_first(value.get("container-title")),
        publisher=str(value.get("publisher") or "").strip() or None,
        doi=doi,
        arxiv_id=arxiv_id_from_doi(doi),
        source_url=str(value.get("URL") or f"https://doi.org/{doi}").strip(),
        pdf_url=pdf_url,
        citation_count=(
            int(value["is-referenced-by-count"])
            if isinstance(value.get("is-referenced-by-count"), int)
            else None
        ),
        references=references,
        query_id=query_id,
        provider_rank=rank,
    )


class CrossrefProvider(ResilientHTTPProvider):
    name = "crossref"

    async def search(self, query: SearchQuery, limit: int) -> list[ProviderPaper]:
        params: dict[str, object] = {
            "query.bibliographic": query.text,
            "rows": min(max(1, int(limit)), 100),
            "select": (
                "DOI,title,author,abstract,published-print,published-online,issued,"
                "container-title,publisher,type,URL,link,is-referenced-by-count,reference"
            ),
        }
        filters: list[str] = []
        if query.provider_filters.get("year_from"):
            filters.append(f"from-pub-date:{query.provider_filters['year_from']}-01-01")
        if query.provider_filters.get("year_to"):
            filters.append(f"until-pub-date:{query.provider_filters['year_to']}-12-31")
        if filters:
            params["filter"] = ",".join(filters)
        payload = await self._get_json(f"{_BASE_URL}/works", params=params)
        message = payload.get("message") if isinstance(payload, Mapping) else None
        items = message.get("items") if isinstance(message, Mapping) else None
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            return []
        return [
            parsed
            for rank, item in enumerate(items, start=1)
            if isinstance(item, Mapping)
            and (parsed := _paper(item, query_id=query.query_id, rank=rank)) is not None
        ]

    async def get_paper(self, paper_id: str) -> ProviderPaper | None:
        doi = paper_id.removeprefix("https://doi.org/").removeprefix("DOI:")
        payload = await self._get_json(f"{_BASE_URL}/works/{quote(doi, safe='')}")
        message = payload.get("message") if isinstance(payload, Mapping) else None
        return _paper(message, query_id=None, rank=1) if isinstance(message, Mapping) else None

    async def references(self, paper_id: str, limit: int) -> list[ProviderPaper]:
        paper = await self.get_paper(paper_id)
        if paper is None:
            return []
        results: list[ProviderPaper] = []
        for rank, doi in enumerate(paper.references[: min(limit, 20)], start=1):
            item = await self.get_paper(doi)
            if item is not None:
                results.append(item.model_copy(update={"provider_rank": rank}))
        return results

    async def citations(self, paper_id: str, limit: int) -> list[ProviderPaper]:
        # Crossref does not expose a complete reverse-citation traversal in the
        # Works endpoint. Other configured providers supply forward citations.
        return []


__all__ = ["CrossrefProvider"]

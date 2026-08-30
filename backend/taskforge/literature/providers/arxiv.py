"""arXiv Atom API adapter."""

from __future__ import annotations

import re
from xml.etree import ElementTree

from ...research_protocol import SearchQuery
from ..models import ProviderPaper
from .base import ResilientHTTPProvider

_API_URL = "https://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"
_QUERY_STOP = frozenset(
    {
        "about",
        "address",
        "and",
        "apply",
        "are",
        "based",
        "discuss",
        "find",
        "for",
        "from",
        "how",
        "literature",
        "looking",
        "methods",
        "paper",
        "papers",
        "provide",
        "propose",
        "recent",
        "review",
        "research",
        "show",
        "survey",
        "that",
        "the",
        "this",
        "using",
        "what",
        "which",
        "with",
        "you",
        "know",
    }
)


def _text(node: ElementTree.Element, tag: str) -> str:
    child = node.find(tag)
    return " ".join((child.text or "").split()) if child is not None else ""


def _parse_feed(xml: str, *, query_id: str | None) -> list[ProviderPaper]:
    root = ElementTree.fromstring(xml)
    results: list[ProviderPaper] = []
    for rank, entry in enumerate(root.findall(f"{_ATOM}entry"), start=1):
        title = _text(entry, f"{_ATOM}title")
        source_url = _text(entry, f"{_ATOM}id")
        arxiv_id = source_url.rstrip("/").rsplit("/", 1)[-1]
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
        if not title or not arxiv_id:
            continue
        authors = [
            _text(author, f"{_ATOM}name")
            for author in entry.findall(f"{_ATOM}author")
            if _text(author, f"{_ATOM}name")
        ]
        published = _text(entry, f"{_ATOM}published")
        year = int(published[:4]) if published[:4].isdigit() else None
        doi = _text(entry, f"{_ARXIV}doi") or None
        journal = _text(entry, f"{_ARXIV}journal_ref") or None
        pdf_url: str | None = None
        for link in entry.findall(f"{_ATOM}link"):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href")
                break
        results.append(
            ProviderPaper(
                provider="arxiv",
                provider_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=_text(entry, f"{_ATOM}summary"),
                year=year,
                publication_type="preprint",
                venue=journal,
                publisher="arXiv",
                doi=doi,
                arxiv_id=arxiv_id,
                source_url=source_url or f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
                query_id=query_id,
                provider_rank=rank,
            )
        )
    return results


def _search_expression(value: str) -> str:
    """Build valid, moderately selective arXiv API syntax from free text."""

    tokens = [
        token
        for token in re.findall(r"[A-Za-z0-9]+", value.casefold())
        if len(token) > 2 and token not in _QUERY_STOP
    ]
    unique = list(dict.fromkeys(tokens))
    if not unique:
        return f'all:"{value.replace(chr(34), " ")[:200]}"'
    # Requiring three informative terms removes prompt boilerplate while still
    # allowing terminology variants to enter through other providers.
    return " AND ".join(f"all:{token}" for token in unique[:3])


class ArxivProvider(ResilientHTTPProvider):
    name = "arxiv"

    async def search(self, query: SearchQuery, limit: int) -> list[ProviderPaper]:
        text = await self._get_text(
            _API_URL,
            params={
                "search_query": _search_expression(query.text),
                "start": 0,
                "max_results": min(max(1, int(limit)), 100),
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
        )
        return _parse_feed(text, query_id=query.query_id)

    async def get_paper(self, paper_id: str) -> ProviderPaper | None:
        text = await self._get_text(
            _API_URL,
            params={"id_list": paper_id, "start": 0, "max_results": 1},
        )
        papers = _parse_feed(text, query_id=None)
        return papers[0] if papers else None

    async def references(self, paper_id: str, limit: int) -> list[ProviderPaper]:
        return []

    async def citations(self, paper_id: str, limit: int) -> list[ProviderPaper]:
        return []


__all__ = ["ArxivProvider"]

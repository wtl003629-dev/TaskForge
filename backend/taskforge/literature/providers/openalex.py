"""OpenAlex Works API adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

from ...research_protocol import SearchQuery
from ..models import ProviderPaper
from .base import ResilientHTTPProvider

_BASE_URL = "https://api.openalex.org"
_SCHOLARLY_TYPES = {
    "article",
    "book-chapter",
    "dissertation",
    "peer-review",
    "preprint",
    "review",
}


def _abstract(index: object) -> str:
    if not isinstance(index, Mapping):
        return ""
    positioned: list[tuple[int, str]] = []
    for token, positions in index.items():
        if not isinstance(positions, Sequence) or isinstance(positions, str):
            continue
        for position in positions:
            if isinstance(position, int):
                positioned.append((position, str(token)))
    return " ".join(token for _, token in sorted(positioned))


def _identifier(value: object) -> str | None:
    text = str(value or "").strip()
    return text.rsplit("/", 1)[-1] if text else None


def _paper(value: Mapping[str, Any], *, query_id: str | None, rank: int) -> ProviderPaper | None:
    title = str(value.get("title") or value.get("display_name") or "").strip()
    raw_id = str(value.get("id") or "").strip()
    publication_type = str(value.get("type") or "").strip().casefold() or None
    if (
        not title
        or not raw_id
        or value.get("is_retracted") is True
        or (publication_type is not None and publication_type not in _SCHOLARLY_TYPES)
    ):
        return None
    ids = value.get("ids") if isinstance(value.get("ids"), Mapping) else {}
    primary = (
        value.get("primary_location")
        if isinstance(value.get("primary_location"), Mapping)
        else {}
    )
    source = primary.get("source") if isinstance(primary.get("source"), Mapping) else {}
    authorships = value.get("authorships")
    authors: list[str] = []
    if isinstance(authorships, Sequence) and not isinstance(authorships, str):
        for authorship in authorships:
            author = authorship.get("author") if isinstance(authorship, Mapping) else None
            if isinstance(author, Mapping):
                name = str(author.get("display_name") or "").strip()
                if name:
                    authors.append(name)
    best_location = (
        value.get("best_oa_location")
        if isinstance(value.get("best_oa_location"), Mapping)
        else {}
    )
    doi = str(value.get("doi") or ids.get("doi") or "").strip() or None
    arxiv_id = _identifier(ids.get("arxiv"))
    return ProviderPaper(
        provider="openalex",
        provider_id=_identifier(raw_id) or raw_id,
        title=title,
        authors=authors,
        abstract=_abstract(value.get("abstract_inverted_index")),
        year=(
            value.get("publication_year")
            if isinstance(value.get("publication_year"), int)
            else None
        ),
        language=str(value.get("language") or "").strip().casefold() or None,
        publication_type=publication_type,
        venue=str(source.get("display_name") or "").strip() or None,
        publisher=str(source.get("host_organization_name") or "").strip() or None,
        doi=doi,
        arxiv_id=arxiv_id,
        openalex_id=raw_id,
        source_url=(
            str(primary.get("landing_page_url") or value.get("doi") or raw_id).strip()
            or None
        ),
        pdf_url=str(
            best_location.get("pdf_url") or primary.get("pdf_url") or ""
        ).strip() or None,
        citation_count=(
            value.get("cited_by_count")
            if isinstance(value.get("cited_by_count"), int)
            else None
        ),
        references=[
            item.rsplit("/", 1)[-1]
            for item in value.get("referenced_works", [])[:500]
            if isinstance(item, str) and item.strip()
        ],
        query_id=query_id,
        provider_rank=rank,
    )


class OpenAlexProvider(ResilientHTTPProvider):
    name = "openalex"

    async def search(self, query: SearchQuery, limit: int) -> list[ProviderPaper]:
        mode = str(query.provider_filters.get("_search_mode") or "lexical")
        search_parameter = "search.semantic" if mode == "semantic" else "search"
        params: dict[str, object] = {
            search_parameter: query.text[:2_000],
            "per_page": min(max(1, int(limit)), 50 if mode == "semantic" else 100),
            "select": (
                "id,doi,title,display_name,authorships,abstract_inverted_index,"
                "publication_year,primary_location,best_oa_location,ids,"
                "cited_by_count,referenced_works,relevance_score,language,type,is_retracted"
            ),
        }
        filters: list[str] = []
        if query.provider_filters.get("year_from"):
            filters.append(f"from_publication_date:{query.provider_filters['year_from']}-01-01")
        if query.provider_filters.get("year_to"):
            filters.append(f"to_publication_date:{query.provider_filters['year_to']}-12-31")
        language = str(query.provider_filters.get("_language") or "").strip().casefold()
        if language:
            filters.append(f"language:{language}")
        if filters:
            params["filter"] = ",".join(filters)
        payload = await self._get_json(f"{_BASE_URL}/works", params=params)
        data = payload.get("results") if isinstance(payload, Mapping) else None
        if not isinstance(data, Sequence) or isinstance(data, str):
            return []
        return [
            parsed
            for rank, item in enumerate(data, start=1)
            if isinstance(item, Mapping)
            and (parsed := _paper(item, query_id=query.query_id, rank=rank)) is not None
        ]

    async def get_paper(self, paper_id: str) -> ProviderPaper | None:
        payload = await self._get_json(f"{_BASE_URL}/works/{quote(paper_id, safe=':/')}")
        return _paper(payload, query_id=None, rank=1) if isinstance(payload, Mapping) else None

    async def references(self, paper_id: str, limit: int) -> list[ProviderPaper]:
        payload = await self._get_json(f"{_BASE_URL}/works/{quote(paper_id, safe=':/')}")
        ids = payload.get("referenced_works") if isinstance(payload, Mapping) else None
        if not isinstance(ids, Sequence) or isinstance(ids, str):
            return []
        results: list[ProviderPaper] = []
        for rank, reference_id in enumerate(ids[:limit], start=1):
            if not isinstance(reference_id, str):
                continue
            item = await self.get_paper(reference_id)
            if item is not None:
                results.append(item.model_copy(update={"provider_rank": rank}))
        return results

    async def citations(self, paper_id: str, limit: int) -> list[ProviderPaper]:
        normalised = _identifier(paper_id) or paper_id
        payload = await self._get_json(
            f"{_BASE_URL}/works",
            params={"filter": f"cites:{normalised}", "per-page": min(limit, 100)},
        )
        data = payload.get("results") if isinstance(payload, Mapping) else None
        if not isinstance(data, Sequence) or isinstance(data, str):
            return []
        return [
            parsed
            for rank, item in enumerate(data, start=1)
            if isinstance(item, Mapping)
            and (parsed := _paper(item, query_id=None, rank=rank)) is not None
        ]


__all__ = ["OpenAlexProvider"]

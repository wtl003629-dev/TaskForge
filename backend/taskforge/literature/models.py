"""Provider-neutral models for paper discovery."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator

from ..domain import StrictModel, utc_now
from ..research_protocol import PaperCard, SearchQuery


class ProviderPaper(StrictModel):
    provider: Literal["semantic_scholar", "openalex", "arxiv", "crossref"]
    provider_id: str = Field(min_length=1, max_length=512)
    title: str = Field(min_length=1, max_length=2_000)
    authors: list[str] = Field(default_factory=list, max_length=256)
    abstract: str = Field(default="", max_length=50_000)
    year: int | None = Field(default=None, ge=1000, le=3000)
    language: str | None = Field(default=None, min_length=2, max_length=32)
    publication_type: str | None = Field(default=None, max_length=128)
    venue: str | None = Field(default=None, max_length=1_000)
    publisher: str | None = Field(default=None, max_length=1_000)
    doi: str | None = Field(default=None, max_length=512)
    arxiv_id: str | None = Field(default=None, max_length=128)
    semantic_scholar_id: str | None = Field(default=None, max_length=128)
    openalex_id: str | None = Field(default=None, max_length=256)
    source_url: str | None = Field(default=None, max_length=4_096)
    pdf_url: str | None = Field(default=None, max_length=4_096)
    citation_count: int | None = Field(default=None, ge=0)
    references: list[str] = Field(default_factory=list, max_length=500)
    cited_by: list[str] = Field(default_factory=list, max_length=500)
    query_id: str | None = Field(default=None, max_length=240)
    provider_rank: int = Field(default=1, ge=1, le=10_000)

    @field_validator("title", mode="before")
    @classmethod
    def normalise_title(cls, value: object) -> str:
        return " ".join(str(value).split())

    @field_validator(
        "authors", "references", "cited_by", mode="before"
    )
    @classmethod
    def unique_strings(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            return list(
                dict.fromkeys(str(item).strip() for item in value if str(item).strip())
            )
        return value


class ProviderReport(StrictModel):
    provider: str = Field(min_length=1, max_length=128)
    query_count: int = Field(default=0, ge=0)
    result_count: int = Field(default=0, ge=0)
    request_count: int = Field(default=0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    elapsed_ms: float = Field(default=0.0, ge=0.0)
    failure: str | None = Field(default=None, max_length=2_000)


class DiscoveryResult(StrictModel):
    request_id: str = Field(min_length=1, max_length=240)
    queries: list[SearchQuery] = Field(default_factory=list, max_length=6)
    papers: list[PaperCard] = Field(default_factory=list, max_length=100)
    provider_reports: list[ProviderReport] = Field(default_factory=list, max_length=16)
    total_raw_candidates: int = Field(default=0, ge=0)
    query_rewrite_applied: bool = False
    query_rewrite_failure: str | None = Field(default=None, max_length=1_000)
    created_at: datetime = Field(default_factory=utc_now)


__all__ = ["DiscoveryResult", "ProviderPaper", "ProviderReport"]

"""Application service for open scholarly discovery and citation expansion."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from time import perf_counter
from typing import Protocol

from ..research_protocol import (
    LiteratureLanguagePreference,
    LiteratureRequest,
    PaperCard,
    SearchQuery,
)
from .deduplicator import merge_provider_papers
from .models import DiscoveryResult, ProviderPaper, ProviderReport
from .providers.base import LiteratureProvider, ProviderError
from .query_planner import english_academic_bridge, plan_literature_queries
from .query_rewriter import LiteratureQueryRewriter, QueryRewriteError
from .ranker import rank_papers
from .repository import LiteratureAccess, SQLiteLiteratureRepository


class DensePaperEmbedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LiteratureDiscoveryService:
    """Coordinates providers without giving them host or scope authority."""

    def __init__(
        self,
        repository: SQLiteLiteratureRepository,
        providers: Sequence[LiteratureProvider],
        *,
        results_per_query: int = 20,
        provider_query_limits: dict[str, int] | None = None,
        dense_embedder: DensePaperEmbedder | None = None,
        query_rewriter: LiteratureQueryRewriter | None = None,
    ) -> None:
        if len({provider.name for provider in providers}) != len(providers):
            raise ValueError("literature provider names must be unique")
        if results_per_query < 1 or results_per_query > 100:
            raise ValueError("results_per_query must be between 1 and 100")
        self.repository = repository
        self.providers = tuple(providers)
        self.results_per_query = results_per_query
        limits = provider_query_limits or {
            "semantic_scholar": 2,
            "openalex": 3,
            "arxiv": 2,
            "crossref": 2,
        }
        if any(value < 1 or value > 6 for value in limits.values()):
            raise ValueError("provider query limits must be between 1 and 6")
        self.provider_query_limits = dict(limits)
        self.dense_embedder = dense_embedder
        self.query_rewriter = query_rewriter

    @staticmethod
    def _contains_cjk(value: str) -> bool:
        return any("\u3400" <= character <= "\u9fff" for character in value)

    def _queries_for_provider(
        self,
        provider: LiteratureProvider,
        queries: Sequence[SearchQuery],
        *,
        language_preference: LiteratureLanguagePreference = "balanced",
    ) -> list[SearchQuery]:
        limit = self.provider_query_limits.get(provider.name, 2)
        method_first = provider.name in {
            "semantic_scholar",
            "arxiv",
            "crossref",
        }
        ordered = sorted(
            queries,
            key=lambda item: (
                self._contains_cjk(item.text),
                0 if method_first and item.intent == "method" else 1,
                item.priority,
            ),
        )
        cjk_query = next(
            (item for item in queries if self._contains_cjk(item.text)), None
        )
        selected = ordered[:limit]
        if cjk_query is not None and language_preference != "english_first":
            # A Chinese request must keep one native-language retrieval leg.
            # Otherwise English rewrites occupy every provider slot and the
            # candidate set becomes English-only before ranking even begins.
            selected = [
                cjk_query,
                *(item for item in ordered if item.query_id != cjk_query.query_id),
            ][:limit]
        if provider.name != "openalex" or not selected:
            return selected

        # OpenAlex exposes two materially different retrieval engines.  Use
        # the broadest natural-language query for semantic recall and retain a
        # complementary lexical leg for exact terminology.  The marker stays
        # provider-local and does not change the durable query plan.
        semantic = next(
            (
                query
                for query in queries
                if query.intent == "topic" and not self._contains_cjk(query.text)
            ),
            next(
                (query for query in queries if not self._contains_cjk(query.text)),
                selected[0],
            ),
        )
        lexical = next(
            (query for query in ordered if query.intent == "method"),
            selected[-1],
        )
        if language_preference == "english_first":
            openalex_queries = (
                (semantic, "semantic", None),
                (lexical, "lexical", None),
            )
        elif language_preference == "chinese_first":
            openalex_queries = (
                (cjk_query or lexical, "lexical", "zh"),
                (semantic, "semantic", "zh"),
                (semantic, "semantic", None),
            )
        elif cjk_query is not None:
            openalex_queries = (
                (cjk_query, "lexical", "zh"),
                (semantic, "semantic", None),
            )
        else:
            openalex_queries = (
                (semantic, "semantic", None),
                (lexical, "lexical", None),
            )
        routed: list[SearchQuery] = []
        for query, mode, language in openalex_queries[:limit]:
            if query is None:
                continue
            filters = dict(query.provider_filters)
            filters["_search_mode"] = mode
            if language is not None:
                filters["_language"] = language
            else:
                filters.pop("_language", None)
            routed.append(query.model_copy(update={"provider_filters": filters}))
        return routed

    def _dense_scores(
        self,
        request: LiteratureRequest,
        cards: Sequence[PaperCard],
    ) -> dict[str, float] | None:
        if self.dense_embedder is None or not cards:
            return None
        texts = [f"{card.canonical_title}. {card.abstract[:2_000]}" for card in cards]
        query_text = english_academic_bridge(request.query) or request.query
        query = self.dense_embedder.embed_query(query_text)
        documents = self.dense_embedder.embed_documents(texts)
        query_norm = math.sqrt(sum(value * value for value in query))
        scores: dict[str, float] = {}
        for card, document in zip(cards, documents, strict=True):
            document_norm = math.sqrt(sum(value * value for value in document))
            cosine = (
                sum(left * right for left, right in zip(query, document, strict=True))
                / (query_norm * document_norm)
                if query_norm and document_norm
                else 0.0
            )
            scores[card.paper_id] = (max(-1.0, min(1.0, cosine)) + 1.0) / 2.0
        return scores

    @staticmethod
    def _cache_hits(provider: LiteratureProvider) -> int:
        cache = getattr(provider, "cache", None)
        return int(getattr(cache, "hits", 0))

    @staticmethod
    def _request_count(provider: LiteratureProvider) -> int:
        return int(getattr(provider, "request_count", 0))

    async def _search_provider(
        self,
        provider: LiteratureProvider,
        queries: Sequence[SearchQuery],
    ) -> tuple[list[ProviderPaper], ProviderReport]:
        started = perf_counter()
        before_requests = self._request_count(provider)
        before_hits = self._cache_hits(provider)
        results: list[ProviderPaper] = []
        failures: list[str] = []
        for query in queries:
            try:
                results.extend(await provider.search(query, self.results_per_query))
            except (ProviderError, TimeoutError, ValueError) as exc:
                failures.append(f"{query.query_id}: {type(exc).__name__}: {exc}")
            except Exception as exc:  # provider payloads and adapters are untrusted
                failures.append(f"{query.query_id}: {type(exc).__name__}: {exc}")
        return results, ProviderReport(
            provider=provider.name,
            query_count=len(queries),
            result_count=len(results),
            request_count=max(0, self._request_count(provider) - before_requests),
            cache_hits=max(0, self._cache_hits(provider) - before_hits),
            elapsed_ms=(perf_counter() - started) * 1_000,
            failure="; ".join(failures)[:2_000] or None,
        )

    async def discover(
        self,
        access: LiteratureAccess,
        request: LiteratureRequest,
    ) -> DiscoveryResult:
        self.repository.save_request(access, request)
        queries = plan_literature_queries(request)
        rewrite_failure: str | None = None
        rewrite_applied = False
        if self.query_rewriter is not None:
            try:
                rewritten = await self.query_rewriter.rewrite(request, queries)
                seen: set[str] = set()
                combined: list[SearchQuery] = []
                for query in [queries[0], *rewritten, *queries[1:]]:
                    key = query.text.casefold()
                    if key in seen:
                        continue
                    seen.add(key)
                    combined.append(query)
                    if len(combined) == 6:
                        break
                queries = combined
                rewrite_applied = bool(rewritten)
            except QueryRewriteError as exc:
                rewrite_failure = str(exc)
        provider_outputs = await asyncio.gather(
            *(
                self._search_provider(
                    provider,
                    self._queries_for_provider(
                        provider,
                        queries,
                        language_preference=request.language_preference,
                    ),
                )
                for provider in self.providers
            )
        )
        raw = [paper for papers, _ in provider_outputs for paper in papers]
        reports = [report for _, report in provider_outputs]
        merged = merge_provider_papers(raw)
        dense_scores = await asyncio.to_thread(self._dense_scores, request, merged)
        cards = rank_papers(request, merged, dense_scores=dense_scores)
        result = DiscoveryResult(
            request_id=request.request_id,
            queries=queries,
            papers=cards,
            provider_reports=reports,
            total_raw_candidates=len(raw),
            query_rewrite_applied=rewrite_applied,
            query_rewrite_failure=rewrite_failure,
        )
        self.repository.save_discovery(access, result)
        return result

    @staticmethod
    def _provider_id(card: PaperCard, provider_name: str) -> str | None:
        if provider_name == "semantic_scholar":
            return card.semantic_scholar_id or (f"DOI:{card.doi}" if card.doi else None)
        if provider_name == "openalex":
            return card.openalex_id or (
                f"https://doi.org/{card.doi}" if card.doi else None
            )
        if provider_name == "arxiv":
            return card.arxiv_id
        return card.doi

    async def expand_citations(
        self,
        access: LiteratureAccess,
        request_id: str,
        seed_paper_ids: Sequence[str],
        *,
        include_references: bool = True,
        include_citations: bool = True,
        per_seed_limit: int = 20,
        total_limit: int = 100,
    ) -> DiscoveryResult:
        """Return additional candidates; never mutate a ResearchScope."""

        if not seed_paper_ids:
            raise ValueError("at least one seed paper is required")
        if not include_references and not include_citations:
            raise ValueError("at least one citation direction is required")
        per_seed_limit = min(max(1, per_seed_limit), 20)
        total_limit = min(max(1, total_limit), 100)
        request = self.repository.get_request(access, request_id)
        seeds = [
            self.repository.get_paper(access, paper_id) for paper_id in seed_paper_ids
        ]
        raw: list[ProviderPaper] = []
        reports: list[ProviderReport] = []
        for provider in self.providers:
            started = perf_counter()
            before_requests = self._request_count(provider)
            before_hits = self._cache_hits(provider)
            provider_results: list[ProviderPaper] = []
            failures: list[str] = []
            for seed in seeds:
                provider_id = self._provider_id(seed, provider.name)
                if provider_id is None:
                    continue
                try:
                    if include_references:
                        provider_results.extend(
                            await provider.references(provider_id, per_seed_limit)
                        )
                    if include_citations:
                        provider_results.extend(
                            await provider.citations(provider_id, per_seed_limit)
                        )
                except Exception as exc:  # isolate optional citation traversal
                    failures.append(f"{seed.paper_id}: {type(exc).__name__}: {exc}")
            raw.extend(provider_results)
            reports.append(
                ProviderReport(
                    provider=provider.name,
                    query_count=len(seeds),
                    result_count=len(provider_results),
                    request_count=max(
                        0, self._request_count(provider) - before_requests
                    ),
                    cache_hits=max(0, self._cache_hits(provider) - before_hits),
                    elapsed_ms=(perf_counter() - started) * 1_000,
                    failure="; ".join(failures)[:2_000] or None,
                )
            )
        existing_ids = {
            paper.paper_id for paper in self.repository.list_papers(access, request_id)
        }
        cards = [
            card
            for card in rank_papers(request, merge_provider_papers(raw))
            if card.paper_id not in existing_ids
        ][:total_limit]
        result = DiscoveryResult(
            request_id=request_id,
            queries=[],
            papers=cards,
            provider_reports=reports,
            total_raw_candidates=len(raw),
        )
        # Preserve the original query plan in storage while appending ranked candidates.
        stored_queries = plan_literature_queries(request)
        self.repository.save_discovery(
            access,
            result.model_copy(update={"queries": stored_queries}),
        )
        return result

    async def aclose(self) -> None:
        for provider in self.providers:
            close = getattr(provider, "aclose", None)
            if close is not None:
                await close()
        if self.query_rewriter is not None:
            close = getattr(self.query_rewriter, "aclose", None)
            if close is not None:
                await close()


__all__ = ["LiteratureDiscoveryService"]

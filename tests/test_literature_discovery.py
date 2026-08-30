from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from taskforge.literature import (
    LiteratureAccess,
    LiteratureDiscoveryService,
    SQLiteLiteratureRepository,
    plan_literature_queries,
)
from taskforge.literature.deduplicator import merge_provider_papers
from taskforge.literature.models import ProviderPaper
from taskforge.literature.normalizer import arxiv_id_from_doi
from taskforge.literature.providers.arxiv import ArxivProvider
from taskforge.literature.providers.base import ProviderError, ProviderUnavailableError
from taskforge.literature.providers.crossref import CrossrefProvider
from taskforge.literature.providers.openalex import OpenAlexProvider
from taskforge.literature.providers.openalex import _paper as openalex_paper
from taskforge.literature.providers.unpaywall import UnpaywallResolver
from taskforge.literature.ranker import rank_papers
from taskforge.literature.repository import (
    LiteratureAccessError,
    LiteratureNotFoundError,
)
from taskforge.research_protocol import (
    EvidenceCard,
    LiteratureRequest,
    PaperCard,
    ResearchScope,
    ScopeExpansionRequest,
    SearchQuery,
)


def _provider_paper(
    provider: str,
    provider_id: str,
    *,
    query_id: str | None = None,
) -> ProviderPaper:
    values: dict[str, object] = {
        "provider": provider,
        "provider_id": provider_id,
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": ["Patrick Lewis"],
        "abstract": "Retrieval augmented generation combines parametric and non-parametric memory.",
        "year": 2020,
        "publication_type": "journal-article",
        "venue": "NeurIPS",
        "publisher": "ACM",
        "doi": "10.5555/rag.2020",
        "source_url": f"https://example.test/{provider_id}",
        "query_id": query_id,
    }
    if provider == "semantic_scholar":
        values["semantic_scholar_id"] = provider_id
    elif provider == "openalex":
        values["openalex_id"] = provider_id
    elif provider == "arxiv":
        values["arxiv_id"] = provider_id
    return ProviderPaper.model_validate(values)


def test_query_planner_and_deduplication_are_bounded_and_explainable() -> None:
    request = LiteratureRequest(
        request_id="request-1",
        query="agentic retrieval augmented generation",
        research_questions=["How is evidence quality evaluated?"],
        required_terms=["retrieval"],
        excluded_terms=["medical"],
        result_limit=10,
    )
    queries = plan_literature_queries(request)
    assert 3 <= len(queries) <= 6
    papers = merge_provider_papers(
        [
            _provider_paper("semantic_scholar", "s2-1", query_id=queries[0].query_id),
            _provider_paper("openalex", "W1", query_id=queries[1].query_id),
        ]
    )
    assert len(papers) == 1
    assert papers[0].verification_status == "cross_source_verified"
    assert len(papers[0].provider_ranks) == 2
    ranked = rank_papers(request, papers)
    assert ranked[0].relevance_score > 0
    assert "required terms" in ranked[0].relevance_reason


def test_provider_rank_signal_keeps_a_first_party_top_hit_above_noise() -> None:
    request = LiteratureRequest(
        request_id="rank-signal", query="target retrieval method"
    )
    target = merge_provider_papers(
        [
            _provider_paper(
                "semantic_scholar",
                "target",
                query_id="query-target",
            ).model_copy(
                update={
                    "title": "A Different Vocabulary for the Target Method",
                    "abstract": "Semantically related evidence.",
                    "provider_rank": 1,
                }
            )
        ]
    )[0]
    noise = merge_provider_papers(
        [
            _provider_paper("openalex", "noise", query_id="query-noise").model_copy(
                update={
                    "title": "Target Retrieval Method Keyword Collision",
                    "abstract": "Unrelated use of the same words.",
                    "doi": "10.5555/noise",
                    "provider_rank": 50,
                }
            )
        ]
    )[0]

    ranked = rank_papers(request, [noise, target])

    assert ranked[0].paper_id == target.paper_id


def test_query_planner_adds_bounded_english_bridge_for_chinese_academic_need() -> None:
    queries = plan_literature_queries(
        LiteratureRequest(
            request_id="request-zh",
            query="开放域问答中哪项工作用双编码器做稠密段落检索？",
        )
    )

    assert 3 <= len(queries) <= 6
    assert any(
        "open-domain question answering" in query.text
        and "dense passage retrieval" in query.text
        for query in queries
    )


def test_chinese_request_keeps_relevant_chinese_papers_in_top_results() -> None:
    request = LiteratureRequest(
        request_id="request-bilingual",
        query="检索增强生成方法",
        result_limit=5,
    )
    english = [
        PaperCard(
            paper_id=f"paper-en-{index}",
            canonical_title=f"Retrieval Augmented Generation Method {index}",
            abstract="retrieval augmented generation method",
            provider_ranks={"openalex": 1},
        )
        for index in range(6)
    ]
    chinese = [
        PaperCard(
            paper_id=f"paper-zh-{index}",
            canonical_title=f"检索增强生成方法研究 {index}",
            abstract="面向中文论文的检索增强生成方法研究。",
            provider_ranks={"openalex": 1},
        )
        for index in range(2)
    ]
    dense_scores = {
        **{paper.paper_id: 0.9 for paper in english},
        **{paper.paper_id: 1.0 for paper in chinese},
    }

    ranked = rank_papers(request, [*english, *chinese], dense_scores=dense_scores)

    assert len(ranked) == 5
    assert sum("检索增强" in paper.canonical_title for paper in ranked) >= 2


def test_chinese_bigrams_rank_topic_matches_above_unrelated_chinese_titles() -> None:
    request = LiteratureRequest(
        request_id="request-chinese-lexical-ranking",
        query="中文文献召回与重排序",
        result_limit=2,
    )
    relevant = PaperCard(
        paper_id="paper-zh-relevant",
        canonical_title="多语言检索中的中文文献召回与重排序研究",
        abstract="研究跨语言检索和中文论文排序。",
        language="zh",
        provider_ranks={"openalex": 3},
    )
    unrelated = PaperCard(
        paper_id="paper-zh-unrelated",
        canonical_title="小学古诗课堂教学策略研究",
        abstract="分析语文课堂中的教学活动。",
        language="zh",
        provider_ranks={"openalex": 3},
    )

    ranked = rank_papers(request, [unrelated, relevant])

    assert ranked[0].paper_id == relevant.paper_id
    assert ranked[0].relevance_score > ranked[1].relevance_score


def test_chinese_provider_rank_is_discounted_without_native_topic_overlap() -> None:
    request = LiteratureRequest(
        request_id="request-chinese-provider-noise",
        query="多语言检索增强生成中的中文文献召回与重排序",
        result_limit=2,
    )
    provider_noise = PaperCard(
        paper_id="paper-zh-provider-noise",
        canonical_title="小学古诗课堂教学策略研究",
        abstract="分析语文课堂中的教学活动。",
        language="zh",
        year=2025,
        provider_ranks={"crossref": 1},
    )
    topic_match = PaperCard(
        paper_id="paper-en-topic-match",
        canonical_title="Cross-lingual Retrieval Augmented Generation and Re-ranking",
        abstract="Academic literature retrieval with citation provenance.",
        language="en",
        year=2025,
        provider_ranks={"crossref": 1},
    )

    ranked = rank_papers(request, [provider_noise, topic_match])

    assert ranked[0].paper_id == topic_match.paper_id


def test_chinese_first_selects_more_relevant_chinese_papers_without_low_score_fill() -> None:
    request = LiteratureRequest(
        request_id="request-language-ranking",
        query="检索增强生成方法",
        result_limit=6,
    )
    english = [
        PaperCard(
            paper_id=f"ranking-en-{index}",
            canonical_title=f"Retrieval Augmented Generation Method {index}",
            abstract="retrieval augmented generation method",
            year=2024,
            provider_ranks={"openalex": 1},
        )
        for index in range(8)
    ]
    chinese = [
        PaperCard(
            paper_id=f"ranking-zh-{index}",
            canonical_title=f"检索增强生成方法 中文研究 {index}",
            abstract="面向中文论文的方法研究。",
            language="zh",
            year=2024,
            provider_ranks={"openalex": 1},
        )
        for index in range(4)
    ]
    low_score_chinese = PaperCard(
        paper_id="ranking-zh-low",
        canonical_title="低相关中文候选",
        abstract="无关内容",
        language="zh",
        year=2000,
        provider_ranks={"openalex": 1},
    )
    papers = [*english, *chinese, low_score_chinese]

    balanced = rank_papers(
        request.model_copy(update={"language_preference": "balanced"}), papers
    )
    chinese_first = rank_papers(
        request.model_copy(update={"language_preference": "chinese_first"}), papers
    )
    def chinese_count(result: list[PaperCard]) -> int:
        return sum(paper.language == "zh" for paper in result)

    assert chinese_count(chinese_first) >= 3
    assert chinese_count(chinese_first) >= chinese_count(balanced)
    assert "ranking-zh-low" not in {paper.paper_id for paper in chinese_first}


def test_openalex_oversized_reference_list_is_bounded() -> None:
    paper = openalex_paper(
        {
            "id": "https://openalex.org/W1",
            "title": "Large Review",
            "referenced_works": [
                f"https://openalex.org/W{index}" for index in range(700)
            ],
        },
        query_id="query-1",
        rank=1,
    )
    assert paper is not None
    assert len(paper.references) == 500


def test_arxiv_doi_is_resolved_to_the_same_known_item_identifier() -> None:
    assert arxiv_id_from_doi("https://doi.org/10.48550/arXiv.2310.11511") == (
        "2310.11511"
    )
    assert arxiv_id_from_doi("10.1000/example") is None


@pytest.mark.asyncio
async def test_crossref_adapter_maps_doi_metadata_and_query_filters() -> None:
    captured: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["query"] = str(request.url.query)
        return httpx.Response(
            200,
            json={
                "message": {
                    "items": [
                        {
                            "DOI": "10.1000/example",
                            "title": ["A DOI-Verified Retrieval Paper"],
                            "author": [{"given": "Ada", "family": "Researcher"}],
                            "abstract": "<jats:p>Bounded evidence retrieval.</jats:p>",
                            "published-online": {"date-parts": [[2025, 1, 2]]},
                            "type": "journal-article",
                            "container-title": ["Journal of Retrieval"],
                            "publisher": "Scholarly Press",
                            "URL": "https://doi.org/10.1000/example",
                            "is-referenced-by-count": 17,
                            "reference": [{"DOI": "10.1000/prior"}],
                        }
                    ]
                }
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = CrossrefProvider(client=client, max_retries=0)
    papers = await provider.search(
        SearchQuery(
            text="bounded retrieval",
            provider_filters={"year_from": 2024, "year_to": 2026},
        ),
        5,
    )

    assert len(papers) == 1
    assert papers[0].doi == "10.1000/example"
    assert papers[0].year == 2025
    assert papers[0].authors == ["Ada Researcher"]
    assert papers[0].abstract == "Bounded evidence retrieval."
    assert papers[0].publication_type == "journal-article"
    assert papers[0].publisher == "Scholarly Press"
    assert papers[0].references == ["10.1000/prior"]
    assert "from-pub-date" in captured["query"]
    await client.aclose()


@pytest.mark.asyncio
async def test_crossref_adapter_excludes_reports_and_other_non_paper_records() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "items": [
                        {
                            "DOI": "10.1000/company-report",
                            "title": ["Vendor Product Outlook"],
                            "type": "report",
                            "publisher": "Small Company Ltd",
                        },
                        {
                            "DOI": "10.1000/research-paper",
                            "title": ["A Scholarly Retrieval Study"],
                            "type": "journal-article",
                            "publisher": "Scholarly Press",
                        },
                    ]
                }
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = CrossrefProvider(client=client, max_retries=0)

    papers = await provider.search(SearchQuery(text="retrieval"), 10)

    assert [paper.doi for paper in papers] == ["10.1000/research-paper"]
    await client.aclose()


@pytest.mark.asyncio
async def test_openalex_routes_semantic_and_lexical_search_modes() -> None:
    captured: list[dict[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(dict(request.url.params))
        return httpx.Response(200, json={"results": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAlexProvider(client=client, max_retries=0)
    base = SearchQuery(text="long natural language research need")
    await provider.search(
        base.model_copy(update={"provider_filters": {"_search_mode": "semantic"}}),
        80,
    )
    await provider.search(
        base.model_copy(update={"provider_filters": {"_search_mode": "lexical"}}),
        80,
    )

    assert captured[0]["search.semantic"] == base.text
    assert captured[0]["per_page"] == "50"
    assert "search" not in captured[0]
    assert captured[1]["search"] == base.text
    assert captured[1]["per_page"] == "80"
    await client.aclose()


@pytest.mark.asyncio
async def test_openalex_language_filter_is_sent_and_metadata_is_parsed() -> None:
    captured: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "https://openalex.org/WZH",
                        "title": "中文检索增强生成研究",
                        "language": "zh",
                        "publication_year": 2025,
                        "authorships": [],
                        "primary_location": {},
                        "best_oa_location": {},
                        "referenced_works": [],
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAlexProvider(client=client, max_retries=0)
    papers = await provider.search(
        SearchQuery(
            text="检索增强生成",
            provider_filters={"_language": "zh", "_search_mode": "lexical"},
        ),
        10,
    )

    assert captured["filter"] == "language:zh"
    assert papers[0].language == "zh"
    await client.aclose()


def test_merge_provider_papers_preserves_openalex_language() -> None:
    card = merge_provider_papers(
        [
            _provider_paper("openalex", "WZH").model_copy(
                update={"language": "zh"}
            )
        ]
    )[0]

    assert card.language == "zh"


def test_openalex_excludes_reports_and_retracted_works() -> None:
    base = {
        "id": "https://openalex.org/W1",
        "title": "Candidate work",
        "authorships": [],
        "primary_location": {},
        "best_oa_location": {},
        "referenced_works": [],
    }

    assert openalex_paper({**base, "type": "report"}, query_id=None, rank=1) is None
    assert (
        openalex_paper(
            {**base, "type": "article", "is_retracted": True},
            query_id=None,
            rank=1,
        )
        is None
    )


def test_ranker_defensively_excludes_explicit_non_paper_types() -> None:
    request = LiteratureRequest(request_id="quality-gate", query="retrieval")
    paper = PaperCard(
        paper_id="paper",
        canonical_title="Retrieval research",
        publication_type="journal-article",
        provider_ranks={"crossref": 1},
    )
    company_report = paper.model_copy(
        update={
            "paper_id": "company-report",
            "canonical_title": "Company retrieval outlook",
            "publication_type": "report",
        }
    )

    ranked = rank_papers(request, [company_report, paper])

    assert [item.paper_id for item in ranked] == ["paper"]


def test_ranker_excludes_sparse_crossref_only_metadata() -> None:
    request = LiteratureRequest(request_id="sparse-quality-gate", query="retrieval")
    sparse = PaperCard(
        paper_id="sparse-company-record",
        canonical_title="Company retrieval outlook",
        publication_type="journal-article",
        publisher="Small Company Ltd",
        provider_ranks={"crossref:query-1": 1},
        verification_status="metadata_partial",
    )

    assert rank_papers(request, [sparse]) == []


def test_ranker_requires_source_strength_for_crossref_only_journal_records() -> None:
    request = LiteratureRequest(request_id="publisher-quality-gate", query="retrieval")
    base = PaperCard(
        paper_id="trusted-paper",
        canonical_title="Scholarly retrieval study",
        authors=["Ada Researcher"],
        abstract="A retrieval study.",
        publication_type="journal-article",
        venue="Journal of Retrieval",
        doi="10.1000/trusted",
        publisher="Association for Computing Machinery (ACM)",
        provider_ranks={"crossref:query-1": 1},
        verification_status="provider_verified",
    )
    company_record = base.model_copy(
        update={
            "paper_id": "company-record",
            "doi": "10.1000/company",
            "publisher": "Unknown Small Company Ltd",
        }
    )

    ranked = rank_papers(request, [company_record, base])

    assert [item.paper_id for item in ranked] == ["trusted-paper"]


@pytest.mark.asyncio
async def test_arxiv_adapter_builds_explicit_boolean_query() -> None:
    captured: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(
            200,
            text=(
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
            ),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = ArxivProvider(client=client, max_retries=0)
    await provider.search(
        SearchQuery(text="find papers about dense passage retrieval methods"),
        50,
    )

    assert captured["search_query"] == ("all:dense AND all:passage AND all:retrieval")
    await client.aclose()


@pytest.mark.asyncio
async def test_long_retry_after_opens_provider_circuit_without_sleeping() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, headers={"Retry-After": "3600"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAlexProvider(client=client, max_retries=3)
    query = SearchQuery(text="rate limited query")

    with pytest.raises(ProviderUnavailableError, match="budget exhausted"):
        await provider.search(query, 5)
    with pytest.raises(ProviderUnavailableError, match="circuit open"):
        await provider.search(query, 5)

    assert calls == 1
    assert provider.request_count == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_unpaywall_resolver_returns_only_https_pdf_locations() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["email"] == "researcher@example.com"
        return httpx.Response(
            200,
            json={
                "best_oa_location": {"url_for_pdf": "http://unsafe.test/paper.pdf"},
                "oa_locations": [{"url_for_pdf": "https://open.example.org/paper.pdf"}],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    resolver = UnpaywallResolver(
        email="researcher@example.com",
        client=client,
        max_retries=0,
    )
    assert await resolver.resolve_pdf("10.1000/EXAMPLE") == (
        "https://open.example.org/paper.pdf"
    )
    await client.aclose()


def _repository(tmp_path: Path) -> SQLiteLiteratureRepository:
    return SQLiteLiteratureRepository(tmp_path / "literature.sqlite3")


def _seed_request_and_paper(
    repository: SQLiteLiteratureRepository,
    access: LiteratureAccess,
) -> str:
    request = LiteratureRequest(request_id="request-1", query="bounded research")
    repository.save_request(access, request)
    card = merge_provider_papers([_provider_paper("semantic_scholar", "s2-1")])[0]
    repository.upsert_paper(access, card)
    return card.paper_id


def test_repository_enforces_tenant_owner_and_immutable_scope_versions(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    access = LiteratureAccess("tenant-a", "user-a", "conversation-a")
    paper_id = _seed_request_and_paper(repository, access)
    scope = repository.create_scope(
        access,
        ResearchScope(
            scope_id="scope-1",
            tenant_id=access.tenant_id,
            owner_user_id=access.user_id,
            conversation_id=access.conversation_id or "",
            request_id="request-1",
            selected_paper_ids=[paper_id],
            user_intent="Compare the retrieval method.",
        ),
    )
    confirmed = repository.update_scope(
        access,
        scope.scope_id,
        status="confirmed",
        expected_version=1,
    )
    assert confirmed.scope_version == 2
    assert repository.get_scope(access, scope.scope_id, version=1).status == "draft"
    assert repository.get_scope(access, scope.scope_id).status == "confirmed"

    with pytest.raises(LiteratureNotFoundError):
        repository.get_scope(
            LiteratureAccess("tenant-b", "user-a", "conversation-a"),
            scope.scope_id,
        )
    with pytest.raises(LiteratureNotFoundError):
        repository.get_scope(
            LiteratureAccess("tenant-a", "user-b", "conversation-a"),
            scope.scope_id,
        )


def test_evidence_and_expansion_cannot_silently_escape_scope(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    access = LiteratureAccess("tenant-a", "user-a", "conversation-a")
    paper_id = _seed_request_and_paper(repository, access)
    scope = repository.create_scope(
        access,
        ResearchScope(
            scope_id="scope-1",
            tenant_id=access.tenant_id,
            owner_user_id=access.user_id,
            conversation_id=access.conversation_id or "",
            request_id="request-1",
            selected_paper_ids=[paper_id],
            user_intent="Verify the method.",
            allowed_expansion=True,
        ),
    )
    repository.transition_scope_status(access, scope.scope_id, "confirmed")
    repository.transition_scope_status(access, scope.scope_id, "ingesting")
    repository.transition_scope_status(access, scope.scope_id, "ready")
    repository.save_evidence(
        access,
        [
            EvidenceCard(
                evidence_id="evidence-1",
                scope_id=scope.scope_id,
                scope_version=scope.scope_version,
                paper_id=paper_id,
                source="paper://s2-1",
                snippet="The retriever uses dense passage retrieval.",
            )
        ],
    )
    with pytest.raises(LiteratureAccessError):
        repository.save_evidence(
            access,
            [
                EvidenceCard(
                    evidence_id="evidence-outside",
                    scope_id=scope.scope_id,
                    scope_version=scope.scope_version,
                    paper_id="paper-outside",
                    source="paper://outside",
                    snippet="This must never be persisted.",
                )
            ],
        )
    with pytest.raises(LiteratureNotFoundError):
        repository.request_expansion(
            access,
            ScopeExpansionRequest(
                expansion_id="expansion-missing",
                scope_id=scope.scope_id,
                requested_by="evaluator",
                reason="Unknown papers cannot cross the Host boundary.",
                proposed_paper_ids=["paper-missing"],
            ),
        )
    expansion_card = merge_provider_papers(
        [
            _provider_paper("semantic_scholar", "s2-new").model_copy(
                update={
                    "title": "A Verified Expansion Candidate",
                    "doi": "10.5555/expansion.2026",
                    "semantic_scholar_id": "s2-new",
                }
            )
        ]
    )[0]
    repository.upsert_paper(access, expansion_card)
    repository.request_expansion(
        access,
        ScopeExpansionRequest(
            expansion_id="expansion-1",
            scope_id=scope.scope_id,
            requested_by="evaluator",
            reason="Current evidence does not cover the baseline.",
            proposed_paper_ids=[expansion_card.paper_id],
        ),
    )
    assert repository.get_scope(access, scope.scope_id).selected_paper_ids == [paper_id]


class _FakeProvider:
    def __init__(self, name: str, *, failing: bool = False) -> None:
        self.name = name
        self.failing = failing
        self.request_count = 0
        self.cache = None

    async def search(self, query: SearchQuery, limit: int) -> list[ProviderPaper]:
        self.request_count += 1
        if self.failing:
            raise ProviderError("temporary provider failure")
        provider_id = "s2-1" if self.name == "semantic_scholar" else "W1"
        return [_provider_paper(self.name, provider_id, query_id=query.query_id)]

    async def get_paper(self, paper_id: str) -> ProviderPaper | None:
        return None

    async def references(self, paper_id: str, limit: int) -> list[ProviderPaper]:
        return []

    async def citations(self, paper_id: str, limit: int) -> list[ProviderPaper]:
        return []


def test_chinese_request_routes_native_and_english_queries_to_openalex(
    tmp_path: Path,
) -> None:
    provider = _FakeProvider("openalex")
    service = LiteratureDiscoveryService(_repository(tmp_path), [provider])
    queries = plan_literature_queries(
        LiteratureRequest(
            request_id="request-routed-bilingual",
            query="开放域问答中的稠密段落检索",
            language_preference="balanced",
        )
    )

    routed = service._queries_for_provider(provider, queries)

    assert len(routed) == 2
    assert routed[0].provider_filters["_language"] == "zh"
    assert routed[0].provider_filters["_search_mode"] == "lexical"
    assert not service._contains_cjk(routed[1].text)
    assert routed[1].provider_filters["_search_mode"] == "semantic"
    assert "_language" not in routed[1].provider_filters


def test_chinese_first_openalex_keeps_two_chinese_legs_and_english_semantic_leg(
    tmp_path: Path,
) -> None:
    provider = _FakeProvider("openalex")
    service = LiteratureDiscoveryService(
        _repository(tmp_path),
        [provider],
        provider_query_limits={"openalex": 3},
    )
    queries = plan_literature_queries(
        LiteratureRequest(
            request_id="request-routed-chinese-first",
            query="开放域问答中的稠密段落检索",
            language_preference="chinese_first",
        )
    )

    routed = service._queries_for_provider(
        provider,
        queries,
        language_preference="chinese_first",
    )

    zh_routes = [
        query
        for query in routed
        if query.provider_filters.get("_language") == "zh"
    ]
    assert len(zh_routes) >= 2
    assert any(
        query.provider_filters.get("_search_mode") == "semantic"
        and "_language" not in query.provider_filters
        for query in routed
    )


def test_english_first_openalex_does_not_send_chinese_language_filter(
    tmp_path: Path,
) -> None:
    provider = _FakeProvider("openalex")
    service = LiteratureDiscoveryService(
        _repository(tmp_path),
        [provider],
        provider_query_limits={"openalex": 3},
    )
    queries = plan_literature_queries(
        LiteratureRequest(
            request_id="request-routed-english-first",
            query="开放域问答中的稠密段落检索",
            language_preference="english_first",
        )
    )

    routed = service._queries_for_provider(
        provider,
        queries,
        language_preference="english_first",
    )

    assert routed
    assert all("_language" not in query.provider_filters for query in routed)
    assert any(
        query.provider_filters.get("_search_mode") == "semantic"
        for query in routed
    )


@pytest.mark.asyncio
async def test_discovery_isolates_provider_failure_and_persists_results(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    service = LiteratureDiscoveryService(
        repository,
        [
            _FakeProvider("semantic_scholar"),
            _FakeProvider("openalex"),
            _FakeProvider("arxiv", failing=True),
        ],
    )
    access = LiteratureAccess("tenant-a", "user-a", "conversation-a")
    result = await service.discover(
        access,
        LiteratureRequest(
            request_id="request-1", query="retrieval augmented generation"
        ),
    )
    assert len(result.papers) == 1
    assert result.papers[0].verification_status == "cross_source_verified"
    failed = next(
        report for report in result.provider_reports if report.provider == "arxiv"
    )
    assert failed.failure is not None
    reports = {report.provider: report for report in result.provider_reports}
    assert reports["semantic_scholar"].query_count == 2
    assert reports["openalex"].query_count == 2
    assert reports["arxiv"].query_count == 2
    assert reports["semantic_scholar"].request_count == 2
    assert reports["openalex"].request_count == 2
    assert reports["arxiv"].request_count == 2
    assert (
        repository.list_papers(access, "request-1")[0].paper_id
        == result.papers[0].paper_id
    )

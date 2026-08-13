"""Run the Scope-safe paper-research MCP server over stdio or HTTP."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.config import Settings  # noqa: E402
from taskforge.literature.evidence import ScopeBoundEvidenceService  # noqa: E402
from taskforge.literature.providers import (  # noqa: E402
    ArxivProvider,
    CrossrefProvider,
    OpenAlexProvider,
    SemanticScholarProvider,
)
from taskforge.literature.providers.base import (  # noqa: E402
    SQLiteProviderCache,
)
from taskforge.literature.repository import (  # noqa: E402
    LiteratureAccess,
    SQLiteLiteratureRepository,
)
from taskforge.literature.service import LiteratureDiscoveryService  # noqa: E402
from taskforge.persistent_context import SQLiteKnowledgeStore  # noqa: E402
from taskforge.research_mcp import (  # noqa: E402
    ResearchMCPServer,
    create_mcp_app,
    run_stdio,
)
from taskforge.research_retrieval import ResearchRetrievalService  # noqa: E402
from taskforge.routed_knowledge import RoutedKnowledgeStore  # noqa: E402


def build_server() -> ResearchMCPServer:
    settings = Settings()
    tenant = "local"
    user = "research-client"
    knowledge = SQLiteKnowledgeStore(settings.context_sqlite_path)
    routed = (
        RoutedKnowledgeStore(
            knowledge,
            general_text_backend=settings.general_text_backend,
            semantic_model=settings.semantic_model,
            semantic_cache_path=str(settings.semantic_cache_path),
        )
        if settings.retrieval_routing == "profile"
        else knowledge
    )
    repository = SQLiteLiteratureRepository(settings.literature_sqlite_path)
    evidence = ScopeBoundEvidenceService(
        repository,
        ResearchRetrievalService(
            routed,
            dense_embedder=getattr(routed, "_embedder", None),
        ),
    )
    cache = SQLiteProviderCache(
        settings.literature_cache_path,
        ttl_seconds=settings.literature_cache_ttl_seconds,
    )
    options = {
        "cache": cache,
        "timeout_seconds": settings.literature_provider_timeout_seconds,
        "max_retries": settings.literature_provider_max_retries,
    }
    contact_headers = (
        {"User-Agent": f"TaskForge/0.3 (mailto:{settings.literature_contact_email})"}
        if settings.literature_contact_email
        else {}
    )
    headers = {
        **contact_headers,
        **(
            {"x-api-key": settings.semantic_scholar_api_key.get_secret_value()}
            if settings.semantic_scholar_api_key is not None
            else {}
        ),
    }
    discovery = LiteratureDiscoveryService(
        repository,
        [
            SemanticScholarProvider(headers=headers, min_interval_seconds=1.0, **options),
            OpenAlexProvider(headers=contact_headers, min_interval_seconds=0.1, **options),
            ArxivProvider(headers=contact_headers, min_interval_seconds=3.1, **options),
            CrossrefProvider(headers=contact_headers, min_interval_seconds=0.1, **options),
        ],
        results_per_query=settings.literature_results_per_query,
    )
    return ResearchMCPServer(
        evidence,
        LiteratureAccess(tenant, user),
        discovery=discovery,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--tenant", default="local")
    parser.add_argument("--user", default="research-client")
    args = parser.parse_args()
    server = build_server()
    server.principal = LiteratureAccess(args.tenant, args.user)
    if args.transport == "stdio":
        asyncio.run(run_stdio(server))
    else:
        app: FastAPI = create_mcp_app(server)
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

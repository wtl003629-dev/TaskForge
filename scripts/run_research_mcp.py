"""Run the Scope-safe paper-research MCP server over stdio or HTTP."""

from __future__ import annotations

import argparse
import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.app import create_app  # noqa: E402
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
from taskforge.postgres_runtime import set_request_tenant  # noqa: E402
from taskforge.research_mcp import (  # noqa: E402
    ResearchMCPServer,
    create_mcp_app,
    run_stdio,
)
from taskforge.research_retrieval import ResearchRetrievalService  # noqa: E402
from taskforge.routed_knowledge import RoutedKnowledgeStore  # noqa: E402


def build_server() -> tuple[ResearchMCPServer, FastAPI | None]:
    settings = Settings()
    tenant = "local"
    user = "research-client"
    if settings.database_backend == "postgres":
        # Reuse the application composition root so the standalone MCP
        # process cannot accidentally construct SQLite repositories while the
        # API/worker are on PostgreSQL.  The returned app is kept as the
        # lifecycle owner for the shared pool and provider clients.
        backend_app = create_app(settings)
        container = backend_app.state.container
        return (
            ResearchMCPServer(
                container.scope_evidence,
                LiteratureAccess(tenant, user),
                discovery=container.literature_discovery,
            ),
            backend_app,
        )
    knowledge = SQLiteKnowledgeStore(settings.context_sqlite_path)
    routed = (
        RoutedKnowledgeStore(
            knowledge,
            general_text_backend=settings.general_text_backend,
            semantic_model=settings.semantic_model,
            semantic_model_path=(
                str(settings.semantic_model_path)
                if settings.semantic_model_path is not None
                else None
            ),
            semantic_cache_path=str(settings.semantic_cache_path),
            fastembed_model_cache_root=str(settings.fastembed_model_cache_root),
            semantic_batch_size=settings.semantic_batch_size,
            semantic_device=settings.semantic_device,
            bailian_api_key=(
                settings.bailian_api_key.get_secret_value()
                if settings.bailian_api_key is not None
                else None
            ),
            bailian_base_url=settings.bailian_base_url,
            bailian_model=settings.bailian_model,
            bailian_embedding_dimension=settings.bailian_embedding_dimension,
            bailian_batch_size=settings.bailian_batch_size,
            bailian_timeout_seconds=settings.bailian_timeout_seconds,
            bailian_max_retries=settings.bailian_max_retries,
            bailian_cache_path=str(settings.bailian_cache_path),
            bailian_index_name=settings.bailian_index_name,
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
    return (
        ResearchMCPServer(
            evidence,
            LiteratureAccess(tenant, user),
            discovery=discovery,
        ),
        None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--tenant", default="local")
    parser.add_argument("--user", default="research-client")
    args = parser.parse_args()
    server, backend_app = build_server()
    server.principal = LiteratureAccess(args.tenant, args.user)
    if backend_app is not None:
        # The API normally binds this ContextVar in its tenant dependency. The
        # standalone MCP transport has no HTTP dependency layer, so bind the
        # host-selected tenant before provider/embedding cache operations.
        set_request_tenant(args.tenant)
    if args.transport == "stdio":
        async def serve_stdio() -> None:
            if backend_app is None:
                await run_stdio(server)
                return
            async with backend_app.router.lifespan_context(backend_app):
                await run_stdio(server)

        asyncio.run(serve_stdio())
    else:
        app: FastAPI = create_mcp_app(server)
        if backend_app is not None:
            @asynccontextmanager
            async def backend_lifespan(_: FastAPI):
                async with backend_app.router.lifespan_context(backend_app):
                    yield

            app.router.lifespan_context = backend_lifespan
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

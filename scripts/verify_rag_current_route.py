"""Verify that a rollback is bound to the original RAG route and index.

The zero-argument check validates host configuration and resolved behavior.
Supplying a research scope additionally inspects the persisted corpus and runs
one lexical smoke query against the legacy/current knowledge-base identity.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from taskforge.config import Settings  # noqa: E402
from taskforge.knowledge import AccessContext  # noqa: E402
from taskforge.literature import PostgresLiteratureRepository  # noqa: E402
from taskforge.literature.repository import (  # noqa: E402
    LiteratureAccess,
    SQLiteLiteratureRepository,
)
from taskforge.persistent_context import SQLiteKnowledgeStore  # noqa: E402
from taskforge.postgres_context_store import PostgresContextStores  # noqa: E402
from taskforge.postgres_runtime import PostgresRuntime  # noqa: E402
from taskforge.rag_experiment_profile import (  # noqa: E402
    resolve_rag_experiment_profile,
)
from taskforge.research_retrieval import (  # noqa: E402
    ResearchQuery,
    ResearchRetrievalService,
)


def verify_current_route(
    settings: Settings,
    *,
    scope_id: str | None = None,
    tenant_id: str | None = None,
    user_id: str | None = None,
    query: str | None = None,
    expected_evidence_id: str | None = None,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    profile = resolve_rag_experiment_profile(
        settings.rag_active_profile,
        settings.rag_optimized_ablation,
    )
    checks["active_profile_is_current"] = settings.rag_active_profile == "current"
    checks["resolved_profile_is_current_a"] = profile.label == "current-a"
    checks["legacy_document_identity_is_unchanged"] = (
        profile.document_id("research-paper:scope:paper")
        == "research-paper:scope:paper"
    )
    checks["legacy_knowledge_base_identity_is_unchanged"] = (
        profile.knowledge_base_id("research-scope:scope:v1")
        == "research-scope:scope:v1"
    )
    checks["active_embedding_backend_is_legacy"] = (
        settings.general_text_backend == "fastembed"
    )
    checks["active_embedding_model_is_legacy"] = (
        settings.semantic_model == "BAAI/bge-small-en-v1.5"
    )
    checks["active_embedding_cache_is_legacy"] = (
        settings.database_backend == "postgres"
        or settings.semantic_cache_path.name == "embeddings.sqlite3"
    )
    checks["optimized_features_are_disabled"] = not any(
        (
            profile.retrieval_text_enabled,
            profile.parent_aware_rerank_enabled,
            profile.lineage_diversity_enabled,
            profile.structure_aware_chunking_enabled,
        )
    )

    report: dict[str, Any] = {
        "schema_version": "taskforge.rag_current_route_check.v1",
        "status": "passed" if all(checks.values()) else "failed",
        "database_backend": settings.database_backend,
        "active_profile": settings.rag_active_profile,
        "resolved_profile": profile.label,
        "context_sqlite_path": (
            None
            if settings.database_backend == "postgres"
            else str(settings.context_sqlite_path.resolve())
        ),
        "literature_sqlite_path": (
            None
            if settings.database_backend == "postgres"
            else str(settings.literature_sqlite_path.resolve())
        ),
        "embedding": {
            "backend": settings.general_text_backend,
            "model": settings.semantic_model,
            "cache_path": (
                None
                if settings.database_backend == "postgres"
                else str(settings.semantic_cache_path.resolve())
            ),
        },
        "checks": checks,
        "scope_smoke": None,
    }

    if scope_id is None:
        return report
    if not tenant_id or not user_id or not query:
        raise ValueError(
            "--scope-id requires --tenant-id, --user-id, and --query"
        )

    access = LiteratureAccess(tenant_id, user_id, None)
    with ExitStack() as resources:
        if settings.database_backend == "postgres":
            runtime = PostgresRuntime(
                settings.database_url or "",
                min_size=settings.postgres_pool_min_size,
                max_size=settings.postgres_pool_max_size,
                connect_timeout_seconds=settings.postgres_connect_timeout_seconds,
            )
            resources.callback(runtime.close)
            context_stores = PostgresContextStores(
                settings.database_url or "",
                min_size=settings.postgres_pool_min_size,
                max_size=settings.postgres_pool_max_size,
                connect_timeout=int(settings.postgres_connect_timeout_seconds),
                runtime=runtime,
            )
            resources.callback(context_stores.close)
            repository = PostgresLiteratureRepository(
                settings.database_url or "",
                tenant_id=tenant_id,
                runtime=runtime,
            )
            resources.callback(repository.close)
            knowledge = context_stores.knowledge
        else:
            repository = SQLiteLiteratureRepository(settings.literature_sqlite_path)
            knowledge = SQLiteKnowledgeStore(settings.context_sqlite_path)
            resources.callback(knowledge.close)

        scope = repository.get_scope(access, scope_id)
        current_knowledge_base_id = f"research-scope:{scope.scope_id}:v{scope.scope_version}"
        optimized_knowledge_base_id = resolve_rag_experiment_profile(
            "optimized", settings.rag_optimized_ablation
        ).knowledge_base_id(current_knowledge_base_id)
        principal = AccessContext(tenant_id=tenant_id, user_id=user_id)
        current_chunks = tuple(
            chunk
            for chunk in knowledge.visible_chunks(
                principal,
                knowledge_base_ids=(current_knowledge_base_id,),
                latest_only=True,
            )
            if profile.matches(chunk.metadata)
        )
        optimized_chunks = tuple(
            chunk
            for chunk in knowledge.visible_chunks(
                principal,
                knowledge_base_ids=(optimized_knowledge_base_id,),
                latest_only=True,
            )
            if str(chunk.metadata.get("rag_profile") or "").casefold() == "optimized"
        )

        retrieval = ResearchRetrievalService(
            knowledge,
            graph_enabled=False,
            parent_aware_rerank_enabled=False,
            lineage_diversity_enabled=False,
            experiment_profile=profile,
        )
        result = retrieval.search(
            ResearchQuery(
                query=query,
                top_k=8,
                candidate_k=50,
                knowledge_base_ids=(current_knowledge_base_id,),
            ),
            principal,
        )
        returned_ids = [item.evidence_id for item in result.evidence]
        returned_chunk_ids = [item.chunk_id for item in result.evidence]
        current_chunk_ids = {chunk.chunk_id for chunk in current_chunks}
        smoke_checks = {
            "current_corpus_is_present": bool(current_chunks),
            "query_returned_evidence": bool(result.evidence),
            "all_results_resolve_to_current_corpus": bool(result.evidence)
            and set(returned_chunk_ids) <= current_chunk_ids,
            "optimized_identity_differs": (
                optimized_knowledge_base_id != current_knowledge_base_id
            ),
            "expected_evidence_returned": (
                expected_evidence_id is None
                or expected_evidence_id in returned_ids
            ),
        }
        checks.update({f"scope_{key}": value for key, value in smoke_checks.items()})
        report["scope_smoke"] = {
            "scope_id": scope.scope_id,
            "scope_version": scope.scope_version,
            "query": query,
            "current_knowledge_base_id": current_knowledge_base_id,
            "optimized_knowledge_base_id": optimized_knowledge_base_id,
            "current_visible_chunk_count": len(current_chunks),
            "optimized_visible_chunk_count": len(optimized_chunks),
            "returned_evidence_ids": returned_ids,
            "returned_chunk_ids": returned_chunk_ids,
            "checks": smoke_checks,
        }
        report["status"] = "passed" if all(checks.values()) else "failed"
        return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--scope-id")
    parser.add_argument("--tenant-id")
    parser.add_argument("--user-id")
    parser.add_argument("--query")
    parser.add_argument("--expected-evidence-id")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        settings = Settings(_env_file=args.env_file)
        report = verify_current_route(
            settings,
            scope_id=args.scope_id,
            tenant_id=args.tenant_id,
            user_id=args.user_id,
            query=args.query,
            expected_evidence_id=args.expected_evidence_id,
        )
    except Exception as exc:  # operational check must fail closed
        report = {
            "schema_version": "taskforge.rag_current_route_check.v1",
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

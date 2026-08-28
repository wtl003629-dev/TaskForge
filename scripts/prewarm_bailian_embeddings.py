"""Precompute Bailian document embeddings for the persistent knowledge store."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections.abc import Iterable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.config import Settings  # noqa: E402
from taskforge.knowledge import AccessContext, KnowledgeChunk  # noqa: E402
from taskforge.postgres_context_store import PostgresContextStores  # noqa: E402
from taskforge.postgres_embeddings import PostgresEmbeddingCache  # noqa: E402
from taskforge.postgres_runtime import PostgresRuntime  # noqa: E402
from taskforge.semantic_providers import BailianDenseEmbedder  # noqa: E402


def _searchable_texts(database: Path) -> tuple[int, list[str]]:
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            """
            SELECT text_content, metadata_json
            FROM knowledge_chunks
            ORDER BY tenant_id, chunk_id
            """
        ).fetchall()
    texts: list[str] = []
    for text, raw_metadata in rows:
        metadata = json.loads(raw_metadata)
        if metadata.get("retrieval_role") == "parent":
            continue
        retrieval_text = metadata.get("retrieval_text")
        selected = retrieval_text if isinstance(retrieval_text, str) else text
        cleaned = str(selected).strip()
        if cleaned:
            texts.append(cleaned)
    return len(rows), list(dict.fromkeys(texts))


def _searchable_chunk_texts(
    chunks: Iterable[KnowledgeChunk],
) -> tuple[int, list[str]]:
    """Extract the same retrieval text from authorised PostgreSQL chunks."""

    texts: list[str] = []
    scanned = 0
    for chunk in chunks:
        scanned += 1
        metadata = chunk.metadata
        if metadata.get("retrieval_role") == "parent":
            continue
        retrieval_text = metadata.get("retrieval_text")
        selected = retrieval_text if isinstance(retrieval_text, str) else chunk.text
        cleaned = str(selected).strip()
        if cleaned:
            texts.append(cleaned)
    return scanned, list(dict.fromkeys(texts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--tenant", default="local")
    parser.add_argument("--user", default="demo")
    parser.add_argument(
        "--confirm-external-calls",
        action="store_true",
        help="Acknowledge that knowledge text is sent to Alibaba Cloud Model Studio.",
    )
    args = parser.parse_args()
    if not args.confirm_external_calls:
        raise SystemExit(
            "refusing Bailian calls without --confirm-external-calls"
        )
    settings = Settings()
    if settings.bailian_api_key is None:
        raise SystemExit("TASKFORGE_BAILIAN_API_KEY is not configured")
    started = time.perf_counter()
    if settings.database_backend == "postgres":
        runtime = PostgresRuntime(
            settings.database_url or "",
            min_size=settings.postgres_pool_min_size,
            max_size=settings.postgres_pool_max_size,
            connect_timeout_seconds=settings.postgres_connect_timeout_seconds,
        )
        stores = PostgresContextStores(
            settings.database_url or "",
            min_size=settings.postgres_pool_min_size,
            max_size=settings.postgres_pool_max_size,
            connect_timeout=int(settings.postgres_connect_timeout_seconds),
            runtime=runtime,
        )
        embedding_cache = PostgresEmbeddingCache(
            settings.database_url,
            tenant_id=args.tenant,
            runtime=runtime,
        )
        try:
            chunks = stores.knowledge.visible_chunks(
                AccessContext(tenant_id=args.tenant, user_id=args.user),
                latest_only=True,
            )
            scanned, texts = _searchable_chunk_texts(chunks)
            with BailianDenseEmbedder(
                api_key=settings.bailian_api_key.get_secret_value(),
                base_url=settings.bailian_base_url,
                model_name=settings.bailian_model,
                dimension=settings.bailian_embedding_dimension,
                batch_size=settings.bailian_batch_size,
                timeout_seconds=settings.bailian_timeout_seconds,
                max_retries=settings.bailian_max_retries,
                cache_store=embedding_cache,
                index_name=settings.bailian_index_name,
            ) as embedder:
                vectors = embedder.embed_documents(texts)
        finally:
            stores.close()
            embedding_cache.close()
            runtime.close()
        cache = "PostgreSQL vector.embedding_cache"
    else:
        database = (args.database or settings.context_sqlite_path).resolve()
        cache = (args.cache or settings.bailian_cache_path).resolve()
        scanned, texts = _searchable_texts(database)
        with BailianDenseEmbedder(
            api_key=settings.bailian_api_key.get_secret_value(),
            base_url=settings.bailian_base_url,
            model_name=settings.bailian_model,
            dimension=settings.bailian_embedding_dimension,
            batch_size=settings.bailian_batch_size,
            timeout_seconds=settings.bailian_timeout_seconds,
            max_retries=settings.bailian_max_retries,
            cache_path=cache,
            index_name=settings.bailian_index_name,
        ) as embedder:
            vectors = embedder.embed_documents(texts)
    elapsed_ms = (time.perf_counter() - started) * 1_000
    print(
        json.dumps(
            {
                "status": "ok",
                "database_rows_scanned": scanned,
                "unique_searchable_texts": len(texts),
                "vectors": len(vectors),
                "dimension": settings.bailian_embedding_dimension,
                "elapsed_ms": round(elapsed_ms, 1),
                "cache": str(cache),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

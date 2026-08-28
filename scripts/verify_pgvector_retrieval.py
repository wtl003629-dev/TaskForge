"""Compare SQLite/NumPy, exact pgvector, and optional HNSW RAG retrieval.

The query file contains already-generated 1024-dimensional vectors and the
SQLite/NumPy reference ranking frozen from the authoritative cache. This
keeps the check independent of model downloads and deliberately does not
change TaskForge's embedding, chunking, or RAG logic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rag_reference import (  # noqa: E402
    APPLICATION_MODEL_NAME,
    DIMENSION,
    numpy_rank,
    sha256_file,
    sqlite_document_vectors,
)

Query = tuple[str, list[float], tuple[str, ...]]


def _load_queries(path: Path) -> tuple[list[Query], dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    metadata = value if isinstance(value, dict) else {}
    raw_queries = value.get("queries") if isinstance(value, dict) else value
    if not isinstance(raw_queries, list) or not raw_queries:
        raise ValueError("queries JSON must contain a non-empty list")
    result: list[Query] = []
    for index, item in enumerate(raw_queries):
        if not isinstance(item, dict):
            raise ValueError(f"query {index} must be an object")
        query_id = str(item.get("id") or index)
        vector = item.get("vector")
        if not isinstance(vector, list) or len(vector) != DIMENSION:
            raise ValueError(f"query {query_id!r} must contain {DIMENSION} values")
        values = [float(number) for number in vector]
        if any(not math.isfinite(number) for number in values) or not any(values):
            raise ValueError(f"query {query_id!r} contains invalid values")
        raw_reference = item.get("sqlite_numpy_top_k", ())
        if not isinstance(raw_reference, list):
            raise ValueError(f"query {query_id!r} has an invalid SQLite reference")
        reference_ids = tuple(
            str(entry["chunk_id"]) if isinstance(entry, dict) else str(entry)
            for entry in raw_reference
        )
        result.append((query_id, values, reference_ids))
    return result, metadata


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(format(value, ".9g") for value in vector) + "]"


def _fetch(
    cursor: Any,
    *,
    vector: list[float],
    tenant_id: str,
    model: str,
    acl_principals: list[str],
    top_k: int,
    approximate: bool,
    ef_search: int,
) -> list[dict[str, Any]]:
    if approximate:
        cursor.execute(f"SET LOCAL hnsw.ef_search = {ef_search}")
        cursor.execute("SET LOCAL enable_seqscan = off")
    else:
        # HNSW is an index scan. Disable index/bitmap scans for a deterministic
        # exact baseline; the ORDER BY remains the same cosine expression.
        cursor.execute("SET LOCAL enable_indexscan = off")
        cursor.execute("SET LOCAL enable_bitmapscan = off")
    literal = _vector_literal(vector)
    cursor.execute(
        """
        SELECT ke.chunk_id, 1.0 - (ke.embedding <=> %s::vector) AS score
          FROM vector.knowledge_embeddings AS ke
          JOIN taskforge.knowledge_chunks AS kc
            ON kc.tenant_id = ke.tenant_id AND kc.chunk_id = ke.chunk_id
         WHERE ke.tenant_id = %s
           AND kc.tenant_id = %s
           AND ke.model = %s
           AND ke.dimension = %s
           AND kc.acl_json ?| %s::text[]
           AND (kc.valid_from IS NULL OR kc.valid_from <= CURRENT_TIMESTAMP)
           AND (kc.valid_until IS NULL OR kc.valid_until > CURRENT_TIMESTAMP)
         ORDER BY ke.embedding <=> %s::vector, ke.chunk_id
         LIMIT %s
        """,
        (literal, tenant_id, tenant_id, model, DIMENSION, acl_principals, literal, top_k),
    )
    rows = cursor.fetchall()
    return [{"chunk_id": str(row[0]), "score": float(row[1])} for row in rows]


def _recall(actual: list[str], reference: list[str], k: int) -> float:
    expected = set(reference[:k])
    if not expected:
        return 1.0
    return len(set(actual[:k]) & expected) / len(expected)


def _mrr(actual: list[str], reference: list[str], k: int) -> float:
    expected = set(reference[:k])
    for rank, chunk_id in enumerate(actual[:k], start=1):
        if chunk_id in expected:
            return 1.0 / rank
    return 0.0


def _ndcg(actual: list[str], reference: list[str], k: int) -> float:
    expected = set(reference[:k])
    if not expected:
        return 1.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(actual[:k], start=1)
        if chunk_id in expected
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(expected)) + 1))
    return dcg / ideal if ideal else 1.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _metric_block(cases: list[dict[str, Any]], method: str) -> dict[str, float]:
    rows = [case[method] for case in cases]
    result: dict[str, float] = {}
    for k in (5, 10, 50):
        result[f"recall_at_{k}"] = sum(row[f"recall_at_{k}"] for row in rows) / len(rows)
    result["mrr_at_10"] = sum(row["mrr_at_10"] for row in rows) / len(rows)
    result["ndcg_at_8"] = sum(row["ndcg_at_8"] for row in rows) / len(rows)
    result["ndcg_at_10"] = sum(row["ndcg_at_10"] for row in rows) / len(rows)
    result["agent_visible_recall_at_8"] = sum(row["recall_at_8"] for row in rows) / len(rows)
    return result


def compare(
    database_url: str,
    *,
    queries: list[Query],
    tenant_id: str,
    model: str,
    acl_principals: list[str],
    top_k: int,
    ef_search: int,
    sqlite_source_root: Path | None = None,
    include_hnsw: bool = True,
) -> dict[str, Any]:
    if sqlite_source_root is not None:
        documents = sqlite_document_vectors(
            sqlite_source_root,
            tenant_id=tenant_id,
            acl_principals=acl_principals,
        )
        checked_queries: list[Query] = []
        for query_id, vector, stored_reference in queries:
            computed = numpy_rank(vector, documents, top_k=top_k)
            computed_ids = tuple(item["chunk_id"] for item in computed)
            if stored_reference and stored_reference[:top_k] != computed_ids:
                raise ValueError(
                    f"SQLite fixture drift for query {query_id!r}; regenerate the frozen set"
                )
            checked_queries.append((query_id, vector, computed_ids))
        queries = checked_queries
    if any(not reference for _, _, reference in queries):
        raise ValueError("queries must contain sqlite_numpy_top_k or --sqlite-source-root")
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("retrieval comparison requires the PostgreSQL extra") from exc
    cases: list[dict[str, Any]] = []
    exact_latencies: list[float] = []
    hnsw_latencies: list[float] = []
    with psycopg.connect(database_url, autocommit=False) as connection:
        for query_id, vector, reference in queries:
            with connection.cursor() as cursor:
                cursor.execute("SELECT set_config('taskforge.tenant_id', %s, true)", (tenant_id,))
                started = time.perf_counter()
                exact = _fetch(
                    cursor,
                    vector=vector,
                    tenant_id=tenant_id,
                    model=model,
                    acl_principals=acl_principals,
                    top_k=top_k,
                    approximate=False,
                    ef_search=ef_search,
                )
                exact_latency = (time.perf_counter() - started) * 1000.0
            connection.rollback()
            if include_hnsw:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT set_config('taskforge.tenant_id', %s, true)", (tenant_id,))
                    started = time.perf_counter()
                    hnsw = _fetch(
                        cursor,
                        vector=vector,
                        tenant_id=tenant_id,
                        model=model,
                        acl_principals=acl_principals,
                        top_k=top_k,
                        approximate=True,
                        ef_search=ef_search,
                    )
                    hnsw_latency = (time.perf_counter() - started) * 1000.0
            else:
                hnsw = []
                hnsw_latency = 0.0
            connection.rollback()
            exact_ids = [item["chunk_id"] for item in exact]
            hnsw_ids = [item["chunk_id"] for item in hnsw]
            exact_metrics = {
                f"recall_at_{k}": _recall(exact_ids, list(reference), k)
                for k in (5, 8, 10, 50)
            }
            exact_metrics.update(
                mrr_at_10=_mrr(exact_ids, list(reference), 10),
                ndcg_at_8=_ndcg(exact_ids, list(reference), 8),
                ndcg_at_10=_ndcg(exact_ids, list(reference), 10),
            )
            hnsw_metrics: dict[str, float] = {}
            if include_hnsw:
                hnsw_metrics = {
                    f"recall_at_{k}": _recall(hnsw_ids, exact_ids, k)
                    for k in (5, 8, 10, 50)
                }
                hnsw_metrics.update(
                    mrr_at_10=_mrr(hnsw_ids, exact_ids, 10),
                    ndcg_at_8=_ndcg(hnsw_ids, exact_ids, 8),
                    ndcg_at_10=_ndcg(hnsw_ids, exact_ids, 10),
                )
            exact_latencies.append(exact_latency)
            hnsw_latencies.append(hnsw_latency)
            case = {
                "query_id": query_id,
                "sqlite_numpy": list(reference),
                "exact": exact,
                "exact_metrics_vs_sqlite_numpy": exact_metrics,
                "exact_latency_ms": exact_latency,
                "same_top_k_sqlite_numpy_pg_exact": exact_ids == list(reference[:top_k]),
            }
            if include_hnsw:
                case.update(
                    hnsw=hnsw,
                    hnsw_metrics_vs_pgvector_exact=hnsw_metrics,
                    hnsw_latency_ms=hnsw_latency,
                    same_top_k_pg_exact_hnsw=exact_ids == hnsw_ids,
                )
            cases.append(case)
    baseline_case = {
        "sqlite_numpy": {
            **{f"recall_at_{k}": 1.0 for k in (5, 8, 10, 50)},
            "mrr_at_10": 1.0,
            "ndcg_at_8": 1.0,
            "ndcg_at_10": 1.0,
        }
    }
    metrics: dict[str, Any] = {
        "sqlite_numpy": _metric_block([baseline_case] * len(cases), "sqlite_numpy"),
        "postgres_exact_vs_sqlite_numpy": _metric_block(
            [{"exact": case["exact_metrics_vs_sqlite_numpy"]} for case in cases], "exact"
        ),
    }
    latency_ms: dict[str, float] = {
        "postgres_exact_p50": _percentile(exact_latencies, 50),
        "postgres_exact_p95": _percentile(exact_latencies, 95),
    }
    if include_hnsw:
        metrics["postgres_hnsw_vs_postgres_exact"] = _metric_block(
            [{"hnsw": case["hnsw_metrics_vs_pgvector_exact"]} for case in cases], "hnsw"
        )
        latency_ms.update(
            postgres_hnsw_p50=_percentile(hnsw_latencies, 50),
            postgres_hnsw_p95=_percentile(hnsw_latencies, 95),
        )
    result = {
        "tenant_id": tenant_id,
        "model": model,
        "dimension": DIMENSION,
        "top_k": top_k,
        "hnsw_ef_search": ef_search,
        "hnsw_enabled": include_hnsw,
        "acl_principals": acl_principals,
        "query_count": len(cases),
        "bailian_api_calls": 0,
        "database_query_count": len(cases) * (2 if include_hnsw else 1),
        "reference": "SQLite + NumPy exact cosine over the frozen authorized corpus",
        "metrics": metrics,
        "latency_ms": latency_ms,
        "same_top_k_sqlite_numpy_pg_exact_count": sum(
            case["same_top_k_sqlite_numpy_pg_exact"] for case in cases
        ),
        "cases": cases,
    }
    if include_hnsw:
        result["same_top_k_pg_exact_hnsw_count"] = sum(
            case["same_top_k_pg_exact_hnsw"] for case in cases
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--sqlite-source-root", type=Path, default=None)
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--acl", action="append", dest="acl_principals", default=[])
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument(
        "--exact-only",
        action="store_true",
        help="run the pre-HNSW exact consistency gate without querying the HNSW index",
    )
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    queries, metadata = _load_queries(args.queries)
    tenant_id = (args.tenant_id or metadata.get("tenant_id") or "").strip()
    acl_principals = sorted(set(args.acl_principals or metadata.get("acl_principals", [])))
    model = (args.model or metadata.get("model") or APPLICATION_MODEL_NAME).strip()
    top_k = int(args.top_k or metadata.get("top_k") or 50)
    if not tenant_id:
        raise SystemExit("--tenant-id or fixture tenant_id is required")
    if not acl_principals:
        raise SystemExit("--acl or fixture acl_principals is required")
    sqlite_source_root = args.sqlite_source_root.resolve() if args.sqlite_source_root else None
    if sqlite_source_root is not None and isinstance(metadata.get("source_files"), dict):
        expected_files = metadata["source_files"]
        for name in ("context.sqlite3", "embeddings-bailian-v4-1024.sqlite3"):
            expected_hash = expected_files.get(name)
            actual_path = sqlite_source_root / name
            if expected_hash and sha256_file(actual_path) != expected_hash:
                raise SystemExit(
                    f"SQLite source drift for {name}; regenerate the frozen RAG query set"
                )
    if not 1 <= top_k <= 100 or not 1 <= args.ef_search <= 10_000:
        raise SystemExit("--top-k must be 1..100 and --ef-search must be 1..10000")
    database_url = (args.database_url or os.getenv("TASKFORGE_DATABASE_URL", "")).strip()
    if not database_url:
        raise SystemExit("--database-url or TASKFORGE_DATABASE_URL is required")
    report = compare(
        database_url,
        queries=queries,
        tenant_id=tenant_id,
        model=model,
        acl_principals=acl_principals,
        top_k=top_k,
        ef_search=args.ef_search,
        sqlite_source_root=sqlite_source_root,
        include_hnsw=not args.exact_only,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report is not None:
        args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.report.resolve().write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Alibaba Cloud Model Studio embedding route

TaskForge can use Alibaba Cloud Model Studio (Bailian) only for dense
embeddings while retaining its existing PDF parsing, Parent-Child chunking,
BM25, reranking, context expansion, ACL filtering, and Child citation path.

## Active candidate

- Provider: Alibaba Cloud Model Studio OpenAI-compatible embeddings API
- Model: `text-embedding-v4`
- Dimension: 1024
- Synchronous document batch: 10
- Cache: `.taskforge/embeddings-bailian-v4-1024.sqlite3`
- Index identity: `knowledge-bailian-text-embedding-v4-1024-v1`

The local `.env` contains the API key. Never copy the key into this document,
`.env.example`, logs, test reports, or Git history.

## Ingestion and query behavior

New PDF ingestion precomputes every searchable Child vector before reporting
the document as indexed. A failed provider request leaves the document in a
failed ingestion state instead of deferring the failure to the first query.

The query path embeds only the query, builds its in-memory matrix from cached
document vectors, and reuses the existing per-corpus in-memory index. The
SQLite cache is content-addressed by provider, model, dimension, embedding
kind, and text hash.

Prewarm an existing persistent knowledge store:

```powershell
.\.venv\Scripts\python.exe scripts\prewarm_bailian_embeddings.py `
  --confirm-external-calls
```

The acknowledgement is required because searchable knowledge text is sent to
Alibaba Cloud Model Studio.

## Evaluation evidence

The real-PDF locked20 run uses the same English QASPER split, MinerU cache,
Parent-Child parameters, candidate count, visible count, graph setting, and
local reranker ensemble as the FastEmbed control.

| Metric | FastEmbed/BGE-small | Bailian v4 |
| --- | ---: | ---: |
| Recall@10 | 0.7900 | 0.7900 |
| NDCG@8 | 0.5029 | 0.5029 |
| Visible Recall@8 | 0.7567 | 0.7567 |
| Citation localization@8 | 0.8500 | 0.8500 |
| Candidate Child Recall@1 | 0.3067 | 0.2267 |
| Query p50 | 3808 ms | 2462 ms |
| Query p95 | 6232 ms | 3563 ms |

The downstream reranker recovered identical final English results, while the
first-stage candidate head regressed. Therefore this is a latency-oriented,
user-authorized switch, not an accuracy-promotion claim. The four-case
Chinese/English smoke passed, but a representative frozen multilingual set is
still required for a quality claim.

Reports:

- `eval/reports/bailian-text-embedding-v4-locked20-v1.json`
- `eval/reports/bailian-locked20-comparison-v1.json`
- `eval/reports/bailian-multilingual-smoke-v1.json`

## Rollback

Restore these values and restart the API/worker:

```dotenv
TASKFORGE_GENERAL_TEXT_BACKEND=fastembed
TASKFORGE_SEMANTIC_MODEL=BAAI/bge-small-en-v1.5
TASKFORGE_SEMANTIC_CACHE_PATH=.taskforge/embeddings.sqlite3
```

Do not delete the original FastEmbed cache or
`knowledge-fastembed-bge-small-v1` identity. Bailian failures must never use a
FastEmbed query vector against a Bailian document index; rollback switches the
entire embedding route.

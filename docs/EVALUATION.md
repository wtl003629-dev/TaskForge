# TaskForge RAG evaluation protocol

TaskForge never treats an attractive benchmark number as evidence unless the
corpus version, license, relevance labels, retrieval configuration, and raw
per-case predictions are reproducible.

## Dataset policy

The source catalog lives in `eval/rag_datasets.json`. Large third-party data is
not committed. Automated artifacts use immutable source revisions, SHA-256,
HTTPS host allowlists, response-size limits, and explicit license metadata.

Recommended commercial-compatible M0 sources are TAT-DQA, DUDE, QASPER and
MuSiQue (all CC BY 4.0). TAT-QA is the low-cost table/text baseline.
MMLongBench-Doc is optional, CC BY-NC 4.0, and must only be used after an
operator explicitly accepts the non-commercial research restriction. Dataset
license metadata is not legal advice and does not automatically grant rights
to redistribute every upstream source document.

List catalog entries:

```powershell
python scripts\fetch_rag_eval.py --list
```

Fetch the pinned TAT-QA development labels:

```powershell
python scripts\fetch_rag_eval.py --dataset tatqa-dev
```

The optional non-commercial labels require an explicit flag:

```powershell
python scripts\fetch_rag_eval.py `
  --dataset mmlongbench-labels `
  --accept-noncommercial
```

## Locked split and attribution

- Keep tuning and locked-test case IDs in versioned manifests.
- Never tune weights or prompts on the locked split.
- Preserve dataset ID, source revision, license, original case ID, and source
  URL in every normalized case.
- Store downloaded corpora under `.taskforge/eval-cache`; this path is ignored.
- Publish only derived aggregate metrics and permitted small fixtures unless a
  source license explicitly allows redistribution with attribution.

## Retrieval metrics

`taskforge.rag_evaluation` computes deterministic macro metrics over stable
evidence IDs:

- Recall@K: proportion of required evidence units retrieved;
- MRR@K: rank of the first required evidence unit;
- nDCG@K: binary relevance ranking quality;
- per-category Recall@K for table, text, cross-page and multi-hop slices;
- missing predictions are scored as zero, never dropped.

Answer evaluation includes normalized exact match and token F1. Later dataset
adapters add ANLS, numeric execution accuracy, support F1 and unanswerable
accuracy without changing the retrieval contract.

## Required ablation

Every reported improvement must include the same locked cases and these stages:

```text
lexical baseline
-> structure-aware chunks
-> dense retrieval
-> dense + sparse RRF
-> reranker
-> graph route (multi-hop slice only)
```

The raw report must include case-level predictions, failures, latency, index
revision and configuration hashes. Graph retrieval is retained only if it
improves the locked multi-hop slice rather than the aggregate through leakage.

## Verified local runs and claim boundary

On 2026-08-04 the pinned 100-case TAT-QA split was rerun through the exact
three-stage pipeline. The raw manifest, case predictions and metrics are in
`.taskforge/eval-runs/rag-tatqa-locked-20260804-final/` (ignored by Git because
third-party-derived evaluation artifacts can be large):

| Stage | Backend | Recall@10 | p50 | p95 |
|---|---|---:|---:|---:|
| lexical BM25 | Python BM25 | 0.658333 | 166.506 ms | 272.097 ms |
| dense+sparse RRF | local Qdrant | 0.248333 | 275.337 ms | 318.717 ms |
| RRF + lexical rerank | local Qdrant | 0.318333 | 279.657 ms | 340.636 ms |

This run is deliberately labelled `degraded_nonsemantic`: the dense channel
uses a deterministic hash embedder so the whole experiment works without a
model download or API key. The lower hybrid scores are a negative result, not
an improvement claim. Local Qdrant execution and server-side RRF are real;
semantic quality is not. This is an evaluation-script path, not the product
application retrieval path. A FastEmbed or OpenAI embedding experiment must be
rerun on the same locked IDs before promoting a semantic configuration.

This run covers only three of the six required ablation stages: lexical,
hash-vector dense+sparse RRF, and fallback reranking. It does not supply a
production semantic-dense result, a structure-only isolation, or a graph-route
quality result, so the full promotion gate remains unmet. No LLM was called to
produce these retrieval metrics. In particular, module/fake-driver coverage
for the Neo4j adapter is not benchmark evidence.

The Agent trajectory suite is separate:

```powershell
.\.venv\Scripts\python.exe scripts\run_eval.py `
  --output .taskforge\eval-report.json
```

It validates deterministic task success, required/forbidden tools, approval,
terminal state, step budgets and safety hard-fails. The full Python regression
suite additionally covers PDF/table ingestion, tenant/ACL isolation, native
function-call HTTP contracts, fixed multi-role review, crash recovery,
execution leases and human-only final decisions. None of those offline or
mock-HTTP tests proves real-model planning quality; they also do not prove that
optional PostgreSQL, Neo4j, remote Qdrant or MCP services are wired into the
application or available live. Once credentials are
available, run the explicitly billable smoke test and report it under a
separate `live_openai_responses` heading:

```powershell
.\.venv\Scripts\python.exe scripts\run_live_openai_smoke.py --confirm-live-call
```

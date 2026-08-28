# BGE-M3 Embedding Rollout

## Current decision

The locked20 comparison did not clear the promotion gate. The live route is
therefore still the original FastEmbed route:

```ini
TASKFORGE_GENERAL_TEXT_BACKEND=fastembed
TASKFORGE_SEMANTIC_MODEL=BAAI/bge-small-en-v1.5
TASKFORGE_SEMANTIC_CACHE_PATH=.taskforge/embeddings.sqlite3
```

The comparison artifact is
`eval/reports/bge-m3-locked20-comparison-v1.json` (final candidate run:
`eval/reports/bge-m3-locked20-v2-dynamic-length.json`). The BGE-M3 candidate had
the same retrieval metrics as the control but a much higher CPU p95 latency,
so locked100 is intentionally deferred.

## Candidate configuration

BGE-M3 is loaded locally from the D drive and uses dense output only:

```ini
TASKFORGE_GENERAL_TEXT_BACKEND=flagembedding
TASKFORGE_SEMANTIC_MODEL=BAAI/bge-m3
TASKFORGE_SEMANTIC_MODEL_PATH=D:/my-coding/TaskForge/.taskforge/model-cache/flagembedding/bge-m3-pytorch
TASKFORGE_SEMANTIC_BATCH_SIZE=8
TASKFORGE_SEMANTIC_DEVICE=auto
TASKFORGE_SEMANTIC_CACHE_PATH=.taskforge/embeddings-bge-m3-v1.sqlite3
```

The model-qualified dense index name is `knowledge-bge-m3-v1`; the control
index name is `knowledge-fastembed-bge-small-v1`. They must never share a
vector cache or index. PDF parsing, chunking, BM25, reranking, and Top-K stay
unchanged for this experiment.

## Controlled evaluation

Run the evaluator with the frozen real-PDF manifest and the same parent-child
parameters used by the control:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_qasper_direct_upload.py `
  --dataset .taskforge\eval-cache\qasper-dev-v0.3.json `
  --split eval\splits\qasper-dev-clean-holdout-100-v2.json `
  --limit 20 --offset 0 --backend flagembedding `
  --semantic-model BAAI/bge-m3 `
  --semantic-model-path D:\my-coding\TaskForge\.taskforge\model-cache\flagembedding\bge-m3-pytorch `
  --semantic-device cpu --semantic-batch-size 8 `
  --pdf-manifest .taskforge\eval-cache\qasper-clean-holdout-real-pdfs-v3.json `
  --pdf-parser-backend mineru --mineru-base-url http://127.0.0.1:8001 `
  --mineru-expected-version 3.4.4 `
  --mineru-cache-root .taskforge\eval-cache\mineru-shared-v1 `
  --pdf-chunking-mode parent_child `
  --pdf-parent-target-tokens 2000 --pdf-parent-max-tokens 3000 `
  --pdf-child-target-tokens 400 --pdf-child-max-tokens 500 `
  --pdf-child-overlap-tokens 60 --candidate-k 50 --agent-visible-k 8
```

Promote only when Visible Recall@8 and NDCG@8 each improve by at least two
percentage points, Recall@10 and citation localization do not regress, English
does not materially regress, p95 stays within 2x, and cache/index isolation
checks pass.

## Rollback

Restore the three control values above and restart the worker/API process. The
old `embeddings.sqlite3` cache and current knowledge-base identity remain in
place; BGE-M3 artifacts can stay on D for another trial without affecting the
control route.

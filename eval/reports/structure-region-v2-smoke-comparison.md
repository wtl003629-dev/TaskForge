# Structure-region v2 smoke comparison

Status: rejected as a baseline candidate after the balanced smoke test. The
product retrieval default remains the frozen Flat 2000 pipeline.

## Test contract

- Corpus: the frozen 30-English + 30-Chinese real-PDF corpus.
- Parser: MinerU 3.4.4.
- Questions: the same deterministic 10 English + 10 Chinese questions.
- Scope: paper-scoped retrieval.
- Shared retrieval stack: BM25 + Bailian `text-embedding-v4`, RRF, and one
  `qwen3-rerank` pass.
- Control: Flat 2000 characters with zero overlap.
- Candidate: unchanged Flat primary lane plus sparse 1,800-character
  structure-region auxiliary chunks.

## Results

| Metric | Flat control | Structure v2 | Delta |
| --- | ---: | ---: | ---: |
| Recall@10 | 0.8750 | 0.8333 | -0.0417 |
| Recall@50 | 0.9500 | 0.9500 | 0.0000 |
| MRR@10 | 0.6389 | 0.5198 | -0.1190 |
| NDCG@10 | 0.6360 | 0.4270 | -0.2090 |
| p50 latency | 265.0 ms | 463.0 ms | +198.0 ms |
| p95 latency | 274.5 ms | 4,531.6 ms | +4,257.2 ms |

Language Recall@10 changed from 0.8500 to 0.8667 for English and from 0.9000
to 0.8000 for Chinese. English Recall rose slightly, but its MRR@10 fell from
0.6167 to 0.4230. Chinese MRR@10 fell from 0.6611 to 0.6167.

## Finding

The auxiliary regions occupied an average of 5.5 of the English Top-10 slots
and 5.0 of the Chinese Top-10 slots. They often represented content already
covered by a Flat chunk, so the reranker spent scarce top positions on a second
projection of the same source region. Recall@50 stayed unchanged while ranking
quality dropped sharply, showing that the main failure was candidate
competition and duplicate evidence rather than corpus coverage.

The full 177-question evaluation was intentionally not run because the
candidate failed the predeclared smoke-test early-stop criterion.

## Artifacts

- Candidate: `mixed-mineru-dual-v2-30zh-30en-bailian-paper-smoke20-v1.json`
- Control: `mixed-mineru-flat2000-30zh-30en-bailian-paper-smoke20-dual-v2-control-v1.json`

# PDF RAG pipeline

TaskForge treats a PDF as a document container, not as a text file. The
production ingestion boundary is:

```text
PDF bytes
  -> Native Parser (born-digital fast path)
  -> Parse Quality Gate
  -> MinerU sidecar when OCR/layout/table/visual recovery is required
  -> parser-neutral DocumentBlock[]
  -> Parent -> Child projection
  -> title/section-enriched Child BM25 + Dense -> multi-query RRF
  -> Child reranker -> bounded Parent-aware reranker -> lineage diversity
  -> query-centred Child evidence window
  -> Parent read, Child citation identity
```

## Direct text and OCR

Born-digital PDFs are first parsed locally with `pypdf` and `pdfplumber`.
Scanned/image-only PDFs do not silently produce an empty index: the Native
Parser reports `ocr_required`, and `auto` routing sends them to MinerU when the
sidecar is configured. A native parse also records embedded image XObjects. If
an image has no trusted textual representation, quality remains
`visual_pending` instead of claiming that the figure was understood.

The quality report records page coverage, garbled characters, repeated
headers/footers, reading-order warnings, orphan captions, empty tables,
unparsed visuals, OCR use, parser name and parser version. Quality thresholds
are development routing thresholds and must be frozen independently for a
locked evaluation.

## MinerU boundary

MinerU runs as a separate service. TaskForge talks to `/health` and
`/file_parse`; raw JSON is cached by PDF SHA-256 plus parser configuration.
The configured runtime version must equal `TASKFORGE_MINERU_EXPECTED_VERSION`.
MinerU schemas never enter retrieval directly: both legacy `content_list` and
`content_list_v2` are normalized into the same `DocumentBlock` contract.

Recommended locked settings for the default D-drive pipeline deployment:

```dotenv
TASKFORGE_PDF_PARSER_BACKEND=auto
TASKFORGE_MINERU_BASE_URL=http://127.0.0.1:8001
TASKFORGE_MINERU_EXPECTED_VERSION=3.4.4
TASKFORGE_MINERU_BACKEND=pipeline
TASKFORGE_MINERU_PARSE_METHOD=auto
TASKFORGE_MINERU_EFFORT=high
TASKFORGE_MINERU_CONCURRENCY=2
```

`hybrid-engine` is an optional higher-cost ablation and requires downloading
MinerU's separate VLM model bundle. It is not needed for the baseline parser,
and it is not the default on an 8 GiB GPU. Figure/chart semantics remain a
separate visual-evidence stage so the text-only DeepSeek writer never receives
pixels directly.

Use loopback HTTP or HTTPS. TaskForge rejects non-loopback plain HTTP and
enforces PDF size, page, response, retry and concurrency bounds. The repository
includes a GPU sidecar under `deploy/mineru/`, derived from the official
release recipe but pinned to `mineru[core]==3.4.4`. Do not use the official
example's moving `latest` image for locked results. Both its container
healthcheck and TaskForge verify `/health.version == 3.4.4`.

MinerU's current license is a custom MinerU Open Source License based on Apache
2.0 with additional commercial thresholds and an attribution obligation for
third-party online services. Review the exact version's license and show the
required attribution before offering this as an online service.

## Hierarchy and evidence identity

- `Block` is the parser-neutral unit with page, bbox, reading order, type,
  text/structured content, content hash and optional visual artifact locator.
- `Parent` is the section-scale reading context (target 2,000 tokens,
  maximum 3,000 tokens).
- Flat mode uses a 2,000-character target and zero overlap by default; optional
  same-page whole-Block overlap and raw sliding windows are experimental only.
- `Child` is the retrieval and citation unit (target 400 tokens, maximum 500,
  60-token whole-block overlap). Its authoritative body remains separate from
  the deterministic title/section-enriched text used for indexing.
- Tables, charts, formulas, images, code and algorithms are atomic. Captions
  bind to their adjacent atomic block.
- Only retrieval units enter BM25/Dense retrieval. In Parent–Child mode,
  `paper_read` expands a Child to its same-document, same-version Parent;
  citation verification deliberately checks the Child, not the wider Parent.

### Optional Flat-primary + Child-auxiliary route

The production control remains unchanged. For an isolated experiment, set
`TASKFORGE_PDF_CHUNKING_MODE=hybrid` and
`TASKFORGE_RESEARCH_DUAL_ROUTE_ENABLED=true`. Ingestion then stores a Flat
primary lane and a Parent/Child auxiliary lane in the same document version.
The retriever searches Flat Top-30 and Child Top-20 independently, merges the
two candidate lists, and keeps a small Flat head in the returned ranking as a
deterministic query-level fallback. Parent context is attached only after a
Child is selected; it does not replace the Child citation text. The six
dual-route knobs are lane budgets, first/optional tail multilingual rerank
budgets, and Flat fallback-head size, so they can be tuned without changing
the legacy path.

The dual route is opt-in and validated as a pair: enabling it without
`hybrid` chunking, or selecting `hybrid` without enabling the route, fails
configuration validation. This prevents a document from being written with
two lanes while the search service silently uses the old single-lane path.

For an offline direct-upload evaluation, the equivalent flags are:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_qasper_direct_upload.py `
  --rag-profile optimized --rag-ablation c `
  --pdf-chunking-mode hybrid --dual-route `
  --dual-route-flat-candidate-k 30 --dual-route-child-candidate-k 20 `
  --dual-route-flat-head-k 2
```

The command is an experiment only; it does not promote the optimized profile
or rewrite the current production index.

For CJK/cross-lingual Dual evaluation, configure a real multilingual
cross-encoder.  The supported local FastEmbed checkpoint is
`jinaai/jina-reranker-v2-base-multilingual`; the evaluator can select it with:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_qasper_direct_upload.py `
  --rag-profile optimized --rag-ablation c `
  --pdf-chunking-mode hybrid --dual-route `
  --multilingual-reranker-backend fastembed `
  --multilingual-reranker-model jinaai/jina-reranker-v2-base-multilingual `
  --dual-route-rerank-candidate-k 10 `
  --dual-route-min-confidence 0.35
```

On CPU, the default first-pass rerank budget is 10 candidates. The remaining
Dual candidate tail stays available for recall and score fusion without
invoking the expensive cross-encoder. A second pass over up to 20 tail
candidates is available for explicitly selected difficult-query experiments;
the production-safe default is
`TASKFORGE_RESEARCH_DUAL_ROUTE_TAIL_RERANK_CANDIDATE_K=0`.

If the optional multilingual checkpoint is absent or cannot be loaded, Dual
does not invoke the configured English reranker.  It keeps the Flat/RRF order
and applies the query-level Flat fallback when coverage is missing or the
fused confidence is below the configured threshold.  Parent-aware sorting is
disabled for Dual; Parent data is reserved for final context expansion.

The previous locked 100-case QASPER A/B promoted the flat baseline:
with identical MinerU 3.4.4 parsing, local embedding/reranker, Candidate@50,
an eight-card Agent head and the original Query, flat reached paragraph
Recall@1/5/10/50 `0.2728/0.7367/0.8625/0.9830`. Agent-visible Recall@8 was
`0.8250`, with no additional presentation-window loss. The full Parent–Child
locked report reached `0.7022/0.8447` at Recall@5/10 and Agent-visible
Recall@8=`0.7938`. Those figures describe the pre-Parent-aware ranking chain,
not the current implementation. Parent–Child is now the application default;
its new title enrichment, Parent-aware rerank and lineage diversity have not
yet been evaluated and no uplift is claimed.
These are paragraph-aligned metrics; page overlap is not used. The current
top-8 reports are [`flat-v2-top8`](../eval/reports/qasper-real-pdf-locked100-current-original-flat-v2-top8.json)
and [`parent-child-v2-top8`](../eval/reports/qasper-real-pdf-locked100-current-original-parent-child-v2-top8.json).

A local no-API 20-case chunk-strategy screen tested Flat 500/1000/1500/2000-
character targets, whole-Block overlap, and raw same-page sliding windows.
Flat 500 reduced Recall@5 to `0.5350`; sliding 500/100 and 1000/200 failed the
Gold alignment gate; sliding 2000/400 improved Recall@10 but reduced Recall@5.
No strategy improved both Recall@5 and Recall@10 in that historical screen, so
Flat 2000/0 remains the frozen comparison baseline. The screen's @5/@10 ordering is unchanged by the production Top-8
cut; the locked v2 report additionally measures Agent-visible Recall@8.
The machine-readable screen is
[`qasper-pdf-chunk-strategy-screen20-v1`](../eval/reports/qasper-pdf-chunk-strategy-screen20-v1.json).

Parent–Child parameter screening also tested smaller and larger Parents,
Children and overlaps. The original 2,000/3,000 Parent + 400/500 Child +
60-token overlap remained the best of the tested hierarchy settings, but still
trailed Flat 2000/0 at both Recall@5 and Recall@10. The hierarchy report is
[`qasper-pdf-parent-child-screen20-v2-top8`](../eval/reports/qasper-real-pdf-screen20-parent2000-3000-child400-500-overlap60-v2-top8.json),
and the promotion decision is frozen in
[`chunking-gate-v2-top8`](../eval/reports/qasper-pdf-chunking-gate-v2-top8.json).

The text-only DeepSeek model never receives raw image pixels. It receives only
trusted caption/OCR/table/LaTeX/chart-analysis text plus page/bbox/artifact
provenance. A separately configured OpenAI-compatible VLM may convert an image
or chart into a validated `VisualEvidence` object containing axes, legends,
data points, nodes, edges, a textual rendering, confidence and warnings. Its
result is cached by image SHA-256, exact model ID and prompt version. Failure is
preserved as `visual_pending`; it is never replaced by an invented description.

## Query and ranking order

`ResearchQuery` accepts the original query plus up to two variants. Each query
runs BM25 and Dense retrieval over an enriched projection containing document
title, heading path and, for obvious backward references, a bounded previous
context. RRF merges retrieval methods and query variants before the first
Cross-Encoder pass over Candidate@K. The leading candidates then load validated
same-document/same-version Parents, build a bounded previous/current/following
window, and receive a second score. Configurable weights fuse the Child,
Parent-context and original retrieval signals; a soft lineage step discourages
one Parent from occupying the whole Agent evidence head. Any missing Parent or
second-pass failure falls back to the first ranking. A normal Child remains the
returned citation unit, while `paper_read` expands it to its Parent without
changing citation identity.

Language routing is explicit and does not replace the validated English
profile. English-only corpora use the existing English embedding and
Cross-Encoder. When the query or indexed paper contains enough CJK text, the
retriever selects the optional multilingual embedding/reranker pair
(`intfloat/multilingual-e5-large` and
`jinaai/jina-reranker-v2-base-multilingual`) when configured. The dense-index
cache includes the model identity, and a Chinese request without that pair is
reported as `multilingual_fallback` rather than being counted as a multilingual
result. English-only operator heuristics are disabled on the multilingual
route until they have their own multilingual evaluation.

The configured local model pair has passed both route selection and a real-model
two-case smoke test: Chinese and cross-lingual queries select `multilingual`
and rank the expected local evidence first, while the unit regression confirms
that an English-only corpus keeps the `english` model path. This is not a
Chinese benchmark or a quality-uplift claim; see
[`multilingual-retrieval-smoke-v2`](../eval/reports/multilingual-retrieval-smoke-v2.json)
and [`multilingual-routing-smoke-v1`](../eval/reports/multilingual-routing-smoke-v1.json).

`TASKFORGE_RESEARCH_QUERY_EXPANSION_MODE` controls the ablation:

- `original`: original query only (portable default);
- `keyword`: original plus one deterministic local keyword/entity query;
- `synonym`: original plus one constrained semantic paraphrase;
- `full`: original plus semantic paraphrase plus keyword/entity query.

The optional expander rejects variants that drop detected entities, numbers,
negations or comparison terms. Expansion failure falls back to the original
query and is recorded; it does not block retrieval. No model-specific absolute
score threshold is used for evidence sufficiency.

## Evaluation order

All previously published QASPER upload numbers based on page overlap are
invalid for paragraph retrieval. The values and every comparison derived from
them must not be used as current results.

The new sequence is:

1. prepare and hash real PDFs with `prepare_qasper_real_pdfs.py`;
2. run `evaluate_qasper_direct_upload.py` on a paper-disjoint locked split;
3. inspect Gold-to-Child alignment coverage and exclude/report unaligned units;
4. report only paragraph Recall@1/5/10/50;
5. compare `original`, `keyword`, `synonym`, and `full` with the same PDFs, split, parser
   version, frozen query-variant manifest, Child IDs and ranking budgets;
6. attribute misses to candidate generation, reranking or presentation window;
7. run answer and four-Agent evaluation only after retrieval is frozen.

```powershell
.\.venv\Scripts\python.exe scripts\prepare_qasper_real_pdfs.py --help
.\.venv\Scripts\python.exe scripts\generate_qasper_query_variants.py --help
.\.venv\Scripts\python.exe scripts\evaluate_qasper_direct_upload.py --help
.\.venv\Scripts\python.exe scripts\evaluate_qasper_corpus_native.py --help
```

The evaluator must score only Child content aligned to Gold paragraphs. Page
overlap is diagnostic provenance only and is never an acceptance metric.
The default alignment gate requires at least 90% aligned Gold units and 90%
alignment-eligible cases. If it fails, headline Recall values are written as
`null`; lower-bound and eligible-subset values remain explicitly diagnostic.
Expanded-query runs require `--query-variants` with a manifest bound to the
locked split SHA-256; live LLM rewrites are never generated inside a scored
run.

The locked 100-case deterministic `keyword` ablation produced exactly the same
Candidate@50, Recall@1/5/10/50 and failure-stage counts as the original Query,
so the default remains `original`. A zero-API 20-case screen of the
pre-generated synonym manifest also matched the original Query at Recall@5,
Recall@10 and Agent-visible Recall@8, so a full synonym run was not started.
The reports are [`qasper-query-expansion-locked100-v1`](../eval/reports/qasper-query-expansion-locked100-v1.json)
and [`qasper-query-expansion-synonym-screen20-v1`](../eval/reports/qasper-query-expansion-synonym-screen20-v1.json).

The current MinerU 3.4.4 100-case run passes the alignment gate: Gold-unit
alignment is `97.03%` and `97/100` cases are fully eligible. With the original
Query, flat chunks, Candidate@50 and the local Cross-Encoder, formal
Recall@1/5/10/50 is `0.2728/0.7367/0.8625/0.9830`; the eight-window Agent-visible
diagnostic is `0.8250`. The same locked inputs with Parent–Child produce
`0.7022/0.8447` at Recall@5/10, so that ablation remains opt-in. The
machine-readable reports are linked above; no page-overlap score is used.

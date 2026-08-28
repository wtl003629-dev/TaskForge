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

### TAT-QA task scope: provided context versus global discovery

TAT-QA is a question-answering benchmark over a *given* hybrid context: one
table and its associated paragraphs are supplied with the question.  TaskForge
therefore reports two different tasks and never promotes one against the other:

- `provided_hybrid_context` applies the case's input `parent_document_id` as a
  trusted pre-ranking filter.  It measures evidence selection and numerical QA
  inside the context supplied by TAT-QA; the scope is not derived from answer or
  relevance labels.
- `global_discovery` searches every normalized TAT-QA context in one knowledge
  base.  It is an additional open-corpus stress test.  Many original questions
  contain no company or report identity, so this score is not comparable to
  TAT-QA paper results.

On the parent-document-disjoint 102-case validation split, the frozen
structured Pair pipeline scores Recall@10/Candidate@50 `0.6716/0.8088` in
`global_discovery`, versus `0.9902/0.9902` in
`provided_hybrid_context`.  The latter has p95 `36.8 ms` and category Recall@10
of `1.0` for arithmetic, count, multi-span and table, and `0.9545` for text.
The large difference is task scope, not a reranker gain.  The promotion gate
rejects comparisons whose retrieval scope differs.

The explicit CLI selector is:

```powershell
.\.venv\Scripts\python.exe scripts\run_rag_experiment.py `
  --dataset tatqa `
  --tatqa-context-mode provided_hybrid_context `
  --development-sweep `
  --stages lexical_bm25 `
  --output .taskforge\eval-runs\tatqa-provided-context
```

Document recall is not operand selection.  The provided-context run can place
the complete table in Top-10 while still failing to identify the rows and cells
needed by an arithmetic program.  TaskForge therefore persists
`retrieved_table_units_by_hit`, including the original evidence-hit rank and
row/cell metadata, instead of inferring coordinate rank from de-duplicated
document IDs.

The separate query-slot diagnostic uses only the question and supplied table:
it classifies a constrained operator, extracts metric/year terms, and scores
row labels, column headers and cell mentions under a fixed ten-cell budget.
TagOp's pinned heuristic `mapping` output is consulted only afterwards for
scoring.  On the 77 mapping-eligible cases in the document-disjoint validation
split, complete-table context coverage is `0.9870`, explicit coordinate hits
from the retriever are `0.0`, and the label-free query-slot selector reaches
cell/row recall `0.8858/0.9464`.  The cell result is `0.9658` for arithmetic,
`1.0` for table lookup, `0.6553` for multi-span and `0.5417` for count.  These
are diagnostic figures, not official TAT-QA gold metrics: TagOp generated the
mappings heuristically, and the hidden split remains unopened.

For the 47 arithmetic/count cases with mapped table operands, the constrained
host executor can execute every gold derivation and reproduce every released
answer.  Complete supplied context therefore has a program-oracle upper bound
of `1.0`; the fixed ten-cell query selector fully covers the mapped operands in
`0.8723` of cases, giving the same `0.8723` retrieved-slots + gold-program
upper bound.  This uses the gold derivation and is explicitly not end-to-end
accuracy.  It isolates the remaining gap: program generation and the seven
count/multi-span/ambiguous slot-selection misses, rather than table discovery
or calculator capability.

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_tatqa_mapping_retrieval.py `
  --annotations .taskforge\eval-cache\tatqa_dataset_tagop_train_870accc4.json `
  --predictions .taskforge\eval-runs\tatqa-group-validation-102-pair-provided-context-hit-aligned-v2\predictions.jsonl `
  --document-k 10 `
  --evidence-hit-k 10 `
  --query-slot-k 10 `
  --output eval\reports\tatqa-group-validation-102-pair-query-slots-program-oracle-v4.json
```

The selector is an isolated table-context branch.  It does not rewrite the
query sent to global text, PDF or cross-document retrieval, so its current
diagnostic improvement cannot mask a regression in those scenarios.  Any
future default integration still has to pass their locked retrieval gates.

A query-aware logistic cell reranker was trained only on the 11,834-case,
1,971-parent fit complement, with a separate 600-case parent-disjoint tuning
split.  The frozen 2:1 learned/heuristic blend was then evaluated with three
distinct training seeds on the 77 mapping-eligible validation cases.  All
three seeds moved macro cell recall from `0.8858` to `0.9459`, including count
`0.5417 -> 0.8958` and multi-span `0.6553 -> 0.8182`, but span fell
`1.0 -> 0.9474`.  That `-0.0526` regression exceeds the `0.03` category guard,
so the candidate is rejected and the frozen label-free heuristic remains the
default.  The validation result was not used for another weight sweep and the
hidden split was not opened.  The machine-readable decision is
`eval/reports/tatqa-slot-reranker-validation-gate-v1.json`.

### Coordinate-preserving TAT-QA table cleaning

TAT-QA cleaning is a separate, label-free search-representation ablation.  It
never replaces the released table or its zero-based coordinates.  Cleaned rows
carry `row_source_indices`, while raw `table_rows` remain available for TagOp
mapping and official evidence evaluation.  The implemented rules are NFKC and
whitespace normalization, empty-row removal, repeated-header removal, safe
consecutive exact-duplicate folding, hierarchical header merging, missing
value normalization, and numeric/currency/percent/scale metadata.  Global
non-consecutive duplicates are deliberately preserved because repeated
`Total` rows can be legitimate business data.

On the pinned train JSON, the v2 audit covers 2,201 tables and 20,728 rows.  It
removes 14 empty rows, 14 repeated header rows and one consecutive duplicate,
while preserving 110 non-consecutive duplicate rows.  It detects 925 two-level
headers and records 44,786 numeric cells.  The source and rule-level counts are
in `eval/reports/tatqa-train-table-cleaning-audit-v2.json`.

The same code revision, validation cases, retrieval budgets and structured
Pair stage were run with only `tatqa_table_cleaning` changed:

| Context | Metric | Control | Cleaned | Delta |
|---|---|---:|---:|---:|
| Global discovery | Candidate@50 | 0.8088 | 0.8088 | 0.0000 |
| Global discovery | Recall@10 | 0.6716 | 0.6716 | 0.0000 |
| Global discovery | row/cell Recall@10 | 0.6806 | 0.6806 | 0.0000 |
| Provided context | Candidate@50 | 0.9902 | 0.9902 | 0.0000 |
| Provided context | Recall@10 | 0.9902 | 0.9902 | 0.0000 |
| Provided context | row/cell Recall@10 | 0.9884/0.9861 | 0.9884/0.9861 | 0.0000 |

All category deltas are zero.  Global p95 was `154.6 -> 164.3 ms` and provided
context p95 was `32.2 -> 30.1 ms`; these single-run latency changes are not a
statistical performance claim.  The cleaning branch passes non-regression but
has no measurable quality gain, so it remains opt-in and does not replace the
frozen baseline.  The artifact-verifying decision is
`eval/reports/tatqa-table-cleaning-ablation-gate-v2.json`.  Enable it only for
an explicit ablation with `--tatqa-table-cleaning`; other datasets cannot set
the flag, so QASPER and cross-document retrieval are unaffected by construction.

Answer evaluation can opt into the same frozen selector by prepending its
ten-cell plan to the unchanged full evidence.  Evidence IDs and retrieval order
do not change, and the model is told to verify the plan against the full table.
The live CLI defaults TAT-QA to its official `provided_hybrid_context` task;
`--tatqa-context-mode global-discovery` remains an explicit stress-test option.
This flag has been contract-tested with a fake provider only; no paid run has
been made for the new context transform.

```powershell
.\.venv\Scripts\python.exe scripts\run_rag_answer_eval.py `
  --dataset tatqa `
  --tatqa-context-mode provided-hybrid-context `
  --tatqa-query-slot-context `
  --tatqa-query-slot-k 10 `
  --retriever tatqa_frozen_pair_rerank `
  --answer-contract online-cited-v1 `
  --output .taskforge\eval-runs\tatqa-query-slots-live `
  --confirm-live-call
```

Answer evaluation keeps the historical normalized exact match and token F1, but
a score is no longer allowed to hide the failure stage. New v1.3 answer runs
record the gold evidence IDs, the complete candidate list (`R_candidate`), the
actual Top-K retrieval list (`R_topk`), IDs presented inside the context budget
(`P`), model citation IDs (`C`), end-to-end latency, provider token usage when
reported, fallback use and a stable failure bucket. The report exposes
`candidate_retrieval`, `evidence_retrieval` (true Top-K), and
`presented_context` as separate metric groups. Provider failures remain case
rows scored as zero rather than deleting the whole run.
Each row also records `failure_stage` with the first observable cause:
`candidate_missing`, `top10_ranking_failure`, `context_coverage_failure`,
`reasoning_failure`, `format_or_scale_failure`, or `execution_error`.

For TAT-QA, the released evaluator's numeric/scale and multi-span semantics are
also available as an offline rescore. This does not call a model and must be
run on existing immutable predictions before interpreting online F1:

```powershell
.\.venv\Scripts\python.exe scripts\recalculate_tatqa_answer_metrics.py `
  --run .taskforge\eval-runs\tatqa-online-full100-pair-v1-20260810 `
  --output eval\reports\tatqa-online-full100-pair-v1-offline-recalc.json
```

The upstream implementation is [TAT-QA's `tatqa_metric.py`](https://github.com/NExTplusplus/TAT-QA/blob/master/tatqa_metric.py).

### Frozen online negative baseline (2026-08-10)

The 100-case train-heldout Pair and BM25 runs are frozen audit artifacts, not
tuning data. After offline rescoring with the released TAT-QA metric:

| Retriever | Candidate@50 | Retrieval Recall@10 | TAT-QA EM | TAT-QA F1 |
|---|---:|---:|---:|---:|
| BM25 | 0.6367 | 0.4733 | 0.1700 | 0.2740 |
| Pair rule | 0.6667 | 0.5183 | 0.1800 | 0.2830 |

The Pair rule improves retrieval by `+0.0450` and official answer F1 by
`+0.0090`, but this is not a promotion result: the generic online token-F1
comparison was `0.2862` versus `0.2887`, and the heldout set must not be used for
further parameter tuning. The repeat-30 run was stopped before publication.

The offline failure-stage projection for Pair is: 41 cases with a missing
Candidate@50, 15 with a Top-10 ranking miss, and 34 with full presented evidence
but an answer/reasoning miss; 10 are exact-match successes. This confirms that
candidate generation is the largest first-order bottleneck, while the
remaining answer gap requires a separate reasoning/scale path.

### Parent-document disjoint validation and hidden split

The two online artifacts above remain audit-only. A tuning run must use a
different split whose cases are selected in whole parent documents, so a
question from the same report cannot appear in both the tuning and audit
sets. The balanced offline validation manifest is:

```text
eval\splits\tatqa-train-group-validation-102-v2.json
```

It contains 102 cases from 17 parent documents and has zero parent overlap with
`tatqa-train-online-heldout-100-v1.json` and
`tatqa-train-online-repeat-30-v1.json`, and meets the declared category floors
(`arithmetic>=30, count>=8, multi-span>=12, table>=20, text>=15`).

A separate hidden manifest is available at
`eval\splits\tatqa-train-group-hidden-102-v2.json`; it is disjoint from the
validation, online-heldout, and repeat manifests and has the same category
floors. It is reserved and must not be inspected for tuning. The earlier
`*-100-v2` and `*-102-v1` files are superseded provisional artifacts and must
not be used for promotion.

The execution order is: (1) rescore existing artifacts offline; (2) run the
same BM25/dense/RRF/rerank matrix on this validation split; (3) inspect
Candidate@50, true Recall@10, presented-context recall, and official TAT-QA
EM/F1 by category; (4) only then change candidate generation. The heldout
100-case audit split remains untouched until the final, one-time confirmation.

The first balanced-validation retrieval comparison is now frozen:

| Stage | Recall@10 | Candidate@50 | Arithmetic@10 | Count@10 | Multi-span@10 | Table@10 | Text@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.5735 | 0.7451 | 0.4872 | 0.4375 | 0.5000 | 0.4750 | 0.9091 |
| BM25 schema/row/cell RRF | 0.5490 | 0.7745 | 0.4359 | 0.4375 | 0.5000 | 0.5000 | 0.8636 |

The multi-granular branch improves candidate recall by `+0.0294`, but loses
`0.0245` Recall@10 and regresses arithmetic and text by more than the `0.03`
protection bound. It is therefore a rejected candidate, not a promoted default.
The run artifacts are
`.taskforge/eval-runs/tatqa-group-validation-102-bm25-offline-v3` and
`.taskforge/eval-runs/tatqa-group-validation-102-table-row-cell-offline-v4`.

The pure-Python BM25 implementation now caches corpus statistics by the full
tenant/ACL/version/knowledge-base scope and scores through an inverted posting
accumulator. It constructs explanations only for the exact Top-N candidates.
The BM25 retrieval payload is byte-for-byte equivalent to the pre-optimization
run, while p50 fell from about `498 ms` to `38 ms`. Multi-representation branch
initialization was also changed from quadratic rescanning to linear grouping;
the same table run fell from about nine minutes wall time to 50 seconds with
identical retrieval output.

The query-typed structured lineage candidate performs materially better on the
same validation split:

| Stage | Recall@10 | Candidate@50 | Arithmetic@10 | Count@10 | Multi-span@10 | Table@10 | Text@10 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.5735 | 0.7451 | 0.4872 | 0.4375 | 0.5000 | 0.4750 | 0.9091 | 74 ms |
| Structured lineage tail | 0.6569 | 0.8088 | 0.6410 | 0.5000 | 0.6154 | 0.5000 | 0.9091 | 206 ms |
| Existing fixed Pair head | **0.6716** | **0.8088** | 0.6410 | **0.6875** | 0.6154 | 0.5000 | 0.9091 | 222 ms |

The structured candidate clears the Candidate@50, overall Recall@10, and table
targets without a category regression. The unchanged Pair head raises Count@10
above the target without altering candidate recall or the other category
scores. This is a stage-three quality pass on the validation split, not a final
promotion: p95 is roughly three times the newly optimized BM25 control, the
hidden split remains unopened, and Pair remains a deterministic rule baseline
rather than a learned reranker. Artifacts are
`.taskforge/eval-runs/tatqa-group-validation-102-structured-lineage-offline-v1`
and `.taskforge/eval-runs/tatqa-group-validation-102-pair-rerank-offline-v1`.

The default live CLI uses `cited_v1`:

```powershell
.\.venv\Scripts\python.exe scripts\run_rag_answer_eval.py `
  --dataset multihop-rag `
  --retriever bm25_source_coverage_rrf `
  --answer-contract cited-v1 `
  --output .taskforge\eval-runs\answer-source-coverage-cited-v1 `
  --confirm-live-call
```

For MultiHop-RAG, `bm25_source_coverage_rrf` is the default non-semantic
retriever and the CLI defaults `max_chunks_per_document` to `1`. Plain `bm25`
remains available as the comparison baseline. Other datasets keep their own
defaults; in particular, this document-diversity setting is not imposed on
TAT-QA tables.

QASPER's original 100-case artifact is retained as a historical general-text
control only. Its cache is extracted
from the official `qasper-train-dev-v0.3.tgz` archive, and the locked split
admits only answerable questions with exact paragraph evidence:

```powershell
\.venv\Scripts\python.exe scripts\run_rag_experiment.py `
  --dataset qasper `
  --qasper-input .taskforge\eval-cache\qasper-dev-v0.3.json `
  --qasper-locked-split eval\splits\qasper-dev-general-100-v1.json `
  --development-sweep `
  --stages lexical_bm25 `
  --chunking
```

For the fair hybrid candidate, run the same locked split once with each of
`bm25`, `qdrant_dense`, `bm25_dense_rrf`, and
`bm25_dense_rrf_rerank`. The semantic runs must include `--semantic`; otherwise
the dense branch is deterministic feature hashing and is valid only as an
offline contract test. `qdrant_rrf*` is retained as a legacy control: its sparse
branch is hashed term frequency, not BM25.

```powershell
.\.venv\Scripts\python.exe scripts\run_rag_answer_eval.py `
  --dataset multihop-rag `
  --retriever bm25_dense_rrf `
  --semantic `
  --answer-contract cited-v1 `
  --output .taskforge\eval-runs\answer-bm25-dense-rrf-cited-v1 `
  --confirm-live-call
```

The model must return exactly `{"answer": "...", "citation_ids": ["..."]}`.
The host computes `V = G ∩ R ∩ P ∩ C`; unknown or invented citations stay in
the precision denominator. A short-answer claim counts as strictly supported
only when the JSON parses, the answer exact-matches gold and `V` is non-empty.
This is deliberately named **strict gold-evidence grounding**: a correct gold
document ID is not a semantic-entailment judgment over every generated claim.
`bare_v1` remains available only for historical answer baselines and reports
grounding as `not_measured`.

Agentic answer evaluation disables hidden automatic context retrieval and uses
`neighbor_window=0`, so only explicit `knowledge_search` receipts count as
presented evidence. Host fallback is off by default. Enabling
`--allow-agentic-host-fallback` discloses every affected case and makes the run
fail the promotion gate; it is useful for diagnosis, not for the headline
agentic score.

Later dataset adapters can add ANLS, numeric execution accuracy and
unanswerable accuracy without changing the R/P/C contract.

## Required ablation

Every reported improvement must include the same locked cases and these stages:

```text
lexical baseline
-> structure-aware chunks
-> dense retrieval
-> BM25 + dense RRF
-> reranker
-> graph route (multi-hop slice only)
```

The raw report must include case-level predictions, failures, latency, index
revision and configuration hashes. Graph retrieval is retained only if it
improves the locked multi-hop slice rather than the aggregate through leakage.

## Paired promotion gate

Do not compare two console summaries by eye. Answer-eval v1.1 manifests pin the
ordered case IDs, dataset/index identity, prompt and tool-schema hashes,
effective configuration, budgets, source hashes and artifact hashes. The comparison command verifies
those artifacts before it computes a fixed-seed paired bootstrap confidence
interval:

```powershell
.\.venv\Scripts\python.exe scripts\compare_rag_answer_runs.py `
  --comparison retriever `
  --baseline bm25=.taskforge\eval-runs\answer-bm25-cited-v1 `
  --candidate hybrid=.taskforge\eval-runs\answer-hybrid-cited-v1 `
  --output .taskforge\eval-runs\promotion-hybrid.json
```

The default hard gate requires all of the following: token-F1 `+0.03`, evidence
recall `+0.03`, paired 95% CI lower bound at least zero, no category with at
least ten cases regressing more than `0.03`, candidate p95 at most `2×` the
baseline, zero execution errors and zero fallback cases. For `--comparison
agentic`, the pair must be naive→agentic with the same retriever and the p95
limit is `2.5×`. Old artifacts missing the v1.1 disclosures are rejected rather
than silently treated as passing runs.

The bootstrap measures paired case uncertainty, not model-run variance. For a
resume claim, run each live configuration at least three times and report every
trial; do not describe a single paid run as stable or statistically final.

### Retrieval-only cross-scenario gate

Retrieval changes are promoted with the stricter retrieval gate, rather than by
copying the headline aggregate from a TAT-QA run. It verifies the hashes and
ordered case IDs in `manifest.json`, the dataset and locked-split identity,
`top_k`/`candidate_k`, ACL/tenant filters, chunking settings, stage predictions,
and the metrics/predictions artifact hashes. Recall@10 and Candidate Recall@50
are recomputed from the per-case `relevant_ids` and `retrieved_ids`; a reported
aggregate cannot override a malformed row or an inaccessible filter probe.

```powershell
\.venv\Scripts\python.exe scripts/compare_rag_retrieval_runs.py `
  --stage bm25_source_coverage_rrf `
  --baseline locked=.taskforge\eval-runs\baseline `
  --candidate table-plan=.taskforge\eval-runs\table-plan `
  --output .taskforge\eval-runs\promotion-retrieval.json
```

The default policy allows at most a `0.01` absolute dataset Recall@10 or
Candidate@50 drop, at most `0.03` degradation for categories with at least ten
cases, and a `1.2x` warm-query p95 limit. A fixed-seed paired bootstrap must not
have a negative Recall@10 lower bound. Use
`--require-identical-retrieval` for an unaffected profile: every case's ordered
retrieved ID sequence must remain identical. This is the mechanism that stops a
TAT-QA table change from silently degrading MultiHop, PDF, QASPER, or a security
probe; those profiles are run as separate candidates against their own locked
baseline and every candidate must pass.

For a profile-local budget such as table-numeric's `1.2s` ceiling, use
`--max-p95-ms 1200` together with the default `--max-p95-ratio 1.2`. When both
are present, both limits are enforced: an isolated table profile may have a
larger absolute budget, but it cannot silently multiply its own baseline
latency beyond the relative budget.

Changing a model, reranker, or retrieval strategy is allowed only as the single
declared ablation variable. The command still records the stage descriptor and
artifact hashes in its JSON report; do not mix model changes with chunking,
filters, budgets, or query expansion in one promotion claim.

For an in-run ablation whose stage names differ, replace `--stage` with
`--baseline-stage lexical_bm25 --candidate-stage bm25_tatqa_query_plan_rrf`.
The same dataset artifact can then be compared without copying or rewriting
prediction rows.

For the full cross-scenario check, create a matrix whose entries point at the
locked baseline and candidate artifacts:

```powershell
\.venv\Scripts\python.exe scripts/compare_rag_retrieval_matrix.py `
  --matrix eval/retrieval-promotion-matrix.json `
  --output .taskforge\eval-runs\promotion-matrix.json
```

Each matrix scenario has `name`, `baseline`, `candidate` and an optional
`policy`. `policy.profile` restricts the comparison to rows selected into that
profile; use `require_identical_retrieval: true` for an unaffected profile.
Use `max_p95_ms` for an isolated profile such as `table_numeric`; pair it with
`max_p95_ratio` when the same scenario also needs a relative guard. The matrix
returns success only when every listed profile passes.

The synthetic PDF smoke set has no sampled split file; its suite hash plus the
manifest's ordered case IDs is treated as its locked-set identity. It remains
a smoke gate, not evidence for production-scale PDF recall.

The original `eval/retrieval-promotion-matrix-20260809.json` remains a historical
negative-control matrix and is intentionally red: it retains slow or
inconclusive candidates. The current promotion matrix is
`eval/retrieval-promotion-matrix-current-20260810.json`; its six active
scenarios plus the whole-TAT-QA target row (seven total) pass in
`.taskforge/eval-runs/promotion-matrix-current-20260810.json`.
It includes explicit whole-MultiHop floors (Recall@10 `≥0.90`, Candidate@50
`≥0.98`, p95 `≤940.5ms`) in addition to paired non-regression checks. The
QASPER 100-case general-text control is pinned at Recall@10 `0.220595`, and the
PDF smoke slice passes. The QASPER number is a lexical baseline, not a claim
about answer quality; its adapter uses only exact full-text paragraph evidence.

#### Product-aligned retained-capability baseline (2026-08-11)

The primary offline gate is now
`eval/retrieval-retained-capabilities-20260811.json`. It measures the retrieval
capabilities the TaskForge product actually promises, rather than treating the
unofficial TAT-QA global-discovery stress task as the product's headline KPI.
All four scenarios passed the hash-verified, case-paired gate in
`eval/reports/retrieval-retained-capabilities-20260811.json`:

| Retained capability | Locked data | Control Recall@10 / Candidate@50 | Current Recall@10 / Candidate@50 | Interpretation |
|---|---|---:|---:|---|
| supplied-context table/numeric evidence | TAT-QA 102 document-disjoint validation cases | `0.9755 / 0.9755` | `0.9902 / 0.9902` | structured table branch passes; p95 `72.3 ms` |
| general long-document evidence | QASPER 100 cases | `0.2206 / 0.3621` | same frozen BM25 floor | regression guard only; quality remains weak |
| identifiable cross-document evidence | MultiHop-RAG 100 held-out cases, `cross_document` profile | `0.8408 / 0.9744` | `0.9199 / 0.9893` | source-coverage routing passes with positive paired CI |
| PDF ingestion/table routing | 12 synthetic PDF cases | `1.0000 / 1.0000` | `1.0000 / 1.0000` | smoke test only, not production-scale quality evidence |

The QASPER run also tested `bm25_parent_child` as a non-promoted exploration.
It moved Recall@10 from `0.220595` to `0.227054` and Candidate@50 from
`0.362054` to `0.394970`, while p95 increased from `88.4 ms` to `191.3 ms`.
That small gain is recorded as a negative/inconclusive result; it is not the
default and the general-text row is deliberately described as a minimum floor.

Run the retained-capability gate with:

```powershell
.\.venv\Scripts\python.exe scripts\compare_rag_retrieval_matrix.py `
  --matrix eval\retrieval-retained-capabilities-20260811.json `
  --output eval\reports\retrieval-retained-capabilities-20260811.json
```

`global_discovery` artifacts and the older TAT-QA-heavy promotion matrix remain
available for architecture stress testing and historical comparisons. They
must not be used to reject a product release that keeps the four retained
capabilities above their own locked gates, and they must not be compared with
official TAT-QA answer leaderboard numbers.

For TAT-QA diagnosis, run the non-promotable oracle report before changing the
retriever:

```powershell
\.venv\Scripts\python.exe scripts/diagnose_tatqa_retrieval.py `
  --run .taskforge\eval-runs\tatqa-locked-table-tail-seed50-20260809 `
  --stage bm25_dense_tatqa_query_table_candidate_rrf `
  --dataset .taskforge\eval-cache\tatqa_dataset_dev.json `
  --output .taskforge\eval-runs\tatqa-oracle-diagnostic.json
```

The report separates O0 real evidence retrieval, O1 gold-parent routing, O2
table/paragraph section accessibility and O3 perfect Top-10 reordering of the
existing Candidate@50 set. O1--O3 are upper bounds only; they are not allowed
in a headline score or a promotion gate. It also emits a deterministic
operation/year/comparator/scale QueryPlan so a later numeric retriever can be
ablated independently of chunking and ranking.

The locked shape-aware run produced
`.taskforge/eval-runs/tatqa-oracle-shape-aware-final-routing-20260810.json`:
O0 real Recall@10 is `0.753333`, O1 parent accessibility `0.97`, O2 section
accessibility `0.93`, and O3 candidate-reordering upper bound `0.90`. The gap
between O0 and O3 is therefore a ranking/selection opportunity, not evidence
that gold labels can be used in retrieval.

The corresponding experimental stage is `bm25_tatqa_query_plan_rrf`. It is
opt-in and combines a compact QueryPlan rewrite with a conditional table-cell
branch (cell-branch RRF weight is configurable and defaults to `0.25`); the
default config does not include it. The current best optimization candidate
uses weight `0.5`: Recall@10 `0.638333`, Candidate@50 `0.753333`, Count@10
`0.541667`, with no category regressions and profile-local p95 `443.4ms`. Its
paired Recall@10 CI is still inconclusive (`[-0.010, 0.075]`), so the strict
gate does not promote it. It is also not a global default: its p95 is about
`3.41x` the same-run BM25 profile, so the shared-profile ratio gate correctly
rejects it. The artifact is
`.taskforge/eval-runs/promotion-query-plan-cell05-strict-20260809.json`.

Because the installed FastEmbed model registry does not expose BGE-M3, the
first model-only fallback used the supported `BAAI/bge-base-en-v1.5`. Its
table-profile RRF raises Candidate@50 from `0.744318` to `0.799242`, but Recall
CI remains inconclusive, Count and text categories regress, and nDCG decreases;
it is therefore not promoted. The raw comparison is in
`.taskforge/eval-runs/promotion-tatqa-bge-base-profile-20260809.json`.
Increasing the BM25 RRF weight to `2.0` removes the Candidate@50 gain and
still leaves Count/Text regressions, so simple branch weighting is rejected.
The weighted artifact is `.taskforge/eval-runs/tatqa-opt-bge-base-profile-bm25x2-20260809`.

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

The current runner can additionally execute `qdrant_dense`,
`bm25_dense_rrf`, and `bm25_dense_rrf_rerank` from the versioned stage list.
RRF uses configurable `rrf_k`, BM25 weight, and dense weight; both branches
receive the identical pre-ranking security/version filter and candidate budget.
When BM25 metadata fields are configured, the dense document representation
also includes those fields while the returned evidence payload remains the
original chunk. These capabilities do not retroactively change the verified
2026-08-04 negative result above; they require a separate semantic locked-set
run.

That semantic retrieval run was completed on 2026-08-09 against the same
100-case TAT-QA split with structure-aware chunking and
`BAAI/bge-small-en-v1.5`. Its artifact is
`.taskforge/eval-runs/rag-tatqa-bm25-dense-rrf-semantic-20260809/`:

| Stage | Recall@10 | nDCG@10 | p95 |
|---|---:|---:|---:|
| BM25 | 0.658333 | 0.529914 | 177.110 ms |
| semantic dense | 0.628333 | 0.489587 | 61.809 ms |
| BM25 + dense RRF | 0.698333 | 0.563363 | 197.958 ms |
| RRF + lexical fallback rerank | 0.548333 | 0.432874 | 172.390 ms |

The genuine hybrid candidate improved Recall@10 by `0.04` and nDCG@10 by
`0.03345`; its p95 was about `1.12x` BM25 with a `20.85 ms` increment. No
category regressed: arithmetic improved `0.05`, count `0.125`, multi-span
`0.025`, and table/text were unchanged. The fallback lexical reranker is a
clear negative result and must not be promoted.

This is still retrieval-only evidence. A fixed-seed, 10,000-resample paired
bootstrap over the 100 cases produced Recall@10 delta CI `[-0.02, 0.10]` and
nDCG@10 delta CI `[-0.01123, 0.07974]`. Since both lower bounds are below zero,
the candidate is promising but not statistically stable enough for a final
promotion claim. It also has not yet passed the live answer-level cited-v1 gate.

### Cross-document retrieval result

The MultiHop-RAG weakness was tested separately on a 100-case development split
and a disjoint 100-case held-out split. The selected retriever first runs a
global BM25 search, then adds one metadata-restricted BM25 branch for every
publication/source explicitly named in the query, and fuses document-level
rankings with RRF. Source branches are derived only from query text and indexed
metadata, never from gold evidence labels. Every branch receives the same
tenant, ACL, knowledge-base and version filters.

| Split / stage | Recall@10 | Recall@25 | nDCG@10 | p95 |
|---|---:|---:|---:|---:|
| development BM25 | 0.845000 | 0.882500 | 0.693360 | 2327 ms |
| development source coverage | 0.898333 | 0.970000 | 0.756750 | 1228 ms |
| held-out BM25 | 0.815833 | 0.858333 | 0.729073 | 1578 ms |
| held-out source coverage | 0.900000 | 0.951667 | 0.808741 | 1208 ms |

On the held-out split, Recall@10 improved by `0.084167`, Recall@25 by
`0.093333`, and nDCG@10 by `0.079668`. A fixed-seed 10,000-resample paired
bootstrap gave a Recall@10 delta CI of `[0.049167, 0.122500]`; 22 cases
improved and one regressed. The comparison, inference and temporal slices all
improved. These results justify making source coverage the MultiHop-RAG
answer-eval default, but they are retrieval-only evidence and do not prove an
answer-quality gain until a live cited-answer run passes the answer gate.
They also do not change the FastAPI application's default SQLite/in-memory
lexical context store; the promoted scope here is the retrieval experiment and
answer-evaluation path, not an unmeasured production-backend replacement.

The rejected development candidates are also retained as negative results:
`max_chunks_per_document=1` alone improved Recall@10 by only `0.0075`; naive
multi-query RRF fell to `0.841667`; and the generic MiniLM learned reranker fell
to `0.804167`. A table-aware count router improved the TAT-QA optimization split
but regressed the held-out split from `0.658333` to `0.653333`, so table routing
and learned reranking remain opt-in experiments rather than defaults.

### Profile-isolated locked gates (2026-08-09/10)

The current promotion matrix evaluates each retrieval profile independently and
also runs unchanged profiles through the same artifact. The locked TAT-QA
shape-aware candidate is profile-routed: explicit table/count/comparison
queries use compact QueryPlan + numeric scan + parent-context closure, while
generic text/date queries stay on lexical BM25. On the 100-case locked split:

| TAT-QA stage | Recall@10 | Candidate@50 | Count Recall@10 | Multi-span Recall@10 | p95 |
|---|---:|---:|---:|---:|---:|
| lexical BM25 control | 0.658333 | 0.786667 | 0.250000 | 0.491667 | 166.1 ms |
| routed shape-aware closure | **0.753333** | **0.900000** | **0.500000** | **0.641667** | 185.9 ms |

The paired gate passes (Recall@10 CI lower bound `+0.045`, p95 ratio `1.119`),
with no text or table-category regression. This remains an offline lexical
diagnostic; it is not evidence that a live semantic model has been verified.

For MultiHop-RAG, ordinary source-coverage RRF improved the cross-document
slice, but its 37-case paired bootstrap was inconclusive (point delta `+0.045`,
CI `[-0.0068, +0.0992]`). The follow-up single-variable candidate protects the
top three lexical hits and applies source-coverage RRF to the remaining budget
(`bm25_source_coverage_anchor_rrf`). The profile router also recognizes two or
more real source labels present in the query; labels come from corpus metadata,
not a dataset name. On the same locked 100 cases, 78 queries use the
cross-document profile and the candidate passes both the profile gate and the
whole-MultiHop protection line: Recall@10 `0.8542 → 0.9158`, Candidate@50
`0.9767 → 0.9883`, paired CI lower bound `+0.0300`, and p95 `421.0 ms`.
Comparison, inference, and temporal slices all improved. The general-text
profile was required to be retrieval-identical and passed.

The matrix may still report `passed=false` because older rejected experiments
(slow parent scan, generic reranker, and earlier semantic candidates) remain as
negative controls. Only the newly added locked profile rows are eligible for
promotion; historical red rows are not silently relabelled.

### Unified retrieval regression at Candidate@50

On 2026-08-09 both locked 100-case suites were rerun with the same chunk size,
overlap, security/version filters, `top_k=[1,5,10]`, `candidate_k=50`, per-case
predictions and latency schema. The derived, hash-pinned summary is
`eval/retrieval-benchmark-20260809.json`.

| Dataset / stage | Recall@10 | Candidate Recall@50 | nDCG@10 | p95 |
|---|---:|---:|---:|---:|
| TAT-QA BM25 | 0.658333 | 0.786667 | 0.529914 | 121.9 ms |
| TAT-QA semantic dense | 0.628333 | 0.811667 | 0.489587 | 198.3 ms |
| TAT-QA BM25 + dense RRF | **0.688333** | 0.826667 | 0.566772 | 390.9 ms |
| TAT-QA RRF + learned rerank | **0.688333** | **0.840000** | **0.594032** | 17771.3 ms |
| TAT-QA table-aware query router | 0.653333 | 0.791667 | 0.529232 | 177.0 ms |
| MultiHop BM25 | 0.820833 | 0.969167 | 0.730845 | 987.1 ms |
| MultiHop source coverage RRF | **0.901667** | **0.988333** | **0.809536** | **940.5 ms** |
| MultiHop multi-query RRF | 0.689167 | 0.969167 | 0.566981 | 2440.4 ms |
| MultiHop BM25 + learned rerank | 0.751667 | 0.971667 | 0.636150 | 6190.3 ms |

The learned reranker improves TAT-QA early ordering (Recall@1 and nDCG) but
does not improve Recall@10 and raises p95 from `391 ms` to `17.8 s`. On
MultiHop it reduces Recall@10 by `0.069166` while raising p95 to `6.19 s`.
The generic MiniLM model is therefore rejected as an online default. Source
coverage remains the only cross-document candidate that improves Recall@10,
Candidate Recall@50, nDCG@10 and p95 together.

This is a regression suite, not a new pristine test set: the locked cases have
already been observed in earlier experiments. The artifacts support
reproducibility and engineering decisions, but not a claim of untouched-set
generalisation. No LLM API was called; FastEmbed embedding and cross-encoder
inference ran locally.

The fresh real-semantic locked control
`.taskforge/eval-runs/tatqa-locked-semantic-bge-small-20260810` uses
`BAAI/bge-small-en-v1.5`, `candidate_k=50`, and chunking. Its BM25+dense RRF
stage reaches Recall@10 `0.688333`, Candidate@50 `0.826667`, and nDCG@10
`0.566772`, but the paired CI lower bound is `-0.005` and p95 is `171.0 ms`
versus BM25 `84.7 ms` (`2.02x`). The semantic stage therefore fails the
promotion gate and remains a negative control; it does not replace the
profile-routed table candidate.

### TAT-QA hierarchy diagnostics

The runner now records `retrieved_parent_ids` and a separate hierarchical
diagnostic for every retrieval stage. Parent Recall is exact at the
context/document level. `weak_operand_recall_at_k` is deliberately weaker: it
measures overlap of answer/derivation numbers and content terms with retrieved
text. TAT-QA does not expose canonical cell-coordinate labels in this adapter,
so this field must not be described as strict cell Recall.

On the 100-case optimization split, the new candidates produced:

| Stage | Recall@10 | Candidate@50 | Parent Recall@10 | Weak operand@10 | p95 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.608333 | 0.753333 | 0.800000 | 0.690283 | 82.2 ms |
| parent → child BM25 | 0.603333 | 0.743333 | 0.780000 | 0.678546 | 69.3 ms |
| table multi-representation max | 0.618333 | 0.758333 | 0.780000 | 0.693656 | 209.1 ms |
| count-adaptive table max | **0.633333** | 0.753333 | 0.800000 | **0.707227** | 178.1 ms |

The strict parent-child route is therefore rejected as a default because it
loses `0.005` Recall@10 and parent Recall. Max fusion is retained as a
diagnostic/opt-in candidate: it improves Count Recall@10 on this split, but
regresses Multi-span and table slices. The next optimization should be
query-type-gated fusion or an explicit cell/row annotation adapter. The
count-adaptive route is the first query-type-gated result: it improves the
optimization split, but on the locked 100-case split it stayed at Recall@10
`0.658333` (Candidate@50 `0.801667` versus BM25 `0.786667`). It remains an
opt-in experiment until a second disjoint holdout confirms the gain. A learned
reranker should only be tested after Candidate Recall@50 is stable.

#### Current TAT-QA candidate/ranking ablation (2026-08-09)

The current optimization artifact is
`.taskforge/eval-runs/tatqa-opt-query-context-20260809`. It keeps the same
100-case optimization split, semantic model, filters, and `candidate_k=50` for
every stage:

| Stage | Recall@10 | Candidate@50 | Count@10 | Multi-span@10 | p95 |
|---|---:|---:|---:|---:|---:|
| semantic BM25 query-RRF + dense | 0.698333 | 0.805000 | 0.416667 | 0.560606 | 328 ms |
| + same-context table/paragraph coverage | **0.698333** | **0.880000** | 0.416667 | 0.560606 | 303 ms |
| + dual lexical/dense query variants + coverage | 0.698333 | 0.880000 | 0.416667 | 0.560606 | 587 ms |
| + MiniLM Top-20 reranker | 0.721667 | 0.880000 | 0.625000 | 0.598485 | 1,875 ms |

The context-coverage stage is the current candidate-generation default for
further experiments: it improves Candidate@50 by `0.075` without changing the
Top-10 head or violating the filter contract. It still misses the `0.90`
candidate gate. MiniLM improves early ordering but fails the `1.2 s` p95 gate;
the Top-10 budget did not recover the latency or add Recall@10.

The domain reranker provider is implemented and trained from the pinned
TAT-QA train file, but its optimization artifact
`.taskforge/eval-runs/tatqa-opt-domain-rerank-top20-20260809` is a negative
result (Recall@10 `0.668333` versus `0.698333`). It remains an opt-in provider,
not a promoted model. FastEmbed 0.8.0 does not list BGE-M3 or ColBERT models;
the large BGE trial exceeded the ten-minute experiment budget and was stopped,
so no BGE-M3/ColBERT metric is claimed.

#### Latest structure-aware candidate scan (2026-08-09)

The next candidate-generation pass adds repeated-header table sections,
query-plan facts, a bounded numeric table scan and parent-context routing. On
the same 100-case optimization split, the best cached artifact
`.taskforge/eval-runs/tatqa-opt-query-plan-parent-scan-cache-20260809` reports
Recall@10 `0.713333` and Candidate@50 `0.856667`, versus BM25's `0.608333` and
`0.753333`. The gain is broad on the table (`+0.190476`), count (`+0.166667`),
arithmetic (`+0.090909`) and multi-span (`+0.131579`) slices, with no text-slice
drop in this artifact. It is still not promoted: p95 is `386.0 ms` versus the
same-run BM25 control's `174.6 ms` (`2.21x`), exceeding the strict `1.2x`
relative budget even though it remains below the profile's `1.2 s` absolute
cap. Candidate@50 also remains below the `0.90` target. The checked-in matrix
records this as `table_numeric_parent_scan_strict_rejected` rather than
silently selecting it.

The candidate scan is profile-routed and constructible on non-table corpora,
but inactive outside `table_numeric`. General-text, cross-document, PDF and
QASPER rows therefore remain separate matrix scenarios; a TAT-QA improvement
cannot be promoted until every listed scenario passes its own case-level,
category, latency and security checks. The current strict matrix report is
`.taskforge/eval-runs/promotion-matrix-20260809-strict.json` and is red because
the older table/cross-document controls still have inconclusive bootstrap
lower bounds, while general-text, PDF and QASPER controls pass.

#### Locked TAT-QA shape-aware gate (2026-08-09)

The final locked diagnostic is
`.taskforge/eval-runs/tatqa-locked-parent-scan-context1-shapes-final-routing-20260809`.
It uses only corpus-derived metadata: bounded prose linked by `table_uid`,
year-column sparsity, temporal row shape and segment-like row labels. Generic
date/numeric text lookups stay on the default profile; only explicit
structured/count/comparison queries use the table branch. No answer, evidence
ID or evaluation label is used for routing.

| Stage | Recall@10 | Candidate@50 | Count Recall@10 | Multi-span Recall@10 | p95 |
|---|---:|---:|---:|---:|---:|
| lexical BM25 control | 0.658333 | 0.786667 | 0.250000 | 0.491667 | 166.1 ms |
| profile-routed shape-aware scan | **0.753333** | **0.900000** | **0.500000** | **0.641667** | 185.9 ms |

The paired gate passes: Candidate@50 `0.90`, Recall@10 `0.753333`, paired CI
lower bound `+0.045`, no category failures, p95 ratio `1.119`, and ACL/security
checks are green. General text remains `0.900` Recall@10 and table Recall@10
does not drop. This is still an offline/degraded lexical diagnostic rather
than a semantic-model promotion; the next required check is the same stage
with the locked semantic embedder and the full general-text, cross-document,
PDF and QASPER matrix.

The optimization artifacts above remain non-promotable. The locked diagnostic
is reported separately and is not a configuration promotion; no artifact is
silently treated as a common-retriever win.

The first deterministic ranking follow-up is also retained as a negative
control. On the same locked split, the opt-in TAT-QA feature reranker scored
Recall@10 `0.723333` (pure features) and `0.733333` (base-RRF/feature blend at
weight `0.2`) versus the shape-aware base `0.753333`. The blend also reduced
the count slice from `0.50` to `0.40`, so neither variant passes the paired
non-regression gate. This is evidence against promoting a hand-written
reranker, not evidence that the profile router should be removed.

#### Controlled table-profile lookup extension (2026-08-10)

The next locked comparison keeps the same candidate branches and enables the
table Profile for generic lookup signals such as components, respective
values/amounts, and year/period queries. Narrative signals such as ``what
caused`` remain on the general branch. This is query/corpus routing, not a
dataset-name check.

In `.taskforge/eval-runs/tatqa-locked-parent-scan-context1-shapes-table-profile-20260810`,
the opt-in stage
`bm25_tatqa_query_plan_parent_scan_closure_table_profile_rrf` reaches Recall@10
`0.778333`, Candidate@50 `0.920000`, Multi-span Recall@10 `0.716667`, and p95
`98.44 ms`. Against the same-run shape-aware control, the paired gate reports
Recall delta `+0.025`, Candidate delta `+0.020`, text Recall unchanged at
`0.900000`, and p95 ratio `1.0005`; all gate checks pass. The broader
"all table Profile" variant is retained as a negative control because it
reduces text Recall and exceeds the relative latency budget.

The active matrix now contains fifteen scenarios, including the table-profile
lookup gate and the whole-TAT-QA target row. The stage remains opt-in until the same locked split is rerun with
the configured semantic embedder and the complete matrix remains green.

The follow-up deterministic rerank on that candidate pool is deliberately not
in the matrix: a conservative feature blend (`0.05`) reached Recall@10
`0.773333` but nDCG@10 `0.617177`, below the candidate base `0.618927`, and
lost `0.005` Recall@10. This negative result keeps the ranking gate honest;
the next ranking candidate must be a real learned model or a separately
validated branch-aware ranker.

The explicit domain-reranker interface was also exercised with the locally
trained `taskforge-tatqa-linear-v1` artifact. It retained Candidate@50
`0.920000` but reduced Recall@10 from `0.778333` to `0.738333`, including a
Count drop from `0.50` to `0.30`; its paired gate failed and it is not in the
promotion matrix. A supplied cross-encoder can use the same stage contract
without changing Profile routing.

The real local cross-encoder follow-up is recorded in
`.taskforge/eval-runs/tatqa-locked-parent-scan-context1-shapes-table-profile-cross-encoder-20260810`.
Using `Xenova/ms-marco-MiniLM-L-6-v2` only for table-profile lookup queries,
it reduced Recall@10 from `0.778333` to `0.768333`, nDCG@10 by `0.023040`, and
the table slice by `0.05`; p95 increased from `152.0 ms` to `460.6 ms`
(`3.03x`). The paired gate
`.taskforge/eval-runs/tatqa-table-profile-cross-encoder-gate-20260810.json`
therefore fails on recall CI, category non-regression and latency. This is a
real learned-model measurement, not a reason to promote the model: the
candidate pool is already strong enough that this generic reranker mostly
reorders relevant table evidence away from the Top-10 while adding material
latency.

The next candidate-generation step follows corpus-derived table/paragraph
lineage without changing the Top-10 head. The stage
`bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_rrf` reserves
12 Candidate@50 tail slots for query-relevant, ACL-filtered evidence from the
top 20 seed parents. In
`.taskforge/eval-runs/tatqa-locked-table-profile-lineage-v2-20260810`, it keeps
Recall@10 `0.778333`, nDCG@10 `0.618927`, and every category Recall unchanged,
while Candidate@50 increases from `0.920000` to `0.940000`. The paired
Candidate delta CI is `[+0.005,+0.040]`; p95 is `195.2 ms` versus the same-run
control's `210.9 ms`. The 11-scenario matrix remains green, including whole
MultiHop, exact general-text/QASPER controls, PDF layout and ACL probes.

FastEmbed SPLADE++ was then exercised as a real learned sparse complement, not
as a BM25 replacement. The locked artifact
`.taskforge/eval-runs/tatqa-locked-table-profile-lineage-splade-20260810`
raises Candidate@50 only from `0.940000` to `0.943333`, while Recall@10 falls
to `0.763333`, table Recall falls by `0.05`, and p95 rises from `111.7 ms` to
`199.7 ms` (`1.79x`). Its paired gate
`.taskforge/eval-runs/tatqa-table-profile-lineage-splade-gate-20260810.json`
fails recall CI, category non-regression and latency. SPLADE therefore remains
an opt-in diagnostic stage and is not part of the active matrix.

The promoted candidate-tail follow-up adds a query-typed structured table
index without changing the stable Top-10 head. It preserves multi-row headers,
row/cell values, normalized numbers, years, units, scale, sign, parent table
and parent-paragraph lineage. Count, arithmetic and multi-span queries use
separate structured branches. The final wrapper may replace at most two
low-ranked Candidate@50 tail items with ACL-filtered missing paragraphs from a
table already present in the Top-10; it cannot introduce a new table into the
candidate pool.

On the same locked 100-case split,
`.taskforge/eval-runs/tatqa-locked-table-profile-lineage-structured-v3-20260810`
keeps Recall@10 `0.778333`, nDCG@10 `0.618927`, Count Recall@10 `0.500000`,
Multi-span Recall@10 `0.716667`, and every category Top-10 result unchanged.
Candidate@50 improves from `0.940000` to `0.950000`: arithmetic rises from
`0.975000` to `1.000000` and multi-span from `0.925000` to `0.950000`, with
count, table and text unchanged. The repaired evidence is the parent paragraph
for the purchase-obligations-within-five-years case and the parent paragraph
for the 2018/2019 prepaid-expenses case. The paired candidate-recall delta CI
is `[0.000,0.025]`; p95 is `143.0 ms` versus `147.6 ms` for the same-run
lineage control. The gate artifact is
`.taskforge/eval-runs/tatqa-table-profile-lineage-structured-gate-20260810.json`.
Both the table-profile slice and whole-TAT-QA structured rows are now in the
15-scenario matrix, which remains green. This is a candidate-coverage gain,
not a claim of better Top-10 ranking or answer accuracy.

The next isolated ranking step operates on that frozen Candidate@50 set. The
stage
`bm25_tatqa_query_plan_parent_scan_closure_table_profile_lineage_pair_rerank_rrf`
recognizes explicit temporal/count relationship forms (for example, how many
years/quarters, in which years, how many as at a date, or how many items are
included within a charge). It may move one existing
`same_parent_evidence_closure` hit to rank 10 only when the hit scores at least
`0.24` and shares a parent with an existing Top-10 hit. It does not use answer
labels, add a document, change Candidate@50 membership, or route narrative
queries.

On the same locked artifact
`.taskforge/eval-runs/tatqa-locked-table-profile-lineage-pair-rerank-v1-20260810`,
the structured control scores Recall@10 `0.778333`, Candidate@50 `0.950000`
and nDCG@10 `0.618927`; the pair-rerank candidate scores `0.803333`,
`0.950000` and `0.627135`. Count Recall@10 rises from `0.500000` to
`0.600000`, Multi-span from `0.716667` to `0.741667`, and arithmetic, table
and text remain unchanged. Candidate@50 membership is identical for every
case. The Recall delta is `+0.025` with paired 95% CI `[+0.005,+0.055]`;
p95 is `141.3 ms` versus the same-run control's `186.0 ms`. The gate artifact
is `.taskforge/eval-runs/tatqa-table-profile-lineage-pair-rerank-gate-20260810.json`.
The matrix now includes both the table-profile slice (whose frozen Candidate@50
baseline is `0.947674`) and whole-TAT-QA row (Candidate@50 `0.950000`); all 15
scenarios pass. This is an early-ranking gain on a fixed candidate set, not an
answer-generation result or a live-model claim.

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

### QASPER document-disjoint hierarchy ablation (2026-08-11)

The earlier 100-case QASPER artifact remains a historical control only. The
current tuning artifact uses the official normalized training data, a fixed
200-case split grouped by paper, and `provided_document_context`; no paper is
shared across the tuning and validation manifests. The input, split SHA-256,
case IDs, filters, model name and candidate budget are recorded in every run.

| Stage | Recall@10 | Candidate@50 | nDCG@10 | p50 | p95 |
|---|---:|---:|---:|---:|---:|
| B0 BM25 | 0.5170 | 0.9068 | 0.2956 | 5.66 ms | 12.27 ms |
| B1 hierarchy/title BM25 | 0.5299 | 0.9434 | 0.3130 | 7.39 ms | 14.51 ms |
| B2 real BGE dense | 0.6282 | 0.9738 | 0.4027 | 7.16 ms | 10.04 ms |
| B3 BM25+dense candidate union | 0.5299 | 0.9679 | 0.3130 | 13.99 ms | 21.05 ms |
| B4 section-parent supplementation | 0.4905 | 0.6221 | 0.2860 | 18.11 ms | 26.47 ms |
| B5 section-parent RRF | 0.5633 | 0.6246 | 0.3669 | 19.67 ms | 29.44 ms |

The machine-readable matrix is
`eval/qasper-hierarchical-ablation-20260811.json`; its paired bootstrap
report is `eval/reports/qasper-hierarchical-ablation-20260811.json`. B2 is the
only candidate in this matrix that clears the current QASPER quality,
non-negative-CI and p95 gates. The dense model is the local
`BAAI/bge-small-en-v1.5`; hash vectors are not used for these quality numbers.
The QASPER benchmark uses an explicit exact-cosine in-memory index to avoid
local Qdrant client overhead; this is an evaluation backend and does not alter
the generic Qdrant production contract.

Failure diagnostics for B0 and B2 classify candidate misses separately from
Top-10 ranking misses and record vocabulary mismatch, title/abstract/section
dependency, same-section versus cross-section evidence, long/truncated
evidence and evidence-alignment failures. The alignment count is zero on the
normalized locked cases (unmatched raw annotations are excluded by the adapter),
which is recorded rather than silently treated as a retrieval success.

#### QASPER learned-reranker Top-N scan (2026-08-11)

The in-memory dense backend now honors `retrieval.rerank_top_k`: only the
highest dense candidates are sent to the Cross-Encoder and the remaining
Candidate@50 tail keeps its dense order. This fixes a previous measurement
bug where the config value was recorded but all candidates were scored. The
same 200-case split gives:

| Rerank prefix | Recall@10 | nDCG@10 | p50 | p95 | scored pairs |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.7070 | 0.4928 | 229.05 ms | 490.01 ms | 3,178 |
| **20** | **0.7341** | **0.5108** | **274.41 ms** | **549.59 ms** | **3,935** |
| 24 | 0.7432 | 0.5113 | 389.35 ms | 728.75 ms | 4,635 |

Candidate@50 remains `0.9738` for all three runs. Top-20 is the recommended
quality/latency point for the QASPER `general_text` route. On the independent
paper-disjoint validation split it improves Recall@10 from `0.6367` to
`0.7223`; the paired 95% CI for the delta is `[+0.0379, +0.1338]`. Its
validation p95 is `693.24 ms`, so this remains an explicit route-level opt-in,
not a global product default. The auditable scan is
`eval/reports/qasper-rerank-topn-20260811.json`, and the validation gate is
`eval/reports/qasper-validation-rerank-top20-20260811.json`.

The promoted evaluation choice is therefore B2, routed only for `general_text`.
The four-scenario regression matrix
`eval/retrieval-retained-capabilities-b2-20260811.json` passes: TAT-QA
provided-context, QASPER, MultiHop cross-document and synthetic PDF/ACL probes
remain within their independent gates.

#### Evidence-graph feature rerank audit (2026-08-11)

The graph stage is evaluated as a separate, candidate-preserving ablation over
the same real-BGE + MiniLM Candidate@50 output. Its graph contains bounded
same-section, adjacency and shared-entity edges. The first implementation
incorrectly mixed uncalibrated cross-encoder scores with graph features and is
retained as a negative regression. The fixed implementation uses rank-based
calibration; graph/entity/section/adjacency/PPR weights are selected only on
the 200-case tuning split, and the paper-disjoint validation split is used for
confirmation.

| Split / stage | Recall@10 | Candidate@50 | nDCG@10 | p95 |
|---|---:|---:|---:|---:|
| frozen validation Top-20 base | 0.7223 | 0.9627 | 0.4856 | 639.6 ms |
| validation fixed Top-30 base | 0.7549 | 0.9627 | 0.5021 | 951.9 ms |
| validation fixed Top-30 + graph | **0.7610** | **0.9627** | **0.5069** | 2.25 ms incremental |
| validation adaptive Top-20/30 base | 0.7256 | 0.9627 | 0.4871 | 928.4 ms |
| validation adaptive Top-20/30 + graph | 0.7468 | 0.9627 | 0.4950 | 2.13 ms incremental |

The graph stage reuses the immutable base-stage response within the same run,
so its latency is incremental. Fixed Top-30 plus graph clears the `0.75`
Recall target, the `+0.02` nDCG target, the unchanged Candidate@50 gate and the
1.2-second route budget. It is retained only for the QASPER/general-text opt-in
route and does not replace table, cross-document or PDF retrieval. The audit is
`eval/reports/qasper-graph-tuned-top30-20260811.json`.

The adaptive budget performs genuine separate inference batches: score Top-20,
then score ranks 21--30 only when the Top-1/Top-2 cross-encoder margin is below
`0.7`. The threshold was selected on tuning. On validation it escalated 35% of
cases, reduced scored pairs from 5,794 to 4,645 and reduced mean base latency by
103.6 ms. However, graph Recall@10 fell by `0.0142` and nDCG@10 by `0.0119`
versus fixed Top-30; p95 improved by only 23.4 ms. It therefore fails promotion
and remains an opt-in negative ablation. The full audit is
`eval/reports/qasper-adaptive-rerank-20260811.json`.

The P3 expansion probe used the same validation split with a same-paper,
one-hop, ten-slot budget. Recall@10 stayed at `0.7320`, Candidate Recall@50
slipped from `0.9627` to `0.9609`, and the incremental graph p95 was `1.9 ms`;
it is therefore retained as a gated API, not enabled in the recommended route.
Its evidence is in
`.taskforge/eval-runs/qasper-train-validation-200-graph-expansion-v2-20260811`.

As a no-tuning generalization check, the separate 200-case paper-disjoint
validation manifest gives B0 `0.5967 / 0.9024` and B2 `0.6367 / 0.9627`
(Recall@10 / Candidate@50, p95 `10.92 ms`). The point improvement is positive,
but its paired 95% Recall CI is `[-0.0295, +0.1094]`; this validation artifact
is therefore reported as uncertainty evidence, not used to retune or to claim
a stronger promotion gate than the locked tuning result.

## Paper-research evaluation stack (2026-08-12)

The paper product has three separate metric boundaries. Results from one row
must never be presented as results from another row.

| Layer | Unit | Current evidence | Gate |
|---|---|---|---|
| Open discovery | papers over the public scholarly universe | 100 needs / 792 known arXiv qrels | **failed** |
| Scope-bound retrieval | passages inside user-selected papers | 414 locked assets across four routes | passed |
| End to end | discovery, user PDF upload, Scope, evidence and four roles | upload-bound API regression; historical 30-task/1 live run used the retired abstract fallback | current upload boundary passed; full semantic gate must be rerun |

### Open discovery

The product response is a recommendation card, not an evidence object: title,
HTTPS source link, year/authors when available, and one short description.
The live evaluator therefore also reports `recommendation_link_coverage` and
`short_description_coverage` over Top-10. Full abstracts remain internal
provider metadata and are not returned by `POST /api/literature/search`.

`eval/literature-discovery-benchmark-100.json` contains 50 PaSa
RealScholar, 30 LitSearch, and 20 TaskForge-authored bilingual actual-needs
queries. The pinned ScholarGym source SHA-256 is checked by
`scripts/prepare_literature_discovery_benchmark.py`. The benchmark has 792
known relevant arXiv IDs; its qrels are incomplete, so unlabelled results are
not automatically irrelevant.

The full anonymous-provider live run made real Semantic Scholar, OpenAlex,
arXiv and Crossref requests. It produced 7,254 raw candidates and 3,815 ranked
results, with zero duplicate and zero unverifiable-card rate, but only
`0.001/0.001` Paper Recall@20/50, `0.001` Precision@10 and `0.0022` nDCG@10.
There were 336 provider/query-group failures. This fails the `0.80/0.90`,
`0.60`, and `0.70` quality targets and is retained as a negative live result,
not a resume claim. The scored report is
`eval/reports/literature-discovery-full100-live.json`; the raw scoring revision
is preserved as `literature-discovery-full100-live-raw-v1.json`.

The observed boundary is specific: anonymous API rate limits and the then-
untranslated Chinese queries left Crossref as the dominant surviving source,
which returned lexical collisions. The code now includes global polite pacing,
contact/key configuration and a conservative bilingual terminology bridge.
Those changes are not promoted until a formal Provider quota reruns the same
frozen 100-query file. The six title-shaped smoke queries are only an interface
regression.

### Scope-bound passage retrieval

Every pre-schema-2 QASPER upload report is retired. Those reports used page
overlap as relevance; all scores, ablations, trained/calibrated rerankers and
promotion decisions derived from that proxy are historical artifacts, not
paragraph-retrieval evidence. They must not be used for acceptance, model
selection, project claims, or answer evaluation. The files are retained only
for auditability; see `eval/reports/QASPER_RETRIEVAL_DEPRECATION.md`.

The frozen `eval/reports/research-scope-retrieval-gate-current.json` matrix is:

| Route | Recall@10 | Candidate@50 |
|---|---:|---:|
| TAT-QA provided context | 0.9902 | 0.9902 |
| QASPER B2 | 0.6282 | 0.9738 |
| MultiHop cross-document | 0.9199 | 0.9893 |
| PDF layout smoke | 1.0000 | 1.0000 |

These retained-capability runs evaluate bounded evidence retrieval over
normalized benchmark contexts. They do not all replay a binary PDF through the
HTTP upload path and therefore must not be presented as upload-path recall.
Search snippets and discovery abstracts are recommendation metadata and are
never accepted as bounded evidence.

The product-facing gate starts at a user-supplied binary PDF. The strict
schema-2.3 evaluator freezes the PDF checksum cohort and parser, maps each Gold
Evidence paragraph to actual retrieval Child text, scores every legal QASPER
annotation set separately, and takes the maximum legal Recall. Parser failures
remain zero-recall rows. Only Recall@1/5/10/50 may be a headline metric.

The native-parser preflight remains a historical diagnostic and is not a
formal result. The current MinerU 3.4.4 locked 100-case run passes the frozen
  gate with `97.03%` Gold-unit alignment and `97/100` fully eligible cases. On
  identical inputs, the frozen Flat comparison with the original Query and local
Cross-Encoder reports paragraph Recall@1/5/10/50
`0.2728/0.7367/0.8625/0.9830`; Agent-visible Recall@8 is `0.8250`. Against
the same-parser, same-chunk, no-reranker locked baseline, Recall@10 improves by
`2.70` percentage points and Recall@5 by approximately `2.68` points; the exact
comparison is frozen in `eval/reports/qasper-pdf-reranker-uplift-v1.json`.
Failure attribution is 3 candidate misses, 17 Top-10 reranking misses and 80
successful cases, with no additional presentation-window loss. The Parent–Child
  A/B reports `0.7022/0.8447` at Recall@5/10 and Agent-visible Recall@8=`0.7938`.
  The current Parent-Child default adds title-enriched indexing, Parent-aware
  reranking and lineage diversity; it has not been rerun, so those historical
  figures remain comparison evidence rather than a claim about the new chain. The machine-readable
reports are
`eval/reports/qasper-real-pdf-locked100-current-original-flat-v2-top8.json` and
`eval/reports/qasper-real-pdf-locked100-current-original-parent-child-v2-top8.json`.

A separate no-API 20-case chunk-strategy screen tested Flat targets from 500 to
2,000 characters and same-page sliding windows. Flat 500 reached only
`0.5350/0.7817` at Recall@5/10; sliding 500/100 and 1000/200 failed the Gold
alignment gate; sliding 2000/400 traded lower Recall@5 for higher Recall@10.
Because no configuration improved both @5 and @10, the default remains Flat
2,000 characters with zero overlap. See
`eval/reports/qasper-pdf-chunk-strategy-screen20-v1.json`.

Parent–Child was then screened with four token budgets, including Child
300/400, 400/500, 500/650 and 600/800 plus proportional overlap. All passed the
alignment gate, but every configuration was below Flat 2000/0 at both @5 and
@10. The hierarchy remains available for `paper_read` context expansion and as
an opt-in ablation; it is not the retrieval default. See
`eval/reports/qasper-real-pdf-screen20-parent2000-3000-child400-500-overlap60-v2-top8.json`.

The remaining formal ablations are:

| ID | Parser | Chunking | Query/ranking addition |
|---|---|---|---|
| A0 | Native | flat | original query, no reranker/supplement |
| A1 | MinerU 3.4.4 | flat | parser change only |
| A2 | MinerU 3.4.4 | Parent–Child | hierarchy change only |
| A3 | same | Parent–Child | constrained synonym query |
| A4 | same | Parent–Child | keyword/entity query |
| A5 | same | Parent–Child | Cross-Encoder over full Candidate@50 |
| A6 | same | Parent–Child | one directed supplementary round |
| A7 | same | Parent–Child | separate VLM visual extraction |

Run the matrix with `scripts/run_qasper_pdf_ablation.py`. Expanded-query runs
require a split-hashed, pre-generated manifest; A7 additionally requires an
explicit visual-call acknowledgement. No query generation or PDF acquisition
occurs inside a scored run. A configuration may be frozen for the locked split
only after validation, parser-quality and alignment gates pass.

The deterministic keyword-only query ablation on the current locked 100-case
Flat track matched the original Query exactly at Candidate@50 and all four
Recall@K values, so it was not promoted. See
`eval/reports/qasper-query-expansion-locked100-v1.json`.

### End-to-end and Token

`eval/paper-research-e2e-cases-30.json` has 10 single-paper, 10 comparison and
10 survey tasks. The deterministic provider report executes all 30 lifecycle
paths with 30/30 pass, four-protocol completion 1.0, Scope escape 0 and cited
Evidence-ID resolution 1.0. Maximum estimated handoff size was 443 Token per
task and p95 was 1.39 seconds. Its 120 provider calls are local fixture calls;
external API calls are zero. It proves API/state/protocol invariants only; it
does not measure semantic citation entailment.

The separate real DeepSeek `deepseek-v4-flash` paired task used live scholarly
discovery and all four model roles. Its total Token fell from 212,874 to 62,186
(`70.79%`), structured cross-role payload was approximately 2,366 Token, and
Scope escape was 0. This is one paired task, not a 30-task model-quality claim.
Reports are `eval/reports/paper-research-e2e-30-deterministic.json`,
`paper-research-business-e2e-live.json`, and
`paper-research-business-e2e-prebudget-live.json`.

### QASPER cited-answer E2E

Existing QASPER answer reports are development diagnostics only: their
retrieved-evidence input came from the retired page-proxy evaluation, early
versions selected only one annotation, and the answer model and semantic judge
were not independent. The later multi-reference validation run still used a
non-independent judge and one answer Agent, not the complete Planner,
Evaluator, Writer and Critic product path. None is a current project result.

The answer evaluator now accepts strict schema 2.0/2.1/2.2/2.3 upload reports
and refuses retrieved-evidence execution unless the Gold→Child alignment gate
passed. It restores Yes/No labels, evaluates all answerable annotations,
separates citation failure from answer reasoning failure and keeps Oracle Gold
Evidence as a separately labelled upper bound. The current retrieval contract
is frozen at Flat 2000/0, Candidate@50 and Agent-visible Top-8; a new live
four-Agent run is intentionally deferred because it requires a DeepSeek API
call and must not be confused with the historical run below. The final
semantic judge should be independent or calibrated against blinded human
labels.

A historical 100-case four-Agent direct-answer replay is retained for audit:
`eval/reports/qasper-four-agent-e2e-live-direct-answer-a1-final-v1.json`. It
used a frozen Parent–Child retrieval trace rather than the current Flat default
and reports deterministic Token F1 `0.4761` (the old baseline delta was
`+36.35` percentage points), with semantic judging explicitly auxiliary and
same-model. These values are historical and are not a current-config E2E
claim.

Reproduction:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_literature_discovery_benchmark.py
.\.venv\Scripts\python.exe scripts\evaluate_literature_discovery.py `
  --cases eval\literature-discovery-benchmark-100.json `
  --output eval\reports\literature-discovery-full100-live.json `
  --state-dir .taskforge\eval-runs\literature-discovery-full100-live

.\.venv\Scripts\python.exe scripts\prepare_paper_research_e2e_benchmark.py
.\.venv\Scripts\python.exe scripts\evaluate_paper_research_e2e.py
.\.venv\Scripts\python.exe scripts\run_paper_research_e2e.py

# Billable only after the strict retrieval report has status=complete.
.\.venv\Scripts\python.exe scripts\evaluate_qasper_answer_e2e.py `
  --retrieval-report <strict-schema-2.3-report.json> `
  --split eval\splits\qasper-dev-clean-holdout-100-v2.json `
  --output eval\reports\qasper-answer-e2e-strict.json `
  --confirm-live-call
.\.venv\Scripts\python.exe scripts\evaluate_qasper_answer_e2e.py `
  --retrieval-report <strict-schema-2.3-report.json> `
  --split eval\splits\qasper-dev-clean-holdout-100-v2.json `
  --evidence-source oracle `
  --output eval\reports\qasper-answer-e2e-strict-oracle.json `
  --confirm-live-call

# Billable four-Agent replay; intentionally deferred while API spend is frozen.
.\.venv\Scripts\python.exe scripts/evaluate_qasper_four_agent_e2e.py `
  --retrieval-report eval\reports\qasper-real-pdf-locked100-current-original-flat-v2-top8.json `
  --split eval\splits\qasper-dev-clean-holdout-100-v2.json `
  --output eval\reports\qasper-four-agent-e2e-current.json `
  --confirm-live-call
```

## Final selected-paper retrieval baseline (2026-08-28)

The frozen baseline for question answering after a user explicitly selects a
paper is `eval/baselines/paper-scoped-flat-bailian-v1.json`. It covers 30
English and 30 Chinese real PDFs with 177 annotated questions. The fixed path
is MinerU 3.4.4, Flat 2000/0, BM25 plus Bailian `text-embedding-v4`, RRF, and
Bailian `qwen3-rerank`, with a per-paper `knowledge_base_id` filter.

The overall result is Recall@10 `0.9262`, MRR@10 `0.6366`, and NDCG@10
`0.6551`. English Recall@10 is `0.9015`; Chinese Recall@10 is `0.9500`.
This baseline is valid only for selected-paper QA. The unchanged global
60-paper report remains the discovery control and must not be compared against
the paper-scoped score as if both tasks had the same information available.

The controlled Dual candidate (Flat primary plus 400/500-Token structured
Child auxiliary retrieval) was rejected on the same 177 questions. It reduced
overall Recall@10 from `0.9262` to `0.8870`, Chinese Recall@10 from `0.9500`
to `0.8667`, MRR@10 from `0.6366` to `0.5785`, and increased p95 from
`384.0 ms` to `448.0 ms`. English Recall@10 improved only slightly to
`0.9080`. The decision artifact is
`eval/reports/paper-scoped-flat-vs-dual-v1.json`; the frozen Flat baseline is
unchanged.

Reproduction:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_mixed_dual_mineru.py `
  --mode flat `
  --scope paper `
  --output eval\reports\mixed-mineru-flat2000-30zh-30en-bailian-paper-scoped-final-v1.json `
  --state-dir .taskforge\eval-runs\mixed-mineru-flat2000-30zh-30en-bailian-all-v1
```

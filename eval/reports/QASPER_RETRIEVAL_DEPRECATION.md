# QASPER retrieval report deprecation

All QASPER retrieval reports produced before strict schema 2 are historical
artifacts only. In particular, reports whose `evaluation_type` is
`qasper_direct_pdf_upload_retrieval` measured page overlap and must not be used
for acceptance, tuning, promotion, answer evaluation, project claims, or a
resume.

This also invalidates comparative claims, training labels, calibration and
promotion decisions derived from those page-proxy reports. Retaining the files
does not make their reported values valid; they exist only for provenance and
debugging.

Valid new runs must use one of these disjoint tracks:

- `qasper_corpus_native_retrieval`: official paragraph/caption IDs;
- `qasper_synthetic_pdf_parser_regression`: generated PDF parser regression;
- `qasper_real_pdf_upload_retrieval`: checksum-pinned real-PDF cohort.

Upload-track rows must contain `gold_alignments` and score only strict
Gold-paragraph content projected onto retrieved Child chunk IDs. Page overlap
is diagnostic metadata at most and is never a relevance judgement. The answer
evaluator rejects the legacy report type explicitly.

Current status: the schema-2.1 native-parser real-PDF preflight aligned 77.54%
of Gold units and made 83/100 cases fully eligible. It failed the frozen
90%/90% alignment gate, so its headline Recall@1/5/10/50 fields are `null`.

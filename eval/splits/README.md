# QASPER split policy

QASPER retrieval, reranker, alignment-calibration, and answer evaluations must
be disjoint by paper ID, not only by question ID. Validate every combination
before use:

```powershell
.\.venv\Scripts\python.exe scripts\validate_qasper_splits.py `
  eval\splits\qasper-dev-general-100-v1.json `
  eval\splits\qasper-validation-upload-50-v2.json
```

`qasper-validation-upload-50.json` is retained only to reproduce historical
page-proxy reports. It overlaps `qasper-dev-general-100-v1.json` by paper and
must not be used for tuning, calibration, promotion, or final claims. Its
paper-disjoint replacement is `qasper-validation-upload-50-v2.json`.

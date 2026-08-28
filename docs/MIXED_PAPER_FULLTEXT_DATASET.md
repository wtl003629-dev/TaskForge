# 中英文论文混合全文数据集

`mixed-paper-fulltext-v1` 将两类全文记录合并为一个可直接用于 RAG 检索、切块和向量化的数据集：

- 英文：QASPER train + validation，共 1,169 篇全文论文；
- 中文：`chinese-ai-oa-jos-v2`，共 1,000 篇通过主题、页数、文本长度、中文比例和去重检查的 AI 相关中文全文论文；
- 合计：2,169 篇全文论文。

正式数据位于：

```text
.taskforge/datasets/mixed-paper-fulltext-v1/
```

## 文件

| 文件 | 内容 |
|---|---|
| `corpus.jsonl.gz` | 2,169 条全文记录，统一包含 `document_id`、`paper_id`、`language`、`title`、`abstract`、`text`、`document_type`、`source_dataset` 等字段 |
| `papers.jsonl.gz` | 同一批论文的元数据，不含全文 `text` |
| `queries.jsonl.gz` | 从原数据集中保留的 QASPER 英文问题 |
| `qrels.jsonl.gz` | 与上述英文问题对应的 QASPER qrels |
| `manifest.json` | 数量、来源、SHA-256、授权和评测范围 |

重新构建：

```powershell
.\.venv\Scripts\python.exe scripts/build_mixed_paper_fulltext_dataset.py `
  --english-dir .taskforge/datasets/bilingual-paper-corpus-v1 `
  --chinese-dir .taskforge/datasets/chinese-ai-oa-jos-v2 `
  --output-dir .taskforge/datasets/mixed-paper-fulltext-v1
```

## 与原双语数据集的区别

原来的 `bilingual-paper-corpus-v1` 包含 1,169 篇英文全文和 100,000 条中文标题/摘要记录。这个新版本只混合全文，故没有把中文摘要记录复制进来；这样做可以避免在 RAG 评测中把“摘要命中”误当作“全文命中”。

QASPER 的英文查询和 qrels 可以直接用于新语料中的英文部分。中文论文目前没有对应的人工相关性标注，因此脚本不会生成或猜测中文 qrels；如需评测中文 Recall@K、MRR 或 NDCG，应另外构建中文问题集和人工/模型辅助审校的 qrels。

英文 QASPER 记录保留 `CC BY 4.0` 信息。中文记录保留论文页面、PDF 地址和质量字段，但公开下载地址不等于可再分发授权，使用或发布数据前应遵守《软件学报》及原始站点条款。

## 随机规模检索验证

使用固定种子对中英文分别按 1:1 抽取 10、20、30、40、50 篇论文，每篇论文生成一个检索问题；英文优先使用 QASPER 问题，中文使用论文标题探针。脚本会分别运行 BM25 和已配置的百炼语义路径，并将每组样本、单条结果和汇总报告写入 `eval/reports/`：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_mixed_fulltext_random_samples.py `
  --backend bailian `
  --with-reranker `
  --confirm-external-calls `
  --output-dir eval\reports\mixed-fulltext-random-rag-bailian-reranker-v1
```

该验证是论文级 Recall/MRR/nDCG 的随机 smoke test，不等价于中文人工 qrels 的证据段落评测。

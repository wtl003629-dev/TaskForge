# 中英文论文混合数据集

`bilingual-paper-corpus-v1` 是由仓库已有的公开数据构建的本地派生数据集：

- 英文：QASPER train + validation，共 1,169 篇全文论文、3,598 个问题；
- 中文：NeuCLIR-CSL，共 100,000 篇标题/摘要记录，其中包含全部 1,907 篇有正相关性标注的文献；
- 中文检索评测：NeuCLIR-Tech 的 110 个中文主题和对应 qrels；
- 总计：101,169 篇文献、3,708 个查询、15,231 条归一化 qrels。

正式数据位于：

```text
.taskforge/datasets/bilingual-paper-corpus-v1/
```

文件说明：

| 文件 | 内容 |
|---|---|
| `corpus.jsonl.gz` | 文档记录；英文为全文，中文为标题/摘要 |
| `queries.jsonl.gz` | QASPER 英文问题与 NeuCLIR 中文主题 |
| `qrels.jsonl.gz` | 统一的 `query_id/document_id/relevance` 记录 |
| `manifest.json` | 数量、来源、采样种子、SHA-256 和限制说明 |

重新构建默认版本：

```powershell
.\.venv\Scripts\python.exe scripts/build_bilingual_paper_dataset.py `
  --chinese-limit 100000 `
  --output-dir .taskforge/datasets/bilingual-paper-corpus-v1
```

要包含 CSL 的全部约 395,927 条记录：

```powershell
.\.venv\Scripts\python.exe scripts/build_bilingual_paper_dataset.py `
  --chinese-limit 0 `
  --output-dir .taskforge/datasets/bilingual-paper-corpus-all-v1
```

采样是按固定种子和文档 ID 的 SHA-256 排序确定的，可重复生成。中文 CSL
是摘要级数据，不应当描述为中文全文 PDF；如需中文全文，需要另行取得有
授权的 PDF，并通过现有 PDF 入库链路解析。

来源与授权信息以 `manifest.json` 及各来源的原始条款为准；重新分发派生
文件时应保留来源、版本和许可证说明。

如需将这批中文全文与英文 QASPER 全文放在同一个检索语料中，请使用
`mixed-paper-fulltext-v1`，详见
[`MIXED_PAPER_FULLTEXT_DATASET.md`](MIXED_PAPER_FULLTEXT_DATASET.md)。

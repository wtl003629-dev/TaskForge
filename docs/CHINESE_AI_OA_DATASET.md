# 中文 AI 开放全文数据集

当前版本：`chinese-ai-oa-jos-v2`

数据位于：

```text
.taskforge/datasets/chinese-ai-oa-jos-v2/
```

## 内容

- 来源：`Journal of Software / 软件学报` 公开期刊页面；
- 时间范围：2000–2026；
- AI 相关候选：1,241 篇；
- 通过全文质量门禁：1,152 篇；
- 主语料保留：1,000 篇；
- 全文平均抽取长度：约 41,882 字符；
- 每篇均有本地 PDF、来源页面、PDF URL、DOI/CSTR（若页面提供）和 SHA-256。

## 文件

| 文件 | 内容 |
|---|---|
| `corpus.jsonl.gz` | 1,000 篇可直接用于 RAG 的中文全文记录 |
| `papers.jsonl.gz` | 不含正文的论文元数据 |
| `pdfs/` | 下载的原始 PDF |
| `candidates.json` | 全部候选论文及筛选分数 |
| `records.jsonl` | 下载、解析和质量门禁结果 |
| `quality_report.json` | 可审计的逐篇质量报告 |
| `manifest.json` | 版本、数量、门禁和来源说明 |

## 质量门禁

候选必须满足：标题或摘要命中 AI 控制词；标题不是前言、通知、勘误等非研究文章；PDF 至少 4 页；可提取正文至少 6,000 字符；抽取正文中文比例至少 0.12；同一 JOS 文章 ID 和相同 PDF SHA-256 去重。

## 重新构建

```powershell
.\.venv\Scripts\python.exe scripts\build_chinese_ai_oa_jos_dataset.py `
  --limit 1000 `
  --candidate-limit 1300 `
  --min-year 2000 `
  --max-year 2026 `
  --output-dir .taskforge/datasets/chinese-ai-oa-jos-v2
```

脚本会复用已下载文件并支持中断后继续。公开可下载不等于允许再分发，使用或发布数据时应保留来源 URL，并遵守期刊网站和论文各自的版权/许可条款。

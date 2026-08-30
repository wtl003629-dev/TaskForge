# TaskForge PDF RAG 流程

本文说明 PDF 从进入 TaskForge 到成为可引用证据的完整过程。技术标识保留英文，说明文字使用中文。

## 总体链路

```text
PDF 或 Zotero 全文
  → 文件与权限校验
  → 原生解析器 / MinerU
  → 统一 DocumentBlock
  → 分块与证据身份生成
  → BM25 与向量索引
  → 查询召回与 RRF 融合
  → qwen3-rerank 重排
  → 去重与质量过滤
  → Evidence Card
```

RAG 只处理已经进入当前 `ResearchScope`、成功解析并建立索引的论文。论文标题或摘要不能在全文缺失时冒充正文证据。

## PDF 获取方式

全文可以来自：

- 合法开放获取 PDF 自动下载；
- 用户手动上传；
- 本机 Zotero 只读同步。

TaskForge 会检查文件大小、PDF 魔数、解析结果和论文身份。Zotero 条目必须通过 DOI 或标题、年份匹配，避免把错误附件绑定到所选论文。

## 解析器选择

可直接提取文本的 PDF 先使用 `pypdf` 和 `pdfplumber`。如果页面是扫描图像、阅读顺序异常、表格或布局恢复不足，`auto` 路由会调用 MinerU。

推荐配置：

```dotenv
TASKFORGE_PDF_PARSER_BACKEND=auto
TASKFORGE_MINERU_BASE_URL=http://127.0.0.1:8001
TASKFORGE_MINERU_EXPECTED_VERSION=3.4.4
TASKFORGE_MINERU_BACKEND=pipeline
TASKFORGE_MINERU_PARSE_METHOD=auto
TASKFORGE_MINERU_EFFORT=high
```

MinerU 作为独立服务运行。原始解析结果按 PDF 哈希和解析配置缓存，再统一转换为 `DocumentBlock`，不会把 MinerU 私有 Schema 直接传入检索层。

## 解析质量检查

质量报告会记录：

- 页面覆盖率；
- 乱码比例；
- 重复页眉和页脚；
- 阅读顺序异常；
- 空表格和孤立图注；
- 未解析图片；
- 是否使用 OCR；
- 解析器名称和版本。

解析失败时论文保持失败状态，不会建立空索引，也不会静默改用摘要。

## 统一文档结构

`DocumentBlock` 是解析器无关的最小结构，包含：

- 页码和 bbox；
- 阅读顺序；
- 标题、段落、表格、公式、图片、代码等类型；
- 文本或结构化内容；
- 内容哈希；
- 相邻块和章节关系。

表格、公式、图片、代码和算法块会尽量保持原子性，图注会绑定到相邻对象。

## 分块方式

TaskForge 支持：

- `flat`：固定长度文本块；
- `parent_child`：较大的 Parent 作为阅读上下文，较小的 Child 作为检索和引用单位；
- `hybrid`：Flat 主通道加 Child 辅助通道；
- `sliding`：滑动窗口实验通道。

当前本地运行配置使用 `parent_child`。用户选定论文后的正式冻结评测基线使用 `Flat 2000/0`，两者不能当作完全相同的链路。修改分块方式后必须重新解析或重建相关论文索引。

在 `parent_child` 模式中：

- Child 用于检索、排序和引用；
- Parent 只用于补充阅读上下文；
- `paper_read` 可以从 Child 展开到同文档、同版本的 Parent；
- 引用核验仍检查 Child 原文，不能用更宽的 Parent 替代。

## 向量化与索引

当前本地百炼组合为：

```text
Embedding：text-embedding-v4
维度：1024
Reranker：qwen3-rerank
生成模型：qwen-plus
```

正文在入库时完成向量化。缓存键包含 Provider、模型、维度、文本类型和内容哈希，避免把不同模型的向量混用。

`BAAI/bge-small-en-v1.5` 和 BM25-only 路径仍可作为本地替代配置，但不能与百炼索引直接混合查询。

## 查询与排序

当前证据检索顺序为：

```text
用户问题
  → Scope、tenant、ACL、版本和全文状态过滤
  → BM25 召回
  → Dense 向量召回
  → RRF 合并 Candidate@50
  → qwen3-rerank
  → 结构和来源补充排序
  → 重复与低质量片段过滤
  → 返回 Evidence Cards
```

Agent 默认只看到有限数量的证据卡，完整候选轨迹保留在 Host 和离线评测中，避免把大段检索轨迹塞入模型上下文。

## 证据质量过滤

入库和返回前会过滤：

- 参考文献、Bibliography、Works Cited 等目录；
- 只有标题、关键词、作者邮箱或模板字段的片段；
- `relevant doc 1` 等占位表格；
- 空文本和过短无意义内容；
- 同一论文内高度重合或重复的片段；
- 解析得到的元数据包装文本。

前端默认显示短证据片段，用户可以按需查看完整原文。截断只影响展示，不会改变 Evidence ID 和引用关系。

## Evidence ID

Evidence ID 绑定：

- tenant 和 Scope；
- Scope 版本；
- 论文和文档版本；
- chunk；
- 原文位置与哈希。

因此 Writer 不能自行编造 Evidence ID，也不能把其他论文或旧 Scope 的证据带入当前报告。

## 当前评测基线

selected-paper 冻结基线覆盖 60 篇真实中英文 PDF 和 177 个问题，使用 MinerU 3.4.4、Flat 2000/0、BM25、百炼 `text-embedding-v4`、RRF 和 `qwen3-rerank`。

| 指标 | 结果 |
|---|---:|
| 整体 Recall@10 | 92.62% |
| 中文 Recall@10 | 95.00% |
| 英文 Recall@10 | 90.15% |

完整冻结配置见 [`paper-scoped-flat-bailian-v1.json`](../eval/baselines/paper-scoped-flat-bailian-v1.json)。评测任务限定为用户已经选定论文后的全文问答，不代表开放论文发现效果。

## 相关配置

```dotenv
TASKFORGE_GENERAL_TEXT_BACKEND=bailian
TASKFORGE_BAILIAN_MODEL=text-embedding-v4
TASKFORGE_BAILIAN_EMBEDDING_DIMENSION=1024
TASKFORGE_RESEARCH_RERANKER_BACKEND=bailian
TASKFORGE_BAILIAN_RERANK_MODEL=qwen3-rerank
TASKFORGE_PDF_CHUNKING_MODE=parent_child
TASKFORGE_RESEARCH_DUAL_ROUTE_ENABLED=false
```

详细评测规则见 [评测说明](EVALUATION.md)。

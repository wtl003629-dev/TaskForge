# TaskForge 论文研究流程

本文说明 TaskForge 当前的论文发现、全文入库、RAG 证据检索和四 Agent 报告生成流程。协议名、环境变量和工具名保留英文，说明文字使用中文。

## 整体流程

```text
用户提出问题
  → 多来源发现候选论文
  → 用户选择论文
  → 系统创建 ResearchScope
  → 获取、同步或上传全文
  → 解析、分块并建立索引
  → 在已选论文中检索证据
  → Planner / Evaluator / Writer / Critic
  → 生成报告草稿
  → 人工核对引用
```

“发现论文”和“在论文中找证据”是两个不同阶段：前者只处理候选元数据，后者只使用已经成功解析和索引的全文。

## 论文发现

当前接入四个开放文献来源：

- Semantic Scholar
- OpenAlex
- arXiv
- Crossref

系统同时保留中文原始查询和英文改写查询。中文问题还会增加 OpenAlex `language:zh` 检索通道，最后合并、去重并重新排序。

用户可以设置：

- 返回 30、50 或 100 篇候选论文；
- 综合、中文优先或英文优先；
- 年份范围；
- 必须包含和需要排除的词；
- 是否根据引用和被引关系扩展候选。

候选卡片会展示作者、年份、期刊或出版社、引用量、DOI、来源链接和可核对状态。系统会优先保留 DOI、arXiv ID、OpenAlex ID 等可验证标识，降低把普通公司文章误当作学术论文的概率。

## 选择论文与 ResearchScope

用户确认论文后，Host 创建 `ResearchScope`。它记录：

- 用户选择和排除的论文；
- Scope 版本；
- tenant、user 和 conversation；
- 每篇论文的全文与索引状态。

Agent 只能读取 Scope，不能自行添加或删除论文。用户批准扩展后会生成新的 Scope 版本，旧报告仍绑定原版本。

## 全文获取与入库

系统按以下顺序准备全文：

1. 尝试下载合法开放获取 PDF；
2. 如果论文受登录或权限限制，可以从本机 Zotero 只读同步；
3. 用户也可以手动上传 PDF；
4. 系统校验文件、解析正文并建立检索索引；
5. 无法获得全文的论文会保留原因和来源链接，但不会用摘要代替正文。

Zotero 只负责使用用户已有权限保存论文。TaskForge 不读取账号密码，也不会绕过登录、付费墙或版权限制。详细配置见 [Zotero 接入说明](ZOTERO_MCP.md)。

## RAG 在流程中的位置

RAG 不是第五个 Agent，而是四 Agent 共用的证据层：

```text
全文解析和索引
    ↓
Planner 决定要查什么
    ↓
Evaluator 调用 paper_search 检索证据
    ↓
Writer 使用 Evidence ID 写报告
    ↓
Critic 对照原文检查引用
```

检索只在当前 `ResearchScope` 内执行。候选经过 BM25、向量召回、RRF 融合和 reranker 后，返回带页码、章节、论文 ID 和 `Evidence ID` 的证据卡。

## 四 Agent 职责

| Agent | 主要任务 | 结构化输出 |
|---|---|---|
| Planner | 把问题拆成最多 3 个子问题、4 项证据要求和 5 个提纲项 | `PlannerHandoff` |
| Evaluator | 根据计划检索全文，筛选证据并记录缺口 | `EvidenceLedger` |
| Writer | 只使用已返回的 Evidence ID 撰写报告 | `DraftArtifact`、`ClaimManifest` |
| Critic | 检查事实、逻辑和引用，提出保留、修改、删除或补证建议 | `ReviewPatch`、verdict |

角色之间不复制整篇论文。原文保存在 Host，交接内容主要是 ID、短片段和结构化状态。

## 证据与引用

每张证据卡至少包含：

- `Evidence ID`
- 论文 ID 与标题
- 页码和章节
- 证据类型
- 原文片段
- 检索得分与来源

系统在入库阶段过滤参考文献目录、模板表格和明显占位内容；在检索阶段继续合并重复或高度重合的片段。

Writer 的主要结论必须绑定 Evidence ID。Critic 可以读取对应原文并执行引用核验。没有证据支持的结论会被删除、修改或标记为需要补证。

## 报告状态

四角色完成后，报告进入：

```text
waiting_human_review
```

这表示草稿和引用已经生成，正在等待用户核对，并不代表系统已经替用户确认所有学术结论。

## 主要 API

```text
POST /api/literature/search
POST /api/literature/expand-citations
POST /api/research/scopes
POST /api/research/scopes/{scope_id}/ingest
PUT  /api/research/scopes/{scope_id}/papers/{paper_id}/pdf
POST /api/research/scopes/{scope_id}/papers/{paper_id}/zotero
POST /api/research/evidence/search
GET  /api/research/scopes/{scope_id}/evidence/{evidence_id}
POST /api/research/scopes/{scope_id}/agent-run
GET  /api/zotero/status
GET  /api/zotero/items
```

## 论文 MCP

论文研究能力也可以通过 stdio 或 HTTP `/mcp` 暴露，当前包含 8 个工具：

```text
literature_search
literature_expand
literature_get
scope_get
paper_search
paper_read
citation_verify
scope_expansion_request
```

Scope 创建、修改和确认仍由 Host API 负责，不向 Agent 开放写权限。

启动 MCP Server：

```powershell
.\.venv\Scripts\python.exe scripts\run_research_mcp.py `
  --transport stdio --tenant local --user researcher

.\.venv\Scripts\python.exe scripts\run_research_mcp.py `
  --transport http --host 127.0.0.1 --port 8765
```

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q
cd frontend
pnpm build
```

当前 selected-paper 全文检索基线和复现口径见 [评测说明](EVALUATION.md)，PDF 解析与检索细节见 [PDF RAG 流程](PDF_RAG_PIPELINE.md)。

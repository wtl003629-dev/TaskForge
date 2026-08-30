# TaskForge

[![CI](https://github.com/wtl003629-dev/TaskForge/actions/workflows/ci.yml/badge.svg)](https://github.com/wtl003629-dev/TaskForge/actions/workflows/ci.yml)

TaskForge 是一个面向论文检索、全文问答和研究报告生成的 **论文研究助手**。用户输入研究问题后，系统会从 Semantic Scholar、OpenAlex、arXiv 和 Crossref 查找候选论文；用户选定范围后，系统获取可用全文、建立索引、找出可引用的证据片段，再生成一份等待人工核对的中文报告。

```text
提出问题 -> 多源检索 -> 选择论文 -> 获取/同步全文
         -> 证据检索 -> 生成中文报告 -> 人工核对引用
```

为了避免模型自行扩大研究范围，用户选中的论文会被固化为不可由 Agent 改写的 `ResearchScope`。Planner、Evaluator、Writer、Critic 只在这个范围内计划、筛选证据、写作和审校。底层仍是 Provider-neutral、权限受控、可恢复的 Agent Harness：模型提出 `ToolRequest`，宿主负责身份、权限、执行、幂等、checkpoint 和证据账本。

## 当前能运行什么

论文研究工作台是当前产品主线，不再以旧版的通用运行台或企业审查页作为主要入口。目前可以：

- 用中文或英文检索论文，并选择“综合、中文优先、英文优先”；中文查询会额外走 OpenAlex 中文通道，再与英文语义检索结果合并；
- 合并 Semantic Scholar、OpenAlex、arXiv、Crossref 的结果，按 DOI、标题等信息去重，并展示作者、出版来源、引用量和可核对状态；
- 自动下载合法开放获取的 PDF；受限论文可从本机 Zotero 只读同步，也可手动上传 PDF；
- 对成功解析的全文建立索引，过滤重复片段、参考文献目录和明显的占位内容；
- 在已选论文内提问，查看短版证据卡和完整原文，并生成带编号引用的中文研究报告；
- 让 Planner、Evaluator、Writer、Critic 完成结构化交接，最终进入 `waiting_human_review`，由用户核对引用。

同一套 `AgentRuntime` 仍保留通用研究、代码库诊断、文档审阅和企业变更审查能力，供开发与评测使用；它们不是当前前端的主产品入口。

完整演示链路是：

```text
Task -> governed retrieval/grep -> ToolResult receipt
     -> artifact_write proposal -> human approval
     -> durable artifact/evidence -> final answer

ReviewCase -> intake -> [compliance, risk] -> decision recommendation
           -> waiting_human_review -> human approve/reject

PaperResearch -> planner -> evaluator -> writer -> critic
              -> waiting_human_review -> human approve/reject
```

Demo Provider 是确定性离线状态机，用来证明真实工具、审批、持久化和恢复链路；它不会伪装成 LLM。设置凭据后可显式切换到 OpenAI Responses Provider。Native function call 会被归一化为同一个 `ToolRequest`，工具输出通过 `previous_response_id + function_call_output` 续接。

为避免把“有代码”误写成“已上线”，本文统一使用四层状态：

| 层级 | 当前含义 | 当前可证明的范围 |
|---|---|---|
| 模块/契约存在 | 有实现、Schema 或 migration | 只能证明接口和静态设计存在 |
| fake/离线/本地已测 | 单测、模拟 HTTP、fake driver、Demo Provider 或本机进程通过 | 证明受控输入下的契约与本地集成，不证明真实服务质量 |
| 产品主链已接入 | FastAPI/Workbench 的默认或可配置运行路径会调用该能力 | 仍不等于外部依赖、生产认证或规模化部署已验证 |
| live 已验证 | 使用真实凭据和真实外部服务显式执行并留存结果 | 只有这一层才能声明对应真实服务链路通过 |

当前默认使用 PostgreSQL，已覆盖 Task/Profile/Run、操作队列、编排、ReviewCase、Verification、Knowledge/Memory、文献仓库、Provider cache 和 embedding cache；选择 `TASKFORGE_DATABASE_BACKEND=sqlite` 才会启用 SQLite 兼容/测试路径，PostgreSQL 连接失败时不会回退到 SQLite。RAG 会先执行 tenant/ACL、Scope 版本、有效期和知识库过滤，pgvector 提供 exact cosine 主路径和显式 opt-in 的 HNSW 路径。现有数据库升级时需依次应用 `migrations/postgres/` 下的迁移脚本。

## 核心能力

- 有界 Agent Loop：step budget、结构化失败、普通工具错误可观察并允许模型恢复；
- Tool Gateway：严格 JSON Schema、allowlist、风险分级、超时与输出上限；
- 审批与幂等：写入/外部/破坏性能力暂停，同一 call/key 换参会 fail closed；
- Durable checkpoint：Task、Profile、Run、pending approval 和 receipt 可跨进程重载；SQLite 与 PostgreSQL 共享同一业务契约；
- 持久化上下文：Knowledge/Memory 支持 SQLite 兼容路径和 PostgreSQL RLS 路径，支持重启恢复、版本替换、过期与租户/ACL/scope 过滤；
- Durable Worker：排队执行、原子 claim、租约心跳、owner/token/version/expiry CAS、显式瞬态 Provider 失败退避重试、末次租约恢复核对与 dead letter；
- MCP client（固定握手式旧版 `2025-11-25`）：仅从宿主 JSON 配置挂载 allowlist 工具，执行前仍经过本地风险、审批和 Schema；
- 审计与指标：append-only 事件、secret-like 字段拒绝、run/tool 成功率、p50/p95、token/cost 与 safety 计数；
- 安全代码检索：不执行 shell；限制工作区、glob、regex、文件大小、结果数和时间；
- RAG：产品主链在 tenant-first ACL、版本/有效期过滤后，按查询与可见语料选择通用 BM25/可选 BGE dense、表格数值 BM25+特征重排、跨文档 source-coverage RRF 或 PDF 结构字段+邻接召回；选择结果和实际 backend 会写入上下文/工具回执；
- 结构化 PDF RAG：`pypdf/pdfplumber` 段落与表格提取，并持久化 block 类型、表格行、页码、bbox 与邻接 provenance；默认在线路径使用结构字段 BM25 和相邻块扩展，本地 Qdrant named dense/sparse 与 server-side RRF 仍属于离线/显式 opt-in 路径；
- 通用文本图谱实验：`LocalEvidenceGraph` 在候选集内使用文档/章节/邻接/实体结构做可审计重排，并提供受限 1--2 跳候选扩展接口；仅 `general_text` 路由 opt-in，未替换表格、跨文档和 PDF 默认路径；
- Memory：tenant/org/user/agent/task 五级 scope、过期时间和 provenance；
- 多角色编排：固定有向无环图、角色 capability、RoleRun 尝试/恢复、分层上下文、handoff、proposed/verified fact 与一次性 host verification receipt；引用有据的 claim（其 evidence refs 全部来自该角色本次运行真实检索到的 knowledge_search 回执）由宿主自动签发 `authority=tool` receipt 并置为 verified，随后向下游依赖角色创建 handoff；未检索到引用的 claim 保持 `model_untrusted`/`proposed`；
- 业务决策边界：模型只能提交结构化建议，case 状态和最终批准由宿主状态机与人工身份控制；
- 持久化基础设施：PostgreSQL 是默认 durable backend，SQLite 仅由 `TASKFORGE_DATABASE_BACKEND=sqlite` 显式选择；本地 Compose 的 PostgreSQL/pgvector 已完成启动、连接和索引验证，Neo4j 1/2-hop 图检索及图/向量 RRF 融合仍默认关闭；
- Vue Workbench：Profile/Skill 选择、inline/queued Run、Job 轮询、轨迹、Tool Call、Evidence、批准/拒绝、Audit/Metrics 与脱敏 MCP 状态；
- 离线评测：task success、工具使用、终态、步数和 safety hard-fail 指标。

## 目录

```text
backend/taskforge/
  app.py               FastAPI 与身份/审批边界
  runtime.py           Provider-neutral Agent loop
  domain.py            Task/Run/Step/Tool 领域契约
  tooling.py           Tool registry 与 capability policy
  security.py          安全 grep/read/calculator
  knowledge.py         ACL/版本感知知识检索
  memory.py            作用域记忆
  context.py           带引用和预算的上下文组装
  checkpoints.py       SQLite durable snapshots
  persistent_context.py SQLite Knowledge/Memory backend
  operations.py        Queue lease/CAS、审计与指标
  worker.py            Durable worker
  mcp.py               受控 MCP Streamable HTTP 客户端
  openai_provider.py   显式启用的 Responses HTTP provider
  demo.py              无网络、可恢复的离线 Provider
  evaluation.py        轨迹级评测
  document_ingestion.py PDF/表格与 provenance 摄取
  hybrid_retrieval.py  BM25/Qdrant/RRF/rerank 检索
  rag_profiles.py      基于查询/可见语料特征的四类路由
  routed_knowledge.py  在线 KnowledgeStore 检索路由适配层
  orchestration.py     多角色计划、RoleRun、事实、handoff 与私有记忆
  case_runtime.py      固定角色计划到 AgentRuntime 的安全桥接
  review_cases.py      企业审查业务状态机与人工决策
  review_service.py    四角色业务协调 saga
  postgres_context.py  可选 PostgreSQL context adapter
  graph_retrieval.py   可选 Neo4j 图检索与融合 gate
frontend/              Vue 3 + TypeScript 工作台
eval/cases.json        可复现离线用例
docs/                  架构与威胁模型
```

## 快速开始（推荐使用 Docker Compose）

当前 Compose 会启动前端、后端、PostgreSQL 和 Worker。仓库按下面的同级目录组织时，可以同时初始化 TaskForge 与 PatchPilot 的数据库：

```text
my-coding/
  TaskForge/
  PatchPilot/
```

首次启动先复制环境变量模板，并至少填写三个 PostgreSQL 密码。真实模型的 API Key 只放在本地 `.env`，不要提交到 Git：

```powershell
cd D:\my-coding\TaskForge
Copy-Item .env.example .env
# 编辑 .env，填写 TASKFORGE_POSTGRES_ADMIN_PASSWORD、
# TASKFORGE_POSTGRES_TASKFORGE_PASSWORD、TASKFORGE_POSTGRES_PATCHPILOT_PASSWORD
docker compose up -d --build
docker compose ps
```

启动后访问：

- 前端：`http://127.0.0.1:5173`
- 后端 API：`http://127.0.0.1:8000`
- API 文档：`http://127.0.0.1:8000/docs`

默认 `TASKFORGE_PROVIDER=demo`，适合验证界面和流程。需要生成真实研究报告时，请在 `.env` 中配置百炼、OpenAI 或 DeepSeek；当前本地开发组合是 PostgreSQL + 阿里云百炼 `qwen-plus`。

### 使用论文研究助手

1. 输入研究问题，选择结果数量和语言偏好；
2. 从候选列表中选论文，核对作者、期刊/出版社和来源链接；
3. 保存清单后，让系统尝试开放获取下载；受限全文可从 Zotero 同步或手动上传；
4. 等全文完成解析后，在已选论文中提问并查看证据；
5. 点击“生成研究报告”，等待状态进入“待核对”，再逐条检查引用。

“找到 10 条证据”表示找到了 10 个论文片段，不是 10 篇论文；“覆盖 100%”只表示每篇已选论文至少命中一个片段，不代表全文都已覆盖。

## 从源码启动（Windows PowerShell）

后端要求 Python 3.11+：

```powershell
cd D:\my-coding\TaskForge
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
python -m uvicorn taskforge.app:create_app --factory --reload
```

如需执行 `execution_mode=queued` 的任务，再开一个 Worker 终端：

```powershell
cd D:\my-coding\TaskForge
.\.venv\Scripts\Activate.ps1
python scripts\run_worker.py
```

`POST /api/runs` 省略 `execution_mode` 时仍为 `inline`；设置为 `queued` 时返回 `202 + pending RunState`。可通过 `/api/runs/{run_id}/job`、`/audit` 与 `/api/metrics?run_id=...` 查看恢复和观测状态。抵达 `waiting_approval` 后，Job 已完成本次 operation；审批由 API 内联恢复，不会重放已持久化 receipt。

打开另一个终端启动前端（Node 20+ / pnpm）：

```powershell
cd D:\my-coding\TaskForge\frontend
pnpm install --frozen-lockfile
$env:VITE_ENABLE_MOCK_FALLBACK="false"
pnpm dev
```

访问 `http://127.0.0.1:5173`。API 文档位于 `http://127.0.0.1:8000/docs`。

默认身份头为 `X-TaskForge-Tenant: local` 和 `X-TaskForge-User: demo`，仅适合本地演示，不等于生产认证。

## 切换阿里云百炼 Provider

复制 `.env.example` 为 `.env`，填写：

```dotenv
TASKFORGE_PROVIDER=bailian
TASKFORGE_BAILIAN_API_KEY=...
TASKFORGE_BAILIAN_CHAT_MODEL=qwen-plus
```

百炼路径使用 OpenAI-compatible Chat Completions，并把结构化工具调用接回同一套宿主权限和证据链。API Key 或模型名缺失时服务会直接报错，不会悄悄改用 Demo Provider。

## 切换真实 OpenAI Provider

复制 `.env.example` 为 `.env`，填写：

```dotenv
TASKFORGE_PROVIDER=openai
TASKFORGE_OPENAI_API_KEY=...
TASKFORGE_OPENAI_MODEL=你的可用模型名
```

未同时提供 key 和 model 时服务会 fail fast，不会静默回退到 mock。Provider 实现遵循 OpenAI 官方的 [function calling flow](https://developers.openai.com/api/docs/guides/function-calling)：应用执行函数并回传与 `call_id` 对应的输出；模型从未获得 Python callable 或宿主权限。

通用 OpenAI 路径的单测仍只覆盖离线 runtime 和模拟 HTTP 契约，不代表真实 OpenAI 模型效果。填好 key 与 model 后，可显式执行一条真实、可能计费的 native tool-calling 冒烟测试：

```powershell
.\.venv\Scripts\python.exe scripts\run_live_openai_smoke.py --confirm-live-call
```

该脚本要求模型调用一次受控 `calculator`，再通过 `previous_response_id + function_call_output` 完成回答；未带 `--confirm-live-call` 时会在任何网络请求前退出。只有这条脚本真实通过后，才能声明当前凭据、模型和网络环境的 live API 链路已验证。

真实模型端到端验证已跑通论文发现、全文检索和四角色报告链路，最终进入 `waiting_human_review`。经过上下文压缩后，同任务 Token 消耗下降约 71%，引用仍绑定在用户选定的论文范围内。

## 切换 DeepSeek Provider

DeepSeek 走 OpenAI 兼容的 Chat Completions 协议（`/chat/completions`），不是 Responses API。复制 `.env.example` 为 `.env`，填写：

```dotenv
TASKFORGE_PROVIDER=deepseek
TASKFORGE_DEEPSEEK_API_KEY=...
TASKFORGE_DEEPSEEK_MODEL=deepseek-chat
```

同样 fail fast：未同时提供 key 和 model 时服务会拒绝启动。续接不依赖服务端会话状态；每次请求由宿主根据 checkpoint 重建完整消息历史（assistant `tool_calls` + `tool` role 回执），因此可安全跨重启重放，不会重复执行工具。

真实冒烟测试（同样需要 `--confirm-live-call`）：

```powershell
.\.venv\Scripts\python.exe scripts\run_live_deepseek_smoke.py --confirm-live-call
```

该脚本要求模型调用一次受控 `calculator`，再通过完整消息历史续接完成回答；只有真实通过后才能声明当前 DeepSeek 凭据、模型和网络环境的 live API 链路已验证。

## 当前评测结果

README 只展示当前论文研究主链的正式基线；历史实验和未采用方案保留在评测目录中，不再放在项目首页。

### 已选论文全文检索

当前冻结基线覆盖 30 篇中文和 30 篇英文真实 PDF，共 177 个标注问题。固定链路为 MinerU 3.4.4 解析、Flat 2000/0 分块、BM25 + 百炼 `text-embedding-v4`、RRF 融合和百炼 `qwen3-rerank`，并按用户选中的单篇论文做硬过滤。

| 指标 | 当前结果 |
|---|---:|
| 整体 Recall@10 | **92.62%** |
| 中文 Recall@10 | **95.00%** |
| 英文 Recall@10 | **90.15%** |

这组结果适用于“用户已经选定论文后的全文问答”，不能与全库论文发现混为同一任务。冻结配置、数据范围、哈希和回归门槛见 [`eval/baselines/paper-scoped-flat-bailian-v1.json`](eval/baselines/paper-scoped-flat-bailian-v1.json)。

### 报告生成

- 真实模型四角色链路已完成 Planner、Evaluator、Writer、Critic 全流程，并进入人工核对阶段；
- 在同一历史配对评测中，端到端答案 Token F1 相对初始基线提升 **36.35 个百分点**，README 不再展示绝对答案分数；
- 上下文与交接压缩后，同任务 Token 消耗下降约 **71%**；
- 报告引用必须来自当前 `ResearchScope` 内已经返回的 Evidence ID。

完整评测协议和复现方式见 [`docs/EVALUATION.md`](docs/EVALUATION.md) 与 [`docs/PAPER_RESEARCH_AGENT.md`](docs/PAPER_RESEARCH_AGENT.md)。

```powershell
python scripts\run_eval.py --output .taskforge\eval-report.json
```

## 持久知识与 Memory

默认 `TASKFORGE_DATABASE_BACKEND=postgres`，必须配置 `TASKFORGE_DATABASE_URL`；切换到 SQLite 仅用于显式兼容测试或迁移工具。首次使用 PostgreSQL 前应应用 `migrations/postgres/` 中的 schema，并完成数据、RLS、备份和回滚门禁。用户 Memory 可通过 `/api/memory` 创建、检索，并删除自己拥有的 user/agent/task scope 记录；tenant/org 共享记录的可见性不会自动授予删除权。Agent 只能经 `memory_remember` 能力写入，并受 profile、审批、幂等和宿主绑定的 tenant/scope 约束。

将工作区内 UTF-8 文档安全摄取到知识库：

```powershell
python scripts\ingest_knowledge.py docs\ARCHITECTURE.md `
  --knowledge-base taskforge --version 2026-08-04 --version-order 1
```

摄取器拒绝路径穿越、symlink/reparse point、二进制、凭据型文件和超限输入，并原子替换同一文档版本。默认演示文档位于 `taskforge`；四角色企业审查 profile 只检索 `enterprise-review`，所以要让审查角色获得政策证据，operator 必须把对应政策文档另行摄取到该知识库，不能依赖用户输入的“证据 ID”代替真实检索。

`postgres_runtime.py`、各 PostgreSQL store、`migrations/postgres/002_taskforge_runtime.sql` 与迁移工具提供完整 PostgreSQL 契约。若 operator 要在独立测试库验证它，必须先应用 Compose 初始化脚本或按以下顺序执行，任一步失败即停止：

```powershell
psql "$env:TASKFORGE_POSTGRES_ADMIN_DSN" -v ON_ERROR_STOP=1 `
  -f migrations/postgres/002_taskforge_runtime.sql
python scripts/migrate_sqlite_to_postgres.py --dry-run
python scripts/migrate_sqlite_to_postgres.py --execute `
  --database-url "$env:TASKFORGE_DATABASE_URL"
python scripts/migrate_sqlite_to_postgres.py --verify `
  --database-url "$env:TASKFORGE_DATABASE_URL"

python scripts/freeze_rag_query_set.py `
  --output ".taskforge\migration\rag-query-vectors.json" `
  --tenant-id local --acl tenant --acl user:demo

python scripts/verify_pgvector_retrieval.py `
  --queries ".taskforge\migration\rag-query-vectors.json" `
  --sqlite-source-root ".taskforge" `
  --database-url "$env:TASKFORGE_DATABASE_URL" `
  --tenant-id local --acl tenant --acl user:demo `
  --exact-only `
  --report ".taskforge\migration\rag-pgvector-exact-report.json"

# Only after the exact migration/RAG gate passes:
psql "$env:TASKFORGE_POSTGRES_ADMIN_DSN" -v ON_ERROR_STOP=1 `
  -f migrations/postgres/003_taskforge_hnsw.sql

# Existing databases need this narrow grant before Zotero/PDF re-indexing:
psql "$env:TASKFORGE_POSTGRES_ADMIN_DSN" -v ON_ERROR_STOP=1 `
  -f migrations/postgres/004_literature_evidence_replace.sql

python scripts/verify_pgvector_retrieval.py `
  --queries ".taskforge\migration\rag-query-vectors.json" `
  --sqlite-source-root ".taskforge" `
  --database-url "$env:TASKFORGE_DATABASE_URL" `
  --tenant-id local --acl tenant --acl user:demo `
  --report ".taskforge\migration\rag-pgvector-report.json"
```

应用角色没有 DDL 权限，迁移工具以只读方式打开 SQLite，并以批次、幂等冲突处理和内容/状态/主键/外键校验导入 PostgreSQL。`freeze_rag_query_set.py` 不会重新调用百炼，只冻结现有 1024 维查询缓存并记录 SQLite+NumPy Top-K 基线；`verify_pgvector_retrieval.py` 再比较 SQLite+NumPy、PostgreSQL exact 与 HNSW，记录 Recall@5/10/50、MRR@10、NDCG@8/10、Agent-visible Recall 和 P50/P95 延迟。只有 schema、迁移校验、exact/HNSW 对照、备份和恢复演练全部通过，才能把对应部署标记为 live。

## 受控 MCP

论文研究能力已封装为独立 MCP Server，支持 stdio 与 HTTP `/mcp`，向 TaskForge、Claude Code 和 Hermes 暴露 8 个工具：`literature_search`、`literature_expand`、`literature_get`、`scope_get`、`paper_search`、`paper_read`、`citation_verify`、`scope_expansion_request`。其中 `paper_search` 强制要求 ready Scope；Scope 创建、修改和确认不作为 Agent 工具暴露。

### Zotero 论文库（无需在 TaskForge 手动下载）

受限全文可以由用户在论文网站登录后，使用 Zotero Connector 保存到本机 Zotero；TaskForge 通过只读 Zotero MCP 读取该条目的元数据和附件全文，再建立自己的检索索引。Connector/Zotero 负责合法获取附件，TaskForge 不绕过登录、付费墙或版权限制，也不会让模型直接调用写入工具。Windows 本地桥接服务和 Docker 配置示例见 [`docs/ZOTERO_MCP.md`](docs/ZOTERO_MCP.md) 与 [`config/mcp.zotero.example.json`](config/mcp.zotero.example.json)。

```powershell
.\.venv\Scripts\python.exe scripts\run_research_mcp.py --transport stdio `
  --tenant local --user researcher

.\.venv\Scripts\python.exe scripts\run_research_mcp.py --transport http `
  --host 127.0.0.1 --port 8765
```

本机 Hermes `0.15.1` 已实际连接 stdio Server 并发现 8/8 工具；Claude Code `2.1.158` 已识别项目级 [`.mcp.json`](.mcp.json)，状态为首次交互批准前的 `Pending approval`。仓库自动测试覆盖 stdio JSON-RPC dispatcher、HTTP JSON-RPC 和 TaskForge MCP Client 互操作。配置细节见 [`docs/PAPER_RESEARCH_AGENT.md`](docs/PAPER_RESEARCH_AGENT.md)。

下文是 TaskForge 作为远程 MCP 客户端连接其他服务时的独立安全边界，不要和上面的论文 MCP Server 混为一谈。

远程 MCP 默认完全关闭。可从 `config/mcp.example.json` 复制配置并设置 `TASKFORGE_MCP_CONFIG_PATH`；配置文件必须把 server 绑定到已存在的 Agent profile，并为每个远程工具声明本地 policy，模型、远端 description 和 annotations 都不能修改权限。例如：

```json
{
  "servers": [{
    "profile_ids": ["research-agent"],
    "server": {
      "namespace": "docs",
      "endpoint": "https://mcp.example.com/mcp",
      "enabled": true,
      "allowed_tools": ["search"],
      "tool_policies": {
        "search": {
          "risk": "external",
          "side_effecting": false,
          "requires_approval": true,
          "description": "Search an operator-approved documentation service."
        }
      },
      "secret_env_var": "TASKFORGE_DOCS_MCP_TOKEN"
    }
  }]
}
```

通用远程客户端固定实现握手式协议修订 `2025-11-25` 的 initialize、分页 `tools/list` 和 `tools/call`。它每次请求做 DNS/IP preflight、禁重定向、流式限制响应大小，并在服务端选择 SSE 时明确 fail closed；当前只实现 JSON response 子集。远端返回 `isError=true` 会记为失败；side-effecting 工具若没有声明必填 string `idempotency_key`，会在挂载前被拒绝。这个 preflight 与 httpx 的实际连接解析不是 IP pinning，不能单独消除 DNS rebinding TOCTOU，生产部署仍需 egress proxy/firewall。MCP 官方在 2026-07-28 已把 current revision 更新为无连接态、per-request metadata 的 `2026-07-28`，本仓库尚未实现该新版，不能宣称“最新 MCP 全兼容”。

## Docker Compose

```powershell
docker compose up -d --build
docker compose ps
```

前端映射到 `5173`，后端映射到 `8000`，独立 Worker 与 API 共享 artifact volume；Compose 默认启动 PostgreSQL/pgvector，数据 bind 到 D 盘，并由独立数据库和应用角色承载运行状态。需要 SQLite 时，显式设置 `TASKFORGE_DATABASE_BACKEND=sqlite` 并使用源码启动路径。当前开发环境已实际验证 frontend、backend、postgres 和 worker 均能健康启动，PostgreSQL 持久化与 HNSW 索引可用；这属于本地集成验证，不等同于生产高并发或公网部署验收。

## 部署说明

- 本地 Compose 已验证 PostgreSQL/pgvector、后端、前端和 Worker 可以完整启动；
- 默认身份头只用于本地开发，公网部署前需要接入正式认证和 RBAC；
- 开放发现卡片是候选论文信息，只有成功解析并建立索引的全文才会用于证据和报告；
- 开放获取下载和 Zotero 同步都遵守原网站权限，不绕过登录、付费墙或版权限制；
- 横向扩容和公网高并发部署仍需要在目标服务器上完成压测与安全配置。

进一步设计见 [架构说明](docs/ARCHITECTURE.md) 与 [威胁模型](docs/THREAT_MODEL.md)。

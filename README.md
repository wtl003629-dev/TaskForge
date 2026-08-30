# TaskForge

[![CI](https://github.com/wtl003629-dev/TaskForge/actions/workflows/ci.yml/badge.svg)](https://github.com/wtl003629-dev/TaskForge/actions/workflows/ci.yml)

TaskForge 当前定位为一个 **交互式论文研究 Agent**：系统先根据用户需求执行多源开放文献发现，由用户选择研究论文；Host 随后创建不可由 Agent 改写的 `ResearchScope`，在所选论文内执行结构感知证据检索，最后由 Planner、Evaluator、Writer、Critic 四角色完成计划、筛证、写作和逐条审校。

```text
研究需求 -> 开放论文发现 -> 用户选文 -> ResearchScope
         -> PDF/摘要摄取 -> 有界证据检索 -> 四 Agent -> 人工复核
```

底层仍是 Provider-neutral、权限受控、可恢复的 Agent Harness：模型只提出 `ToolRequest`，宿主负责身份、权限、Schema、执行、幂等、checkpoint、证据账本和评测。论文原文不在角色间复制，模型也不是授权系统。

## 当前能运行什么

论文研究工作台是当前产品主链；同一套 `AgentRuntime` 也保留下列可配置能力，无需复制核心循环：

- 论文研究：四源候选发现、PaperCard 选择/排除、Scope 安全摄取、有界证据卡、四角色结构化交接；
- 通用研究与报告：ACL 过滤知识检索、作用域记忆召回、审批后生成报告；
- 代码库诊断：安全 `workspace_grep`、按行读取、证据化结论；
- 文档审阅：真实 PDF/表格分块、本地文档与知识库检索、版本/来源保留、结构化交付；
- 企业变更审查：受理、合规、风险、决策四角色固定 DAG，最终只生成 `model_untrusted` 建议，必须由人工批准或拒绝。

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

当前默认使用 PostgreSQL，正式 PostgreSQL 路径已覆盖 Task/Profile/Run、操作队列、编排、ReviewCase、Verification、Knowledge/Memory、文献仓库、Provider cache 和 embedding cache；选择 `TASKFORGE_DATABASE_BACKEND=sqlite` 才会启用 SQLite 兼容/测试路径，PostgreSQL 连接失败时不会回退到 SQLite。RAG 仍在 tenant/ACL、Scope 版本、有效期和知识库过滤后执行，pgvector 提供 exact cosine 主路径和显式 opt-in 的 HNSW 路径。真实 PostgreSQL/pgvector 验收需按 `../migration/README.md` 完成后，才能把对应能力标为 live。

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
- 持久化基础设施：PostgreSQL 是默认 durable backend，SQLite 仅由 `TASKFORGE_DATABASE_BACKEND=sqlite` 显式选择；Neo4j 1/2-hop 图检索和图/向量 RRF 融合仍默认关闭；未连接真实 PostgreSQL 服务时不宣称 live；
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

## 本地启动（Windows PowerShell）

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

`/api/review-cases` 的通用 `execution` 状态不会从 API key 推断成功；论文研究的真实 DeepSeek 业务验证保存在独立、不可混用的 E2E 报告中。当前一条完整四角色任务从预优化 `212,874` Token 降至 `62,186` Token，Scope 越界为 0；这证明该任务链路执行成功，不证明所有研究问题的回答质量或生产 SLA。

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

## 评测

论文研究采用三层互不替代的报告：

| 层级 | 数据量 | 当前结果 |
|---|---:|---|
| 开放论文发现 | 100 个真实需求、792 个已知相关 arXiv 标签 | 本机匿名 Provider live：Recall@20/50 `0.001/0.001`，**质量门禁失败**；336 个 Provider/查询组失败暴露了限流和跨语言查询短板 |
| 用户选文后的证据检索 | 414 个锁定资产 | TAT-QA `0.9902/0.9902`、QASPER B2 `0.6282/0.9738`、MultiHop `0.9199/0.9893`、PDF `1.0/1.0`（Recall@10/Candidate@50） |
| 用户直接上传 PDF 后的核心召回 | QASPER real-PDF strict track | MinerU 3.4.4 locked 100 题的冻结 Flat 对照通过 90%/90% 对齐门禁，段落 Recall@1/5/10/50 为 `0.2728/0.7367/0.8625/0.9830`，Agent-visible Recall@8 为 `0.8250`；旧 Parent–Child A/B 的 Recall@5/10 为 `0.7022/0.8447`、Agent-visible Recall@8 为 `0.7938`。当前默认已改为标题增强、Parent-aware 二次重排和 lineage diversity 的 Parent–Child 链路，但尚未重跑，未声称超过冻结对照（见 [chunking gate](eval/reports/qasper-pdf-chunking-gate-v2-top8.json)） |
| 端到端 | 用户上传硬边界回归 + 历史四 Agent 100 题 direct-answer replay | 当前 upload → PDF indexing → bounded evidence 回归通过；历史四 Agent Token F1 为 `0.4761`（旧基线 +36.35 个百分点），但使用旧 Parent–Child trace，仅作历史记录；当前 Flat 配置的 live E2E 暂不重复消耗 API |

此前 clean holdout 的页级重合数字已经全部作废，不能作为段落召回、验收或简历成果。当前链路采用原生文本快速抽取、MinerU 3.4.4 结构化回退、可选独立 VLM；flat 2000 字符、零 overlap、原始 Query 是当前锁定默认，生产搜索只向 Agent 暴露 8 个查询中心窗口。相较同解析/分块、无精排的单 Query locked baseline，Cross-Encoder 使段落 Recall@10 提升 2.70 个百分点、Recall@5 提升约 2.68 个百分点；Flat overlap、滑动窗口、多组 Parent–Child 参数和规则关键词查询均未稳定提升 Recall@5/@10，因此保留为显式 ablation。段落 Recall 只在 Gold→Child 对齐门通过时发布，详见 [`docs/PDF_RAG_PIPELINE.md`](docs/PDF_RAG_PIPELINE.md)、[`eval/reports/qasper-pdf-reranker-uplift-v1.json`](eval/reports/qasper-pdf-reranker-uplift-v1.json)、[`eval/reports/qasper-pdf-chunking-gate-v2-top8.json`](eval/reports/qasper-pdf-chunking-gate-v2-top8.json)、[`eval/reports/qasper-query-expansion-locked100-v1.json`](eval/reports/qasper-query-expansion-locked100-v1.json) 与 [`eval/reports/QASPER_RETRIEVAL_DEPRECATION.md`](eval/reports/QASPER_RETRIEVAL_DEPRECATION.md)。

开放发现只返回标题、来源链接和一句话介绍，不自动下载论文。用户选择并上传 PDF 后才建立 RAG；未上传论文禁止用摘要回退。开放发现低分不是有界检索低分，也不会被标题型冒烟覆盖。离线 20 题同义改写筛选与原始 Query 的 Recall@5/10、Agent-visible Recall@8 完全相同，因此没有进行全量同义改写或 API 调用；D 盘真实多语言模型两题 smoke 正确选择 multilingual route 并排在预期证据首位，但不作为中文质量提升结论。历史报告分别见：

- [`eval/reports/literature-discovery-full100-live.json`](eval/reports/literature-discovery-full100-live.json)
- [`eval/reports/paper-research-e2e-30-deterministic.json`](eval/reports/paper-research-e2e-30-deterministic.json)
- [`eval/reports/paper-research-business-e2e-live.json`](eval/reports/paper-research-business-e2e-live.json)
- [`eval/reports/qasper-query-expansion-synonym-screen20-v1.json`](eval/reports/qasper-query-expansion-synonym-screen20-v1.json)
- [`eval/reports/multilingual-retrieval-smoke-v2.json`](eval/reports/multilingual-retrieval-smoke-v2.json)

完整定位、复现命令和口径见 [`docs/PAPER_RESEARCH_AGENT.md`](docs/PAPER_RESEARCH_AGENT.md) 与 [`docs/EVALUATION.md`](docs/EVALUATION.md)。

```powershell
python scripts\run_eval.py --output .taskforge\eval-report.json
```

内置三类离线 case：research evidence、repository grep、approval denied。安全违规是 hard failure，不会被平均任务分数抵消。

真实 PDF/表格与混合检索的可复现实验：

```powershell
# 默认生成并评测 3 份合成 PDF / 12 个问题
python scripts\run_rag_experiment.py

# 下载已锁定且校验 SHA-256 的 TAT-QA labels 后，跑固定 100 case
python scripts\fetch_rag_eval.py --dataset tatqa-dev
python scripts\run_rag_experiment.py --dataset tatqa
```

当前主评测按产品需要拆成四个隔离场景：TAT-QA 题目自带上下文中的表格/数值证据、QASPER 长文档、MultiHop-RAG 可识别跨文档证据，以及合成 PDF/权限冒烟。历史 retained-capability 控制结果分别为 Recall@10 `0.9902`、`0.2206`、`0.9199` 和 `1.0000`；QASPER 的 `0.2206` 只代表旧的 100 题 BM25 审计控制，不是当前文档隔离调优结果。PDF 仅有 12 题，不能作为生产质量证明。完整矩阵见 [保留能力基线](eval/retrieval-retained-capabilities-20260811.json) 和 [评测协议](docs/EVALUATION.md)。

 QASPER 的新文档隔离调优集已经固定为 200 个训练题（按论文划分，历史 100 题不参与调参）。B0 BM25 为 Recall@10 `0.5170` / Candidate@50 `0.9068`；通过论文标题、章节上下文和真实本地 BGE small 向量，B2 达到 `0.6282` / `0.9738`，最新同代码产物 p95 `10.04 ms`，配对 bootstrap CI 下界为正。对同一 Candidate@50 只重排 Top-20 后，QASPER Recall@10 达到 `0.7341`，p95 `549.6 ms`；独立验证集为 `0.7223`，配对 CI 下界为 `+0.0379`。该 reranker 只作为 `general_text` 路由的显式 opt-in，不替换表格、跨文档或 PDF 路由。完整 B0-B5 消融（含未晋级的 B1/B3/B4/B5 负结果）见 [QASPER 消融矩阵](eval/qasper-hierarchical-ablation-20260811.json)；Top-N 扫描见 [QASPER reranker 报告](eval/reports/qasper-rerank-topn-20260811.json)；四场景回归门禁见 [B2 四场景报告](eval/reports/retrieval-retained-capabilities-b2-20260811.json)。
在此基础上，候选保持型 `LocalEvidenceGraph` 使用训练集选择的图/实体/章节/邻接/PPR 权重，在独立 validation 上达到 Recall@10 `0.7610`、nDCG@10 `0.5069`，Candidate@50 保持 `0.9627`；Top-30 交叉编码器基线 p95 为 `951.9 ms`，图重排增量 p95 为 `2.25 ms`，ACL 违规为 `0`。该结果只晋级为 `general_text/QASPER` opt-in，不替换表格、跨文档或 PDF 路由；Top-30 配置与独立产物见 [图谱重排 Top-30 报告](eval/reports/qasper-graph-tuned-top30-20260811.json)，旧权重审计仍保留在 [图谱重排报告](eval/reports/qasper-graph-feature-20260811.json)。

Top-20 低置信度升级 Top-30 的两段式 Cross-Encoder 预算也已实现并真实分批推理。训练划分使用 Top-1/Top-2 分差 `<0.7` 选择阈值；独立 validation 减少 `19.83%` 的打分对、平均延迟降低 `103.6 ms`，但图路线 Recall@10 从 `0.7610` 降至 `0.7468`、nDCG@10 降至 `0.4950`，因此只保留为 opt-in 负消融，不替换固定 Top-30 默认质量配置。详见 [自适应重排报告](eval/reports/qasper-adaptive-rerank-20260811.json)。

旧 TAT-QA 全库发现压力测试仍保留为负结果：BM25 Recall@10 为 `0.658333`，无语义 hash 向量的 Qdrant RRF 为 `0.248333`，再加词法 rerank 后为 `0.318333`。它不是 TAT-QA 官方任务口径，只证明实验路径可执行，不能证明混合检索优于词法基线，也不能与官方答案榜直接比较。离线/模拟 HTTP 通过不等于真实模型通过。

## 持久知识与 Memory

默认 `TASKFORGE_DATABASE_BACKEND=postgres`，必须配置 `TASKFORGE_DATABASE_URL`；切换到 SQLite 仅用于显式兼容测试或迁移工具。首次使用 PostgreSQL 前应完成 `../migration/README.md` 中的 schema、数据、RLS、备份和回滚门禁。用户 Memory 可通过 `/api/memory` 创建、检索，并删除自己拥有的 user/agent/task scope 记录；tenant/org 共享记录的可见性不会自动授予删除权。Agent 只能经 `memory_remember` 能力写入，并受 profile、审批、幂等和宿主绑定的 tenant/scope 约束。

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
  --output "..\migration\rag-query-vectors.json" `
  --tenant-id local --acl tenant --acl user:demo

python scripts/verify_pgvector_retrieval.py `
  --queries "..\migration\rag-query-vectors.json" `
  --sqlite-source-root ".taskforge" `
  --database-url "$env:TASKFORGE_DATABASE_URL" `
  --tenant-id local --acl tenant --acl user:demo `
  --exact-only `
  --report "..\migration\rag-pgvector-exact-report.json"

# Only after the exact migration/RAG gate passes:
psql "$env:TASKFORGE_POSTGRES_ADMIN_DSN" -v ON_ERROR_STOP=1 `
  -f migrations/postgres/003_taskforge_hnsw.sql

# Existing databases need this narrow grant before Zotero/PDF re-indexing:
psql "$env:TASKFORGE_POSTGRES_ADMIN_DSN" -v ON_ERROR_STOP=1 `
  -f migrations/postgres/004_literature_evidence_replace.sql

python scripts/verify_pgvector_retrieval.py `
  --queries "..\migration\rag-query-vectors.json" `
  --sqlite-source-root ".taskforge" `
  --database-url "$env:TASKFORGE_DATABASE_URL" `
  --tenant-id local --acl tenant --acl user:demo `
  --report "..\migration\rag-pgvector-report.json"
```

应用角色没有 DDL 权限，迁移工具以只读方式打开 SQLite，并以批次、幂等冲突处理和内容/状态/主键/外键校验导入 PostgreSQL。`freeze_rag_query_set.py` 从不重新调用百炼，只冻结现有 1024 维查询缓存并记录 SQLite+NumPy Top-K 基线；`verify_pgvector_retrieval.py` 再比较 SQLite+NumPy、PostgreSQL exact 与 HNSW，记录 Recall@5/10/50、MRR@10、NDCG@8/10、Agent-visible Recall 和 P50/P95 延迟。完整真实验收仍以 `../migration/README.md` 的 live 结果为准。

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
docker compose up --build
```

前端映射到 `5173`，后端映射到 `8000`，独立 Worker 与 API 共享 artifact volume；Compose 默认启动 PostgreSQL，数据 bind 到 D 盘，并由两个独立数据库和应用角色承载运行状态。需要 SQLite 时，显式设置 `TASKFORGE_DATABASE_BACKEND=sqlite` 并使用本地启动路径。当前开发机的 Docker Desktop 引擎不可用，因此仓库内 Compose/Dockerfile 已完成配置校验但未在本机完成镜像构建和真实 PostgreSQL 启动验证。

## 已知边界

- 产品主链已把 PostgreSQL（默认）与 SQLite（显式兼容）durable store 都接入应用配置；PostgreSQL 启动失败会 fail closed，不回退 SQLite。真实 PostgreSQL/RLS/pgvector、Neo4j 与远程 MCP 仍分别以各自 live 验收为准；
- 默认 `general_text` 使用 FastEmbed BGE-small；设置 `TASKFORGE_GENERAL_TEXT_BACKEND=bm25` 可显式回退到 BM25。上传链路默认使用 Jina + MiniLM 归一化集成以优先 Recall，BGE-M3 零样本在 20 题验证上为负结果，领域微调入口已提供但需 GPU 才适合完整训练；本地 Qdrant/hash 与图重排仍是评测或显式 opt-in 路径；真实远程 Qdrant、PostgreSQL、Neo4j 与远程 MCP 均无 live 成功声明；
- 演示知识只为 `local` tenant 加载显式 allowlist 文档，其他 tenant 会检索为空而不是跨租户回退；
- API header 是演示 identity，需要在生产前替换为可信认证与 RBAC；
- Worker 已有 SQLite/PostgreSQL lease/CAS；审批 API 的业务写入在 PostgreSQL 路径使用数据库事务，横向扩容仍需完成真实并发验收；
- 开放发现卡片不是证据；只有用户上传并成功解析的 PDF 才能进入 ready Scope。历史摘要回退 E2E 已失效，需按新上传协议重跑；
- 论文 MCP Server 已验证 stdio/HTTP JSON-RPC 与本地客户端互操作；通用远程 MCP Client 仍是 JSON response 子集。没有模型生成 shell 或容器代码执行；四角色是已接入产品 API 的宿主固定 DAG，不是开放式群聊；
- 100 题匿名 Provider 开放发现质量门禁当前失败，本轮主要伴随 Semantic Scholar/OpenAlex/arXiv 限流和中文查询未翻译；代码已加入礼貌全局限速、联系身份、API Key 入口和保守的中英学术术语桥接，但在正式配额下用同一数据重跑前，不能在简历中声称 Paper Recall/Precision/nDCG 达标；
- RoleRun SQLite 租约会在 provider/tool dispatch 前重新 fencing，能阻止失去租约的旧 worker 执行工具；但已发出的 provider HTTP 请求无法撤回，进程停顿跨过租期时仍可能产生重复模型调用或费用，不能宣称 provider exactly-once；
- PostgreSQL 路径把 Queue/审计/checkpoint 放入同一数据库体系，但 provider HTTP 请求和下游副作用仍不具备 exactly-once；下游必须自行尊重 idempotency key；
- 创建 queued Run 尚无客户端请求幂等键；网络结果不明确时不会伪造 mock 成功，但调用方仍需先按业务请求标识查询再决定是否重试；
- Artifact 写入必须审批，但它不是通用代码执行沙箱。

进一步设计见 [架构说明](docs/ARCHITECTURE.md) 与 [威胁模型](docs/THREAT_MODEL.md)。

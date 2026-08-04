# TaskForge

TaskForge 是一个 **Provider-neutral、权限受控、可恢复的通用 Agent Runtime**。它参考了截图项目的 Agent Profile、RAG、分层记忆、工具与审批等产品能力，同时采用 Coding Agent 的 Harness 思路：模型只提出 `ToolRequest`，宿主负责权限、参数校验、执行、幂等、checkpoint 和评测。

它不是“套一个聊天页面”，也不把模型当授权系统。

## 当前能运行什么

同一套 `AgentRuntime` 通过配置切换三类 Agent，无需复制核心循环：

- 研究与报告：ACL 过滤知识检索、作用域记忆召回、审批后生成报告；
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
```

Demo Provider 是确定性离线状态机，用来证明真实工具、审批、持久化和恢复链路；它不会伪装成 LLM。设置凭据后可显式切换到 OpenAI Responses Provider。Native function call 会被归一化为同一个 `ToolRequest`，工具输出通过 `previous_response_id + function_call_output` 续接。

为避免把“有代码”误写成“已上线”，本文统一使用四层状态：

| 层级 | 当前含义 | 当前可证明的范围 |
|---|---|---|
| 模块/契约存在 | 有实现、Schema 或 migration | 只能证明接口和静态设计存在 |
| fake/离线/本地已测 | 单测、模拟 HTTP、fake driver、Demo Provider 或本机进程通过 | 证明受控输入下的契约与本地集成，不证明真实服务质量 |
| 产品主链已接入 | FastAPI/Workbench 的默认或可配置运行路径会调用该能力 | 仍不等于外部依赖、生产认证或规模化部署已验证 |
| live 已验证 | 使用真实凭据和真实外部服务显式执行并留存结果 | 只有这一层才能声明对应真实服务链路通过 |

当前产品主链使用 SQLite Knowledge/Memory 与词法检索，并接入单 Agent 和固定四角色审查流程；离线回归使用 Demo Provider。Qdrant 混合检索仍是本地评测路径，PostgreSQL/Neo4j 是可选模块但尚未接入应用配置主链，OpenAI Provider 只有模拟 HTTP 契约测试。仓库尚未宣称任何真实模型、PostgreSQL、Neo4j 或远程 MCP 的 live 成功。

## 核心能力

- 有界 Agent Loop：step budget、结构化失败、普通工具错误可观察并允许模型恢复；
- Tool Gateway：严格 JSON Schema、allowlist、风险分级、超时与输出上限；
- 审批与幂等：写入/外部/破坏性能力暂停，同一 call/key 换参会 fail closed；
- SQLite checkpoint：Task、Profile、Run、pending approval 和 receipt 可跨进程重载；
- 持久化上下文：Knowledge/Memory 默认写入独立 SQLite，支持重启恢复、版本替换、过期与租户/ACL/scope 过滤；
- Durable Worker：排队执行、原子 claim、租约心跳、owner/token/version/expiry CAS、显式瞬态 Provider 失败退避重试、末次租约恢复核对与 dead letter；
- MCP client（固定握手式旧版 `2025-11-25`）：仅从宿主 JSON 配置挂载 allowlist 工具，执行前仍经过本地风险、审批和 Schema；
- 审计与指标：append-only 事件、secret-like 字段拒绝、run/tool 成功率、p50/p95、token/cost 与 safety 计数；
- 安全代码检索：不执行 shell；限制工作区、glob、regex、文件大小、结果数和时间；
- RAG：产品主链使用 tenant-first ACL、版本/有效期和词法检索，并保留可替换的 hybrid score 接口；
- 结构化 PDF RAG：`pypdf/pdfplumber` 段落与表格提取、页码/邻接 provenance；BM25 + 本地 Qdrant named dense/sparse + RRF + 可替换 reranker 当前属于离线评测路径；
- Memory：tenant/org/user/agent/task 五级 scope、过期时间和 provenance；
- 多角色编排：固定有向无环图、角色 capability、RoleRun 尝试/恢复、分层上下文、handoff、proposed/verified fact 与一次性 host verification receipt；引用有据的 claim（其 evidence refs 全部来自该角色本次运行真实检索到的 knowledge_search 回执）由宿主自动签发 `authority=tool` receipt 并置为 verified，随后向下游依赖角色创建 handoff；未检索到引用的 claim 保持 `model_untrusted`/`proposed`；
- 业务决策边界：模型只能提交结构化建议，case 状态和最终批准由宿主状态机与人工身份控制；
- 可选基础设施：PostgreSQL context adapter、Neo4j 1/2-hop 图检索和图/向量 RRF 融合均默认关闭，未连接真实服务时不宣称 live；
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
python -m uvicorn taskforge.app:app --reload
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

仓库单测只覆盖离线 runtime 和模拟 HTTP 契约，不代表真实模型效果。填好 key 与 model 后，可显式执行一条真实、可能计费的 native tool-calling 冒烟测试：

```powershell
.\.venv\Scripts\python.exe scripts\run_live_openai_smoke.py --confirm-live-call
```

该脚本要求模型调用一次受控 `calculator`，再通过 `previous_response_id + function_call_output` 完成回答；未带 `--confirm-live-call` 时会在任何网络请求前退出。只有这条脚本真实通过后，才能声明当前凭据、模型和网络环境的 live API 链路已验证。

`/api/review-cases` 的 `execution` 响应把状态拆为四项：`provider_configured`、`contract_tested_mock`、`live_smoke_verified` 与 `business_e2e_verified`。配置 key/model 只会令第一项为 `true`；模拟 HTTP 契约测试也不等于真实调用。当前尚未实现可校验、可持久化的 live smoke 与业务 E2E 验证记录，因此即使本地脚本曾成功，API 中后两项仍保持 `false`，不会从环境变量或 API key 推断成功。

## 评测

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

当前本机锁定 TAT-QA 结果是一个必须保留的负结果：BM25 Recall@10 为 `0.658333`，无语义 hash 向量的 Qdrant RRF 为 `0.248333`，再加词法 rerank 后为 `0.318333`。因此它只证明本地 Qdrant/RRF 路径可执行，不能证明混合检索优于词法基线，更不能证明生产语义检索质量。原始结果、负结果解释和第三方数据许可边界见 [评测协议](docs/EVALUATION.md)。离线/模拟 HTTP 通过不等于真实模型通过。

## 持久知识与 Memory

默认 `TASKFORGE_CONTEXT_BACKEND=sqlite`。用户 Memory 可通过 `/api/memory` 创建、检索，并删除自己拥有的 user/agent/task scope 记录；tenant/org 共享记录的可见性不会自动授予删除权。Agent 只能经 `memory_remember` 能力写入，并受 profile、审批、幂等和宿主绑定的 tenant/scope 约束。

将工作区内 UTF-8 文档安全摄取到知识库：

```powershell
python scripts\ingest_knowledge.py docs\ARCHITECTURE.md `
  --knowledge-base taskforge --version 2026-08-04 --version-order 1
```

摄取器拒绝路径穿越、symlink/reparse point、二进制、凭据型文件和超限输入，并原子替换同一文档版本。默认演示文档位于 `taskforge`；四角色企业审查 profile 只检索 `enterprise-review`，所以要让审查角色获得政策证据，operator 必须把对应政策文档另行摄取到该知识库，不能依赖用户输入的“证据 ID”代替真实检索。

`postgres_context.py` 与两阶段 migration 提供可选 PostgreSQL 契约。若 operator 要在独立测试库验证它，必须严格按以下顺序执行，任一步失败即停止：

```powershell
psql "$env:TASKFORGE_POSTGRES_DSN" -v ON_ERROR_STOP=1 `
  -f migrations/postgres/001_context.sql
psql "$env:TASKFORGE_POSTGRES_DSN" -v ON_ERROR_STOP=1 `
  -f migrations/0002_context_postgres.sql
```

第一步创建 context schema、表、基础索引与初始 RLS；第二步在这些对象之上加固策略和补充索引，不能倒序或只执行第二步。当前应用的 `TASKFORGE_CONTEXT_BACKEND` 仍只选择 `sqlite`/`memory`，PostgreSQL adapter 只通过注入式 fake DB-API 测试；即使 SQL 成功执行，也不等于产品主链已接入或 live RLS 已验证。

## 受控 MCP

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

客户端固定实现握手式协议修订 `2025-11-25` 的 initialize、分页 `tools/list` 和 `tools/call`。它每次请求做 DNS/IP preflight、禁重定向、流式限制响应大小，并在服务端选择 SSE 时明确 fail closed；当前只实现 JSON response 子集。远端返回 `isError=true` 会记为失败；side-effecting 工具若没有声明必填 string `idempotency_key`，会在挂载前被拒绝。这个 preflight 与 httpx 的实际连接解析不是 IP pinning，不能单独消除 DNS rebinding TOCTOU，生产部署仍需 egress proxy/firewall。MCP 官方在 2026-07-28 已把 current revision 更新为无连接态、per-request metadata 的 `2026-07-28`，本仓库尚未实现该新版，不能宣称“最新 MCP 全兼容”。仓库测试使用模拟 HTTP，没有调用真实 MCP 服务。

## Docker Compose

```powershell
docker compose up --build
```

前端映射到 `5173`，后端映射到 `8000`，独立 Worker 与 API 共享 SQLite/artifact 命名 volume；Python 容器根文件系统只读，仅 `.taskforge` volume 可写。当前开发机没有 Docker，因此仓库内 Compose/Dockerfile 已编写但未在本机完成镜像构建验证。

## 已知边界

- 产品主链当前只把 SQLite `memory`/`sqlite` context、case、orchestration 和 Operations 接入应用配置；PostgreSQL adapter 仅做 fake DB-API 契约测试，Neo4j retriever 仅做 fake-driver 测试且 gate 默认关闭，两者都不是当前应用主链；
- 本地 Qdrant/hash 混合检索属于评测脚本路径，锁定结果显著低于 BM25；FastEmbed/OpenAI embedding 仅有注入或模拟测试。真实语义模型、远程 Qdrant、PostgreSQL、Neo4j 与远程 MCP 均无 live 成功声明；
- 演示知识只为 `local` tenant 加载显式 allowlist 文档，其他 tenant 会检索为空而不是跨租户回退；
- API header 是演示 identity，需要在生产前替换为可信认证与 RBAC；
- Worker 已有 SQLite lease/CAS；审批 API 的并发锁仍只覆盖单进程，横向扩容审批需要数据库级 claim；
- MCP 仅在宿主显式配置后启用，当前 JSON-only，测试只使用模拟 HTTP；没有模型生成 shell 或容器代码执行。多角色是已经接入产品 API 的宿主固定 DAG，但不是开放式、可动态配置的自主 Agent 群聊，且尚未用真实模型运行；
- RoleRun SQLite 租约会在 provider/tool dispatch 前重新 fencing，能阻止失去租约的旧 worker 执行工具；但已发出的 provider HTTP 请求无法撤回，进程停顿跨过租期时仍可能产生重复模型调用或费用，不能宣称 provider exactly-once；
- Queue/审计/checkpoint 分属本地 SQLite 事务，尚不是跨库原子 exactly-once；下游副作用仍必须自行尊重 idempotency key；
- 创建 queued Run 尚无客户端请求幂等键；网络结果不明确时不会伪造 mock 成功，但调用方仍需先按业务请求标识查询再决定是否重试；
- Artifact 写入必须审批，但它不是通用代码执行沙箱。

进一步设计见 [架构说明](docs/ARCHITECTURE.md) 与 [威胁模型](docs/THREAT_MODEL.md)。

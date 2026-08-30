# TaskForge

[![CI](https://github.com/wtl003629-dev/TaskForge/actions/workflows/ci.yml/badge.svg)](https://github.com/wtl003629-dev/TaskForge/actions/workflows/ci.yml)

TaskForge 是一个论文研究助手：帮你找论文、获取全文、检索证据，并生成带引用的中文研究报告。

```text
提出问题 → 检索论文 → 选择论文 → 获取全文 → 查找证据 → 生成报告 → 人工核对
```

## 主要功能

- 从 Semantic Scholar、OpenAlex、arXiv 和 Crossref 检索论文；
- 支持中文、英文查询，以及“综合 / 中文优先 / 英文优先”；
- 自动合并重复结果，并展示作者、出版来源、引用量和来源链接；
- 自动下载合法开放获取 PDF，也支持 Zotero 同步和手动上传；
- 只在已选论文的完整正文中检索，不用摘要冒充全文证据；
- 过滤重复片段、参考文献目录和无意义占位内容；
- 生成带编号引用的中文报告，最终交给用户核对。

## 快速开始

推荐使用 Docker Compose。当前 Compose 会启动：

- 前端：`http://127.0.0.1:5173`
- 后端：`http://127.0.0.1:8000`
- PostgreSQL + pgvector
- 后台 Worker

仓库默认按下面的同级目录组织：

```text
my-coding/
  TaskForge/
  PatchPilot/
```

首次启动：

```powershell
cd D:\my-coding\TaskForge
Copy-Item .env.example .env

# 编辑 .env，填写三个 PostgreSQL 密码
docker compose up -d --build
docker compose ps
```

默认使用 Demo Provider。生成真实报告时，可以在 `.env` 中启用阿里云百炼：

```dotenv
TASKFORGE_PROVIDER=bailian
TASKFORGE_BAILIAN_API_KEY=...
TASKFORGE_BAILIAN_CHAT_MODEL=qwen-plus
TASKFORGE_GENERAL_TEXT_BACKEND=bailian
TASKFORGE_RESEARCH_RERANKER_BACKEND=bailian
```

OpenAI 和 DeepSeek 也可以通过 `.env.example` 中的对应配置启用。API Key 只放在本地 `.env`，不要提交到 Git。

## 使用流程

1. 输入研究问题，选择结果数量和语言偏好；
2. 从候选列表中选择论文，并核对作者、期刊或出版社；
3. 保存论文清单，让系统获取和解析全文；
4. 在已选论文中提问，查看返回的证据片段；
5. 点击“生成研究报告”，等待报告进入“待核对”状态。

“找到 10 条证据”表示找到了 10 个论文片段，不是 10 篇论文；“覆盖 100%”表示每篇已选论文至少命中一个片段，不代表全文都已覆盖。

## Zotero 同步

对于需要登录才能下载的论文，可以先用 Zotero Connector 保存到本机 Zotero。TaskForge 通过只读接口同步元数据和附件全文，不会绕过登录、付费墙或版权限制。

配置方法见 [Zotero 接入说明](docs/ZOTERO_MCP.md)。

## 当前评测结果

已选论文全文检索基线覆盖 30 篇中文和 30 篇英文真实 PDF，共 177 个标注问题。检索使用 MinerU、BM25、百炼 `text-embedding-v4` 和 `qwen3-rerank`。

| 指标 | 结果 |
|---|---:|
| 整体 Recall@10 | **92.62%** |
| 中文 Recall@10 | **95.00%** |
| 英文 Recall@10 | **90.15%** |

这组结果针对“用户已经选定论文后的全文问答”，不代表全库论文发现效果。正式基线见 [`paper-scoped-flat-bailian-v1.json`](eval/baselines/paper-scoped-flat-bailian-v1.json)。

报告生成方面：

- 四角色报告链路已经真实跑通并进入人工核对阶段；
- 端到端答案 Token F1 相对初始基线提升 **36.35 个百分点**；
- 上下文优化后，同任务 Token 消耗下降约 **71%**。

## 本地开发

后端要求 Python 3.11+：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
python -m uvicorn taskforge.app:create_app --factory --reload
```

前端要求 Node 20+ 和 pnpm：

```powershell
cd frontend
pnpm install --frozen-lockfile
pnpm dev
```

## 项目结构

```text
backend/taskforge/   后端、Agent Runtime、论文检索与 RAG
frontend/            Vue 3 研究工作台
migrations/          PostgreSQL / pgvector 迁移
scripts/             启动、评测和数据处理脚本
eval/                冻结数据、基线和评测结果
docs/                架构与详细说明
```

## 详细文档

- [论文研究流程](docs/PAPER_RESEARCH_AGENT.md)
- [RAG 与 PDF 处理](docs/PDF_RAG_PIPELINE.md)
- [评测方法](docs/EVALUATION.md)
- [系统架构](docs/ARCHITECTURE.md)
- [安全设计](docs/THREAT_MODEL.md)

当前 Compose 已在本地验证前端、后端、PostgreSQL/pgvector 和 Worker 可以完整启动。公网部署前仍需接入正式认证，并完成目标服务器上的安全配置和压力测试。

# TaskForge 安全设计

本文说明 TaskForge 当前信任哪些组件、主要防范什么风险，以及公网部署前还需要补充什么。技术标识保留英文，说明文字使用中文。

## 信任边界

以下内容都视为不可信输入：

- 用户问题和上传文件；
- 模型输出；
- RAG 检索到的论文文本；
- 外部文献 Provider 返回的数据；
- Zotero 条目和 Markdown；
- 远程 MCP 工具结果。

以下能力属于 Host 边界：

- 身份和 tenant；
- ResearchScope；
- 工具白名单和风险策略；
- 文件路径与工作区根目录；
- 数据库角色与 RLS；
- checkpoint、receipt 和 Evidence ID；
- 最终人工批准。

核心原则是：`ToolRequest` 只是模型提出的请求，不代表它已经获得执行权限。

## 主要风险与控制

| 风险 | 当前控制 |
|---|---|
| RAG Prompt Injection | 检索文本标记为不可信证据，不能修改工具、Scope 或身份 |
| 伪造引用 | Host 只接受真实检索回执中出现过的 Evidence ID |
| 跨论文取证 | 检索前强制检查 ResearchScope、论文 ID 和版本 |
| 跨租户数据泄漏 | PostgreSQL 查询先绑定 tenant，再执行 ACL 和 Scope 过滤 |
| 路径穿越和敏感文件读取 | 拒绝绝对路径、`..`、symlink/reparse point、凭据型文件和超限输入 |
| 模型生成任意 Shell | 论文研究工具不提供通用 Shell；工具参数使用结构化 Schema |
| 重复副作用 | side-effecting 工具要求 idempotency key，调用指纹和回执持久化 |
| Worker 重复执行 | 原子 claim、owner、lease token、version 和 expiry 共同 fencing |
| MCP SSRF | MCP 默认关闭，Host 配置 endpoint 和 allowlist，并限制重定向、私网地址和响应大小 |
| Zotero 内容污染 | 条目必须匹配论文身份，正文会过滤元数据包装、参考目录和占位内容 |
| 模型自行批准结果 | Agent 只能生成草稿和 verdict，最终状态由 Host 和用户决定 |

## ResearchScope

`ResearchScope` 是论文研究最重要的权限边界：

- 由用户选择论文后通过 Host 创建；
- Agent 只有读取权；
- 每次修改产生新版本；
- Evidence ID 绑定 Scope 及其版本；
- Scope 关闭后不能继续读取正文；
- 扩展请求必须由用户批准。

发现阶段的标题和摘要不能直接进入 Scope 证据库。只有成功获取、解析和索引的全文可以用于报告。

## 文件与 PDF

上传和下载路径会检查：

- 文件大小；
- PDF 魔数和解析结果；
- 工作区边界；
- symlink 和 reparse point；
- 二进制及凭据型文件；
- 文档身份匹配。

扫描 PDF 或解析失败不会产生空的“成功索引”。如果需要 OCR 但 MinerU 不可用，论文会保持失败状态。

## 证据与报告

Writer 的结论必须引用 Evidence ID。Host 会核对：

- Evidence ID 是否真实存在；
- 是否来自本次运行的检索回执；
- 是否属于当前 Scope 和版本；
- 是否绑定正确论文；
- 引用原文是否仍可读取。

Critic 可以要求修改、删除或补充证据，但不能用范围外知识替换缺失证据。

## 数据库与 Worker

PostgreSQL 是默认持久化后端。数据库角色分离迁移权限与应用权限，应用角色不拥有 DDL 权限。

Worker 使用租约执行 queued Run。失去租约的 Worker 不能继续写入完成状态，但已经发出的 Provider HTTP 请求无法可靠撤回，因此外部副作用仍需要业务幂等。

## MCP

远程 MCP 工具只有在 Host 配置中显式启用并加入 allowlist 后才会挂载。模型不能自行添加 MCP Server。

MCP Schema 和结果会限制大小并去除不可信描述。具有副作用的工具必须声明 `idempotency_key`，否则在挂载阶段就会被拒绝。

## 身份与审批

本地开发使用请求头模拟 tenant 和 user，适合本机调试，不是公网认证方案。

公网部署前需要：

- 正式登录和 token 校验；
- RBAC 与管理员角色；
- 高风险操作二次认证；
- tenant 隔离和密钥轮换；
- 审批记录与审计保留策略。

## 部署安全

本地 Compose 已验证前端、后端、PostgreSQL/pgvector 和 Worker 可以健康启动。公网部署还需要：

- HTTPS 和反向代理；
- API 限流；
- egress proxy 或防火墙；
- 加密备份和恢复演练；
- 集中日志与告警；
- 容器资源限制；
- Provider 费用和 Token 预算；
- 并发和故障恢复压测；
- Prompt Injection、数据泄露和拒绝服务红队测试。

系统架构见 [架构说明](ARCHITECTURE.md)，论文工作流见 [论文研究流程](PAPER_RESEARCH_AGENT.md)。

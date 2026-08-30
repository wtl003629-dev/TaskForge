# Zotero MCP 本地接入

这条链路用于把 Zotero 作为论文资料库：用户在浏览器中使用 Zotero Connector 保存论文，`zotero-mcp` 读取 Zotero 元数据和全文，TaskForge 再把可用内容纳入自己的研究流程。正常使用不需要用户手动下载 PDF 或在 TaskForge 再次上传。

## 权限边界

- 元数据（标题、作者、DOI、摘要、来源等）和全文是两种不同能力。元数据可用不代表附件全文可读。
- Connector 能否保存内容取决于页面、站点和 Zotero Connector 的可访问权限。
- 付费墙、登录限制、版权限制或未保存附件不会被绕过。此时 MCP 只能返回已有元数据，或提示没有可读全文。
- 当前配置只挂载只读工具，不包含新增、修改、删除、标签或笔记写入工具。

## Windows 启动

先安装并运行 Zotero Desktop，再安装浏览器 Zotero Connector。项目提供固定本机地址的启动脚本：

```powershell
.\scripts\run_zotero_mcp.ps1
```

脚本使用 `uvx` 固定运行 `zotero-mcp-server==0.11.0`，绑定 `127.0.0.1:8766`，使用 `streamable-http`，并设置 `ZOTERO_LOCAL=true`、`ZOTERO_SEARCH_BACKEND=sqlite`、`FASTMCP_JSON_RESPONSE=true`、`ZOTERO_MCP_TOOLSETS=none`。如果存在默认数据目录，会自动设置 `ZOTERO_DB_PATH` 指向本机 `Zotero/zotero.sqlite`。脚本不会打印密钥；本地模式默认不需要 Zotero Web API 密钥。

如果 Zotero 本地 API 返回 403，请在 Zotero 的“设置 → 高级”中启用“允许本机其他应用程序与 Zotero 通信”，然后重启 Zotero 和本脚本。该开关只允许本机读取，不会把 23119 端口暴露到公网。

## Docker / TaskForge 配置

将 `config/mcp.zotero.example.json` 复制为宿主配置（或由部署流程加载），按需把 `enabled` 改为 `true`，并设置 `TASKFORGE_MCP_CONFIG_PATH` 指向它。配置使用：

```text
http://host.docker.internal:8766/mcp
```

它显式放行本地 HTTP、私网和 TCP 8766，只允许搜索、最近项目、元数据、全文、子项和 PDF 页读取工具。`profile_ids` 为空，因此这些工具不会暴露给研究模型；TaskForge 宿主根据用户点击确定性地执行“匹配条目 → 分页读取 → 入库”。长 PDF 会按页分批读取并保留 `## Page N` 页码标记，而不是依赖一次超长模型上下文。TaskForge 仍会执行自己的 MCP 握手、allowlist、schema、输出大小和网络边界校验。

## 当前验证状态

已使用 `zotero-mcp-server==0.11.0` 完成 TaskForge 客户端的真实 Streamable HTTP 握手，协商协议版本 `2025-11-25`，并发现配置中的 6 个只读工具；Docker 后端也已验证可访问宿主机 `host.docker.internal:8766/mcp`。兼容的关键设置是 `FASTMCP_JSON_RESPONSE=true`，缺少它时服务会返回 TaskForge 当前明确拒绝的 SSE 响应。

空馆藏也属于有效连接状态：TaskForge 会显示“已连接”，论文列表为空，待用户通过 Connector 保存论文后即可检索和同步。真实论文保存、附件自动获取、分页全文读取和付费附件行为仍需在用户实际登录目标站点并保存一篇论文后验收；付费墙不会被绕过。

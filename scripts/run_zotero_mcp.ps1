[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

# Keep the bridge local-only.  ZOTERO_LOCAL can be overridden explicitly, but
# defaults to the Desktop/Connector-backed local Zotero API.
if (-not $env:ZOTERO_LOCAL) {
    $env:ZOTERO_LOCAL = 'true'
}
# Prefer the local SQLite metadata path. This avoids requiring a Zotero.org
# Web API key and keeps the bridge on the same machine as Zotero Desktop.
if (-not $env:ZOTERO_SEARCH_BACKEND) {
    $env:ZOTERO_SEARCH_BACKEND = 'sqlite'
}
if (-not $env:ZOTERO_DB_PATH) {
    $defaultDbPath = Join-Path $env:USERPROFILE 'Zotero\zotero.sqlite'
    if (Test-Path -LiteralPath $defaultDbPath -PathType Leaf) {
        $env:ZOTERO_DB_PATH = (Resolve-Path -LiteralPath $defaultDbPath).Path
    }
}
$env:FASTMCP_JSON_RESPONSE = 'true'
$env:ZOTERO_MCP_TOOLSETS = 'none'

& uvx --from 'zotero-mcp-server==0.11.0' zotero-mcp serve `
    --transport streamable-http `
    --host 127.0.0.1 `
    --port 8766

if ($LASTEXITCODE -ne 0) {
    throw "zotero-mcp exited with code $LASTEXITCODE"
}

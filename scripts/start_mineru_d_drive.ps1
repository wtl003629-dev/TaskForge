[CmdletBinding()]
param(
    [string]$Root = "D:\TaskForge\mineru",
    [int]$Port = 8001,
    [switch]$EnableVlmPreload
)

$ErrorActionPreference = "Stop"
$rootPath = [System.IO.Path]::GetFullPath($Root)
if (-not $rootPath.StartsWith("D:\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "MinerU root must be on D:; received $rootPath"
}
if ($Port -lt 1 -or $Port -gt 65535) {
    throw "Port must be between 1 and 65535"
}

$cache = Join-Path $rootPath "cache"
$env:TEMP = Join-Path $rootPath "temp"
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $cache "pip"
$env:HF_HOME = Join-Path $cache "huggingface"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $cache "huggingface\hub"
$env:MODELSCOPE_CACHE = Join-Path $cache "modelscope"
$env:TORCH_HOME = Join-Path $cache "torch"
$env:PYTHONPYCACHEPREFIX = Join-Path $cache "python-bytecode"
$env:MINERU_TOOLS_CONFIG_JSON = Join-Path $cache "config\mineru.json"
$env:MINERU_API_OUTPUT_ROOT = Join-Path $rootPath "output"
$env:MINERU_MODEL_SOURCE = "local"

$mineruApi = Join-Path $rootPath "runtime\venv\Scripts\mineru-api.exe"
if (-not (Test-Path -LiteralPath $mineruApi -PathType Leaf)) {
    throw "MinerU is not installed on D:. Run scripts\setup_mineru_d_drive.ps1 first."
}
if (-not (Test-Path -LiteralPath $env:MINERU_TOOLS_CONFIG_JSON -PathType Leaf)) {
    throw "MinerU models are not downloaded. Rerun setup with -DownloadModels."
}

$vlmPreload = if ($EnableVlmPreload) { "true" } else { "false" }
& $mineruApi --host 127.0.0.1 --port $Port --enable-vlm-preload $vlmPreload

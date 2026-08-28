[CmdletBinding()]
param(
    [string]$Root = "D:\TaskForge\mineru",
    [string]$Python = "D:\my-coding\TaskForge\.venv\Scripts\python.exe",
    [ValidateSet("auto", "huggingface", "modelscope")]
    [string]$ModelSource = "auto",
    [switch]$DownloadModels
)

$ErrorActionPreference = "Stop"
$rootPath = [System.IO.Path]::GetFullPath($Root)
if (-not $rootPath.StartsWith("D:\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "MinerU root must be on D:; received $rootPath"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python 3.10-3.13 executable not found: $Python"
}

$runtime = Join-Path $rootPath "runtime"
$venv = Join-Path $runtime "venv"
$cache = Join-Path $rootPath "cache"
$temp = Join-Path $rootPath "temp"
$output = Join-Path $rootPath "output"
$config = Join-Path $cache "config\mineru.json"
foreach ($path in @(
    $runtime,
    $cache,
    $temp,
    $output,
    (Split-Path -Parent $config),
    (Join-Path $cache "pip"),
    (Join-Path $cache "huggingface"),
    (Join-Path $cache "modelscope"),
    (Join-Path $cache "torch"),
    (Join-Path $cache "python-bytecode")
)) {
    New-Item -ItemType Directory -Path $path -Force | Out-Null
}

# Every known package/model/temp write path is redirected before creating the
# environment. The runtime virtualenv itself also lives on D:.
$env:TEMP = $temp
$env:TMP = $temp
$env:PIP_CACHE_DIR = Join-Path $cache "pip"
$env:HF_HOME = Join-Path $cache "huggingface"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $cache "huggingface\hub"
$env:MODELSCOPE_CACHE = Join-Path $cache "modelscope"
$env:TORCH_HOME = Join-Path $cache "torch"
$env:PYTHONPYCACHEPREFIX = Join-Path $cache "python-bytecode"
$env:MINERU_TOOLS_CONFIG_JSON = $config
$env:MINERU_API_OUTPUT_ROOT = $output

if (-not (Test-Path -LiteralPath (Join-Path $venv "Scripts\python.exe"))) {
    & $Python -m venv $venv
}
$mineruPython = Join-Path $venv "Scripts\python.exe"
$mineruApi = Join-Path $venv "Scripts\mineru-api.exe"
$modelDownloader = Join-Path $venv "Scripts\mineru-models-download.exe"

& $mineruPython -m pip install --upgrade pip
& $mineruPython -m pip install --require-virtualenv "mineru[core]==3.4.4"
$installedVersion = & $mineruPython -c "from mineru.version import __version__; print(__version__)"
if ($installedVersion.Trim() -ne "3.4.4") {
    throw "MinerU version mismatch: expected 3.4.4, received $installedVersion"
}

if ($DownloadModels) {
    $env:MINERU_MODEL_SOURCE = $ModelSource
    & $modelDownloader -s $ModelSource -m all
    if (-not (Test-Path -LiteralPath $config -PathType Leaf)) {
        throw "MinerU model download did not create the D-drive configuration"
    }
}

[PSCustomObject]@{
    status = "ready"
    version = $installedVersion.Trim()
    runtime = $venv
    mineru_api = $mineruApi
    config = $config
    cache = $cache
    output = $output
    models_downloaded = [bool]$DownloadModels
} | ConvertTo-Json

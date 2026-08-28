[CmdletBinding()]
param(
    [string]$DataRoot = "D:\TaskForge\mineru"
)

$ErrorActionPreference = "Stop"
$resolvedRoot = [System.IO.Path]::GetFullPath($DataRoot)
if (-not $resolvedRoot.StartsWith("D:\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "MinerU data root must be on D:; received $resolvedRoot"
}

$dockerSettings = Join-Path $env:APPDATA "Docker\settings-store.json"
if (-not (Test-Path -LiteralPath $dockerSettings -PathType Leaf)) {
    throw "Docker Desktop settings were not found: $dockerSettings"
}
$settings = Get-Content -LiteralPath $dockerSettings -Raw | ConvertFrom-Json
$dockerWslRoot = [string]$settings.CustomWslDistroDir
if (-not $dockerWslRoot.StartsWith("D:\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw (
        "Docker Desktop's image disk is not configured on D: (current: $dockerWslRoot). " +
        "Before building MinerU, open Docker Desktop > Settings > Resources > " +
        "Advanced > Disk image location, move it to D:\DockerDesktop, Apply, " +
        "then run this check again."
    )
}
$dockerDisk = Join-Path $dockerWslRoot "disk\docker_data.vhdx"
if (-not (Test-Path -LiteralPath $dockerDisk -PathType Leaf)) {
    throw "Docker Desktop D-drive disk was not found: $dockerDisk"
}

foreach ($path in @(
    $resolvedRoot,
    (Join-Path $resolvedRoot "cache"),
    (Join-Path $resolvedRoot "output")
)) {
    New-Item -ItemType Directory -Path $path -Force | Out-Null
}

$env:TASKFORGE_MINERU_DATA_ROOT = $resolvedRoot.Replace("\", "/")
docker compose -f deploy/mineru/compose.yaml --profile mineru config --quiet

[PSCustomObject]@{
    status = "ready"
    mineru_data_root = $resolvedRoot
    docker_disk_on_d = $true
    docker_disk = $dockerDisk
} | ConvertTo-Json

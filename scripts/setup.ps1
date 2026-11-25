param([string]$EngineArchive, [switch]$SkipEngine, [switch]$Offline)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv is required. See the installation links in README.md.'
}
if ($EngineArchive) { $EngineArchive = (Resolve-Path -LiteralPath $EngineArchive).ProviderPath }
Push-Location -LiteralPath $root
try {
    $syncArgs = @('sync', '--locked', '--python', '3.11')
    if ($Offline) { $syncArgs += '--offline' }
    & uv @syncArgs
    if ($LASTEXITCODE -ne 0) { throw "uv sync failed with exit code $LASTEXITCODE" }
    if (-not $SkipEngine) {
        if ($Offline -and -not $EngineArchive -and -not (Test-Path 'third_party/focus-stack/1.5/focus-stack/focus-stack.exe')) {
            throw 'Pass -EngineArchive with a local release ZIP or use -SkipEngine.'
        }
        & (Join-Path $PSScriptRoot 'setup_focus_stack.ps1') -Archive $EngineArchive
    }
} finally { Pop-Location }

param([string]$Archive)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = Split-Path -Parent $PSScriptRoot
$pin = Get-Content -LiteralPath (Join-Path $root 'src/pollen_edf/resources/engine.json') -Raw | ConvertFrom-Json
$exe = Join-Path $root $pin.installed_path
$packages = Join-Path $root 'third_party/focus-stack'
$destination = Join-Path $packages $pin.release

function Confirm-Engine([string]$Path) {
    $hash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    if ($hash -ne $pin.binary_sha256) { throw "Unexpected focus-stack EXE SHA-256: $hash" }
    $start = [Diagnostics.ProcessStartInfo]::new($Path, '--help')
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardError = $true
    $process = [Diagnostics.Process]::Start($start)
    try {
        $helpTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(10000)) { $process.Kill(); throw 'focus-stack --help timed out.' }
        $helpText = $helpTask.GetAwaiter().GetResult()
        if ($process.ExitCode -ne 1 -or $helpText -notmatch '--no-align' -or $helpText -notmatch '--depthmap') {
            throw "focus-stack --help failed with exit code $($process.ExitCode). $helpText"
        }
    } finally { $process.Dispose() }
    Write-Output "focus-stack $($pin.release) verified at $Path"
    Write-Output "Executable SHA-256 $hash"
}

if (Test-Path -LiteralPath $exe -PathType Leaf) {
    Confirm-Engine $exe
    return
}
if (Test-Path -LiteralPath $destination) { throw "Engine destination is incomplete: $destination" }
if (-not [Environment]::Is64BitOperatingSystem) { throw 'The official focus-stack asset requires Windows x64.' }

$stage = Join-Path $packages ('.install-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $stage -Force | Out-Null
try {
    if (-not $Archive) {
        $Archive = Join-Path $stage 'release.zip'
        Invoke-WebRequest -UseBasicParsing -Uri $pin.archive_url -OutFile $Archive -TimeoutSec 180
    }
    $hash = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash
    if ($hash -ne $pin.archive_sha256) { throw "Unexpected focus-stack ZIP SHA-256: $hash" }
    $expanded = Join-Path $stage 'expanded'
    Expand-Archive -LiteralPath $Archive -DestinationPath $expanded
    Confirm-Engine (Join-Path $expanded 'focus-stack/focus-stack.exe') | Out-Null
    $packageRoot = (Resolve-Path -LiteralPath $packages).ProviderPath
    $source = (Resolve-Path -LiteralPath $expanded).ProviderPath
    $target = [IO.Path]::GetFullPath((Join-Path $packageRoot $pin.release))
    $prefix = $packageRoot.TrimEnd('\') + '\'
    if (-not ($source.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) -and
              $target.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase))) { throw 'Invalid install paths.' }
    try { [IO.Directory]::Move($source, $target) }
    catch [IO.IOException] { if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw } }
    Confirm-Engine $exe
} finally {
    $resolvedStage = (Resolve-Path -LiteralPath $stage).ProviderPath
    $resolvedPackages = (Resolve-Path -LiteralPath $packages).ProviderPath.TrimEnd('\') + '\'
    if ($resolvedStage.StartsWith($resolvedPackages, [StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $resolvedStage -Recurse -Force
    }
}

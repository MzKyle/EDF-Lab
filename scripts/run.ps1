$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run scripts/setup.ps1 first.' }
if ($args.Count -eq 0) { & $python -m pollen_edf gui }
else { & $python -m pollen_edf @args }
exit $LASTEXITCODE

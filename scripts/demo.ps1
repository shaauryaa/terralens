# TerraLens live demo (Windows PowerShell). CPU only, no API keys, no downloads.
# Usage (from the repo root):  powershell -ExecutionPolicy Bypass -File scripts\demo.ps1 [--skip-tests] [--limit N]
# See DEMO.md and scripts/demo.py for what each step does.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Offline: use only what is already in the Hugging Face cache; never download.
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

& $python scripts\demo.py @args
exit $LASTEXITCODE

#Requires -Version 5.1
<#
.SYNOPSIS
    Side-Step Easy Installer for Windows.
.DESCRIPTION
    Downloads and installs Side-Step using uv (the fast Python
    package manager).  Handles:
      - uv installation (if missing)
      - Python 3.11 (via uv)
      - Side-Step clone + dependency sync
      - CUDA 12.8 PyTorch wheels (automatic via pyproject.toml)
      - Model checkpoint download (from HuggingFace)
.NOTES
    Run from any directory.  Creates a folder in the current directory:

        ./Side-Step/         (standalone training toolkit)

    Requirements:
      - Windows 10/11
      - NVIDIA GPU with CUDA support (for training)
      - Git installed and on PATH
      - ~12 GB free disk (models + deps)
#>

param(
    [string]$InstallDir = ".",
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# ── Colours ──────────────────────────────────────────────────────────────
function Write-Step  { param($m) Write-Host "`n==> $m" -ForegroundColor Cyan }
function Write-Ok    { param($m) Write-Host "  [OK] $m" -ForegroundColor Green }
function Write-Warn  { param($m) Write-Host "  [WARN] $m" -ForegroundColor Yellow }
function Write-Fail  { param($m) Write-Host "  [FAIL] $m" -ForegroundColor Red }

# ── Banner ───────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  ███████ ██ ██████  ███████       ███████ ████████ ███████ ██████" -ForegroundColor Cyan
Write-Host "  ██      ██ ██   ██ ██            ██         ██    ██      ██   ██" -ForegroundColor Cyan
Write-Host "  ███████ ██ ██   ██ █████   █████ ███████    ██    █████   ██████" -ForegroundColor Cyan
Write-Host "       ██ ██ ██   ██ ██                 ██    ██    ██      ██" -ForegroundColor Cyan
Write-Host "  ███████ ██ ██████  ███████       ███████    ██    ███████ ██" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Standalone Installer (v1.0.0-beta)" -ForegroundColor Green
Write-Host ""

# ── Pre-flight: Git ──────────────────────────────────────────────────────
Write-Step "Checking prerequisites"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Fail "Git is not installed or not on PATH."
    Write-Host "  Install from: https://git-scm.com/download/win"
    exit 1
}
Write-Ok "Git found: $(git --version)"
Write-Ok "Install directory: $(Resolve-Path -LiteralPath $InstallDir)"

# Optional Python visibility check (uv manages Python, this is informational)
if (Get-Command python -ErrorAction SilentlyContinue) {
    try {
        $pyv = python --version 2>$null
        if ($pyv) { Write-Ok "System Python found: $pyv (uv will still manage 3.11)" }
    } catch {}
} else {
    Write-Warn "System Python not found on PATH (this is okay; uv will provision Python 3.11)."
}

# ── Install uv if missing ───────────────────────────────────────────────
Write-Step "Checking for uv (fast Python package manager)"

if (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Ok "uv found: $(uv --version)"
} else {
    Write-Host "  Installing uv..."
    try {
        irm https://astral.sh/uv/install.ps1 | iex
        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            $uvPath = Join-Path $env:USERPROFILE ".local\bin"
            $env:Path += ";$uvPath"
        }
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            Write-Fail "uv installation completed but command was not found on PATH."
            Write-Host "  Try opening a new PowerShell window, then run:"
            Write-Host "    irm https://astral.sh/uv/install.ps1 | iex"
            Write-Host "  Or install manually: https://docs.astral.sh/uv/getting-started/installation/"
            exit 1
        }
        Write-Ok "uv installed: $(uv --version)"
    } catch {
        Write-Fail "Could not install uv automatically."
        Write-Host "  Manual install: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    }
}

# ── Clone or verify Side-Step ────────────────────────────────────────────
Write-Step "Setting up Side-Step"

$sideDir = Join-Path $InstallDir "Side-Step"
if (Test-Path $sideDir) {
    Write-Ok "Side-Step directory exists: $sideDir"
} else {
    Write-Host "  Cloning Side-Step..."
    git clone "https://github.com/koda-dernet/Side-Step.git" $sideDir
    Write-Ok "Cloned to $sideDir"
}

# ── Install Python 3.11 via uv ──────────────────────────────────────────
Write-Step "Ensuring Python 3.11 is available"

try {
    uv python install 3.11
    Write-Ok "Python 3.11 ready"
} catch {
    Write-Warn "uv python install failed -- assuming Python 3.11 is already available"
}

# ── Install Side-Step dependencies ───────────────────────────────────────
Write-Step "Installing Side-Step dependencies (this may take a few minutes)"
Write-Host "  PyTorch with CUDA 12.8 will be downloaded automatically."
Write-Host "  First run downloads ~5 GB of wheels.`n"

Set-Location $sideDir
try {
    uv sync
    Write-Ok "Side-Step dependencies installed"
} catch {
    Write-Fail "Dependency sync failed. Check the output above for errors."
    Write-Host "  Common fix: ensure you have enough disk space and a stable internet connection."
    exit 1
}

# ── Download model checkpoints ───────────────────────────────────────────
$checkpointsDir = Join-Path $sideDir "checkpoints"

if (-not $SkipModels) {
    Write-Step "Downloading model checkpoints"

    if (Test-Path (Join-Path $checkpointsDir "acestep-v15-turbo")) {
        Write-Ok "Checkpoints already downloaded"
    } else {
        Write-Host "  This downloads ~8 GB of model weights from HuggingFace.`n"
        try {
            Set-Location $sideDir
            uv run huggingface-cli download ACE-Step/ACE-Step-v1-5-turbo --local-dir (Join-Path $checkpointsDir "acestep-v15-turbo")
            uv run huggingface-cli download ACE-Step/ACE-Step-v1-5-base --local-dir (Join-Path $checkpointsDir "acestep-v15-base")
            Write-Ok "Model checkpoints downloaded to $checkpointsDir"
        } catch {
            Write-Warn "Automatic download failed. You can download manually later:"
            Write-Host "    uv run huggingface-cli download ACE-Step/ACE-Step-v1-5-turbo --local-dir checkpoints/acestep-v15-turbo"
            Write-Host "    uv run huggingface-cli download ACE-Step/ACE-Step-v1-5-base --local-dir checkpoints/acestep-v15-base"
        }
    }
} else {
    Write-Warn "Skipping model download (--SkipModels)."
    Write-Host "  Download later with:"
    Write-Host "    cd $sideDir"
    Write-Host "    uv run huggingface-cli download ACE-Step/ACE-Step-v1-5-turbo --local-dir checkpoints/acestep-v15-turbo"
}

# ── Summary ──────────────────────────────────────────────────────────────
Write-Host "`n"
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Installation complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Side-Step:    $sideDir"
Write-Host "  Checkpoints:  $checkpointsDir"
Write-Host ""
Write-Host "  Quick start:"
Write-Host "    cd `"$sideDir`""
Write-Host "    sidestep.bat              # Pick wizard or GUI"
Write-Host "    sidestep.bat --gui        # Launch GUI directly"
Write-Host "    sidestep.bat fixed --help # CLI help"
Write-Host ""
Write-Host "  Or use uv directly:"
Write-Host "    uv run sidestep"
Write-Host ""
Write-Host "  Alternative (pip users):"
Write-Host "    pip install -r requirements.txt"
Write-Host "    python train.py"
Write-Host ""
Write-Host "  IMPORTANT:"
Write-Host "    - Never rename checkpoint folders"
Write-Host "    - First run will ask where your checkpoints are"
Write-Host ""
Write-Host "  If you get CUDA errors, check:"
Write-Host "    uv run python -c `"import torch; print(torch.cuda.is_available())`""
Write-Host ""
Write-Host "  If uv is not recognized in new terminals:"
Write-Host "    - Close and reopen PowerShell"
Write-Host "    - Or run: $env:Path += ';$env:USERPROFILE\.local\bin'"
Write-Host ""

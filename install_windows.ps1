#Requires -Version 5.1
<#
.SYNOPSIS
    Side-Step Easy Installer for Windows.
.DESCRIPTION
    Downloads and installs Side-Step + ACE-Step 1.5 side-by-side using
    uv (the fast Python package manager).  Handles:
      - uv installation (if missing)
      - Python 3.11 (via uv)
      - ACE-Step 1.5 clone (for model checkpoints and optional vanilla mode)
      - Side-Step clone + dependency sync (standalone training toolkit)
      - CUDA 12.8 PyTorch wheels (automatic via pyproject.toml)
      - Model checkpoint download
.NOTES
    Run from any directory.  Creates two sibling folders in the current
    directory:

        ./ACE-Step-1.5/     (base repo -- checkpoints, vanilla support)
        ./Side-Step/         (standalone training toolkit)

    Requirements:
      - Windows 10/11
      - NVIDIA GPU with CUDA support (for training)
      - Git installed and on PATH
      - ~15 GB free disk (models + deps)
#>

param(
    [string]$InstallDir = ".",
    [switch]$SkipModels,
    [switch]$SkipACEStep
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
Write-Host "  Standalone Installer (v0.9.0-beta)" -ForegroundColor Green
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

# ── Clone or verify ACE-Step 1.5 ────────────────────────────────────────
$aceDir = Join-Path $InstallDir "ACE-Step-1.5"

if (-not $SkipACEStep) {
    Write-Step "Setting up ACE-Step 1.5 (for checkpoints and optional vanilla mode)"

    if (Test-Path $aceDir) {
        Write-Ok "ACE-Step directory exists: $aceDir"
    } else {
        Write-Host "  Cloning ACE-Step 1.5..."
        git clone "https://github.com/ace-step/ACE-Step-1.5.git" $aceDir
        Write-Ok "Cloned to $aceDir"
    }
} else {
    Write-Warn "Skipping ACE-Step clone (--SkipACEStep). Vanilla training will not be available."
}

# ── Clone or verify Side-Step ────────────────────────────────────────────
Write-Step "Setting up Side-Step (standalone training toolkit)"

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

# ── Optional: install ACE-Step deps (for vanilla mode + generation) ──────
if (-not $SkipACEStep -and (Test-Path $aceDir)) {
    Write-Step "Installing ACE-Step dependencies (for vanilla mode and generation)"
    Write-Host "  This is optional but enables vanilla training and the Gradio UI.`n"

    try {
        Set-Location $aceDir
        uv sync
        Write-Ok "ACE-Step dependencies installed"
    } catch {
        Write-Warn "ACE-Step dependency sync failed."
        Write-Host "  Side-Step training still works standalone."
        Write-Host "  Re-run later if you need ACE-Step generation UI/extra tooling:"
        Write-Host "    cd $aceDir && uv sync"
    }
    Set-Location $sideDir
}

# ── Download model checkpoints ───────────────────────────────────────────
if (-not $SkipModels -and (Test-Path $aceDir)) {
    Write-Step "Downloading model checkpoints"

    $checkpointsDir = Join-Path $aceDir "checkpoints"
    if (Test-Path (Join-Path $checkpointsDir "acestep-v15-turbo")) {
        Write-Ok "Checkpoints already downloaded"
    } else {
        Write-Host "  This downloads ~8 GB of model weights.`n"
        try {
            Set-Location $aceDir
            uv run acestep-download
            Write-Ok "Model checkpoints downloaded"
        } catch {
            Write-Warn "Automatic download failed. You can download manually later with:"
            Write-Host "    cd $aceDir && uv run acestep-download"
        }
        Set-Location $sideDir
    }
} elseif ($SkipModels) {
    Write-Warn "Skipping model download (--SkipModels)."
    Write-Host "  Download later from the ACE-Step directory:"
    Write-Host "    cd $aceDir && uv run acestep-download"
} else {
    Write-Warn "No ACE-Step directory -- cannot download models."
    Write-Host "  Download them manually or re-run without --SkipACEStep."
}

# ── Summary ──────────────────────────────────────────────────────────────
Write-Host "`n"
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Installation complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Side-Step:   $sideDir"
if (-not $SkipACEStep) {
    Write-Host "  ACE-Step:    $aceDir"
}
Write-Host ""
Write-Host "  Quick start (from Side-Step directory):"
Write-Host "    cd `"$sideDir`""
Write-Host "    uv run python train.py              # Interactive wizard (first run = setup)"
Write-Host "    uv run python train.py fixed --help # CLI help"
Write-Host ""
Write-Host "  IMPORTANT:"
Write-Host "    - Never rename checkpoint folders"
Write-Host "    - First run will ask where your checkpoints are"
Write-Host "    - Fine-tune training requires the original base model too"
Write-Host ""
if (-not $SkipACEStep) {
    Write-Host "  Generate music (from ACE-Step directory):"
    Write-Host "    cd `"$aceDir`""
    Write-Host "    uv run acestep --share              # Gradio UI"
    Write-Host ""
}
Write-Host "  If you get CUDA errors, check:"
Write-Host "    uv run python -c `"import torch; print(torch.cuda.is_available())`""
Write-Host ""
Write-Host "  If uv is not recognized in new terminals:"
Write-Host "    - Close and reopen PowerShell"
Write-Host "    - Or run: $env:Path += ';$env:USERPROFILE\.local\bin'"
Write-Host ""

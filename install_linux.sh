#!/usr/bin/env bash
# ====================================================================
#  Side-Step Easy Installer (Linux / macOS)
#
#  Usage:
#    curl -sSL <url>/install.sh | bash
#    -- or --
#    chmod +x install.sh && ./install.sh
#
#  Options:
#    --skip-models    Skip downloading model checkpoints
#    --dir <path>     Install directory (default: current directory)
# ====================================================================
set -euo pipefail

# ── Colours ─────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'
step()  { echo -e "\n${CYAN}==> $1${NC}"; }
ok()    { echo -e "  ${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "  ${YELLOW}[WARN]${NC} $1"; }
fail()  { echo -e "  ${RED}[FAIL]${NC} $1"; }

# ── Parse args ──────────────────────────────────────────────────────
INSTALL_DIR="."
SKIP_MODELS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-models) SKIP_MODELS=true; shift ;;
        --dir) INSTALL_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Banner ──────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}  ███████ ██ ██████  ███████       ███████ ████████ ███████ ██████${NC}"
echo -e "${CYAN}  ██      ██ ██   ██ ██            ██         ██    ██      ██   ██${NC}"
echo -e "${CYAN}  ███████ ██ ██   ██ █████   █████ ███████    ██    █████   ██████${NC}"
echo -e "${CYAN}       ██ ██ ██   ██ ██                 ██    ██    ██      ██${NC}"
echo -e "${CYAN}  ███████ ██ ██████  ███████       ███████    ██    ███████ ██${NC}"
echo ""
echo -e "  ${GREEN}Standalone Installer (v1.0.0-beta)${NC}"
echo ""

# ── Pre-flight: Git ─────────────────────────────────────────────────
step "Checking prerequisites"

if ! command -v git &>/dev/null; then
    fail "Git is not installed."
    echo "  Install with: sudo apt install git  (or brew install git on macOS)"
    exit 1
fi
ok "Git found: $(git --version)"

# ── Install uv if missing ──────────────────────────────────────────
step "Checking for uv (fast Python package manager)"

if command -v uv &>/dev/null; then
    ok "uv found: $(uv --version)"
else
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the env so uv is on PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
    if command -v uv &>/dev/null; then
        ok "uv installed: $(uv --version)"
    else
        fail "uv installation completed but command not found."
        echo "  Try: export PATH=\"\$HOME/.local/bin:\$PATH\""
        echo "  Or install manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
fi

# ── Clone or verify Side-Step ───────────────────────────────────────
step "Setting up Side-Step"

SIDE_DIR="$INSTALL_DIR/Side-Step"
if [[ -d "$SIDE_DIR" ]]; then
    ok "Side-Step directory exists: $SIDE_DIR"
else
    echo "  Cloning Side-Step..."
    git clone "https://github.com/koda-dernet/Side-Step.git" "$SIDE_DIR"
    ok "Cloned to $SIDE_DIR"
fi

# ── Install Python 3.11 via uv ─────────────────────────────────────
step "Ensuring Python 3.11 is available"

uv python install 3.11 2>/dev/null && ok "Python 3.11 ready" || warn "Assuming Python 3.11 is already available"

# ── Install dependencies ────────────────────────────────────────────
step "Installing Side-Step dependencies (this may take a few minutes)"
echo "  PyTorch with CUDA 12.8 will be downloaded automatically."
echo "  First run downloads ~5 GB of wheels."
echo ""

cd "$SIDE_DIR"
if uv sync; then
    ok "Side-Step dependencies installed"
else
    fail "Dependency sync failed. Check the output above."
    exit 1
fi

# ── Download model checkpoints ──────────────────────────────────────
CKPT_DIR="$SIDE_DIR/checkpoints"

if [[ "$SKIP_MODELS" == "false" ]]; then
    step "Downloading model checkpoints"

    if [[ -d "$CKPT_DIR/acestep-v15-turbo" ]]; then
        ok "Checkpoints already downloaded"
    else
        echo "  This downloads ~8 GB of model weights from HuggingFace."
        echo ""
        if uv run huggingface-cli download ACE-Step/ACE-Step-v1-5-turbo --local-dir "$CKPT_DIR/acestep-v15-turbo" && \
           uv run huggingface-cli download ACE-Step/ACE-Step-v1-5-base --local-dir "$CKPT_DIR/acestep-v15-base"; then
            ok "Model checkpoints downloaded to $CKPT_DIR"
        else
            warn "Automatic download failed. Download manually later:"
            echo "    uv run huggingface-cli download ACE-Step/ACE-Step-v1-5-turbo --local-dir checkpoints/acestep-v15-turbo"
            echo "    uv run huggingface-cli download ACE-Step/ACE-Step-v1-5-base --local-dir checkpoints/acestep-v15-base"
        fi
    fi
else
    warn "Skipping model download (--skip-models)."
    echo "  Download later with:"
    echo "    cd $SIDE_DIR"
    echo "    uv run huggingface-cli download ACE-Step/ACE-Step-v1-5-turbo --local-dir checkpoints/acestep-v15-turbo"
fi

# ── Summary ─────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Installation complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "  Side-Step:    $SIDE_DIR"
echo "  Checkpoints:  $CKPT_DIR"
echo ""
echo "  Quick start:"
echo "    cd \"$SIDE_DIR\""
echo "    ./sidestep.sh              # Pick wizard or GUI"
echo "    ./sidestep.sh --gui        # Launch GUI directly"
echo "    ./sidestep.sh fixed --help # CLI help"
echo ""
echo "  Or use uv directly:"
echo "    uv run sidestep"
echo ""
echo "  Alternative (pip users):"
echo "    pip install -r requirements.txt"
echo "    python train.py"
echo ""
echo "  IMPORTANT:"
echo "    - Never rename checkpoint folders"
echo "    - First run will ask where your checkpoints are"
echo ""
echo "  If you get CUDA errors, check:"
echo "    uv run python -c \"import torch; print(torch.cuda.is_available())\""
echo ""

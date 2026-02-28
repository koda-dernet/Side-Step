#!/usr/bin/env bash
# ====================================================================
#  Side-Step Launcher (Linux / macOS)
#
#  Usage:
#    ./sidestep.sh              # Interactive menu
#    ./sidestep.sh gui           # Skip menu, launch GUI directly
#    ./sidestep.sh train ...    # Skip menu, direct CLI mode
# ====================================================================
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv &>/dev/null; then
    echo "[FAIL] uv not found. Install it: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# If args provided, pass through directly (CLI mode)
if [[ $# -gt 0 ]]; then
    uv run sidestep "$@"
    exit $?
fi

# Interactive menu
CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
echo ""
echo -e "  ${CYAN}░▒▓ S I D E · S T E P ▓▒░${NC}"
echo -e "  ${GREEN}v1.0.0-beta${NC}"
echo ""
echo -e "  ${YELLOW}[1]${NC}  Wizard   — Interactive CLI training wizard"
echo -e "  ${YELLOW}[2]${NC}  GUI      — Launch web GUI in browser"
echo ""
read -rp "  Pick [1/2]: " choice

case "$choice" in
    2|gui|GUI)
        echo ""
        echo "  Launching GUI..."
        uv run sidestep gui
        ;;
    *)
        echo ""
        echo "  Launching wizard..."
        uv run sidestep
        ;;
esac

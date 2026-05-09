#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

step()   { echo -e "\n${CYAN}▸ $1${NC}"; }
ok()     { echo -e "  ${GREEN}✓ $1${NC}"; }
warn()   { echo -e "  ${YELLOW}⚠ $1${NC}"; }
fail()   { echo -e "  ${RED}✗ $1${NC}"; }

MODEL="${MODEL:-Qwen/Qwen2-1.5B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_TOKENS="${MAX_TOKENS:-256}"
SKIP_DEPS="${SKIP_DEPS:-false}"
SKIP_TESTS="${SKIP_TESTS:-false}"

step "Preflight checks"

if ! command -v uv &>/dev/null; then
    fail "uv not found — install it: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
ok "uv: $(uv --version)"

PYTHON="$(uv python find 3.10 2>/dev/null || uv python find 3.11 2>/dev/null || uv python find 3.12 2>/dev/null || true)"
if [ -z "$PYTHON" ]; then
    warn "No compatible Python found — uv will install one automatically"
fi

step "Setting up local environment (.venv)"
if [ "$SKIP_DEPS" = "false" ]; then
    if [ ! -d ".venv" ]; then
        uv venv .venv
        ok "Created .venv"
    fi
    uv pip install -r requirements.txt
    ok "Dependencies installed in .venv"
else
    if [ ! -d ".venv" ]; then
        uv venv .venv
        ok "Created .venv"
    fi
    warn "Skipped install (SKIP_DEPS=true)"
fi

VENV_PYTHON=".venv/bin/python"

step "Running tests"
if [ "$SKIP_TESTS" = "false" ]; then
    if $VENV_PYTHON -m pytest tests/ -v --tb=short; then
        ok "All tests passed"
    else
        warn "Some tests failed — continuing anyway (tests may require vLLM/GPU)"
    fi
else
    warn "Skipped (SKIP_TESTS=true)"
fi

step "Running experiment configurations"
echo "  Model:         $MODEL"
echo "  Max model len:  $MAX_MODEL_LEN"
echo "  Max tokens:     $MAX_TOKENS"

$VENV_PYTHON scripts/run_experiment.py --all \
    --model "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-tokens "$MAX_TOKENS" \
    --skip-unsupported

ok "Experiments complete — results in results/raw/"

step "Generating reports"
$VENV_PYTHON scripts/run_experiment.py --report-only
ok "Reports generated in results/reports/"

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}  All done!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "  Raw results:  results/raw/"
echo "  Reports:      results/reports/"
echo ""
echo "  Key files:"
echo "    results/reports/full_report.md"
echo "    results/reports/metrics_summary.csv"
echo ""

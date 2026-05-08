#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "========================================="
echo "LLM Inference Configuration Experiment"
echo "========================================="
echo ""

# Default values (override via environment variables)
MODEL="${MODEL:-Qwen/Qwen2-1.5B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_TOKENS="${MAX_TOKENS:-256}"

echo "Model: $MODEL"
echo "Max model length: $MAX_MODEL_LEN"
echo "Max tokens: $MAX_TOKENS"
echo ""

echo "Step 1: Running all configurations..."
python scripts/run_experiment.py --all \
    --model "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-tokens "$MAX_TOKENS" \
    --skip-unsupported

echo ""
echo "Step 2: Generating reports..."
python scripts/run_experiment.py --report-only

echo ""
echo "========================================="
echo "Experiment complete!"
echo "Results: results/raw/"
echo "Reports: results/reports/"
echo "========================================="

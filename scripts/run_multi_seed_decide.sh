#!/usr/bin/env bash
set -euo pipefail

# Multi-seed sequential evaluation and aggregation for oniwa-decide
# Usage: ./scripts/run_multi_seed_decide.sh [steps] [config]
# Example: ./scripts/run_multi_seed_decide.sh 20 standard

STEPS="${1:-20}"
CONFIG="${2:-standard}"
OUTPUT_DIR="logs/experiments/multi_seed_${CONFIG}"
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo " 🧪 ONIWA: Multi-Seed Experimentation Suite"
echo "    Config: ${CONFIG}, Steps: ${STEPS}, Seeds: 42 43 44 45 46"
echo "============================================================"

SEEDS=(42 43 44 45 46)

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "▶️ Running Seed ${SEED}..."
    CKPT_DIR="crates/oniwa-decide/checkpoints/experiments/seed_${SEED}"
    METRICS_FILE="${OUTPUT_DIR}/metrics_seed_${SEED}.json"

    cargo run --release -p oniwa-decide --bin train -- \
        --steps "$STEPS" \
        --config "$CONFIG" \
        --seed "$SEED" \
        --reset \
        --checkpoint-dir "$CKPT_DIR" \
        --output-metrics "$METRICS_FILE"
done

echo ""
echo "📊 Aggregating multi-seed metrics..."
cargo run --release -p oniwa-decide --bin bench -- --iters 20 --output "${OUTPUT_DIR}/hardware_profile.jsonl"

echo "✅ All 5 seeds complete! Results stored in ${OUTPUT_DIR}"

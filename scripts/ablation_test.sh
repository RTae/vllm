#!/usr/bin/env bash
# Ablation study: speculative decoding strategies
# Runs baseline and Eagle3 speculative decoding back-to-back and prints
# a timing summary at the end.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
INFER="$REPO_ROOT/scripts/infer_drivelm_lora.py"

ADAPTER_PATH="${ADAPTER_PATH:-/workspace/vllm/models/Qwen3-VL-8B-Instruct}"
BASE_MODEL="${BASE_MODEL:-/workspace/.hf_home/qwen3-vl-8b/}"
EAGLE3_MODEL="${EAGLE3_MODEL:-/workspace/.hf_home/qwen3-vl-8b-eagle3}"
DATASET="${DATASET:-/workspace/vllm/datasets/DriveLM_nuScenes/split/val}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-5}"

# ── Results directory ────────────────────────────────────────────────────────
RESULTS_DIR="$REPO_ROOT/ablation_results/speculative_decoding"
mkdir -p "$RESULTS_DIR"

SUMMARY_FILE="$RESULTS_DIR/summary.txt"
: > "$SUMMARY_FILE"

# ── Helper ───────────────────────────────────────────────────────────────────
run_experiment() {
  local name="$1"
  shift
  local log_file="$RESULTS_DIR/${name}.log"

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Running: $name"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  local start_ts
  start_ts=$(date +%s)

  "$PYTHON" "$INFER" "$@" 2>&1 | tee "$log_file"

  local end_ts exit_code=${PIPESTATUS[0]}
  end_ts=$(date +%s)
  local elapsed=$(( end_ts - start_ts ))

  if [[ $exit_code -eq 0 ]]; then
    echo "  ✓ $name completed in ${elapsed}s" | tee -a "$SUMMARY_FILE"
  else
    echo "  ✗ $name FAILED after ${elapsed}s (exit $exit_code)" | tee -a "$SUMMARY_FILE"
  fi
}

# ── Experiments ──────────────────────────────────────────────────────────────
COMMON_ARGS=(
  --adapter-path  "$ADAPTER_PATH"
  --base-model    "$BASE_MODEL"
  --dataset       "$DATASET"
  --num-samples   "$NUM_SAMPLES"
)

run_experiment "baseline" \
  "${COMMON_ARGS[@]}"

run_experiment "eagle3_spec_tokens_${NUM_SPEC_TOKENS}" \
  "${COMMON_ARGS[@]}" \
  --eagle3 "$EAGLE3_MODEL" \
  --num-speculative-tokens "$NUM_SPEC_TOKENS"

# ── Print summary ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Ablation summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cat "$SUMMARY_FILE"
echo ""
echo "  Logs saved to: $RESULTS_DIR"

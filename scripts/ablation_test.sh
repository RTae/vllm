#!/usr/bin/env bash
# Ablation study: all inference configurations from README
# Skips SpargeAttention (not yet stable).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
INFER="$REPO_ROOT/scripts/infer_drivelm_lora.py"

ADAPTER_PATH="${ADAPTER_PATH:-/workspace/vllm/models/Qwen3-VL-8B-Instruct}"
BASE_MODEL="${BASE_MODEL:-/workspace/.hf_home/qwen3-vl-8b/}"
DRAFT_MODEL="${DRAFT_MODEL:-/workspace/.hf_home/qwen3-vl-2b-merged/}"
EAGLE3_MODEL="${EAGLE3_MODEL:-/workspace/.hf_home/qwen3-vl-8b-eagle3}"
DATASET="${DATASET:-/workspace/vllm/datasets/DriveLM_nuScenes/split/val}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-5}"

# ── Results directory ────────────────────────────────────────────────────────
RESULTS_DIR="$REPO_ROOT/ablation_results"
mkdir -p "$RESULTS_DIR"

SUMMARY_FILE="$RESULTS_DIR/summary.txt"
: > "$SUMMARY_FILE"

# ── Helper ───────────────────────────────────────────────────────────────────
run_experiment() {
  local name="$1"
  shift
  local log_file="$RESULTS_DIR/${name}.log"
  local metrics_file="$RESULTS_DIR/${name}_metrics.json"

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Running: $name"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  local start_ts
  start_ts=$(date +%s)

  "$PYTHON" "$INFER" "$@" --output-metrics "$metrics_file" 2>&1 | tee "$log_file"

  local end_ts exit_code=${PIPESTATUS[0]}
  end_ts=$(date +%s)
  local elapsed=$(( end_ts - start_ts ))

  if [[ $exit_code -eq 0 ]]; then
    echo "  ✓ $name  wall=${elapsed}s" | tee -a "$SUMMARY_FILE"
    if [[ -f "$metrics_file" ]]; then
      python3 - "$metrics_file" >> "$SUMMARY_FILE" <<'PYEOF'
import json, sys
data = json.load(open(sys.argv[1]))
a = data.get("aggregate", {})
print(f"    throughput       : {a.get('throughput_samples_per_s')} samples/s")
print(f"    e2e latency p50  : {a.get('e2e_latency_p50_s')}s")
print(f"    e2e latency p95  : {a.get('e2e_latency_p95_s')}s")
print(f"    TTFT mean        : {a.get('ttft_mean_s')}s")
print(f"    prefill time mean: {a.get('prefill_time_mean_s')}s")
print(f"    prefill time p50 : {a.get('prefill_time_p50_s')}s")
print(f"    decode time mean : {a.get('decode_time_mean_s')}s")
print(f"    decode time p50  : {a.get('decode_time_p50_s')}s")
print(f"    tokens/s mean    : {a.get('tokens_per_second_mean')}")
print(f"    total gen tokens : {a.get('total_gen_tokens')}")
PYEOF
    fi
  else
    echo "  ✗ $name FAILED after ${elapsed}s (exit $exit_code)" | tee -a "$SUMMARY_FILE"
  fi
}

# ── Shared args ───────────────────────────────────────────────────────────────
COMMON=(
  --adapter-path "$ADAPTER_PATH"
  --base-model   "$BASE_MODEL"
  --dataset      "$DATASET"
  --num-samples  "$NUM_SAMPLES"
)


# ════════════════════════════════════════════════════════════════════════════════
# Section 1: Attention Backends
# Fixed: no APC so only the attention kernel changes.
# ════════════════════════════════════════════════════════════════════════════════
print_section() { echo ""; echo ""; echo "  ▶ $1"; echo "" | tee -a "$SUMMARY_FILE"; echo "  ▶ $1" >> "$SUMMARY_FILE"; }

print_section "BASELINE"

# 0. True baseline – vLLM defaults (Triton, no APC, no SD)
run_experiment "00_baseline_default" \
  "${COMMON[@]}" \
  --attention-backend TRITON_ATTN \
  --no-prefix-caching

print_section "ATTENTION BACKENDS"

# 1. FlashAttention
run_experiment "01b_attn_flash_no_apc" \
  "${COMMON[@]}" \
  --no-prefix-caching

# ════════════════════════════════════════════════════════════════════════════════
# Section 2: Automatic Prefix Caching
# Fixed: Triton backend so only APC changes.
# ════════════════════════════════════════════════════════════════════════════════
print_section "AUTOMATIC PREFIX CACHING"

# 2. Triton – APC enabled
run_experiment "02b_apc_triton_on" \
  "${COMMON[@]}" \
  --attention-backend TRITON_ATTN

# ════════════════════════════════════════════════════════════════════════════════
# Section 3: Speculative Decoding
# Fixed: Triton + no APC so only the SD strategy changes.
# ════════════════════════════════════════════════════════════════════════════════
print_section "SPECULATIVE DECODING (Triton, no APC)"

SD_BASE=(
  "${COMMON[@]}"
  --attention-backend TRITON_ATTN
  --no-prefix-caching
)

# 3a. N-gram
run_experiment "03b_sd_ngram_k${NUM_SPEC_TOKENS}" \
  "${SD_BASE[@]}" \
  --ngram \
  --num-speculative-tokens "$NUM_SPEC_TOKENS"

# 3b. Draft model (merged 2B LoRA)
if [[ -d "$DRAFT_MODEL" ]]; then
  run_experiment "03b_sd_draft_2b_k${NUM_SPEC_TOKENS}" \
    "${SD_BASE[@]}" \
    --speculative-decoding \
    --draft-model "$DRAFT_MODEL" \
    --num-speculative-tokens "$NUM_SPEC_TOKENS"
else
  echo "  ⚠  SKIP 03b_sd_draft_2b: $DRAFT_MODEL not found" | tee -a "$SUMMARY_FILE"
fi

# 3c. EAGLE3
if [[ -d "$EAGLE3_MODEL" ]]; then
  run_experiment "03c_sd_eagle3_k${NUM_SPEC_TOKENS}" \
    "${SD_BASE[@]}" \
    --eagle3 "$EAGLE3_MODEL" \
    --num-speculative-tokens "$NUM_SPEC_TOKENS"
else
  echo "  ⚠  SKIP 03c_sd_eagle3: $EAGLE3_MODEL not found" | tee -a "$SUMMARY_FILE"
fi

# ── Apply all ─────────────────────────────────────────────────────────────────
print_section "APPLY ALL (Eagle3 + FlashAttention + APC)"

if [[ -d "$EAGLE3_MODEL" ]]; then
  run_experiment "04_all_eagle3_flash_apc" \
    "${COMMON[@]}" \
    --eagle3 "$EAGLE3_MODEL" \
    --num-speculative-tokens "$NUM_SPEC_TOKENS"
else
  echo "  ⚠  SKIP 04_all: $EAGLE3_MODEL not found" | tee -a "$SUMMARY_FILE"
fi

# ── Print summary ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Ablation summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cat "$SUMMARY_FILE"
echo ""
echo "  Logs and metrics: $RESULTS_DIR"

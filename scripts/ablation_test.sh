#!/usr/bin/env bash
# Ablation study: all inference configurations
# New in this version:
#   Section 5: Chunked-prefill token budget (--max-num-batched-tokens 16384)
#   Section 6: Sequential scene inference (--sequential-scenes)
#   Section 7: Sparse attention backends (SPARGE_ATTN, XATTN)
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
# Token budget for chunked prefill (DriveLM seqs ~8421 tokens → 2 chunks at 8192 default)
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}"

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
      "$PYTHON" - "$metrics_file" >> "$SUMMARY_FILE" <<'PYEOF'
import json, sys
data = json.load(open(sys.argv[1]))
a = data.get("aggregate", {})
print(f"    throughput       : {a.get('throughput_samples_per_s')} samples/s")
print(f"    e2e latency p50  : {a.get('e2e_latency_p50_s')}s")
print(f"    e2e latency p95  : {a.get('e2e_latency_p95_s')}s")
print(f"    compute time mean: {a.get('request_compute_time_mean_s')}s  (prefill+decode)")
print(f"    compute time p50 : {a.get('request_compute_time_p50_s')}s")
print(f"    TTFT mean        : {a.get('ttft_mean_s')}s")
print(f"    prefill time mean: {a.get('prefill_time_mean_s')}s")
print(f"    prefill time p50 : {a.get('prefill_time_p50_s')}s")
print(f"    decode time mean : {a.get('decode_time_mean_s')}s")
print(f"    decode time p50  : {a.get('decode_time_p50_s')}s")
print(f"    TPOT mean        : {a.get('tpot_mean_s')}s  (~ITL)")
print(f"    TPOT p50         : {a.get('tpot_p50_s')}s")
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
# Section 0: Baseline (Triton, all caches off)
# Everything else is compared against this.
# ════════════════════════════════════════════════════════════════════════════════
print_section() { echo ""; echo ""; echo "  ▶ $1"; echo "" | tee -a "$SUMMARY_FILE"; echo "  ▶ $1" >> "$SUMMARY_FILE"; }

print_section "BASELINE (Triton, all caches off)"

NO_CACHE=(
  --attention-backend TRITON_ATTN
  --no-prefix-caching
  --disable-mm-preprocessor-cache
  --disable-chunked-mm-input
)

run_experiment "00_baseline" \
  "${COMMON[@]}" \
  "${NO_CACHE[@]}"

# ════════════════════════════════════════════════════════════════════════════════
# Section 1: Attention Backends
# Fixed: all caches off, no SD — isolates kernel differences.
# ════════════════════════════════════════════════════════════════════════════════
print_section "ATTENTION BACKENDS (all caches off)"

# 1a. FlashAttention (default backend)
run_experiment "01a_attn_flash" \
  "${COMMON[@]}" \
  --no-prefix-caching \
  --disable-mm-preprocessor-cache \
  --disable-chunked-mm-input

# 1b. FlashInfer
run_experiment "01b_attn_flashinfer" \
  "${COMMON[@]}" \
  --attention-backend FLASHINFER \
  --no-prefix-caching \
  --disable-mm-preprocessor-cache \
  --disable-chunked-mm-input

# 1c. SPARGE_ATTN (topk=0.5) — sparse prefill, paged kernel, no gather
SPARGE_ATTN_TOPK=0.5 \
run_experiment "01c_attn_sparge_topk0.5" \
  "${COMMON[@]}" \
  --attention-backend SPARGE_ATTN \
  --no-prefix-caching \
  --disable-mm-preprocessor-cache \
  --disable-chunked-mm-input

# 1d. XAttention (topk=0.5, stride=8) — block-level importance scoring
XATTN_THRESHOLD=0.5 XATTN_STRIDE=8 \
run_experiment "01d_attn_xattn_topk0.5" \
  "${COMMON[@]}" \
  --attention-backend XATTN \
  --no-prefix-caching \
  --disable-mm-preprocessor-cache \
  --disable-chunked-mm-input

# ════════════════════════════════════════════════════════════════════════════════
# Section 2: Caching Ablation
# Fixed: Triton backend, no SD — add one cache at a time.
# ════════════════════════════════════════════════════════════════════════════════
print_section "CACHING ABLATION (Triton, no SD)"

# 2a. APC only
run_experiment "02a_cache_apc_only" \
  "${COMMON[@]}" \
  --attention-backend TRITON_ATTN \
  --disable-mm-preprocessor-cache \
  --disable-chunked-mm-input

# 2b. MM preprocessor cache only
run_experiment "02b_cache_mm_preprocessor_only" \
  "${COMMON[@]}" \
  --attention-backend TRITON_ATTN \
  --no-prefix-caching \
  --disable-chunked-mm-input

# 2c. Chunked MM input only
run_experiment "02c_cache_chunked_mm_only" \
  "${COMMON[@]}" \
  --attention-backend TRITON_ATTN \
  --no-prefix-caching \
  --disable-mm-preprocessor-cache

# 2d. APC + MM preprocessor cache
run_experiment "02d_cache_apc_and_mm" \
  "${COMMON[@]}" \
  --attention-backend TRITON_ATTN \
  --disable-chunked-mm-input

# 2e. All caches enabled
run_experiment "02e_cache_all" \
  "${COMMON[@]}" \
  --attention-backend TRITON_ATTN

# ════════════════════════════════════════════════════════════════════════════════
# Section 3: Speculative Decoding
# Fixed: Triton + all caches off — isolates SD strategy.
# ════════════════════════════════════════════════════════════════════════════════
print_section "SPECULATIVE DECODING (Triton, all caches off)"

SD_BASE=(
  "${COMMON[@]}"
  "${NO_CACHE[@]}"
)

# 3a. N-gram
run_experiment "03a_sd_ngram_k${NUM_SPEC_TOKENS}" \
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

# ════════════════════════════════════════════════════════════════════════════════
# Section 4: Apply all
# EAGLE3 + FlashAttention + all caches
# ════════════════════════════════════════════════════════════════════════════════
print_section "APPLY ALL (Eagle3 + FlashAttention + all caches)"

if [[ -d "$EAGLE3_MODEL" ]]; then
  run_experiment "04_all_eagle3_flash_all_caches" \
    "${COMMON[@]}" \
    --eagle3 "$EAGLE3_MODEL" \
    --num-speculative-tokens "$NUM_SPEC_TOKENS"
else
  echo "  ⚠  SKIP 04_all: $EAGLE3_MODEL not found" | tee -a "$SUMMARY_FILE"
fi

# ════════════════════════════════════════════════════════════════════════════════
# Section 5: Chunked-prefill token budget
# DriveLM sequences are ~8421 tokens (6 cameras). Default max_num_batched_tokens=8192
# splits each into 2 prefill chunks. Setting 16384 processes in 1 chunk, saving
# one full 28-layer LLM forward pass (~465ms/request).
# ════════════════════════════════════════════════════════════════════════════════
print_section "CHUNKED PREFILL BUDGET (--max-num-batched-tokens)"

# 5a. Default (vLLM heuristic, typically 8192)
run_experiment "05a_batched_default" \
  "${COMMON[@]}" \
  --no-prefix-caching \
  --disable-mm-preprocessor-cache \
  --disable-chunked-mm-input

# 5b. 16384 — fits each DriveLM sequence in 1 prefill step
run_experiment "05b_batched_16384" \
  "${COMMON[@]}" \
  --no-prefix-caching \
  --disable-mm-preprocessor-cache \
  --disable-chunked-mm-input \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS"

# 5c. 16384 + APC: best combined throughput setting
run_experiment "05c_batched_16384_apc" \
  "${COMMON[@]}" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS"

# ════════════════════════════════════════════════════════════════════════════════
# Section 6: Sequential scene inference
# Submits one scene at a time (all questions per scene in one generate() call).
# Guarantees 100% intra-scene APC hit rate — image KV blocks cannot be evicted
# between questions from the same scene.
# Trade-off: lower GPU utilisation (-28% throughput), but 15× lower p50 latency.
# ════════════════════════════════════════════════════════════════════════════════
print_section "SEQUENTIAL SCENE INFERENCE (--sequential-scenes)"

# 6a. Default batch (all 100 at once) — reference for this section
run_experiment "06a_batch_all" \
  "${COMMON[@]}" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS"

# 6b. Sequential scenes (one generate() per scene)
run_experiment "06b_sequential_scenes" \
  "${COMMON[@]}" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --sequential-scenes

# ════════════════════════════════════════════════════════════════════════════════
# Section 7: Sparse attention + 16384 token budget
# Compare sparse backends with the larger token budget for fair comparison.
# At ~8K tokens attention = 0.5% of prefill; gains expected at ≥32K tokens.
# ════════════════════════════════════════════════════════════════════════════════
print_section "SPARSE ATTENTION + 16384 BUDGET (no caches)"

export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 7a. SPARGE_ATTN + 16384
SPARGE_ATTN_TOPK=0.5 \
run_experiment "07a_sparge_topk0.5_b16384" \
  "${COMMON[@]}" \
  --no-prefix-caching \
  --disable-mm-preprocessor-cache \
  --disable-chunked-mm-input \
  --attention-backend SPARGE_ATTN \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS"

# 7b. XATTN + 16384
XATTN_THRESHOLD=0.5 XATTN_STRIDE=8 \
run_experiment "07b_xattn_topk0.5_b16384" \
  "${COMMON[@]}" \
  --no-prefix-caching \
  --disable-mm-preprocessor-cache \
  --disable-chunked-mm-input \
  --attention-backend XATTN \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS"

# ── Print summary ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Ablation summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cat "$SUMMARY_FILE"
echo ""
echo "  Logs and metrics: $RESULTS_DIR"

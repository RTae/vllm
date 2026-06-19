#!/usr/bin/env bash
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
SPEC_TOKEN_SWEEP="${SPEC_TOKEN_SWEEP:-3 5 7}"
# Default token budget for all runs. DriveLM seqs are ~8421 tokens, so 16384
# keeps each request in a single prefill chunk unless a run overrides it.
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}"
# Ground-truth reference file for evaluation.py
VAL_COT="${VAL_COT:-$REPO_ROOT/datasets/DriveLM_nuScenes/refs/val_cot.json}"

# ── Results directory ────────────────────────────────────────────────────────
RESULTS_DIR="$REPO_ROOT/ablation_results"
mkdir -p "$RESULTS_DIR"

SUMMARY_FILE="$RESULTS_DIR/summary.txt"
: > "$SUMMARY_FILE"

# ── Helpers ──────────────────────────────────────────────────────────────────

append_metrics_summary() {
  local metrics_file="$1"

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
}

has_valid_metrics() {
  local metrics_file="$1"

  [[ -s "$metrics_file" ]] || return 1
  "$PYTHON" - "$metrics_file" <<'PYEOF' >/dev/null 2>&1
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
if not isinstance(data, dict) or "aggregate" not in data:
    raise SystemExit(1)
PYEOF
}

# Run evaluation.py on a metrics JSON and append scores to summary
run_eval() {
  local name="$1"
  local metrics_file="$RESULTS_DIR/${name}_metrics.json"
  local pred_file="$RESULTS_DIR/${name}_predictions.json"
  local eval_file="$RESULTS_DIR/${name}_eval.json"

  if [[ ! -f "$metrics_file" ]]; then
    echo "    [eval] SKIP — metrics file not found: $metrics_file" | tee -a "$SUMMARY_FILE"
    return
  fi
  if [[ ! -f "$VAL_COT" ]]; then
    echo "    [eval] SKIP — val_cot.json not found: $VAL_COT" | tee -a "$SUMMARY_FILE"
    return
  fi

  # Convert inference output → evaluation format
  "$PYTHON" "$REPO_ROOT/scripts/convert_metrics_to_eval_format.py" \
    --src "$metrics_file" --dst "$pred_file" 2>/dev/null

  # Run evaluation and capture scores
  eval_output=$("$PYTHON" "$REPO_ROOT/scripts/evaluation.py" \
    --src "$pred_file" --tgt "$VAL_COT" 2>/dev/null)

  accuracy=$(echo "$eval_output" | grep "accuracy score" | awk '{print $NF}')
  match=$(echo "$eval_output" | grep "match score" | awk '{print $NF}')
  final=$(echo "$eval_output" | grep "final combined score" | awk '{print $NF}')

  echo "    accuracy         : $accuracy" | tee -a "$SUMMARY_FILE"
  echo "    match score      : $match" | tee -a "$SUMMARY_FILE"
  echo "    final score      : $final  (accuracy+match+language combined)" | tee -a "$SUMMARY_FILE"

  # Save full eval output
  echo "$eval_output" > "$eval_file"
}
run_experiment() {
  local name="$1"
  shift
  local log_file="$RESULTS_DIR/${name}.log"
  local metrics_file="$RESULTS_DIR/${name}_metrics.json"
  local infer_args=("$@")
  local has_max_batched_tokens=false

  for arg in "${infer_args[@]}"; do
    if [[ "$arg" == "--max-num-batched-tokens" ]]; then
      has_max_batched_tokens=true
      break
    fi
  done

  if [[ "$has_max_batched_tokens" == false ]]; then
    infer_args+=(--max-num-batched-tokens "$MAX_BATCHED_TOKENS")
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Running: $name"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  if has_valid_metrics "$metrics_file"; then
    echo "  ↷ $name  SKIP (existing metrics found)" | tee -a "$SUMMARY_FILE"
    append_metrics_summary "$metrics_file"
    run_eval "$name"
    return
  fi

  local start_ts
  start_ts=$(date +%s)

  "$PYTHON" "$INFER" "${infer_args[@]}" --output-metrics "$metrics_file" 2>&1 | tee "$log_file"

  local end_ts exit_code=${PIPESTATUS[0]}
  end_ts=$(date +%s)
  local elapsed=$(( end_ts - start_ts ))

  if [[ $exit_code -eq 0 ]]; then
    echo "  ✓ $name  wall=${elapsed}s" | tee -a "$SUMMARY_FILE"
    if [[ -f "$metrics_file" ]]; then
      append_metrics_summary "$metrics_file"
      # Run DriveLM evaluation (accuracy, match, language scores)
      run_eval "$name"
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
# Section 0: Base family
# Everything else is compared against this.
# ════════════════════════════════════════════════════════════════════════════════
print_section() { echo ""; echo ""; echo "  ▶ $1"; echo "" | tee -a "$SUMMARY_FILE"; echo "  ▶ $1" >> "$SUMMARY_FILE"; }

print_section "BASE: Baseline"

NO_CACHE=(
  --attention-backend TRITON_ATTN
  --no-prefix-caching
  --disable-mm-preprocessor-cache
  --disable-chunked-mm-input
)

run_experiment "base_baseline" \
  "${COMMON[@]}" \
  "${NO_CACHE[@]}"

# ════════════════════════════════════════════════════════════════════════════════
# Section 1: Attn family
# Fixed: all caches off, no SD — isolates kernel differences.
# ════════════════════════════════════════════════════════════════════════════════
print_section "ATTN: Attention Backends"

# 1a. FlashAttention (default backend)
run_experiment "attn_flashattention" \
  "${COMMON[@]}" \
  --no-prefix-caching \
  --disable-mm-preprocessor-cache \
  --disable-chunked-mm-input

# 1b. FlashInfer
run_experiment "attn_flashinfer" \
  "${COMMON[@]}" \
  --attention-backend FLASHINFER \
  --no-prefix-caching \
  --disable-mm-preprocessor-cache \
  --disable-chunked-mm-input

# 1c. SPARGE_ATTN (topk=0.5) — sparse prefill, paged kernel, no gather
SPARGE_ATTN_TOPK=0.5 \
run_experiment "attn_spargeattn_0p5" \
  "${COMMON[@]}" \
  --attention-backend SPARGE_ATTN \
  --no-prefix-caching \
  --disable-mm-preprocessor-cache \
  --disable-chunked-mm-input

# 1d. XAttention (topk=0.5, stride=8) — block-level importance scoring
XATTN_THRESHOLD=0.5 XATTN_STRIDE=8 \
run_experiment "attn_xattention_0p5" \
  "${COMMON[@]}" \
  --attention-backend XATTN \
  --no-prefix-caching \
  --disable-mm-preprocessor-cache \
  --disable-chunked-mm-input

# ════════════════════════════════════════════════════════════════════════════════
# Section 2: Cache family (Triton)
# Fixed: Triton backend, no SD — add one cache at a time.
# ════════════════════════════════════════════════════════════════════════════════
print_section "CACHE: Triton"

# 2a. APC only
run_experiment "cache_apc_only" \
  "${COMMON[@]}" \
  --attention-backend TRITON_ATTN \
  --disable-mm-preprocessor-cache \
  --disable-chunked-mm-input

# 2b. MM preprocessor cache only
run_experiment "cache_mm_prep" \
  "${COMMON[@]}" \
  --attention-backend TRITON_ATTN \
  --no-prefix-caching \
  --disable-chunked-mm-input

# 2c. Chunked MM input only
run_experiment "cache_chunked_mm" \
  "${COMMON[@]}" \
  --attention-backend TRITON_ATTN \
  --no-prefix-caching \
  --disable-mm-preprocessor-cache

# 2d. APC + MM preprocessor cache
run_experiment "cache_apc_plus_mm" \
  "${COMMON[@]}" \
  --attention-backend TRITON_ATTN \
  --disable-chunked-mm-input

# 2e. All caches enabled
run_experiment "cache_all_caches" \
  "${COMMON[@]}" \
  --attention-backend TRITON_ATTN

# ════════════════════════════════════════════════════════════════════════════════
# Section 3: Cache family (FlashAttention extension)
# Adds the missing cache points under the default FlashAttention backend.
# ════════════════════════════════════════════════════════════════════════════════
print_section "CACHE: FlashAttention"

# 3a. FlashAttention + APC only
run_experiment "cache_flashattention_apc_only" \
  "${COMMON[@]}" \
  --disable-mm-preprocessor-cache \
  --disable-chunked-mm-input

# 3b. FlashAttention + MM preprocessor cache only
run_experiment "cache_flashattention_mm_prep" \
  "${COMMON[@]}" \
  --no-prefix-caching \
  --disable-chunked-mm-input

# 3c. FlashAttention + all caches is provided by combined_flashattention_plus_cache.
echo "  ↳ combined_flashattention_plus_cache provides the FlashAttention + all caches point" | tee -a "$SUMMARY_FILE"

# ════════════════════════════════════════════════════════════════════════════════
# Section 4: SD family (k sweep)
# Fixed: Triton + all caches off — isolates SD strategy across k values.
# ════════════════════════════════════════════════════════════════════════════════
print_section "SD: K Sweep"

SD_BASE=(
  "${COMMON[@]}"
  "${NO_CACHE[@]}"
)

# 4a. N-gram
for spec_k in $SPEC_TOKEN_SWEEP; do
  run_experiment "sd_ngram_k${spec_k}" \
    "${SD_BASE[@]}" \
    --ngram \
    --num-speculative-tokens "$spec_k"
done

# 4b. Draft model (merged 2B LoRA)
if [[ -d "$DRAFT_MODEL" ]]; then
  for spec_k in $SPEC_TOKEN_SWEEP; do
    run_experiment "sd_draft_2b_k${spec_k}" \
      "${SD_BASE[@]}" \
      --speculative-decoding \
      --draft-model "$DRAFT_MODEL" \
      --num-speculative-tokens "$spec_k"
  done
else
  echo "  ⚠  SKIP sd_draft_2b sweep: $DRAFT_MODEL not found" | tee -a "$SUMMARY_FILE"
fi

# 4c. EAGLE3
if [[ -d "$EAGLE3_MODEL" ]]; then
  for spec_k in $SPEC_TOKEN_SWEEP; do
    run_experiment "sd_eagle3_k${spec_k}" \
      "${SD_BASE[@]}" \
      --eagle3 "$EAGLE3_MODEL" \
      --num-speculative-tokens "$spec_k"
  done
else
  echo "  ⚠  SKIP sd_eagle3 sweep: $EAGLE3_MODEL not found" | tee -a "$SUMMARY_FILE"
fi

# ════════════════════════════════════════════════════════════════════════════════
# Section 5: Quant family
# Matches the table's quantization rows: AWQ and AWQ + FlashAttention.
# ════════════════════════════════════════════════════════════════════════════════
print_section "QUANT: Quantization"

AWQ_MODEL="${AWQ_MODEL:-/workspace/.hf_home/qwen3-vl-8b-awq-int4/}"

if [[ -d "$AWQ_MODEL" ]]; then
  # 5a. AWQ — no caches, Triton backend
  run_experiment "quant_awq" \
    --base-model "$AWQ_MODEL" \
    --adapter-path "$ADAPTER_PATH" \
    --dataset      "$DATASET" \
    --num-samples  "$NUM_SAMPLES" \
    --attention-backend TRITON_ATTN \
    --no-prefix-caching \
    --disable-mm-preprocessor-cache \
    --disable-chunked-mm-input

  # 5b. AWQ + FlashAttention — no caches
  run_experiment "quant_awq_plus_flashattention" \
    --base-model "$AWQ_MODEL" \
    --adapter-path "$ADAPTER_PATH" \
    --dataset      "$DATASET" \
    --num-samples  "$NUM_SAMPLES" \
    --no-prefix-caching \
    --disable-mm-preprocessor-cache \
    --disable-chunked-mm-input
else
  echo "  ⚠  SKIP quant_awq sweep: $AWQ_MODEL not found — run quantize_qwen3vl_awq.py first" | tee -a "$SUMMARY_FILE"
fi

# ════════════════════════════════════════════════════════════════════════════════
# Section 6: Combined family
# Matches the table's combined rows.
# ════════════════════════════════════════════════════════════════════════════════
print_section "COMBINED: Production Candidates"

# 6a. FlashAttention + cache
run_experiment "combined_flashattention_plus_cache" \
  "${COMMON[@]}"

# 6b. FlashInfer + cache
run_experiment "combined_flashinfer_plus_cache" \
  "${COMMON[@]}" \
  --attention-backend FLASHINFER

# 6c. Draft 2B + FlashAttention + cache
if [[ -d "$DRAFT_MODEL" ]]; then
  run_experiment "combined_draft2b_plus_flashattention_plus_cache" \
    "${COMMON[@]}" \
    --speculative-decoding \
    --draft-model "$DRAFT_MODEL" \
    --num-speculative-tokens "$NUM_SPEC_TOKENS"
else
  echo "  ⚠  SKIP combined_draft2b_plus_flashattention_plus_cache: $DRAFT_MODEL not found" | tee -a "$SUMMARY_FILE"
fi

# 6d. AWQ + FlashAttention + cache
if [[ -d "$AWQ_MODEL" ]]; then
  run_experiment "combined_awq_plus_flashattention_plus_cache" \
    --base-model   "$AWQ_MODEL" \
    --adapter-path "$ADAPTER_PATH" \
    --dataset      "$DATASET" \
    --num-samples  "$NUM_SAMPLES"
else
  echo "  ⚠  SKIP combined_awq_plus_flashattention_plus_cache: $AWQ_MODEL not found" | tee -a "$SUMMARY_FILE"
fi

# 6e. Eagle3 + FlashAttention + cache
if [[ -d "$EAGLE3_MODEL" ]]; then
  run_experiment "combined_eagle3_plus_flashattention_plus_cache" \
    "${COMMON[@]}" \
    --eagle3 "$EAGLE3_MODEL" \
    --num-speculative-tokens "$NUM_SPEC_TOKENS"
else
  echo "  ⚠  SKIP combined_eagle3_plus_flashattention_plus_cache: $EAGLE3_MODEL not found" | tee -a "$SUMMARY_FILE"
fi

# 6f. AWQ + Eagle3 + FlashAttention + cache
if [[ -d "$AWQ_MODEL" && -d "$EAGLE3_MODEL" ]]; then
  run_experiment "combined_awq_plus_eagle3_plus_flashattention_plus_cache" \
    --base-model   "$AWQ_MODEL" \
    --adapter-path "$ADAPTER_PATH" \
    --dataset      "$DATASET" \
    --num-samples  "$NUM_SAMPLES" \
    --eagle3 "$EAGLE3_MODEL" \
    --num-speculative-tokens "$NUM_SPEC_TOKENS"
else
  echo "  ⚠  SKIP combined_awq_plus_eagle3_plus_flashattention_plus_cache: AWQ and/or Eagle3 model not found" | tee -a "$SUMMARY_FILE"
fi
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Ablation summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cat "$SUMMARY_FILE"
echo ""
echo "  Logs and metrics: $RESULTS_DIR"

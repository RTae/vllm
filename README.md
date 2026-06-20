# FastDriveLM: Accelerating Long-Context Multimodal Inference with Quantization, Sparse Attention, and Speculative Decoding

This repository contains code for FastDriveLM, a collection of techniques to accelerate long-context multimodal inference on Qwen3-VL models. We demonstrate these optimizations on the DriveLM task, which involves generating driving decisions from 6-camera input sequences up to 16k tokens.

## Installation

```bash
uv pip install -e . --torch-backend=auto
uv pip install datasets transformers peft pytest
MAX_JOBS=4 uv pip install flash-attn --no-build-isolation
MAX_JOBS=4 uv pip install minference --no-build-isolation
MAX_JOBS=4 uv pip install --no-deps llmcompressor==0.10.0.2
MAX_JOBS=4 uv pip install --python .venv/bin/python --no-build-isolation -e ./block_sparse_attn -v
MAX_JOBS=4 uv pip install --python .venv/bin/python --no-build-isolation -e ./spas_sage_attn -v
```

## Download Models and Datasets

1. Download datasets:

```bash
./scripts/download_datasets.sh

python scripts/create_drivelm_nuscenes.py
```

2. Download finetuned LoRA adapter model from these links below

| Model | Link |
|---|---|
| Qwen3-VL-8B-Instruct LoRA adapter | [Google Drive](https://drive.google.com/file/d/1I9C5TxNvwl6JQ6nMlwESbN_s6XfIdWxn/view?usp=sharing) |
| Qwen3-VL-2B-Instruct LoRA adapter | [Google Drive](https://drive.google.com/file/d/1wf9eHxhtDiZPNVWutyzizeYvr6ygOk3Z/view?usp=sharing) |

3. Download base models (Qwen3-VL-8B and Qwen3-VL-2B) and EAGLE3 models from Hugging Face or use the provided download script:
```bash
./scripts/download_models.sh
```


## Quantization (AWQ INT4)

### Run AWQ INT4 quantization

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--adapter-path` | `outputs/qwen3vl` | PEFT LoRA adapter directory |
| `--base-model` | auto from adapter_config | Qwen3-VL base model path |
| `--data` | DriveLM val split | Calibration dataset (load_from_disk) |
| `--output-dir` | `outputs/qwen3vl_awq_int4` | Where to save the quantized model |
| `--num-calibration-samples` | 256 | Number of samples for AWQ calibration |
| `--max-seq-length` | 2048 | Max token length for calibration |
| `--num-images` | 6 | Images per calibration sample |
| `--scheme` | `W4A16_ASYM` | Quantization scheme (AWQ INT4 weight-only) |
| `--ignore` | `lm_head`, `visual.*` | Layers kept in full precision |

```bash
python scripts/quantize_qwen3vl_awq.py \
  --adapter-path /workspace/vllm/models/Qwen3-VL-8B-Instruct \
  --base-model /workspace/.hf_home/qwen3-vl-8b/ \
  --data datasets/DriveLM_nuScenes/split/val \
  --output-dir /workspace/.hf_home/qwen3-vl-8b-awq-int4/ \
  --num-calibration-samples 256
```
run with nohup:
```bash
nohup python scripts/quantize_qwen3vl_awq.py \
  --adapter-path /workspace/vllm/models/Qwen3-VL-8B-Instruct \
  --base-model /workspace/.hf_home/qwen3-vl-8b/ \
  --data datasets/DriveLM_nuScenes/split/val \
  --output-dir /workspace/.hf_home/qwen3-vl-8b-awq-int4/ \
  --num-calibration-samples 256 > quantization.log 2>&1 &
echo "PID: $!"
tail -f quantization.log
```

> **Note:** After quantization, copy the base model's tokenizer files to the quantized output
> directory (done automatically if you use the updated `quantize_qwen3vl_awq.py`):
> ```bash
> cp /workspace/.hf_home/qwen3-vl-8b/tokenizer*.json \
>    /workspace/.hf_home/qwen3-vl-8b/vocab.json \
>    /workspace/.hf_home/qwen3-vl-8b/merges.txt \
>    /workspace/.hf_home/qwen3-vl-8b/chat_template.json \
>    /workspace/.hf_home/qwen3-vl-8b-awq-int4/
> ```

### Inference with AWQ INT4 Quantized Model
```bash
python scripts/infer_drivelm_lora.py \
  --base-model /workspace/.hf_home/qwen3-vl-8b-awq-int4/ \
  --adapter-path /workspace/vllm/models/Qwen3-VL-8B-Instruct \
  --dataset /workspace/vllm/datasets/DriveLM_nuScenes/split/val \
  --num-samples 100 \
  --attention-backend TRITON_ATTN \
  --no-prefix-caching \
  --disable-mm-preprocessor-cache \
  --disable-chunked-mm-input \
  --max-num-batched-tokens 16384
```

## Inference

### Baseline
```bash
python scripts/infer_drivelm_lora.py \
  --adapter-path /workspace/vllm/models/Qwen3-VL-8B-Instruct \
  --base-model /workspace/.hf_home/qwen3-vl-8b/ \
  --dataset /workspace/vllm/datasets/DriveLM_nuScenes/split/val \
  --num-samples 100 \
  --attention-backend TRITON_ATTN \
  --no-prefix-caching \
  --disable-mm-preprocessor-cache \
  --disable-chunked-mm-input \
  --max-num-batched-tokens 16384
```

### Recommended default for DriveLM
DriveLM sequences are ~8421 tokens (6 cameras). The `--max-num-batched-tokens 16384`
flag avoids splitting each sequence into 2 prefill chunks, saving one full 28-layer
forward pass per request (**+12% throughput, -21% latency** vs default 8192):

```bash
python scripts/infer_drivelm_lora.py \
  --adapter-path /workspace/vllm/models/Qwen3-VL-8B-Instruct \
  --base-model /workspace/.hf_home/qwen3-vl-8b/ \
  --dataset /workspace/vllm/datasets/DriveLM_nuScenes/split/val \
  --num-samples 100 \
  --max-num-batched-tokens 16384
```

### Speculative decoding
#### With N-gram proposer
```bash
python scripts/infer_drivelm_lora.py \
  --adapter-path /workspace/vllm/models/Qwen3-VL-8B-Instruct \
  --base-model /workspace/.hf_home/qwen3-vl-8b/ \
  --dataset /workspace/vllm/datasets/DriveLM_nuScenes/split/val \
  --num-samples 100 \
  --ngram \
  --num-speculative-tokens 5 \
  --attention-backend TRITON_ATTN \
  --no-prefix-caching \
  --disable-mm-preprocessor-cache \
  --disable-chunked-mm-input \
  --max-num-batched-tokens 16384
```

#### With draft model
```bash
python scripts/infer_drivelm_lora.py \
  --adapter-path /workspace/vllm/models/Qwen3-VL-8B-Instruct \
  --base-model /workspace/.hf_home/qwen3-vl-8b/ \
  --dataset /workspace/vllm/datasets/DriveLM_nuScenes/split/val \
  --num-samples 4 \
  --speculative-decoding \
  --draft-model /workspace/.hf_home/qwen3-vl-2b-merged/ \
  --num-speculative-tokens 5 \
  --attention-backend TRITON_ATTN \
  --no-prefix-caching \
  --disable-mm-preprocessor-cache \
  --disable-chunked-mm-input \
  --max-num-batched-tokens 16384
```

#### With EAGLE proposer
```bash
python scripts/infer_drivelm_lora.py \
  --adapter-path /workspace/vllm/models/Qwen3-VL-8B-Instruct \
  --base-model /workspace/.hf_home/qwen3-vl-8b/ \
  --dataset /workspace/vllm/datasets/DriveLM_nuScenes/split/val \
  --num-samples 100 \
  --eagle3 /workspace/.hf_home/qwen3-vl-8b-eagle3 \
  --num-speculative-tokens 5 \
  --attention-backend TRITON_ATTN \
  --no-prefix-caching \
  --disable-mm-preprocessor-cache \
  --disable-chunked-mm-input \
  --max-num-batched-tokens 16384
```

### Attention Backends

#### FlashAttention (default)
```bash
python scripts/infer_drivelm_lora.py \
  --adapter-path /workspace/vllm/models/Qwen3-VL-8B-Instruct \
  --base-model /workspace/.hf_home/qwen3-vl-8b/ \
  --dataset /workspace/vllm/datasets/DriveLM_nuScenes/split/val \
  --num-samples 100 \
  --no-prefix-caching \
  --disable-mm-preprocessor-cache \
  --disable-chunked-mm-input \
  --max-num-batched-tokens 16384
```

#### XAttention (sparse prefill)
XAttention accelerates long-context prefill by estimating which 128-token KV blocks each query block attends to (cheap strided dot-product), then computing only those selected blocks with `block_sparse_attn`. Decode falls back to paged FlashAttention.

> **Note:** `VLLM_WORKER_MULTIPROC_METHOD=spawn` is required because the `block_sparse_attn` CUDA extension is loaded inside the worker process.

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn python scripts/infer_drivelm_lora.py \
  --adapter-path /workspace/vllm/models/Qwen3-VL-8B-Instruct \
  --base-model /workspace/.hf_home/qwen3-vl-8b/ \
  --dataset /workspace/vllm/datasets/DriveLM_nuScenes/split/val \
  --num-samples 100 \
  --attention-backend XATTN \
  --no-prefix-caching \
  --disable-mm-preprocessor-cache \
  --disable-chunked-mm-input \
  --max-num-batched-tokens 16384
```

Optional tuning via environment variables:

| Variable | Default | Effect |
|---|---|---|
| `XATTN_THRESHOLD` | `0.8` | Fraction of KV blocks to retain (lower = sparser/faster) |
| `XATTN_STRIDE` | `8` | Estimation stride (higher = cheaper estimate) |

```bash
XATTN_THRESHOLD=0.5 XATTN_STRIDE=8 \
  python scripts/infer_drivelm_lora.py \
  --adapter-path /workspace/vllm/models/Qwen3-VL-8B-Instruct \
  --base-model /workspace/.hf_home/qwen3-vl-8b/ \
  --dataset /workspace/vllm/datasets/DriveLM_nuScenes/split/val \
  --num-samples 100 \
  --attention-backend XATTN \
  --no-prefix-caching \
  --disable-mm-preprocessor-cache \
  --disable-chunked-mm-input \
  --max-num-batched-tokens 16384
```

### Caching

#### APC only
```bash
python scripts/infer_drivelm_lora.py \
  --adapter-path /workspace/vllm/models/Qwen3-VL-8B-Instruct \
  --base-model /workspace/.hf_home/qwen3-vl-8b/ \
  --dataset /workspace/vllm/datasets/DriveLM_nuScenes/split/val \
  --num-samples 100 \
  --attention-backend TRITON_ATTN \
  --disable-mm-preprocessor-cache \
  --disable-chunked-mm-input \
  --max-num-batched-tokens 16384
```

#### MM preprocessor cache only
```bash
python scripts/infer_drivelm_lora.py \
  --adapter-path /workspace/vllm/models/Qwen3-VL-8B-Instruct \
  --base-model /workspace/.hf_home/qwen3-vl-8b/ \
  --dataset /workspace/vllm/datasets/DriveLM_nuScenes/split/val \
  --num-samples 100 \
  --attention-backend TRITON_ATTN \
  --no-prefix-caching \
  --disable-chunked-mm-input \
  --max-num-batched-tokens 16384
```

#### All caches enabled
```bash
python scripts/infer_drivelm_lora.py \
  --adapter-path /workspace/vllm/models/Qwen3-VL-8B-Instruct \
  --base-model /workspace/.hf_home/qwen3-vl-8b/ \
  --dataset /workspace/vllm/datasets/DriveLM_nuScenes/split/val \
  --num-samples 100 \
  --attention-backend TRITON_ATTN \
  --max-num-batched-tokens 16384
```

### Apply all
Apply all optimizations together, including EAGLE proposer, FlashAttention backend, and all caches:
```bash
python scripts/infer_drivelm_lora.py \
  --adapter-path /workspace/vllm/models/Qwen3-VL-8B-Instruct \
  --base-model /workspace/.hf_home/qwen3-vl-8b/ \
  --dataset /workspace/vllm/datasets/DriveLM_nuScenes/split/val \
  --num-samples 100 \
  --eagle3 /workspace/.hf_home/qwen3-vl-8b-eagle3 \
  --num-speculative-tokens 5 \
  --max-num-batched-tokens 16384
```

## Ablation Study

```bash
NUM_SAMPLES=1000 ./scripts/ablation_test.sh

# run with nohup
nohup bash -c 'NUM_SAMPLES=1000 ./scripts/ablation_test.sh' > ablation_test.log 2>&1 &
tail -f ablation_test.log
```

lower samples for faster testing:

```bash
NUM_SAMPLES=10 ./scripts/ablation_test.sh
```

Each experiment automatically runs DriveLM quality evaluation after inference. Results are saved to `ablation_results/summary.txt` and individual `*_eval.json` files.
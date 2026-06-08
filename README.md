<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Installation

```bash
uv pip install -e . --torch-backend=auto
uv pip install datasets transformers peft pytest
MAX_JOBS=4 uv pip install flash-attn --no-build-isolation
MAX_JOBS=4 uv pip install minference --no-build-isolation
MAX_JOBS=4 uv pip install --no-deps llmcompressor==0.10.0.2
MAX_JOBS=4 uv pip install --python .venv/bin/python --no-build-isolation -e ./block_sparse_attn -v
MAX_JOBS=4 uv pip install --python .venv/bin/python --no-build-isolation -e ./spas_sage_attn -v\
```

## Download Models and Datasets
1. Download datasets:

```bash
./scripts/download_datasets.sh

python scripts/create_drivelm_nuscenes.py
```

2. Download models:

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
echo "PID: $!"
tail -f ablation_test.log
```

lower samples for faster testing:

```bash
NUM_SAMPLES=10 ./scripts/ablation_test.sh
```

Each experiment automatically runs DriveLM quality evaluation after inference. Results are saved to `ablation_results/summary.txt` and individual `*_eval.json` files.

### Results (1120 samples, ~4.5k tokens/request)

| Experiment | Throughput (samples/s) | TTFT mean (s) | Final Score |
|---|---|---|---|
| **00 Baseline** (Triton, no caches) | 0.998 | 376.6 | 0.6347 |
| **01a Flash** Attention | 1.152 | 320.9 | 0.6428 |
| **01b FlashInfer** | 1.208 | 304.2 | 0.6430 |
| **01c SpargeAttn** topk=0.5 | 0.992 | 373.4 | 0.6457 |
| **01d XAttention** topk=0.5 | 0.998 | 370.1 | 0.6375 |
| **02a APC** only | 2.172 | 225.3 | 0.6357 |
| **02b MM preprocessor cache** only | 1.313 | 339.9 | 0.6466 |
| **02c Chunked MM** only | 1.009 | 368.6 | 0.6346 |
| **02d APC + MM** cache | 4.465 | 85.7 | 0.6443 |
| **02e All caches** | 4.685 | 78.6 | 0.6387 |
| **03a N-gram** SD k=5 | 0.954 | 389.3 | 0.6421 |
| **03b Draft 2B** SD k=5 | 0.736 | 521.6 | 0.6438 |
| **03c EAGLE3** SD k=5 | 1.030 | 359.8 | 0.6425 |
| **04a Default** batched tokens (8192) | 1.162 | 322.2 | 0.6431 |
| **04b 16384** batched tokens | 1.235 | 299.4 | 0.6466 |
| **04c 16384 + APC** | 5.772 | 60.3 | 0.6387 |
| **05a All caches** batch | 5.634 | 62.4 | **0.6500** |
| **05b Sequential scenes** | 1.577 | 0.94 | 0.6448 |
| **06a SpargeAttn** + 16384 | 1.053 | 348.3 | 0.6405 |
| **06b XAttention** + 16384 | 1.046 | 350.6 | 0.6396 |
| **07a Eagle3 + Flash + caches** | 4.527 | 84.3 | 0.6402 |
| **07b Eagle3 + Flash + caches + 16384** | 5.304 | 69.7 | 0.6490 |
| **08a AWQ INT4** no caches | 0.957 | 392.8 | 0.5909 |
| **08b AWQ INT4** all caches | 5.509 | 63.7 | 0.5870 |
| **09 AWQ + Eagle3 + Flash + caches + 16384** | 4.733 | 80.7 | 0.5899 |

**Key findings:**
- Best throughput: `04c` (16384 + APC): **5.77 samples/s** (+478% vs baseline)
- Best quality: `05a` (all caches, batch): final score **0.6500**
- Best throughput+quality: `07b` (Eagle3 + Flash + all caches + 16384): 5.3 samples/s, score 0.6490
- AWQ INT4 reduces quality (~0.59 final score) but enables the same throughput as FP16+caches
- APC is by far the dominant optimization (2–5× throughput, negligible quality loss)
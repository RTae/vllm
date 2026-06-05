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
uv pip install datasets transformers
uv pip install --python .venv/bin/python -e ./spas_sage_attn
uv pip install --python .venv/bin/python --no-build-isolation -e ./spas_sage_attn -v
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

## Inference

### Baseline
```bash
python scripts/infer_drivelm_lora.py \
  --adapter-path /workspace/vllm/models/Qwen3-VL-8B-Instruct \
  --base-model /workspace/.hf_home/qwen3-vl-8b/ \
  --dataset /workspace/vllm/datasets/DriveLM_nuScenes/split/val \
  --num-samples 100 \
  --attention-backend TRITON_ATTN
```

```txt
AGGREGATE METRICS
  samples          : 100
  wall time        : 45.83s
  throughput       : 2.182 samples/s
  e2e latency mean : 20.4989s
  e2e latency p50  : 20.9438s
  e2e latency p95  : 32.814s
  tokens/s mean    : 1.12
  TTFT mean        : 14.1056s
  prefill time mean: 2.6589s
  prefill time p50 : 1.7793s
  decode time mean : 6.3933s
  decode time p50  : 2.5702s
  total gen tokens : 2991
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
  --attention-backend TRITON_ATTN
```

```txt
AGGREGATE METRICS
  samples          : 100
  wall time        : 54.56s
  throughput       : 1.833 samples/s
  e2e latency mean : 25.8072s
  e2e latency p50  : 27.2647s
  e2e latency p95  : 38.9524s
  tokens/s mean    : 0.9
  TTFT mean        : 18.0023s
  prefill time mean: 2.8864s
  prefill time p50 : 1.6377s
  decode time mean : 7.805s
  decode time p50  : 3.3781s
  total gen tokens : 2997
```

#### With draft model
```bash
python scripts/infer_drivelm_lora.py \
  --adapter-path /workspace/vllm/models/Qwen3-VL-8B-Instruct \
  --base-model /workspace/.hf_home/qwen3-vl-8b/ \
  --dataset /workspace/vllm/datasets/DriveLM_nuScenes/split/val \
  --num-samples 100 \
  --speculative-decoding \
  --draft-model /workspace/.hf_home/qwen3-vl-2b/ \
  --num-speculative-tokens 5 \
  --attention-backend TRITON_ATTN
```

```txt
AGGREGATE METRICS
  samples          : 100
  wall time        : 57.49s
  throughput       : 1.739 samples/s
  e2e latency mean : 20.9826s
  e2e latency p50  : 21.5327s
  e2e latency p95  : 35.6791s
  tokens/s mean    : 1.07
  TTFT mean        : 16.472s
  prefill time mean: 1.632s
  prefill time p50 : 1.1161s
  decode time mean : 4.5107s
  decode time p50  : 1.7117s
  total gen tokens : 2925
```

#### With EAGLE proposer
```bash
python /workspace/vllm/scripts/infer_drivelm_lora.py \
  --adapter-path /workspace/vllm/models/Qwen3-VL-8B-Instruct \
  --base-model /workspace/.hf_home/qwen3-vl-8b/ \
  --dataset /workspace/vllm/datasets/DriveLM_nuScenes/split/val \
  --num-samples 100 \
  --eagle3 /workspace/.hf_home/qwen3-vl-8b-eagle3 \
  --num-speculative-tokens 5 \
  --attention-backend TRITON_ATTN
```

```txt
AGGREGATE METRICS
  samples          : 100
  wall time        : 65.06s
  throughput       : 1.537 samples/s
  e2e latency mean : 34.0451s
  e2e latency p50  : 38.5903s
  e2e latency p95  : 49.6095s
  tokens/s mean    : 0.71
  TTFT mean        : 27.797s
  prefill time mean: 13.9762s
  prefill time p50 : 15.2291s
  decode time mean : 6.2481s
  decode time p50  : 2.7681s
  total gen tokens : 2964
```

### Attention Backends

#### FlashAttention (default)
```bash
python scripts/infer_drivelm_lora.py \
  --adapter-path /workspace/vllm/models/Qwen3-VL-8B-Instruct \
  --base-model /workspace/.hf_home/qwen3-vl-8b/ \
  --dataset /workspace/vllm/datasets/DriveLM_nuScenes/split/val \
  --num-samples 100
```

```txt
AGGREGATE METRICS
  samples          : 100
  wall time        : 37.52s
  throughput       : 2.665 samples/s
  e2e latency mean : 15.3654s
  e2e latency p50  : 15.2807s
  e2e latency p95  : 25.5983s
  tokens/s mean    : 1.47
  TTFT mean        : 10.1351s
  prefill time mean: 2.1091s
  prefill time p50 : 1.6195s
  decode time mean : 5.2303s
  decode time p50  : 2.2644s
  total gen tokens : 2989
```

<!-- #### SpargeAttention (sparse prefill)
```bash
python scripts/infer_drivelm_lora.py \
  --adapter-path /workspace/vllm/models/Qwen3-VL-8B-Instruct \
  --base-model /workspace/.hf_home/qwen3-vl-8b/ \
  --dataset /workspace/vllm/datasets/DriveLM_nuScenes/split/val \
  --num-samples 100 \
  --sparge-attn \
  --sparge-topk 0.5
``` -->

### Apply flash attention and speculative decoding (EAGLE proposer)
```bash
python scripts/infer_drivelm_lora.py \
  --adapter-path /workspace/vllm/models/Qwen3-VL-8B-Instruct \
  --base-model /workspace/.hf_home/qwen3-vl-8b/ \
  --dataset /workspace/vllm/datasets/DriveLM_nuScenes/split/val \
  --num-samples 100 \
  --eagle3 /workspace/.hf_home/qwen3-vl-8b-eagle3 \
  --num-speculative-tokens 5
```

## Ablation Study of Speculative Decoding

```bash
NUM_SAMPLES=100 ./scripts/ablation_test.sh

# run with nohup
nohup NUM_SAMPLES=100 ./scripts/ablation_test.sh > ablation_test.log 2>&1 &
tail -f ablation_test.log
```

lower samples for faster testing:

```bash
NUM_SAMPLES=10 ./scripts/ablation_test.sh
```
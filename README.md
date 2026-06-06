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
uv pip install datasets transformers peft
MAX_JOBS=4 uv pip install flash-attn --no-build-isolation
MAX_JOBS=4 uv pip install minference --no-build-isolation
MAX_JOBS=4 uv pip install --python .venv/bin/python --no-build-isolation -e ./block_sparse_attn -v
MAX_JOBS=4 uv pip install --python .venv/bin/python --no-build-isolation -e ./spas_sage_attn -v
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
  --attention-backend TRITON_ATTN \
  --no-prefix-caching \
  --disable-mm-preprocessor-cache \
  --disable-chunked-mm-input
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
  --disable-chunked-mm-input
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
  --disable-chunked-mm-input
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
  --disable-chunked-mm-input
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
  --disable-chunked-mm-input
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
  --disable-chunked-mm-input
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
  --disable-chunked-mm-input
```

#### All caches enabled
```bash
python scripts/infer_drivelm_lora.py \
  --adapter-path /workspace/vllm/models/Qwen3-VL-8B-Instruct \
  --base-model /workspace/.hf_home/qwen3-vl-8b/ \
  --dataset /workspace/vllm/datasets/DriveLM_nuScenes/split/val \
  --num-samples 100 \
  --attention-backend TRITON_ATTN
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
  --num-speculative-tokens 5
```

## Ablation Study of Speculative Decoding

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
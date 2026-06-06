import argparse
import json
import os
from pathlib import Path
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NUM_SPECULATIVE_TOKENS = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run offline inference on a DriveLM-style dataset with a local LoRA adapter.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        required=True,
        help="Path to the local PEFT LoRA adapter directory.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to a Hugging Face dataset directory or a JSON file.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help=(
            "Base model path or Hugging Face model ID. If omitted, the script "
            "uses adapter_config.json's base_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--adapter-name",
        type=str,
        default="drivelm-lora",
        help="Logical LoRA name used in the vLLM request.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional split name when loading a dataset saved with load_from_disk.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Dataset row index to start from.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="Number of samples to run. Use -1 (default) to run all samples in the dataset.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="Maximum sequence length passed to vLLM.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=28 * 28,
        help="Minimum number of pixels used by the multimodal processor.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1280 * 28 * 28,
        help="Maximum number of pixels used by the multimodal processor.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size passed to vLLM.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code for base model loading.",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default=None,
        metavar="BACKEND",
        help=(
            "Override the attention backend. Options: FLASH_ATTN (default), "
            "FLASHINFER, TRITON_ATTN, XATTN (sparse prefill via XAttention). "
            "XATTN and SPARGE_ATTN automatically enable eager mode."
        ),
    )
    parser.add_argument(
        "--no-prefix-caching",
        action="store_true",
        help=(
            "Disable Automatic Prefix Caching (APC). "
            "APC is enabled by default and reuses KV cache for shared prompt prefixes. "
            "Disable to benchmark without APC or when prompts share no prefix."
        ),
    )
    parser.add_argument(
        "--disable-mm-preprocessor-cache",
        action="store_true",
        help=(
            "Disable the multimodal preprocessor cache (sets mm_processor_cache_gb=0). "
            "Enabled by default; disabling forces images to be re-preprocessed each request."
        ),
    )
    parser.add_argument(
        "--disable-chunked-mm-input",
        action="store_true",
        help=(
            "Disable chunked multimodal input (disable_chunked_mm_input=True). "
            "Enabled by default; disabling prevents splitting large mm inputs across chunks."
        ),
    )
    parser.add_argument(
        "--speculative-decoding",
        action="store_true",
        help="Enable speculative decoding using a smaller draft model.",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default=None,
        metavar="JSON_PATH",
        help="Path to save per-request and aggregate metrics as JSON.",
    )
    parser.add_argument(
        "--ngram",
        action="store_true",
        help=(
            "Enable n-gram speculative decoding. No draft model needed. "
            "Uses repeated n-grams from the prompt/context to propose tokens."
        ),
    )
    parser.add_argument(
        "--ngram-prompt-lookup-max",
        type=int,
        default=5,
        help="Maximum n-gram window size for n-gram speculative decoding.",
    )
    parser.add_argument(
        "--ngram-prompt-lookup-min",
        type=int,
        default=1,
        help="Minimum n-gram window size for n-gram speculative decoding.",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default=None,
        help=(
            "Local path to the draft model used for speculative decoding. If "
            "omitted, the script tries to infer a 2B draft model from an 8B target model path."
        ),
    )
    parser.add_argument(
        "--eagle3",
        type=str,
        default=None,
        metavar="EAGLE3_MODEL_PATH",
        help=(
            "Path to an Eagle3 speculative draft head (e.g. "
            "/workspace/.hf_home/qwen3-vl-2b-eagle3). Enables Eagle3 speculative "
            "decoding which achieves much higher acceptance rates than a draft model "
            "because it conditions on the target model hidden states."
        ),
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=DEFAULT_NUM_SPECULATIVE_TOKENS,
        help="Number of speculative tokens proposed by the draft model per step.",
    )
    parser.add_argument(
        "--draft-tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for the speculative draft model.",
    )
    return parser.parse_args()


def load_adapter_config(adapter_path: Path) -> dict:
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing adapter config: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_base_model(adapter_config: dict, override: str | None) -> str:
    base_model = override or adapter_config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(
            "Base model could not be determined. Pass --base-model explicitly."
        )

    base_model_path = Path(base_model).expanduser()
    if base_model_path.exists():
        return str(base_model_path.resolve())

    return base_model


def infer_default_draft_model(base_model: str) -> str | None:
    local_candidates = [
        Path("/workspace/.hf_home/qwen3-vl-2b"),
        Path("/workspace/.hf_home/qwen3-vl-2b/"),
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return str(candidate.resolve())

    snapshot_root = Path(
        "/workspace/.hf_home/hub/models--qwen--Qwen3-VL-2B-Instruct/snapshots"
    )
    if snapshot_root.exists():
        snapshot_dirs = sorted(path for path in snapshot_root.iterdir() if path.is_dir())
        if snapshot_dirs:
            return str(snapshot_dirs[-1].resolve())

    replacements = (
        ("Qwen3-VL-8B-Instruct", "Qwen3-VL-2B-Instruct"),
        ("qwen3-vl-8b", "qwen3-vl-2b"),
        ("Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3-VL-2B-Instruct"),
    )

    for source, target in replacements:
        if source not in base_model:
            continue

        candidate = base_model.replace(source, target)
        candidate_path = Path(candidate).expanduser()
        if candidate_path.exists():
            return str(candidate_path.resolve())
        return candidate

    return None


def resolve_draft_model(base_model: str, override: str | None) -> str:
    draft_model = override or infer_default_draft_model(base_model)
    if not draft_model:
        raise ValueError(
            "Draft model could not be inferred from the target model. Pass --draft-model explicitly."
        )

    draft_model_path = Path(draft_model).expanduser()
    if draft_model_path.exists():
        return str(draft_model_path.resolve())

    raise FileNotFoundError(
        "Draft model was not found locally. Download it first or pass a valid local path with --draft-model."
    )


def load_samples(dataset_path: Path, split: str | None):
    if dataset_path.is_dir():
        from datasets import load_from_disk

        dataset = load_from_disk(str(dataset_path))
        if split is not None:
            return dataset[split]
        return dataset

    if dataset_path.suffix.lower() == ".json":
        with dataset_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    raise ValueError(
        "Dataset must be a load_from_disk directory or a JSON file."
    )


def get_sample_value(sample: dict, *keys, default=None):
    for key in keys:
        if key in sample:
            return sample[key]
    return default


def resolve_image_path(image_path: str, dataset_path: Path) -> Path:
    candidate = Path(image_path)
    if candidate.is_absolute():
        return candidate

    repo_relative = REPO_ROOT / candidate
    if repo_relative.exists():
        return repo_relative

    dataset_relative = dataset_path.parent / candidate
    return dataset_relative


def load_images(sample: dict, dataset_path: Path) -> list[Image.Image]:
    from PIL import Image


    image_paths = get_sample_value(sample, "image_paths", "image")
    if image_paths is None:
        raise KeyError("Sample is missing 'image_paths' or 'image'.")

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    images = []
    for image_path in image_paths:
        resolved_path = resolve_image_path(image_path, dataset_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Image not found: {resolved_path}")
        with Image.open(resolved_path) as image:
            images.append(image.convert("RGB"))
    return images


def extract_question_and_reference(sample: dict) -> tuple[str, str | None]:
    conversations = sample.get("conversations")
    if not conversations:
        raise KeyError("Sample is missing 'conversations'.")

    question = None
    answer = None
    for turn in conversations:
        if turn.get("from") == "human" and question is None:
            question = turn.get("value")
        if turn.get("from") == "gpt" and answer is None:
            answer = turn.get("value")

    if not question:
        raise ValueError("Could not find a human prompt in 'conversations'.")

    return question, answer


def validate_speculative_decoding(args, samples: list[dict], base_model: str) -> None:
    pass


def build_prompt(processor, question: str, num_images: int) -> str:
    placeholders = [{"type": "image"} for _ in range(num_images)]
    messages = [
        {"role": "system", "content": "You are a helpful driving assistant."},
        {
            "role": "user",
            "content": [*placeholders, {"type": "text", "text": question}],
        },
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    args = parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # vLLM forks EngineCore by default; CUDA-linked extensions loaded in the main
    # process conflict with fork. Spawn avoids re-initialising CUDA in the child.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    adapter_config = load_adapter_config(args.adapter_path)
    base_model = resolve_base_model(adapter_config, args.base_model)
    max_lora_rank = adapter_config.get("r", 64)
    speculative_config = None

    if args.speculative_decoding and args.ngram:
        raise ValueError("--speculative-decoding and --ngram are mutually exclusive.")

    if args.eagle3 and (args.speculative_decoding or args.ngram):
        raise ValueError("--eagle3 is mutually exclusive with --speculative-decoding and --ngram.")

    if args.eagle3:
        eagle3_path = str(Path(args.eagle3).expanduser().resolve())
        if not Path(eagle3_path).exists():
            raise FileNotFoundError(
                f"Eagle3 model not found: {eagle3_path}. Download it with:\n"
                "  HF_ENDPOINT=https://hf-mirror.com hf download taobao-mnn/Qwen3-VL-2B-Instruct-Eagle3 "
                "--local-dir /workspace/.hf_home/qwen3-vl-2b-eagle3"
            )
        speculative_config = {
            "method": "eagle3",
            "model": eagle3_path,
            "num_speculative_tokens": args.num_speculative_tokens,
            "draft_tensor_parallel_size": args.draft_tensor_parallel_size,
        }
    elif args.speculative_decoding:
        draft_model = resolve_draft_model(base_model, args.draft_model)
        speculative_config = {
            "method": "draft_model",
            "model": draft_model,
            "num_speculative_tokens": args.num_speculative_tokens,
            "draft_tensor_parallel_size": args.draft_tensor_parallel_size,
            "max_model_len": args.max_model_len,
        }
    elif args.ngram:
        speculative_config = {
            "method": "ngram",
            "num_speculative_tokens": args.num_speculative_tokens,
            "prompt_lookup_max": args.ngram_prompt_lookup_max,
            "prompt_lookup_min": args.ngram_prompt_lookup_min,
        }

    try:
        processor = AutoProcessor.from_pretrained(
            base_model,
            trust_remote_code=args.trust_remote_code,
            local_files_only=True,
        )
    except OSError as error:
        raise RuntimeError(
            "Base model was not found locally. Download it first or pass a local "
            "snapshot path with --base-model."
        ) from error

    dataset = load_samples(args.dataset, args.split)
    if args.num_samples == -1:
        end_index = len(dataset)
    else:
        end_index = min(len(dataset), args.start_index + args.num_samples)
    if args.start_index >= len(dataset):
        raise IndexError(
            f"start-index {args.start_index} is out of range for dataset of size {len(dataset)}"
        )

    samples = [dataset[index] for index in range(args.start_index, end_index)]
    validate_speculative_decoding(args, samples, base_model)

    max_images = max(
        len(get_sample_value(sample, "image_paths", "image", default=[])) or 1
        for sample in samples
    )

    llm = LLM(
        model=base_model,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        mm_processor_kwargs={
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        },
        limit_mm_per_prompt={"image": max_images},
        trust_remote_code=args.trust_remote_code,
        speculative_config=speculative_config,
        attention_backend=args.attention_backend,
        # XATTN and SPARGE_ATTN use Python-level per-sequence iteration and
        # are incompatible with torch.compile/CUDA graphs.
        enforce_eager=args.attention_backend in ("XATTN", "SPARGE_ATTN"),
        # Enable per-request metrics so output.metrics is populated.
        disable_log_stats=False,
        # APC is on by default; --no-prefix-caching turns it off.
        enable_prefix_caching=not args.no_prefix_caching,
        # Multimodal preprocessor cache: disabled → re-preprocess every request.
        mm_processor_cache_gb=0 if args.disable_mm_preprocessor_cache else None,
        # Chunked mm input: default on; --disable-chunked-mm-input turns it off.
        disable_chunked_mm_input=args.disable_chunked_mm_input,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    lora_request = LoRARequest(args.adapter_name, 1, str(args.adapter_path))

    inputs = []
    metadata = []
    for sample in samples:
        images = load_images(sample, args.dataset)
        question, reference = extract_question_and_reference(sample)
        prompt = build_prompt(processor, question, len(images))
        inputs.append(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": images},
            }
        )
        metadata.append(
            {
                "id": sample.get("id"),
                "question": question,
                "reference": reference,
                "num_images": len(images),
            }
        )

    import time
    t_generate_start = time.perf_counter()
    outputs = llm.generate(
        inputs,
        sampling_params=sampling_params,
        lora_request=[lora_request] * len(inputs),
    )
    t_generate_end = time.perf_counter()
    total_wall_time = t_generate_end - t_generate_start

    # ── Collect metrics ───────────────────────────────────────────────────────
    records = []
    for meta, output in zip(metadata, outputs, strict=True):
        m = output.metrics  # RequestStateStats or None
        gen_tokens = len(output.outputs[0].token_ids)

        # Timestamps in RequestStateStats:
        #   arrival_time    – wall-clock (frontend)
        #   first_token_ts  – monotonic (engine core)
        #   last_token_ts   – monotonic (engine core)
        #   first_token_latency – already computed (wall-clock seconds)
        #
        # Do NOT mix clocks. Compute e2e as:
        #   first_token_latency + (last_token_ts - first_token_ts)
        e2e = None
        decode_time = None
        prefill_time = None
        ttft = round(m.first_token_latency, 4) if m and m.first_token_latency else None
        if m and m.first_token_ts and m.last_token_ts and ttft is not None:
            decode_time = round(m.last_token_ts - m.first_token_ts, 4)
            e2e = round(ttft + decode_time, 4)
        if m and m.scheduled_ts and m.first_token_ts:
            prefill_time = round(m.first_token_ts - m.scheduled_ts, 4)
        gpu_work = round(prefill_time + decode_time, 4) if prefill_time is not None and decode_time is not None else None
        # tokens/s per GPU work (excludes queue wait) — the true per-request GPU throughput
        tps = round(gen_tokens / gpu_work, 2) if gpu_work and gpu_work > 0 else None
        # TPOT: time per output token (decode phase only, excluding first token)
        # Standard LLM inference metric equivalent to mean ITL
        tpot = round(decode_time / (gen_tokens - 1), 4) if decode_time and gen_tokens > 1 else None

        record = {
            "id": meta["id"],
            "num_images": meta["num_images"],
            "num_prompt_tokens": None,
            "num_generation_tokens": gen_tokens,
            "e2e_latency_s": e2e,
            "first_token_latency_s": ttft,
            "request_compute_time_s": gpu_work,
            "decode_time_s": decode_time,
            "tpot_s": tpot,
            "prefill_time_s": prefill_time,
            "tokens_per_second": tps,
        }
        records.append({**record, "question": meta["question"], "reference": meta["reference"],
                        "prediction": output.outputs[0].text.strip()})

    # ── Aggregate ─────────────────────────────────────────────────────────────
    valid = [r for r in records if r["e2e_latency_s"] is not None]
    agg = {}
    if valid:
        import statistics
        latencies = [r["e2e_latency_s"] for r in valid]
        tps_list = [r["tokens_per_second"] for r in valid if r["tokens_per_second"] is not None]
        ttfts = [r["first_token_latency_s"] for r in valid if r["first_token_latency_s"] is not None]
        decode_times = [r["decode_time_s"] for r in valid if r["decode_time_s"] is not None]
        tpots = [r["tpot_s"] for r in valid if r["tpot_s"] is not None]
        prefill_times = [r["prefill_time_s"] for r in valid if r["prefill_time_s"] is not None]
        gpu_works = [r["request_compute_time_s"] for r in valid if r["request_compute_time_s"] is not None]
        agg = {
            "num_samples": len(valid),
            "total_wall_time_s": round(total_wall_time, 2),
            "throughput_samples_per_s": round(len(valid) / total_wall_time, 3),
            "e2e_latency_mean_s": round(statistics.mean(latencies), 4),
            "e2e_latency_p50_s": round(statistics.median(latencies), 4),
            "e2e_latency_p95_s": round(sorted(latencies)[min(int(len(latencies) * 0.95), len(latencies) - 1)], 4) if latencies else None,
            "request_compute_time_mean_s": round(statistics.mean(gpu_works), 4) if gpu_works else None,
            "request_compute_time_p50_s": round(statistics.median(gpu_works), 4) if gpu_works else None,
            "decode_time_mean_s": round(statistics.mean(decode_times), 4) if decode_times else None,
            "decode_time_p50_s": round(statistics.median(decode_times), 4) if decode_times else None,
            "tpot_mean_s": round(statistics.mean(tpots), 4) if tpots else None,
            "tpot_p50_s": round(statistics.median(tpots), 4) if tpots else None,
            "prefill_time_mean_s": round(statistics.mean(prefill_times), 4) if prefill_times else None,
            "prefill_time_p50_s": round(statistics.median(prefill_times), 4) if prefill_times else None,
            "tokens_per_second_mean": round(statistics.mean(tps_list), 2) if tps_list else None,
            "ttft_mean_s": round(statistics.mean(ttfts), 4) if ttfts else None,
            "total_gen_tokens": sum(r["num_generation_tokens"] for r in valid),
        }

    # ── Print per-sample outputs and summary ─────────────────────────────────
    for rec in records:
        print("=" * 80)
        print(f"id: {rec['id']}")
        print(f"images: {rec['num_images']}")
        print(f"question: {rec['question']}")
        if rec["reference"] is not None:
            print(f"reference: {rec['reference']}")
        print(f"prediction: {rec['prediction']}")
        if rec["e2e_latency_s"] is not None:
            print(f"[metrics] e2e={rec['e2e_latency_s']}s  ttft={rec['first_token_latency_s']}s  tpot={rec['tpot_s']}s  decode={rec['decode_time_s']}s  gen_tokens={rec['num_generation_tokens']}  tps={rec['tokens_per_second']}")

    if agg:
        print("\n" + "=" * 80)
        print("AGGREGATE METRICS")
        print(f"  samples          : {agg['num_samples']}")
        print(f"  wall time        : {agg['total_wall_time_s']}s")
        print(f"  throughput       : {agg['throughput_samples_per_s']} samples/s")
        print(f"  e2e latency mean : {agg['e2e_latency_mean_s']}s")
        print(f"  e2e latency p50  : {agg['e2e_latency_p50_s']}s")
        print(f"  e2e latency p95  : {agg['e2e_latency_p95_s']}s")
        if agg.get('request_compute_time_mean_s'):
            print(f"  compute time mean: {agg['request_compute_time_mean_s']}s  (prefill + decode, no queue wait)")
        if agg.get('request_compute_time_p50_s'):
            print(f"  compute time p50 : {agg['request_compute_time_p50_s']}s")
        if agg['tokens_per_second_mean']:
            print(f"  tokens/s mean    : {agg['tokens_per_second_mean']}")
        if agg['ttft_mean_s']:
            print(f"  TTFT mean        : {agg['ttft_mean_s']}s")
        if agg.get('prefill_time_mean_s'):
            print(f"  prefill time mean: {agg['prefill_time_mean_s']}s")
        if agg.get('prefill_time_p50_s'):
            print(f"  prefill time p50 : {agg['prefill_time_p50_s']}s")
        if agg.get('decode_time_mean_s'):
            print(f"  decode time mean : {agg['decode_time_mean_s']}s")
        if agg.get('decode_time_p50_s'):
            print(f"  decode time p50  : {agg['decode_time_p50_s']}s")
        if agg.get('tpot_mean_s'):
            print(f"  TPOT mean        : {agg['tpot_mean_s']}s  (decode/(tokens-1), ≈ ITL mean)")
        if agg.get('tpot_p50_s'):
            print(f"  TPOT p50         : {agg['tpot_p50_s']}s")
        print(f"  total gen tokens : {agg['total_gen_tokens']}")

    # ── Save to JSON if requested ─────────────────────────────────────────────
    if args.output_metrics:
        import json as _json
        out_path = Path(args.output_metrics)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"aggregate": agg, "per_request": records}
        with out_path.open("w", encoding="utf-8") as f:
            _json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nMetrics saved to {out_path}")


if __name__ == "__main__":
    main()
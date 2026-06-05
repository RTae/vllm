from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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
        "--sparge-attn",
        action="store_true",
        help=(
            "Use SpargeAttention for prefill (sparse) and FlashAttention for "
            "decode. Requires spas_sage_attn to be installed."
        ),
    )
    parser.add_argument(
        "--sparge-topk",
        type=float,
        default=0.5,
        help=(
            "Fraction of KV blocks retained per head during SpargeAttention "
            "prefill. 1.0 = dense, 0.5 = keep 50%% of blocks (default)."
        ),
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default=None,
        metavar="BACKEND",
        help=(
            "Override the attention backend. Options: FLASH_ATTN (default), "
            "FLASHINFER (no FlashAttention), TRITON_ATTN (no FlashAttention, Triton only). "
            "Ignored when --sparge-attn is set."
        ),
    )
    parser.add_argument(
        "--speculative-decoding",
        action="store_true",
        help="Enable speculative decoding using a smaller draft model.",
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

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    adapter_config = load_adapter_config(args.adapter_path)
    base_model = resolve_base_model(adapter_config, args.base_model)
    max_lora_rank = adapter_config.get("r", 64)
    speculative_config = None

    if args.sparge_attn:
        os.environ["SPARGE_ATTN_TOPK"] = str(args.sparge_topk)

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
        attention_backend=(
            "SPARGE_ATTN" if args.sparge_attn else args.attention_backend
        ),
        # SpargeAttention uses Python-level per-sequence iteration in its
        # forward pass and is incompatible with torch.compile/CUDA graphs.
        enforce_eager=args.sparge_attn or False,
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

    outputs = llm.generate(
        inputs,
        sampling_params=sampling_params,
        lora_request=[lora_request] * len(inputs),
    )

    for meta, output in zip(metadata, outputs, strict=True):
        print("=" * 80)
        print(f"id: {meta['id']}")
        print(f"images: {meta['num_images']}")
        print(f"question: {meta['question']}")
        if meta["reference"] is not None:
            print(f"reference: {meta['reference']}")
        print(f"prediction: {output.outputs[0].text.strip()}")


if __name__ == "__main__":
    main()
"""
Merge a PEFT LoRA adapter into a base model and save the result as a
standalone Hugging Face model directory.

Usage:
    python scripts/merge_lora.py \
        --base-model /workspace/.hf_home/qwen3-vl-2b/ \
        --adapter   /workspace/vllm/models/Qwen3-VL-2B-Instruct \
        --output    /workspace/.hf_home/qwen3-vl-2b-merged/
"""

import argparse
import shutil
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge a PEFT LoRA adapter into a base model.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path to the base model directory.",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to the PEFT LoRA adapter directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the merged model.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Dtype to use when loading and saving (default: bfloat16).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from peft import PeftModel
    from transformers import AutoProcessor, AutoModelForImageTextToText

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model from {args.base_model} ...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from {args.adapter} ...")
    peft_model = PeftModel.from_pretrained(base_model, args.adapter)

    print("Merging LoRA weights into the base model ...")
    merged = peft_model.merge_and_unload()

    print(f"Saving merged model to {output_path} ...")
    merged.save_pretrained(str(output_path), safe_serialization=True)

    print("Copying processor / tokenizer files ...")
    processor = AutoProcessor.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )
    processor.save_pretrained(str(output_path))

    # Also copy any extra config files that AutoProcessor might miss
    for fname in ("generation_config.json", "chat_template.json",
                  "preprocessor_config.json", "video_preprocessor_config.json"):
        src = Path(args.base_model) / fname
        if src.exists():
            shutil.copy2(src, output_path / fname)

    print(f"\nDone. Merged model saved to: {output_path}")
    print("You can now use it as a draft model with --draft-model")


if __name__ == "__main__":
    main()

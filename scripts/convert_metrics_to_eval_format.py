"""
Convert infer_drivelm_lora.py JSON output to the format expected by evaluation.py.

evaluation.py expects:
    List[{"id": "scene_id_frame_id_i", "answer": "..."}]

infer_drivelm_lora.py outputs:
    {"aggregate": {...}, "per_request": [{"id": "...", "prediction": "...", ...}]}

Usage:
    python scripts/convert_metrics_to_eval_format.py \
        --src ablation_results/00_baseline_metrics.json \
        --dst ablation_results/00_baseline_predictions.json
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert vLLM inference metrics JSON to evaluation.py format."
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Input metrics JSON from infer_drivelm_lora.py (--output-metrics).",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=None,
        help="Output predictions JSON for evaluation.py. Defaults to <src stem>_predictions.json.",
    )
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    dst = Path(args.dst) if args.dst else src.parent / (src.stem.replace("_metrics", "") + "_predictions.json")

    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)

    per_request = data.get("per_request", [])
    if not per_request:
        raise ValueError(f"No per_request records found in {src}")

    # Convert: rename 'prediction' → 'answer', keep 'id'
    predictions = [
        {"id": rec["id"], "answer": rec.get("prediction", "")}
        for rec in per_request
    ]

    with dst.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(predictions)} records: {src} → {dst}")


if __name__ == "__main__":
    main()

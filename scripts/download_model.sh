#!/bin/bash

# Check HF_TOKEN environment variable
if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN environment variable is not set."
  echo "Please set HF_TOKEN to your Hugging Face API token and try again."
  exit 1
fi

hf download Qwen/Qwen3-VL-8B-Instruct --local-dir /workspace/.hf_home/qwen3-vl-8b
hf download Qwen/Qwen3-VL-2B-Instruct --local-dir /workspace/.hf_home/qwen3-vl-2b
hf download taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3 --local-dir /workspace/.hf_home/qwen3-vl-8b-eagle3
#!/bin/bash

HF_ENDPOINT=https://hf-mirror.com hf download Qwen/Qwen3-VL-8B-Instruct --local-dir /workspace/.hf_home/qwen3-vl-8b
HF_ENDPOINT=https://hf-mirror.com hf download Qwen/Qwen3-VL-2B-Instruct --local-dir /workspace/.hf_home/qwen3-vl-2b
HF_ENDPOINT=https://hf-mirror.com hf download taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3 --local-dir /workspace/.hf_home/qwen3-vl-8b-eagle3
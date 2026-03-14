#!/bin/bash
# GPU memory estimation constants for shell scripts.
# Python side (hardware.py) computes exact values from model config.
# Bash side approximation: weights (bf16) + KV cache at max_new_tokens=32768, batch=8.
# Llama-3.2-1B: weights ~2.4GB + KV ~8GB + CUDA ~0.5GB = ~11GB -> 11/1B = 11 per B.
# Llama-3.1-8B: weights ~16GB + KV ~33GB + CUDA ~0.5GB = ~49GB -> 49/8B = 6 per B.
# Use conservative value covering smaller models where KV dominates.
FP16_GB_PER_BILLION_PARAMS=2
GPU_FRAMEWORK_OVERHEAD_GB=8
# Approximate total per-worker: param_b * WEIGHT_PLUS_KV_PER_BILLION + CUDA_CONTEXT_GB
WEIGHT_PLUS_KV_PER_BILLION=11
CUDA_CONTEXT_GB=1

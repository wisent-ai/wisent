#!/bin/bash
# GPU memory estimation constants for shell scripts.
# Python side (hardware.py) computes exact values from model config.
# Bash side approximation: weights (bf16) + KV cache at max_new_tokens=32768, batch=8.
# Base estimate ~11GB/B, with 1.5x fragmentation factor = ~16GB/B.
# Llama-3.2-1B: 1*16+1 = 17GB/worker -> 1 worker on 24GB L4.
# Llama-3.1-8B: 8*16+1 = 129GB/worker -> needs A100 80GB.
FP16_GB_PER_BILLION_PARAMS=2
GPU_FRAMEWORK_OVERHEAD_GB=8
# Approximate total per-worker: param_b * WEIGHT_PLUS_KV_PER_BILLION + CUDA_CONTEXT_GB
# Includes 1.5x GPU_FRAGMENTATION_OVERHEAD_FACTOR (matches Python side)
WEIGHT_PLUS_KV_PER_BILLION=16
CUDA_CONTEXT_GB=1
# Max concurrent GCP instances for batch experiments
DEFAULT_MAX_PARALLEL_INSTANCES=8
# Shell exit codes (mirrors Python EXIT_CODE_ERROR)
EXIT_FAILURE=1
EXIT_SUCCESS=0
# Counter / index initialization
COUNTER_INIT=0
# Signal zero for kill-based process liveness check
SIGNAL_ZERO=0
# stderr file descriptor
FD_STDERR=2

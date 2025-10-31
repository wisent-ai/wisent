#!/bin/bash

# Example: Generate responses from a task without any steering or classification
# This script demonstrates basic response generation using different tasks
# and generation parameters.

# Generate responses from arc_easy task
python -m wisent.core.main generate-responses \
    meta-llama/Llama-3.2-1B-Instruct \
    --task arc_easy \
    --num-questions 10 \
    --max-new-tokens 128 \
    --temperature 0.7 \
    --top-p 0.95 \
    --output ./results/arc_easy_responses.json \
    --verbose

# Generate responses with deterministic settings
python -m wisent.core.main generate-responses \
    meta-llama/Llama-3.2-1B-Instruct \
    --task truthfulqa_mc1 \
    --num-questions 20 \
    --max-new-tokens 256 \
    --temperature 0.0 \
    --output ./results/truthfulqa_deterministic.json \
    --device cuda \
    --verbose

# Generate responses with creative settings
python -m wisent.core.main generate-responses \
    meta-llama/Llama-3.2-1B-Instruct \
    --task gsm8k \
    --num-questions 15 \
    --max-new-tokens 512 \
    --temperature 1.0 \
    --top-p 0.9 \
    --output ./results/gsm8k_creative.json \
    --verbose

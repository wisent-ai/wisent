#!/bin/bash

# Example: Create steering vector from lm-eval task (single unified command)
# This script demonstrates using the generate-vector-from-task command which runs
# the complete pipeline in one go:
# 1. Generate contrastive pairs from an lm-eval task
# 2. Collect activations from those pairs
# 3. Create steering vectors from the activations
#
# Usage:
#   bash wisent/examples/cli/steering/create_steering_vector_from_task.sh

python -m wisent.core.main generate-vector-from-task \
    --task hellaswag \
    --trait-label correctness \
    --output ./data/steering_vectors/hellaswag_correctness_vector.json \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --num-pairs 50 \
    --layers 8 \
    --token-aggregation average \
    --prompt-strategy chat_template \
    --method caa \
    --normalize \
    --verbose \
    --timing


#!/bin/bash

# Example: Create steering vector from synthetic contrastive pairs (single unified command)
# This script demonstrates using the generate-vector-from-synthetic command which runs
# the complete pipeline in one go:
# 1. Generate synthetic contrastive pairs for a trait
# 2. Collect activations from those pairs
# 3. Create steering vectors from the activations
#
# Usage:
#   bash wisent/examples/cli/steering/create_steering_vector_from_synthetic.sh

python -m wisent.core.main generate-vector-from-synthetic \
    --trait helpfulness \
    --output ./data/steering_vectors/helpfulness_vector.json \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --num-pairs 20 \
    --similarity-threshold 0.8 \
    --layers 8 \
    --token-aggregation average \
    --prompt-strategy chat_template \
    --method caa \
    --normalize \
    --verbose \
    --timing

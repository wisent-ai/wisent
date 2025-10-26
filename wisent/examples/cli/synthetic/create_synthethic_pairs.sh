#!/bin/bash

# Example: Generate synthetic contrastive pairs for a trait
# This script demonstrates using the generate-pairs command to create
# synthetic contrastive pairs that can later be used for:
# - Collecting activations
# - Creating steering vectors
# - Training classifiers
#
# Usage:
#   bash wisent/examples/cli/synthetic/create_synthethic_pairs.sh

python -m wisent.core.main generate-pairs \
    --trait helpfulness \
    --output ./data/synthetic_pairs/helpfulness_pairs.json \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --num-pairs 20 \
    --similarity-threshold 0.8 \
    --device cpu \
    --verbose \
    --timing

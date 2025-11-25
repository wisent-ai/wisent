#!/bin/bash

# Example: Modify model weights from synthetic contrastive pairs
# This script demonstrates using the modify-weights command with synthetically-generated steering vectors
# Uses kernel-based additive method across multiple layers for stronger steering effect
#
# Usage:
#   bash wisent/examples/cli/weight_modification/modify_weights_from_synthetic.sh

python -m wisent.core.main modify-weights \
    --trait helpfulness \
    --output-dir ./data/modified_models/helpfulness_modified \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --num-pairs 50 \
    --similarity-threshold 0.8 \
    --layers all \
    --method abliteration \
    --strength 1.0 \
    --components self_attn.o_proj mlp.down_proj \
    --use-kernel \
    --max-weight 1.5 \
    --max-weight-position 8.0 \
    --min-weight 0.3 \
    --min-weight-distance 6.0 \
    --normalize-vectors \
    --verbose

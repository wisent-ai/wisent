#!/bin/bash

# Example: Get activations from contrastive pairs
# This script demonstrates how to load existing contrastive pairs from JSON
# and enrich them with neural network activations from specified layers.

python -m wisent.core.main get-activations \
    ./data/test_synthetic.json \
    --output ./data/test_synthetic_with_activations.json \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --layers 8 \
    --token-aggregation average \
    --device cpu \
    --verbose \
    --timing

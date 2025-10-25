#!/bin/bash

# Generate synthetic contrastive pairs from a trait description
# This example shows how to generate pairs by describing a desired trait or behavior.

python -m wisent.core.main generate-pairs \
    --trait "hallucination and making up false information" \
    --num-pairs 30 \
    --output ./data/synthetic_hallucination_pairs.json \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --similarity-threshold 0.8 \
    --verbose \
    --timing

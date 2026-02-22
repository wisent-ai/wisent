#!/bin/bash

# Example: Generate responses with steering vectors
# This script demonstrates how to use steering vectors to control
# model behavior during generation.

# First, train and save a steering vector (training-only mode)
python -m wisent.core.main tasks truthfulqa_mc1 \
    meta-llama/Llama-3.2-1B-Instruct \
    --layer 15 \
    --steering-method CAA \
    --limit 100 \
    --train-only \
    --save-vector ./vectors/truthfulness_vector.pt \
    --output ./results/vector_training \
    --verbose

# Now use the steering vector during generation (inference-only mode)
python -m wisent.core.main tasks truthfulqa_mc1 \
    meta-llama/Llama-3.2-1B-Instruct \
    --layer 15 \
    --steering-method CAA \
    --steering-strength 1.5 \
    --limit 20 \
    --inference-only \
    --load-vector ./vectors/truthfulness_vector.pt \
    --output ./results/steered_generation \
    --verbose

# Use stronger steering with CAA_L2
python -m wisent.core.main tasks arc_easy \
    meta-llama/Llama-3.2-1B-Instruct \
    --layer 12 \
    --steering-method CAA_L2 \
    --steering-strength 2.0 \
    --limit 50 \
    --inference-only \
    --load-vector ./vectors/arc_reasoning_vector.pt \
    --output ./results/strong_steering \
    --device cuda \
    --verbose

# Multi-layer steering with different vectors per layer
python -m wisent.core.main tasks boolq \
    meta-llama/Llama-3.2-1B-Instruct \
    --layers 10 15 20 \
    --steering-method CAA \
    --steering-strength 1.2 \
    --limit 40 \
    --inference-only \
    --vector-dir ./vectors/boolq_multilayer \
    --output ./results/multilayer_steering \
    --verbose

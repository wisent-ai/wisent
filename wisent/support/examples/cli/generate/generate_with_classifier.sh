#!/bin/bash

# Example: Generate responses with real-time classifier monitoring
# This script demonstrates how to use trained classifiers to monitor
# model outputs during generation and detect specific behaviors.

# First, train and save a classifier (training-only mode)
python -m wisent.core.main tasks truthfulqa_mc1 \
    meta-llama/Llama-3.2-1B-Instruct \
    --layer 15 \
    --classifier-type logistic \
    --limit 100 \
    --train-only \
    --save-classifier ./models/hallucination_detector.pt \
    --output ./results/training_logs \
    --verbose

# Now use the trained classifier during generation (inference-only mode)
python -m wisent.core.main tasks truthfulqa_mc1 \
    meta-llama/Llama-3.2-1B-Instruct \
    --layer 15 \
    --limit 20 \
    --inference-only \
    --load-classifier ./models/hallucination_detector.pt \
    --output ./results/monitored_generation \
    --verbose

# Use classifier with custom detection threshold
python -m wisent.core.main tasks arc_easy \
    meta-llama/Llama-3.2-1B-Instruct \
    --layer 12 \
    --classifier-type mlp \
    --limit 50 \
    --inference-only \
    --load-classifier ./models/arc_classifier.pt \
    --threshold 0.7 \
    --output ./results/arc_monitored \
    --device cuda \
    --verbose

# Multi-layer classifier monitoring
python -m wisent.core.main tasks gsm8k \
    meta-llama/Llama-3.2-1B-Instruct \
    --layers 10 15 20 \
    --classifier-type logistic \
    --limit 30 \
    --inference-only \
    --classifier-dir ./models/gsm8k_classifiers \
    --output ./results/multi_layer_monitoring \
    --verbose

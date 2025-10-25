#!/bin/bash

# Use a pre-trained classifier from a saved location
# This example shows how to load a previously trained classifier and use it
# for inference without retraining.

python -m wisent.core.main tasks truthfulqa_mc1 \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --layer 15 \
    --load-classifier ./models/truthfulqa_classifier.pt \
    --inference-only \
    --testing-limit 100 \
    --output ./results/inference_results

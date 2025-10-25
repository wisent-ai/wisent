#!/bin/bash

# Train a classifier and save it for later use
# This example shows how to train a logistic classifier on the truthfulqa_mc1 task
# to detect hallucinations, and save the trained model.

python -m wisent.core.main tasks truthfulqa_mc1 \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --layer 15 \
    --classifier-type logistic \
    --limit 200 \
    --save-classifier ./models/truthfulqa_classifier.pt \
    --output ./results/classifier_training

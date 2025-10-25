#!/bin/bash

# Train a classifier and evaluate it on a benchmark task
# This example trains a classifier on the truthfulqa_mc1 benchmark,
# evaluates its performance, and generates evaluation metrics.

python -m wisent.core.main tasks truthfulqa_mc1 \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --layer 15 \
    --classifier-type logistic \
    --training-limit 200 \
    --testing-limit 100 \
    --ground-truth-method lm-eval-harness \
    --token-aggregation average \
    --detection-threshold 0.6 \
    --output ./results/benchmark_evaluation \
    --evaluation-report ./results/benchmark_evaluation/report.json \
    --verbose

#!/bin/bash

# Example: Optimize training sample size for classifiers
# This script finds the optimal number of training samples needed to achieve
# good performance on a task. It tests different sample sizes and evaluates
# classification accuracy to determine the minimum viable training set size.

# Basic sample size optimization for classification
python -m wisent.core.main optimize-sample-size \
    meta-llama/Llama-3.2-1B-Instruct \
    --task truthfulqa_mc1 \
    --layer 15 \
    --token-aggregation average \
    --sample-sizes 5 10 20 50 100 200 \
    --test-size 100 \
    --save-plot \
    --verbose

# Optimize for steering mode instead of classification
python -m wisent.core.main optimize-sample-size \
    meta-llama/Llama-3.2-1B-Instruct \
    --task arc_easy \
    --layer 10 \
    --token-aggregation final \
    --steering-mode \
    --steering-method CAA \
    --steering-strength 1.5 \
    --sample-sizes 10 25 50 100 200 500 \
    --test-size 200 \
    --save-plot \
    --device cuda \
    --verbose

# Custom sample sizes with specific threshold
python -m wisent.core.main optimize-sample-size \
    meta-llama/Llama-3.2-1B-Instruct \
    --task gsm8k \
    --layer 12 \
    --token-aggregation max \
    --threshold 0.6 \
    --sample-sizes 20 40 80 160 320 \
    --test-size 150 \
    --seed 123 \
    --save-plot \
    --verbose

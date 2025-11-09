#!/bin/bash

# Demonstrate autonomous agent functionality
# Strategy: synthetic_pairs_classifier_steering
#
# Steps:
# 1. Agent creates contrastive pairs synthetically for the desired representation
# 2. Agent trains classifiers and chooses the best one using evaluation
# 3. Agent generates an unsteered response and uses classifier to check correctness
# 4. If incorrect: uses steering to train control vector, creates new response, evaluates until success
# 5. If correct: returns the response

wisent agent "Write a helpful and informative response about climate change" \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --agent-strategy synthetic_pairs_classifier_steering \
    --quality-threshold 0.3 \
    --time-budget 10.0 \
    --max-attempts 3 \
    --verbose

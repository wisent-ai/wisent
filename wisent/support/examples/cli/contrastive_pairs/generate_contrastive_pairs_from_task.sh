#!/bin/bash

# Generate contrastive pairs from an lm-eval task
# This example shows how to extract contrastive pairs from the truthfulqa_mc1 task
# and save them to a JSON file for later use.

python -m wisent.core.main generate-pairs-from-task truthfulqa_mc1 \
    --output ./data/truthfulqa_mc1_pairs.json \
    --limit 50 \
    --verbose

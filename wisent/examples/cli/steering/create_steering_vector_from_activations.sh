#!/bin/bash

# Example: Create steering vectors from enriched pairs (pairs with activations)
# This script demonstrates how to use the create-steering-vector command to generate
# steering vectors from contrastive pairs that already have activations collected.
#
# Prerequisites:
# 1. Have enriched pairs file (output from get-activations command)
#
# Usage:
#   bash wisent/examples/cli/steering/create_steering_vector_from_activations.sh

python -m wisent.core.main create-steering-vector \
    ./data/test_synthetic_with_activations.json \
    --output ./data/steering_vectors/rudeness_vector.json \
    --method caa \
    --normalize \
    --verbose \
    --timing

#!/bin/bash

# Example: Train multiple steering vectors and combine them dynamically
# This script demonstrates the complete multi-steering workflow:
# 1. Train individual steering vectors for different traits
# 2. Combine them with different weights during inference
# 3. Generate responses with the combined steering effect

# Step 1: Train steering vectors for different traits
echo "Training truthfulness vector..."
python -m wisent.core.main generate-vector-from-task \
    --task truthfulqa_mc1 \
    --trait-label truthfulness \
    --output ./vectors/truthfulness.pt \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --num-pairs 100 \
    --layers 15 \
    --token-aggregation average \
    --method caa \
    --normalize \
    --device cuda \
    --verbose

echo "Training helpfulness vector..."
python -m wisent.core.main generate-vector-from-task \
    --task arc_easy \
    --trait-label helpfulness \
    --output ./vectors/helpfulness.pt \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --num-pairs 100 \
    --layers 15 \
    --token-aggregation average \
    --method caa \
    --normalize \
    --device cuda \
    --verbose

echo "Training creativity vector..."
python -m wisent.core.main generate-vector-from-synthetic \
    --behavior "creative and imaginative responses" \
    --opposite-behavior "boring and predictable responses" \
    --num-pairs 50 \
    --trait-label creativity \
    --output ./vectors/creativity.pt \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --layers 15 \
    --token-aggregation average \
    --method caa \
    --normalize \
    --device cuda \
    --verbose

# Step 2: Combine vectors with equal weights
echo "Generating with equal weight combination..."
python -m wisent.core.main multi-steer \
    --vector ./vectors/truthfulness.pt:0.33 \
    --vector ./vectors/helpfulness.pt:0.33 \
    --vector ./vectors/creativity.pt:0.34 \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --layer 15 \
    --method CAA \
    --prompt "Explain quantum computing in simple terms." \
    --max-new-tokens 200 \
    --normalize-weights \
    --save-combined ./vectors/balanced_assistant.pt \
    --device cuda \
    --verbose

# Step 3: Emphasize truthfulness more
echo "Generating with truthfulness emphasis..."
python -m wisent.core.main multi-steer \
    --vector ./vectors/truthfulness.pt:0.7 \
    --vector ./vectors/helpfulness.pt:0.2 \
    --vector ./vectors/creativity.pt:0.1 \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --layer 15 \
    --method CAA \
    --prompt "What are the health benefits of coffee?" \
    --max-new-tokens 200 \
    --normalize-weights \
    --device cuda \
    --verbose

# Step 4: Emphasize creativity for storytelling
echo "Generating with creativity emphasis..."
python -m wisent.core.main multi-steer \
    --vector ./vectors/truthfulness.pt:0.1 \
    --vector ./vectors/helpfulness.pt:0.2 \
    --vector ./vectors/creativity.pt:0.7 \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --layer 15 \
    --method CAA \
    --prompt "Write a short story about a robot learning to paint." \
    --max-new-tokens 300 \
    --normalize-weights \
    --device cuda \
    --verbose

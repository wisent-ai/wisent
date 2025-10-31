#!/bin/bash

# Example: Get activations using all combinations of aggregation and prompt strategies
# This script demonstrates all available activation aggregation strategies and
# prompt construction strategies for collecting neural network activations.

set -e

# Configuration
INPUT_FILE="./data/test_synthetic.json"
OUTPUT_DIR="./data/activations_comparison"
MODEL="meta-llama/Llama-3.2-1B-Instruct"
LAYERS="8,12,15"
DEVICE="cpu"
LIMIT=10

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=================================================="
echo "Testing All Activation Aggregation Strategies"
echo "=================================================="
echo ""

# Token Aggregation Strategies:
# - average: Mean pooling across all tokens
# - final: Use only the last token
# - first: Use only the first token
# - max: Max pooling across all tokens
# - min: Min pooling across all tokens

AGGREGATION_STRATEGIES=("average" "final" "first" "max" "min")

# Prompt Construction Strategies:
# - chat_template: Use model's chat template (default)
# - direct_completion: Direct text completion
# - instruction_following: Instruction-following format
# - multiple_choice: Multiple choice question format
# - role_playing: Role-playing scenario format

PROMPT_STRATEGIES=("chat_template" "direct_completion" "instruction_following" "multiple_choice" "role_playing")

# Test each aggregation strategy with the default prompt strategy
echo "Testing Aggregation Strategies (with chat_template prompt):"
echo "-----------------------------------------------------------"
for agg in "${AGGREGATION_STRATEGIES[@]}"; do
    output_file="${OUTPUT_DIR}/activations_agg_${agg}.json"
    echo "Testing: --token-aggregation $agg"

    python -m wisent.core.main get-activations \
        "$INPUT_FILE" \
        --output "$output_file" \
        --model "$MODEL" \
        --layers "$LAYERS" \
        --token-aggregation "$agg" \
        --prompt-strategy chat_template \
        --device "$DEVICE" \
        --limit "$LIMIT" \
        --verbose \
        --timing

    echo " Saved to: $output_file"
    echo ""
done

echo ""
echo "=================================================="
echo "Testing All Prompt Construction Strategies"
echo "=================================================="
echo ""

# Test each prompt strategy with the default aggregation
echo "Testing Prompt Strategies (with average aggregation):"
echo "-----------------------------------------------------"
for prompt in "${PROMPT_STRATEGIES[@]}"; do
    output_file="${OUTPUT_DIR}/activations_prompt_${prompt}.json"
    echo "Testing: --prompt-strategy $prompt"

    python -m wisent.core.main get-activations \
        "$INPUT_FILE" \
        --output "$output_file" \
        --model "$MODEL" \
        --layers "$LAYERS" \
        --token-aggregation average \
        --prompt-strategy "$prompt" \
        --device "$DEVICE" \
        --limit "$LIMIT" \
        --verbose \
        --timing

    echo " Saved to: $output_file"
    echo ""
done

echo ""
echo "=================================================="
echo "Testing Combined Strategy Variations"
echo "=================================================="
echo ""

# Test some interesting combinations
echo "Testing Combined Strategies:"
echo "----------------------------"

# Combination 1: Final token + Direct completion
echo "1. Final token aggregation + Direct completion prompt"
python -m wisent.core.main get-activations \
    "$INPUT_FILE" \
    --output "${OUTPUT_DIR}/activations_final_direct.json" \
    --model "$MODEL" \
    --layers "$LAYERS" \
    --token-aggregation final \
    --prompt-strategy direct_completion \
    --device "$DEVICE" \
    --limit "$LIMIT" \
    --verbose
echo ""

# Combination 2: First token + Instruction following
echo "2. First token aggregation + Instruction following prompt"
python -m wisent.core.main get-activations \
    "$INPUT_FILE" \
    --output "${OUTPUT_DIR}/activations_first_instruction.json" \
    --model "$MODEL" \
    --layers "$LAYERS" \
    --token-aggregation first \
    --prompt-strategy instruction_following \
    --device "$DEVICE" \
    --limit "$LIMIT" \
    --verbose
echo ""

# Combination 3: Max pooling + Multiple choice
echo "3. Max pooling aggregation + Multiple choice prompt"
python -m wisent.core.main get-activations \
    "$INPUT_FILE" \
    --output "${OUTPUT_DIR}/activations_max_multiplechoice.json" \
    --model "$MODEL" \
    --layers "$LAYERS" \
    --token-aggregation max \
    --prompt-strategy multiple_choice \
    --device "$DEVICE" \
    --limit "$LIMIT" \
    --verbose
echo ""

# Combination 4: Average + Role playing
echo "4. Average aggregation + Role playing prompt"
python -m wisent.core.main get-activations \
    "$INPUT_FILE" \
    --output "${OUTPUT_DIR}/activations_average_roleplaying.json" \
    --model "$MODEL" \
    --layers "$LAYERS" \
    --token-aggregation average \
    --prompt-strategy role_playing \
    --device "$DEVICE" \
    --limit "$LIMIT" \
    --verbose
echo ""

echo ""
echo "=================================================="
echo "Testing Different Layer Configurations"
echo "=================================================="
echo ""

# Single layer
echo "Testing single layer (layer 8):"
python -m wisent.core.main get-activations \
    "$INPUT_FILE" \
    --output "${OUTPUT_DIR}/activations_single_layer.json" \
    --model "$MODEL" \
    --layers "8" \
    --token-aggregation average \
    --prompt-strategy chat_template \
    --device "$DEVICE" \
    --limit "$LIMIT" \
    --verbose
echo ""

# Multiple specific layers
echo "Testing multiple layers (8,12,15):"
python -m wisent.core.main get-activations \
    "$INPUT_FILE" \
    --output "${OUTPUT_DIR}/activations_multi_layers.json" \
    --model "$MODEL" \
    --layers "8,12,15" \
    --token-aggregation average \
    --prompt-strategy chat_template \
    --device "$DEVICE" \
    --limit "$LIMIT" \
    --verbose
echo ""

# All layers (warning: may be slow!)
echo "Testing all layers (may be slow!):"
python -m wisent.core.main get-activations \
    "$INPUT_FILE" \
    --output "${OUTPUT_DIR}/activations_all_layers.json" \
    --model "$MODEL" \
    --layers "all" \
    --token-aggregation average \
    --prompt-strategy chat_template \
    --device "$DEVICE" \
    --limit 5 \
    --verbose
echo ""

echo ""
echo "=================================================="
echo " All activation extraction tests completed!"
echo "=================================================="
echo ""
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Summary of tested configurations:"
echo "- Token Aggregation: ${AGGREGATION_STRATEGIES[*]}"
echo "- Prompt Strategies: ${PROMPT_STRATEGIES[*]}"
echo "- Layer configurations: single, multiple, all"
echo ""

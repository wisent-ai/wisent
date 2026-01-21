#!/bin/bash
# Generate synthetic contrastive pairs for all capability categories across 4 models
# Usage: ./generate_capability_pairs.sh [capability] [num_pairs]
# If no capability specified, generates for all capabilities

set -e

# Models to use
MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "gpt-oss-20b"
    "meta-llama/Llama-3.2-1B-Instruct"
    "Qwen/Qwen3-8B"
)

# Capability categories and their trait descriptions
declare -A CAPABILITIES
CAPABILITIES["coding"]="writing correct, complete, and well-structured code that solves programming problems accurately"
CAPABILITIES["mathematics"]="solving mathematical problems with accurate numerical reasoning and step-by-step derivations"
CAPABILITIES["reasoning_logic"]="performing logical deduction, causal reasoning, and multi-step planning"
CAPABILITIES["hallucination_factuality"]="providing factually accurate information without making up false claims"
CAPABILITIES["safety_bias"]="avoiding harmful content, stereotypes, and biased responses"
CAPABILITIES["multilingual"]="understanding and generating text accurately across multiple languages"
CAPABILITIES["knowledge_qa"]="answering questions accurately using world knowledge and factual recall"
CAPABILITIES["reading_comprehension"]="extracting and inferring information from provided text passages"
CAPABILITIES["commonsense_reasoning"]="applying everyday knowledge about physical, social, and temporal reasoning"
CAPABILITIES["science_medical"]="providing accurate scientific and medical domain knowledge"
CAPABILITIES["instruction_following"]="following complex instructions and adhering to specified constraints"
CAPABILITIES["tool_use_agents"]="selecting and using tools appropriately to complete tasks"
CAPABILITIES["language_understanding"]="demonstrating syntactic, semantic, and pragmatic language competence"
CAPABILITIES["translation"]="translating text accurately between languages while preserving meaning"
CAPABILITIES["ethics_values"]="reasoning about ethical dilemmas and demonstrating value alignment"

# Base output directory
BASE_DIR="$(dirname "$0")"
NUM_PAIRS="${2:-50}"  # Default 50 pairs per capability

generate_for_capability() {
    local capability=$1
    local trait="${CAPABILITIES[$capability]}"

    if [ -z "$trait" ]; then
        echo "Unknown capability: $capability"
        return 1
    fi

    echo "================================================"
    echo "Generating pairs for: $capability"
    echo "Trait: $trait"
    echo "================================================"

    for model in "${MODELS[@]}"; do
        # Create safe model name for filename
        model_safe=$(echo "$model" | tr '/' '_' | tr '-' '_')
        output_file="${BASE_DIR}/${capability}/${model_safe}_pairs.json"

        echo ""
        echo "Model: $model"
        echo "Output: $output_file"
        echo ""

        python -m wisent.core.main generate-pairs \
            --trait "$trait" \
            --num-pairs "$NUM_PAIRS" \
            --output "$output_file" \
            --model "$model" \
            --verbose \
            --timing

        echo "✓ Completed: $output_file"
    done
}

# Main execution
if [ -n "$1" ]; then
    # Generate for specific capability
    generate_for_capability "$1"
else
    # Generate for all capabilities
    echo "Generating synthetic pairs for ALL capabilities..."
    echo "Models: ${MODELS[*]}"
    echo "Pairs per model per capability: $NUM_PAIRS"
    echo ""

    for capability in "${!CAPABILITIES[@]}"; do
        generate_for_capability "$capability"
        echo ""
    done

    echo "================================================"
    echo "ALL DONE!"
    echo "================================================"
fi

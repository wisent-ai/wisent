#!/bin/bash
#
# Comprehensive Steering Optimization with ALL Configuration Examples
#
# This example demonstrates saving generation examples for EVERY configuration tested:
# 1. Optimizes steering parameters (layer, strength, strategy) across tasks
# 2. For EACH configuration tested, generates example responses (unsteered vs steered)
# 3. Saves all examples to see how different configurations affect generation
# 4. Also saves the best steering vector
#
# WARNING: This is MUCH slower than regular optimization because it generates
# text for every single configuration (80 configs × 2 examples = 160 generations)
#
# Usage:
#   bash comprehensive_all_examples.sh
#
# Output:
#   - ./steering_optimization_results/steering_comprehensive_*.json (optimization results)
#   - ./steering_vectors/<task>_layer<N>.pt (best steering vector)
#   - ./steering_vectors/<task>_all_generation_examples.json (examples for ALL configs)

MODEL="meta-llama/Llama-3.2-1B-Instruct"
TASKS="boolq"
LIMIT=20
METHODS="CAA"
DEVICE="cpu"
VECTOR_DIR="./steering_vectors"
NUM_EXAMPLES=1  # Use just 1 example per config to keep it manageable

echo "=========================================================================="
echo "Comprehensive Steering Optimization with ALL Configuration Examples"
echo "=========================================================================="
echo "Model: $MODEL"
echo "Tasks: $TASKS"
echo "Sample limit: $LIMIT per task"
echo "Methods: $METHODS"
echo "Device: $DEVICE"
echo "Vector output: $VECTOR_DIR"
echo "Examples per config: $NUM_EXAMPLES"
echo ""
echo "⚠️  WARNING: This will generate examples for ALL configurations"
echo "    Expected time: ~60-90 minutes for 80 configurations"
echo "=========================================================================="
echo ""

python -m wisent.core.main optimize-steering comprehensive "$MODEL" \
    --tasks $TASKS \
    --methods $METHODS \
    --limit $LIMIT \
    --device $DEVICE \
    --save-best-vector "$VECTOR_DIR" \
    --save-all-generation-examples \
    --num-generation-examples $NUM_EXAMPLES \
    --verbose

echo ""
echo "=========================================================================="
echo "Optimization Complete!"
echo "=========================================================================="
echo "Check results:"
echo "  - Optimization results: ./optimization_results/steering_comprehensive_*.json"
echo "  - Best steering vector: $VECTOR_DIR/<task>_layer<N>.pt"
echo "  - ALL generation examples: $VECTOR_DIR/<task>_all_generation_examples.json"
echo "=========================================================================="

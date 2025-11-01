#!/bin/bash
#
# Comprehensive Steering Optimization with Vector Saving and Generation Examples
#
# This example demonstrates the full steering optimization workflow:
# 1. Optimizes steering parameters (layer, strength, strategy) across tasks
# 2. Saves the best steering vector for each task
# 3. Generates example responses showing steered vs unsteered behavior
#
# Usage:
#   bash comprehensive_with_examples.sh
#
# Output:
#   - ./steering_optimization_results/steering_comprehensive_*.json (optimization results)
#   - ./steering_vectors/<task>_layer<N>.pt (best steering vectors)
#   - ./steering_vectors/<task>_generation_examples.json (steered vs unsteered examples)

MODEL="meta-llama/Llama-3.2-1B-Instruct"
TASKS="boolq"
LIMIT=30
METHODS="CAA"
DEVICE="cpu"
VECTOR_DIR="./steering_vectors"
NUM_EXAMPLES=3

echo "=================================================="
echo "Comprehensive Steering Optimization with Examples"
echo "=================================================="
echo "Model: $MODEL"
echo "Tasks: $TASKS"
echo "Sample limit: $LIMIT per task"
echo "Methods: $METHODS"
echo "Device: $DEVICE"
echo "Vector output: $VECTOR_DIR"
echo "Generation examples: $NUM_EXAMPLES per task"
echo "=================================================="
echo ""

python -m wisent.core.main optimize-steering comprehensive "$MODEL" \
    --tasks $TASKS \
    --methods $METHODS \
    --limit $LIMIT \
    --device $DEVICE \
    --save-best-vector "$VECTOR_DIR" \
    --save-generation-examples \
    --num-generation-examples $NUM_EXAMPLES \
    --verbose

echo ""
echo "=================================================="
echo "Optimization Complete!"
echo "=================================================="
echo "Check results:"
echo "  - Optimization results: ./optimization_results/steering_comprehensive_*.json"
echo "  - Best steering vectors: $VECTOR_DIR/<task>_layer<N>.pt"
echo "  - Generation examples: $VECTOR_DIR/<task>_generation_examples.json"
echo "=================================================="

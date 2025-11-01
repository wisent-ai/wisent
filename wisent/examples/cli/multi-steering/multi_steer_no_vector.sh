#!/bin/bash

# Example: Generate unsteered baseline with multi-steer command
# This script demonstrates how to use multi-steer with NO vectors
# to generate baseline (unsteered) responses for comparison.

# Configuration
MODEL="meta-llama/Llama-3.2-1B-Instruct"
DEVICE="cpu"

echo "=== Unsteered Baseline Generation ==="
echo "Generating response without any steering vectors for comparison."
echo ""

# Example 1: Technical question - baseline response
echo "Example 1: Technical Question (Baseline)"
python -m wisent.core.main multi-steer \
    --model $MODEL \
    --prompt "Explain how neural networks learn from data." \
    --max-new-tokens 150 \
    --device $DEVICE \
    --verbose

echo ""
echo "=== Comparison with Steered Generation ==="
echo "To compare, you would run the same prompt WITH a steering vector:"
echo ""
echo "python -m wisent.core.main multi-steer \\"
echo "    --vector ./steering_vectors/technical.pt:1.0 \\"
echo "    --model $MODEL \\"
echo "    --layer 15 \\"
echo "    --prompt \"Explain how neural networks learn from data.\" \\"
echo "    --max-new-tokens 150 \\"
echo "    --device $DEVICE"
echo ""

# Example 2: Simple question - baseline
echo "Example 2: Simple Question (Baseline)"
python -m wisent.core.main multi-steer \
    --model $MODEL \
    --prompt "What is machine learning?" \
    --max-new-tokens 100 \
    --device $DEVICE \
    --verbose

echo ""
echo "=== Use Case ==="
echo "This baseline generation is useful for:"
echo "  1. Comparing steered vs unsteered outputs"
echo "  2. Measuring the effect of steering vectors"
echo "  3. Establishing a performance baseline"
echo "  4. A/B testing different steering configurations"
echo ""

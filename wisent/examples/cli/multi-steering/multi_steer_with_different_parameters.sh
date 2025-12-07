#!/bin/bash

# Example: Experiment with different multi-steering parameters
# This script demonstrates various ways to combine steering vectors:
# - Different weight normalization strategies
# - Different target norms for stronger/weaker effects
# - CAA steering method
# - Allowing unnormalized weights for amplified effects

# Assume we have pre-trained vectors
TRUTHFULNESS_VECTOR="./vectors/truthfulness.pt"
HELPFULNESS_VECTOR="./vectors/helpfulness.pt"
CREATIVITY_VECTOR="./vectors/creativity.pt"

PROMPT="Explain artificial intelligence to a beginner."
MODEL="meta-llama/Llama-3.2-1B-Instruct"
LAYER=15

# 1. Basic combination with weight normalization
echo "=== Example 1: Normalized weights (sum to 1.0) ==="
python -m wisent.core.main multi-steer \
    --vector $TRUTHFULNESS_VECTOR:0.5 \
    --vector $HELPFULNESS_VECTOR:0.3 \
    --vector $CREATIVITY_VECTOR:0.2 \
    --model $MODEL \
    --layer $LAYER \
    --method CAA \
    --prompt "$PROMPT" \
    --max-new-tokens 150 \
    --normalize-weights \
    --verbose

# 2. Unnormalized weights for stronger combined effect
echo "=== Example 2: Unnormalized weights (stronger effect) ==="
python -m wisent.core.main multi-steer \
    --vector $TRUTHFULNESS_VECTOR:2.0 \
    --vector $HELPFULNESS_VECTOR:1.5 \
    --vector $CREATIVITY_VECTOR:1.0 \
    --model $MODEL \
    --layer $LAYER \
    --method CAA \
    --prompt "$PROMPT" \
    --max-new-tokens 150 \
    --allow-unnormalized \
    --verbose

# 3. Scale combined vector to specific norm
echo "=== Example 3: Target norm scaling ==="
python -m wisent.core.main multi-steer \
    --vector $TRUTHFULNESS_VECTOR:0.5 \
    --vector $HELPFULNESS_VECTOR:0.5 \
    --model $MODEL \
    --layer $LAYER \
    --method CAA \
    --prompt "$PROMPT" \
    --max-new-tokens 150 \
    --target-norm 10.0 \
    --verbose

# 4. Strong scaling for maximum effect
echo "=== Example 4: Very strong scaling ==="
python -m wisent.core.main multi-steer \
    --vector $TRUTHFULNESS_VECTOR:1.0 \
    --vector $HELPFULNESS_VECTOR:1.0 \
    --model $MODEL \
    --layer $LAYER \
    --method CAA \
    --prompt "$PROMPT" \
    --max-new-tokens 150 \
    --target-norm 50.0 \
    --verbose

# 5. Subtle effect with small norm
echo "=== Example 5: Subtle steering ==="
python -m wisent.core.main multi-steer \
    --vector $TRUTHFULNESS_VECTOR:0.5 \
    --vector $HELPFULNESS_VECTOR:0.5 \
    --model $MODEL \
    --layer $LAYER \
    --method CAA \
    --prompt "$PROMPT" \
    --max-new-tokens 150 \
    --target-norm 2.0 \
    --verbose

# 6. Save different combinations for reuse
echo "=== Example 7: Save balanced combination ==="
python -m wisent.core.main multi-steer \
    --vector $TRUTHFULNESS_VECTOR:0.33 \
    --vector $HELPFULNESS_VECTOR:0.33 \
    --vector $CREATIVITY_VECTOR:0.34 \
    --model $MODEL \
    --layer $LAYER \
    --method CAA \
    --prompt "$PROMPT" \
    --max-new-tokens 150 \
    --normalize-weights \
    --save-combined ./vectors/balanced_combo.pt \
    --verbose

echo "=== Example 8: Save truthfulness-focused combination ==="
python -m wisent.core.main multi-steer \
    --vector $TRUTHFULNESS_VECTOR:0.8 \
    --vector $HELPFULNESS_VECTOR:0.2 \
    --model $MODEL \
    --layer $LAYER \
    --method CAA \
    --prompt "$PROMPT" \
    --max-new-tokens 150 \
    --normalize-weights \
    --save-combined ./vectors/truthful_combo.pt \
    --verbose

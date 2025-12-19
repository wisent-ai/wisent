#!/bin/bash

# ===========================================
# Test all extraction strategies in ONE session
# ===========================================

# AWS Settings
INSTANCE_TYPE="g6e.xlarge"

# Model (base model to test)
MODEL="meta-llama/Llama-3.2-1B"

# Task
TASK="boolq"

# Trait label
TRAIT_LABEL="correctness"

# Steering method
METHOD="caa"

# Layer
LAYERS="8"

# Device
DEVICE="cuda:0"

# Number of contrastive pairs
NUM_PAIRS="50"

# Output directories
REMOTE_OUTPUT_DIR="/home/ubuntu/output"
LOCAL_OUTPUT_DIR="/home/bc/Desktop/python/wisent/wisent/comparison/comparison_results"

cd /home/bc/Desktop/python/wisent

# Build command that runs all strategies in one session
CMD="for STRATEGY in chat_mean chat_first chat_last chat_gen_point chat_max_norm chat_weighted role_play mc_balanced; do
    echo '=== Testing strategy:' \$STRATEGY '===';
    wisent generate-vector-from-task \
        --task $TASK \
        --trait-label $TRAIT_LABEL \
        --model $MODEL \
        --num-pairs $NUM_PAIRS \
        --layers $LAYERS \
        --method $METHOD \
        --device $DEVICE \
        --extraction-strategy \$STRATEGY \
        --output $REMOTE_OUTPUT_DIR/${MODEL##*/}_${TASK}_${METHOD}_\${STRATEGY}_vector.json \
        --accept-low-quality-vector \
        --verbose;
done"

echo ""
echo "==========================================="
echo "Running all strategies on AWS (single session)"
echo "==========================================="
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Method: $METHOD"
echo "Strategies: chat_mean chat_first chat_last chat_gen_point chat_max_norm chat_weighted role_play mc_balanced"
echo "==========================================="
echo ""

./run_on_aws.sh \
    --instance-type "$INSTANCE_TYPE" \
    "$CMD" \
    "$LOCAL_OUTPUT_DIR"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "==========================================="
    echo "SUCCESS - All strategies completed"
    echo "Results in: $LOCAL_OUTPUT_DIR"
    echo "==========================================="
else
    echo ""
    echo "==========================================="
    echo "FAILED (exit code $EXIT_CODE)"
    echo "==========================================="
fi

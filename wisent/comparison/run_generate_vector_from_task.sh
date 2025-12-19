#!/bin/bash

# ===========================================
# Configuration - Edit these parameters
# ===========================================

# AWS Settings
INSTANCE_TYPE="g6e.xlarge"

# Model
MODEL="meta-llama/Llama-3.2-1B"

# Task
TASK="boolq"

# Trait label
TRAIT_LABEL="correctness"

# Steering method: caa, prism, pulse, titan
METHOD="caa"

# Layer(s) for steering
LAYERS="8"

# Device
DEVICE="cuda:0"

# Number of contrastive pairs for steering vector
NUM_PAIRS="50"

# Extraction strategy: chat_mean, chat_last, chat_first, chat_gen_point, chat_max_norm, chat_weighted, role_play, mc_balanced
EXTRACTION_STRATEGY="mc_balanced"

# Output directories
REMOTE_OUTPUT_DIR="/home/ubuntu/output"
LOCAL_OUTPUT_DIR="/home/bc/Desktop/python/wisent/wisent/comparison/comparison_results"

# Output filename
OUTPUT_FILENAME="${MODEL##*/}_${TASK}_${METHOD}_vector.json"

# ===========================================
# Run - Don't edit below unless necessary
# ===========================================

cd /home/bc/Desktop/python/wisent

echo ""
echo "=========================================="
echo "Running generate-vector-from-task on AWS"
echo "=========================================="
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Method: $METHOD"
echo "Layers: $LAYERS"
echo "Num pairs: $NUM_PAIRS"
echo "Extraction strategy: $EXTRACTION_STRATEGY"
echo "=========================================="
echo ""

./run_on_aws.sh \
    --instance-type "$INSTANCE_TYPE" \
    "wisent generate-vector-from-task \
        --task $TASK \
        --trait-label $TRAIT_LABEL \
        --model $MODEL \
        --num-pairs $NUM_PAIRS \
        --layers $LAYERS \
        --method $METHOD \
        --device $DEVICE \
        --extraction-strategy $EXTRACTION_STRATEGY \
        --output $REMOTE_OUTPUT_DIR/$OUTPUT_FILENAME \
        --accept-low-quality-vector \
        --verbose" \
    "$LOCAL_OUTPUT_DIR"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "SUCCESS"
    echo "Output saved to: $LOCAL_OUTPUT_DIR/$OUTPUT_FILENAME"
    echo "=========================================="
    exit 0
else
    echo ""
    echo "=========================================="
    echo "FAILED (exit code $EXIT_CODE)"
    echo "=========================================="
    exit 1
fi

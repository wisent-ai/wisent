#!/bin/bash

# ===========================================
# Configuration - Edit these parameters
# ===========================================

# AWS Settings
INSTANCE_TYPE="g6e.2xlarge"

# Model
MODEL="google/gemma-2-9b"

# Layer index
LAYER="12"

# Number of text samples
NUM_SAMPLES="1000"

# Number of top BOS features to detect
TOP_K="12"

# Batch size
BATCH_SIZE="1"

# Device
DEVICE="cuda:0"

# Output directories
REMOTE_OUTPUT_DIR="/home/ubuntu/output"
LOCAL_OUTPUT_DIR="/home/bc/Desktop/python/wisent/wisent/comparison/results"

# Retry settings
MAX_RETRIES=30
RETRY_DELAY=5  # seconds between retries

# ===========================================
# Run - Don't edit below unless necessary
# ===========================================

cd /home/bc/Desktop/python/wisent

for ((attempt=1; attempt<=MAX_RETRIES; attempt++)); do
    echo ""
    echo "=========================================="
    echo "Attempt $attempt of $MAX_RETRIES"
    echo "=========================================="
    echo ""

    ./run_on_aws.sh \
        --model "$MODEL" \
        --instance-type "$INSTANCE_TYPE" \
        "python -m wisent.comparison.detect_bos_features \
            --model $MODEL \
            --layer $LAYER \
            --num-samples $NUM_SAMPLES \
            --top-k $TOP_K \
            --batch-size $BATCH_SIZE \
            --device $DEVICE \
            --output-dir $REMOTE_OUTPUT_DIR" \
        "$LOCAL_OUTPUT_DIR"

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "SUCCESS on attempt $attempt"
        echo "=========================================="
        exit 0
    fi

    echo ""
    echo "=========================================="
    echo "FAILED (exit code $EXIT_CODE) - retrying in ${RETRY_DELAY}s..."
    echo "=========================================="
    sleep $RETRY_DELAY
done

echo ""
echo "=========================================="
echo "FAILED after $MAX_RETRIES attempts"
echo "=========================================="
exit 1

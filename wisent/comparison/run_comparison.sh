#!/bin/bash

# ===========================================
# Configuration - Edit these parameters
# ===========================================

# AWS Settings
INSTANCE_TYPE="g6e.xlarge"

# Model
MODEL="google/gemma-2-2b"

# Tasks (comma-separated)
TASKS="boolq"

# Steering method: caa, sae, fgaa
METHODS="fgaa,caa"

# Steering scales (comma-separated)
SCALES="2.5"

# Layer(s) for steering
LAYERS="12"

# Device
DEVICE="cuda:0"

# Batch sizes
BATCH_SIZE="1"
MAX_BATCH_SIZE="1"

# Train/test split ratio (0.4 = 40% train, 60% test)
TRAIN_RATIO="0.4"

# Evaluation limit (number of examples, or empty for all)
LIMIT="6"

# Number of contrastive pairs for steering vector
NUM_PAIRS="6"

# Output directories
REMOTE_OUTPUT_DIR="/home/ubuntu/output"
LOCAL_OUTPUT_DIR="/home/bc/Desktop/python/wisent/wisent/comparison/comparison_results"

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
        --instance-type "$INSTANCE_TYPE" \
        "python -m wisent.comparison.main \
            --model $MODEL \
            --tasks $TASKS \
            --methods $METHODS \
            --scales $SCALES \
            --layers $LAYERS \
            --device $DEVICE \
            --batch-size $BATCH_SIZE \
            --max-batch-size $MAX_BATCH_SIZE \
            --output-dir $REMOTE_OUTPUT_DIR \
            --train-ratio $TRAIN_RATIO \
            --limit $LIMIT \
            --num-pairs $NUM_PAIRS" \
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

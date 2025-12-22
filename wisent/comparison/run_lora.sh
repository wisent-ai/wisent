#!/bin/bash

# ===========================================
# Configuration - Edit these parameters
# ===========================================

# AWS Settings
INSTANCE_TYPE="g6e.xlarge"

# Model
MODEL="google/gemma-2-2b"

# Task
TASK="boolq"

# Device
DEVICE="cuda:0"

# LoRA parameters
LORA_R="16"
LORA_ALPHA="32"
LORA_DROPOUT="0.05"
LEARNING_RATE="2e-4"
NUM_EPOCHS="3"
BATCH_SIZE="4"
MAX_LENGTH="512"

# Number of training examples
NUM_PAIRS="200"

# Train/test split ratio (0.4 = 40% train, 60% test)
TRAIN_RATIO="0.4"

# Evaluation settings
EVAL_BATCH_SIZE="1"
EVAL_LIMIT="600"

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
        --model "$MODEL" \
        --instance-type "$INSTANCE_TYPE" \
        "python -m wisent.comparison.lora \
            --model $MODEL \
            --task $TASK \
            --device $DEVICE \
            --output-dir $REMOTE_OUTPUT_DIR \
            --num-pairs $NUM_PAIRS \
            --lora-r $LORA_R \
            --lora-alpha $LORA_ALPHA \
            --lora-dropout $LORA_DROPOUT \
            --learning-rate $LEARNING_RATE \
            --num-epochs $NUM_EPOCHS \
            --batch-size $BATCH_SIZE \
            --max-length $MAX_LENGTH \
            --train-ratio $TRAIN_RATIO \
            --eval-batch-size $EVAL_BATCH_SIZE \
            --eval-limit $EVAL_LIMIT" \
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

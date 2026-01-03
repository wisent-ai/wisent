#!/bin/bash

# ===========================================
# Configuration - Edit these parameters
# ===========================================

# AWS Settings
INSTANCE_TYPE="g6e.2xlarge"

# Model
MODEL="meta-llama/Llama-3.1-8B"

# Task
TASK="cb"

# Device
DEVICE="cuda:0"

# ReFT parameters (LoReFT)
LOW_RANK_DIMENSION="4"  # Very small! LoReFT is 10-50x more efficient than LoRA
INTERVENTION_LAYERS="16"  # Middle layer for Llama 8B (32 layers)
LEARNING_RATE="5e-4"
NUM_EPOCHS="10"
BATCH_SIZE="2"
MAX_LENGTH="1024"

# Number of training examples
NUM_PAIRS="100"

# Train/test split ratio (0.8 = 80% train, 20% test)
TRAIN_RATIO="0.8"

# Evaluation settings
EVAL_BATCH_SIZE="1"
EVAL_MAX_BATCH_SIZE="1"
EVAL_LIMIT="100"

# ReFT + Steering settings (set WITH_STEERING="true" to enable)
WITH_STEERING="false"
STEERING_METHOD="caa"  # caa
STEERING_LAYERS="16"
STEERING_NUM_PAIRS="50"
STEERING_SCALES="1.0,2.0,4.0,6.0,8.0,10.0"
EXTRACTION_STRATEGY="mc_completion"

# Output directories
REMOTE_OUTPUT_DIR="/home/ubuntu/output"
LOCAL_OUTPUT_DIR="/home/bc/Desktop/python/wisent/wisent/comparison/results/reft"

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

    # Build command
    CMD="python -m wisent.comparison.reft \
        --model $MODEL \
        --task $TASK \
        --device $DEVICE \
        --output-dir $REMOTE_OUTPUT_DIR \
        --num-pairs $NUM_PAIRS \
        --low-rank-dimension $LOW_RANK_DIMENSION \
        --intervention-layers $INTERVENTION_LAYERS \
        --learning-rate $LEARNING_RATE \
        --num-epochs $NUM_EPOCHS \
        --batch-size $BATCH_SIZE \
        --max-length $MAX_LENGTH \
        --train-ratio $TRAIN_RATIO \
        --eval-batch-size $EVAL_BATCH_SIZE \
        --eval-max-batch-size $EVAL_MAX_BATCH_SIZE"

    # Add eval-limit only if set
    if [ -n "$EVAL_LIMIT" ]; then
        CMD="$CMD --eval-limit $EVAL_LIMIT"
    fi

    # Add steering options if enabled
    if [ "$WITH_STEERING" = "true" ]; then
        CMD="$CMD --with-steering \
            --steering-method $STEERING_METHOD \
            --steering-layers $STEERING_LAYERS \
            --steering-num-pairs $STEERING_NUM_PAIRS \
            --steering-scales $STEERING_SCALES \
            --extraction-strategy $EXTRACTION_STRATEGY"
    fi

    ./run_on_aws.sh \
        --model "$MODEL" \
        --instance-type "$INSTANCE_TYPE" \
        "$CMD" \
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

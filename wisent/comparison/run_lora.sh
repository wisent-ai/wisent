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

# LoRA parameters
LORA_R="16"
LORA_ALPHA="32"
LORA_DROPOUT="0.05"
LEARNING_RATE="5e-5"  # Lower to reduce catastrophic forgetting
NUM_EPOCHS="10"
BATCH_SIZE="1"
MAX_LENGTH="1024"

# Number of training examples (use full 80% of pooled data)
NUM_PAIRS="300"

# Train/test split ratio (0.8 = 80% train, 20% test)
TRAIN_RATIO="0.8"

# Evaluation settings
EVAL_BATCH_SIZE="1"
EVAL_MAX_BATCH_SIZE="1"
# LoRA + Steering settings (set WITH_STEERING="true" to enable)
WITH_STEERING="true"
STEERING_METHOD="caa"  # caa
STEERING_LAYERS="16"
STEERING_NUM_PAIRS="50"
STEERING_SCALES="1.0,2.0,4.0,6.0,8.0,10.0"
EXTRACTION_STRATEGY="mc_completion"

# Output directories
REMOTE_OUTPUT_DIR="/home/ubuntu/output"
LOCAL_OUTPUT_DIR="/home/bc/Desktop/python/wisent/wisent/comparison/results/lora"

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
    CMD="python -m wisent.comparison.lora \
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
        --eval-max-batch-size $EVAL_MAX_BATCH_SIZE"

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

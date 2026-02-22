#!/bin/bash

# Comprehensive Classifier Training Script
# This script demonstrates all available arguments for training classifiers with the Wisent framework

set -e

# Configuration
OUTPUT_DIR="./results/classifier_comparison"
MODEL="meta-llama/Llama-3.2-1B-Instruct"
DEVICE="cpu"
TASK="truthfulqa_mc1"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=================================================="
echo "Classifier Training - All Available Arguments"
echo "=================================================="
echo ""

# =============================================================================
# SECTION 1: Basic Classifier Types
# =============================================================================

echo "SECTION 1: Testing Different Classifier Types"
echo "----------------------------------------------"

# 1.1 Logistic Regression Classifier
echo "1.1 Training Logistic Regression Classifier"
python -m wisent.core.main tasks "$TASK" \
    --model "$MODEL" \
    --layer 15 \
    --classifier-type logistic \
    --limit 50 \
    --device "$DEVICE" \
    --output "$OUTPUT_DIR/logistic" \
    --save-classifier "$OUTPUT_DIR/models/logistic_classifier.pt" \
    --verbose
echo ""

# 1.2 MLP (Multi-Layer Perceptron) Classifier
echo "1.2 Training MLP Classifier"
python -m wisent.core.main tasks "$TASK" \
    --model "$MODEL" \
    --layer 15 \
    --classifier-type mlp \
    --limit 50 \
    --device "$DEVICE" \
    --output "$OUTPUT_DIR/mlp" \
    --save-classifier "$OUTPUT_DIR/models/mlp_classifier.pt" \
    --verbose
echo ""

# =============================================================================
# SECTION 2: Token Aggregation Strategies
# =============================================================================

echo ""
echo "SECTION 2: Testing Token Aggregation Strategies"
echo "------------------------------------------------"

AGGREGATION_STRATEGIES=("average" "final" "first" "max" "min")

for agg in "${AGGREGATION_STRATEGIES[@]}"; do
    echo "2.${agg} Training with --token-aggregation $agg"
    python -m wisent.core.main tasks "$TASK" \
        --model "$MODEL" \
        --layer 15 \
        --classifier-type logistic \
        --token-aggregation "$agg" \
        --limit 30 \
        --device "$DEVICE" \
        --output "$OUTPUT_DIR/aggregation_${agg}" \
        --save-classifier "$OUTPUT_DIR/models/classifier_agg_${agg}.pt" \
        --verbose
    echo ""
done

# =============================================================================
# SECTION 3: Layer Selection
# =============================================================================

echo ""
echo "SECTION 3: Testing Different Layer Configurations"
echo "--------------------------------------------------"

# 3.1 Single Layer
echo "3.1 Training on Single Layer (layer 15)"
python -m wisent.core.main tasks "$TASK" \
    --model "$MODEL" \
    --layer 15 \
    --classifier-type logistic \
    --limit 30 \
    --device "$DEVICE" \
    --output "$OUTPUT_DIR/single_layer" \
    --verbose
echo ""

# 3.2 Different Single Layers
for layer in 8 12 20; do
    echo "3.2.${layer} Training on Layer $layer"
    python -m wisent.core.main tasks "$TASK" \
        --model "$MODEL" \
        --layer "$layer" \
        --classifier-type logistic \
        --limit 20 \
        --device "$DEVICE" \
        --output "$OUTPUT_DIR/layer_${layer}" \
        --save-classifier "$OUTPUT_DIR/models/classifier_layer_${layer}.pt" \
        --verbose
    echo ""
done

# =============================================================================
# SECTION 4: Prompt Construction & Token Targeting Strategies
# =============================================================================

echo ""
echo "SECTION 4: Testing Prompt Construction & Token Targeting"
echo "---------------------------------------------------------"

# Prompt Construction Strategies
PROMPT_STRATEGIES=("multiple_choice" "role_playing" "direct_completion" "instruction_following")

for prompt in "${PROMPT_STRATEGIES[@]}"; do
    echo "4.1.${prompt} Testing --prompt-construction-strategy $prompt"
    python -m wisent.core.main tasks "$TASK" \
        --model "$MODEL" \
        --layer 15 \
        --classifier-type logistic \
        --prompt-construction-strategy "$prompt" \
        --limit 20 \
        --device "$DEVICE" \
        --output "$OUTPUT_DIR/prompt_${prompt}" \
        --verbose
    echo ""
done

# Token Targeting Strategies
TOKEN_STRATEGIES=("choice_token" "continuation_token" "last_token" "first_token" "mean_pooling" "max_pooling")

for token_strat in "${TOKEN_STRATEGIES[@]}"; do
    echo "4.2.${token_strat} Testing --token-targeting-strategy $token_strat"
    python -m wisent.core.main tasks "$TASK" \
        --model "$MODEL" \
        --layer 15 \
        --classifier-type logistic \
        --token-targeting-strategy "$token_strat" \
        --limit 20 \
        --device "$DEVICE" \
        --output "$OUTPUT_DIR/token_target_${token_strat}" \
        --verbose
    echo ""
done

# =============================================================================
# SECTION 5: Detection Threshold Variations
# =============================================================================

echo ""
echo "SECTION 5: Testing Different Detection Thresholds"
echo "--------------------------------------------------"

THRESHOLDS=("0.3" "0.5" "0.6" "0.7" "0.9")

for thresh in "${THRESHOLDS[@]}"; do
    echo "5.${thresh} Testing --detection-threshold $thresh"
    python -m wisent.core.main tasks "$TASK" \
        --model "$MODEL" \
        --layer 15 \
        --classifier-type logistic \
        --detection-threshold "$thresh" \
        --limit 20 \
        --device "$DEVICE" \
        --output "$OUTPUT_DIR/threshold_${thresh}" \
        --save-classifier "$OUTPUT_DIR/models/classifier_thresh_${thresh}.pt" \
        --verbose
    echo ""
done

# =============================================================================
# SECTION 6: Split Ratio & Data Limits
# =============================================================================

echo ""
echo "SECTION 6: Testing Data Split & Limit Options"
echo "----------------------------------------------"

# 6.1 Different split ratios
echo "6.1 Testing different --split-ratio values"
for ratio in 0.6 0.7 0.8 0.9; do
    echo "   Split ratio: $ratio"
    python -m wisent.core.main tasks "$TASK" \
        --model "$MODEL" \
        --layer 15 \
        --classifier-type logistic \
        --split-ratio "$ratio" \
        --limit 30 \
        --device "$DEVICE" \
        --output "$OUTPUT_DIR/split_${ratio}" \
        --verbose
    echo ""
done

# 6.2 Separate training and testing limits
echo "6.2 Testing --training-limit and --testing-limit"
python -m wisent.core.main tasks "$TASK" \
    --model "$MODEL" \
    --layer 15 \
    --classifier-type logistic \
    --training-limit 40 \
    --testing-limit 10 \
    --device "$DEVICE" \
    --output "$OUTPUT_DIR/separate_limits" \
    --verbose
echo ""

# =============================================================================
# SECTION 7: Optimization Options
# =============================================================================

echo ""
echo "SECTION 7: Testing Hyperparameter Optimization"
echo "-----------------------------------------------"

# 7.1 Basic optimization
echo "7.1 Hyperparameter optimization (all layers)"
python -m wisent.core.main tasks "$TASK" \
    --model "$MODEL" \
    --layer 15 \
    --classifier-type logistic \
    --optimize \
    --optimize-layers all \
    --optimize-metric f1 \
    --optimize-max-combinations 20 \
    --limit 30 \
    --device "$DEVICE" \
    --output "$OUTPUT_DIR/optimize_all" \
    --verbose
echo ""

# 7.2 Layer range optimization
echo "7.2 Optimization with specific layer range"
python -m wisent.core.main tasks "$TASK" \
    --model "$MODEL" \
    --layer 15 \
    --classifier-type logistic \
    --optimize \
    --optimize-layers "8-20" \
    --optimize-metric accuracy \
    --optimize-max-combinations 15 \
    --limit 30 \
    --device "$DEVICE" \
    --output "$OUTPUT_DIR/optimize_range" \
    --verbose
echo ""

# 7.3 Different optimization metrics
for metric in accuracy f1 precision recall auc; do
    echo "7.3.${metric} Optimizing for --optimize-metric $metric"
    python -m wisent.core.main tasks "$TASK" \
        --model "$MODEL" \
        --layer 15 \
        --classifier-type logistic \
        --optimize \
        --optimize-layers "10,15,20" \
        --optimize-metric "$metric" \
        --optimize-max-combinations 10 \
        --limit 20 \
        --device "$DEVICE" \
        --output "$OUTPUT_DIR/optimize_${metric}" \
        --verbose
    echo ""
done

# =============================================================================
# SECTION 8: Train-Only and Inference-Only Modes
# =============================================================================

echo ""
echo "SECTION 8: Testing Training and Inference Modes"
echo "------------------------------------------------"

# 8.1 Train-only mode
echo "8.1 Train-only mode (saves model, skips inference)"
python -m wisent.core.main tasks "$TASK" \
    --model "$MODEL" \
    --layer 15 \
    --classifier-type logistic \
    --train-only \
    --save-classifier "$OUTPUT_DIR/models/train_only_classifier.pt" \
    --limit 30 \
    --device "$DEVICE" \
    --output "$OUTPUT_DIR/train_only" \
    --verbose
echo ""

# 8.2 Inference-only mode (requires pre-trained classifier)
echo "8.2 Inference-only mode (loads pre-trained classifier)"
python -m wisent.core.main tasks "$TASK" \
    --model "$MODEL" \
    --layer 15 \
    --classifier-type logistic \
    --inference-only \
    --load-classifier "$OUTPUT_DIR/models/train_only_classifier.pt" \
    --limit 10 \
    --device "$DEVICE" \
    --output "$OUTPUT_DIR/inference_only" \
    --verbose
echo ""

# =============================================================================
# SECTION 9: Normalization Options
# =============================================================================

echo ""
echo "SECTION 9: Testing Normalization Methods"
echo "-----------------------------------------"

NORM_METHODS=("none" "l2_unit" "cross_behavior" "layer_wise_mean")

for norm in "${NORM_METHODS[@]}"; do
    echo "9.${norm} Testing --normalization-method $norm"
    python -m wisent.core.main tasks "$TASK" \
        --model "$MODEL" \
        --layer 15 \
        --classifier-type logistic \
        --normalization-method "$norm" \
        --limit 20 \
        --device "$DEVICE" \
        --output "$OUTPUT_DIR/norm_${norm}" \
        --verbose
    echo ""
done

# =============================================================================
# SECTION 10: Performance Monitoring
# =============================================================================

echo ""
echo "SECTION 10: Testing Performance Monitoring Options"
echo "---------------------------------------------------"

echo "10.1 Training with full performance monitoring"
python -m wisent.core.main tasks "$TASK" \
    --model "$MODEL" \
    --layer 15 \
    --classifier-type logistic \
    --enable-memory-tracking \
    --enable-latency-tracking \
    --track-gpu-memory \
    --detailed-performance-report \
    --export-performance-csv "$OUTPUT_DIR/performance.csv" \
    --show-memory-usage \
    --show-timing-summary \
    --limit 20 \
    --device "$DEVICE" \
    --output "$OUTPUT_DIR/performance_monitoring" \
    --verbose
echo ""

# =============================================================================
# SECTION 11: Activation Saving/Loading
# =============================================================================

echo ""
echo "SECTION 11: Testing Activation Caching"
echo "---------------------------------------"

# 11.1 Save test activations
echo "11.1 Training and saving test activations"
python -m wisent.core.main tasks "$TASK" \
    --model "$MODEL" \
    --layer 15 \
    --classifier-type logistic \
    --save-test-activations "$OUTPUT_DIR/test_activations.npy" \
    --limit 30 \
    --device "$DEVICE" \
    --output "$OUTPUT_DIR/save_activations" \
    --verbose
echo ""

# 11.2 Load test activations
echo "11.2 Training using pre-saved test activations"
python -m wisent.core.main tasks "$TASK" \
    --model "$MODEL" \
    --layer 15 \
    --classifier-type logistic \
    --load-test-activations "$OUTPUT_DIR/test_activations.npy" \
    --limit 30 \
    --device "$DEVICE" \
    --output "$OUTPUT_DIR/load_activations" \
    --verbose
echo ""

# =============================================================================
# SECTION 12: Advanced Combinations
# =============================================================================

echo ""
echo "SECTION 12: Testing Advanced Argument Combinations"
echo "---------------------------------------------------"

# 12.1 Full-featured training
echo "12.1 Full-featured training with multiple options"
python -m wisent.core.main tasks "$TASK" \
    --model "$MODEL" \
    --layer 15 \
    --classifier-type mlp \
    --token-aggregation average \
    --prompt-construction-strategy multiple_choice \
    --token-targeting-strategy choice_token \
    --detection-threshold 0.6 \
    --split-ratio 0.8 \
    --training-limit 40 \
    --testing-limit 10 \
    --normalization-method l2_unit \
    --enable-memory-tracking \
    --enable-latency-tracking \
    --save-classifier "$OUTPUT_DIR/models/full_featured_classifier.pt" \
    --device "$DEVICE" \
    --output "$OUTPUT_DIR/full_featured" \
    --csv-output "$OUTPUT_DIR/full_featured/results.csv" \
    --seed 42 \
    --verbose
echo ""

# 12.2 Optimized training with best practices
echo "12.2 Optimized training with best practices"
python -m wisent.core.main tasks "$TASK" \
    --model "$MODEL" \
    --layer 15 \
    --classifier-type logistic \
    --optimize \
    --optimize-layers "10-20" \
    --optimize-metric f1 \
    --optimize-max-combinations 30 \
    --token-aggregation average \
    --normalization-method cross_behavior \
    --save-classifier "$OUTPUT_DIR/models/optimized_best_practices.pt" \
    --save-test-activations "$OUTPUT_DIR/optimized_activations.npy" \
    --limit 50 \
    --device "$DEVICE" \
    --output "$OUTPUT_DIR/optimized_best_practices" \
    --verbose
echo ""

# =============================================================================
# SECTION 13: Seeds and Reproducibility
# =============================================================================

echo ""
echo "SECTION 13: Testing Seed and Reproducibility"
echo "---------------------------------------------"

for seed in 42 123 456; do
    echo "13.${seed} Training with --seed $seed"
    python -m wisent.core.main tasks "$TASK" \
        --model "$MODEL" \
        --layer 15 \
        --classifier-type logistic \
        --seed "$seed" \
        --limit 20 \
        --device "$DEVICE" \
        --output "$OUTPUT_DIR/seed_${seed}" \
        --save-classifier "$OUTPUT_DIR/models/classifier_seed_${seed}.pt" \
        --verbose
    echo ""
done

# =============================================================================
# SECTION 14: Output Options
# =============================================================================

echo ""
echo "SECTION 14: Testing Different Output Options"
echo "---------------------------------------------"

echo "14.1 Training with CSV output"
python -m wisent.core.main tasks "$TASK" \
    --model "$MODEL" \
    --layer 15 \
    --classifier-type logistic \
    --csv-output "$OUTPUT_DIR/csv_results/results.csv" \
    --limit 20 \
    --device "$DEVICE" \
    --output "$OUTPUT_DIR/csv_output" \
    --verbose
echo ""

echo "14.2 Training with evaluation report"
python -m wisent.core.main tasks "$TASK" \
    --model "$MODEL" \
    --layer 15 \
    --classifier-type logistic \
    --evaluation-report "$OUTPUT_DIR/evaluation_report.json" \
    --limit 20 \
    --device "$DEVICE" \
    --output "$OUTPUT_DIR/eval_report" \
    --verbose
echo ""

echo ""
echo "=================================================="
echo " All Classifier Training Tests Completed!"
echo "=================================================="
echo ""
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Summary of tested configurations:"
echo ""
echo "Classifier Types:"
echo "  - logistic, mlp"
echo ""
echo "Token Aggregation:"
echo "  - average, final, first, max, min"
echo ""
echo "Prompt Construction:"
echo "  - multiple_choice, role_playing, direct_completion, instruction_following"
echo ""
echo "Token Targeting:"
echo "  - choice_token, continuation_token, last_token, first_token, mean_pooling, max_pooling"
echo ""
echo "Normalization Methods:"
echo "  - none, l2_unit, cross_behavior, layer_wise_mean"
echo ""
echo "Optimization Metrics:"
echo "  - accuracy, f1, precision, recall, auc"
echo ""
echo "Special Modes:"
echo "  - train-only, inference-only, optimization, performance monitoring"
echo ""

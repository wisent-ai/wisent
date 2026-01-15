#!/bin/bash
# Benchmark TruthfulQA performance: Base Qwen3-8B vs TITAN-steered

set -e

MODEL_BASE="Qwen/Qwen3-8B"
MODEL_TITAN="wisent-ai/Qwen3-8B-TruthfulQA-TITAN"
TASK="truthfulqa_custom"
NUM_QUESTIONS=100
OUTPUT_DIR="./benchmark_results"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Step 1: Generate responses from BASE model"
echo "=========================================="
python3.11 -m wisent.core.main generate-responses "$MODEL_BASE" \
    --task "$TASK" \
    --num-questions "$NUM_QUESTIONS" \
    --output "$OUTPUT_DIR/base_responses.json" \
    --disable-thinking \
    --verbose

echo ""
echo "=========================================="
echo "Step 2: Generate responses from TITAN model"
echo "=========================================="
python3.11 -m wisent.core.main generate-responses "$MODEL_TITAN" \
    --task "$TASK" \
    --num-questions "$NUM_QUESTIONS" \
    --output "$OUTPUT_DIR/titan_responses.json" \
    --disable-thinking \
    --verbose

echo ""
echo "=========================================="
echo "Step 3: Evaluate BASE model responses"
echo "=========================================="
python3.11 -m wisent.core.main evaluate-responses \
    --input "$OUTPUT_DIR/base_responses.json" \
    --output "$OUTPUT_DIR/base_eval.json" \
    --verbose

echo ""
echo "=========================================="
echo "Step 4: Evaluate TITAN model responses"
echo "=========================================="
python3.11 -m wisent.core.main evaluate-responses \
    --input "$OUTPUT_DIR/titan_responses.json" \
    --output "$OUTPUT_DIR/titan_eval.json" \
    --verbose

echo ""
echo "=========================================="
echo "RESULTS SUMMARY"
echo "=========================================="
echo "Base model:"
cat "$OUTPUT_DIR/base_eval.json"
echo ""
echo "TITAN model:"
cat "$OUTPUT_DIR/titan_eval.json"

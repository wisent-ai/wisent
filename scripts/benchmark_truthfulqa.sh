#!/bin/bash
# Benchmark TruthfulQA performance: Base Qwen3-8B vs GROM-steered

set -e

MODEL_BASE="Qwen/Qwen3-8B"
MODEL_GROM="wisent-ai/Qwen3-8B-TruthfulQA-GROM"
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
echo "Step 2: Generate responses from GROM model"
echo "=========================================="
python3.11 -m wisent.core.main generate-responses "$MODEL_GROM" \
    --task "$TASK" \
    --num-questions "$NUM_QUESTIONS" \
    --output "$OUTPUT_DIR/grom_responses.json" \
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
echo "Step 4: Evaluate GROM model responses"
echo "=========================================="
python3.11 -m wisent.core.main evaluate-responses \
    --input "$OUTPUT_DIR/grom_responses.json" \
    --output "$OUTPUT_DIR/grom_eval.json" \
    --verbose

echo ""
echo "=========================================="
echo "RESULTS SUMMARY"
echo "=========================================="
echo "Base model:"
cat "$OUTPUT_DIR/base_eval.json"
echo ""
echo "GROM model:"
cat "$OUTPUT_DIR/grom_eval.json"

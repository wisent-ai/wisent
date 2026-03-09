#!/bin/bash
# Run quality metrics sweep across multiple benchmarks
# This script runs the optimization pipeline for each benchmark and collects
# quality metrics alongside steering effectiveness (delta) for correlation analysis.
#
# Output: all_trials_metrics_{timestamp}.json for each benchmark in /home/ubuntu/output/
#
# Features:
# - Saves intermediate results after each benchmark to GCS
# - Supports resuming from last completed benchmark
# - Continues on individual benchmark failures (doesn't abort entire sweep)
#
# Usage:
#   ./run_quality_metrics_sweep.sh

# Don't exit on error - we want to continue with other benchmarks
set -uo pipefail

# Configuration
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/ubuntu/output}"
LAYER_RANGE="${LAYER_RANGE:-0-23}"
GCS_BUCKET="${GCS_BUCKET:-wisent-images-bucket}"

# Progress tracking file
PROGRESS_FILE="$OUTPUT_DIR/.sweep_progress"

# Source helper functions (save_intermediate_results, is_benchmark_completed, etc.)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_helpers/sweep_helpers.sh
source "$SCRIPT_DIR/_helpers/sweep_helpers.sh"

# Benchmarks to test (these have meaningful correct/incorrect answer pairs)
BENCHMARKS=(
    "gsm8k"
    "arc_easy"
    "arc_challenge"
    "hellaswag"
    "winogrande"
    "truthfulqa_mc1"
    "piqa"
    "boolq"
    "openbookqa"
    "livecodebench"
)

# Synthetic steering types for validation:
# - "british" = meaningful steering (British vs American English - should have good metrics AND show steering effect)
# - "random" = random pairs (should have BAD metrics AND NO steering effect)
# These validate which metrics actually predict steering effectiveness
SYNTHETIC_TYPES=(
    "british"
    "random"
)

echo "=========================================="
echo "Quality Metrics Sweep"
echo "=========================================="
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo "Layer range: $LAYER_RANGE"
echo "Benchmarks: ${BENCHMARKS[*]}"
echo "Synthetic types: ${SYNTHETIC_TYPES[*]}"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ==========================================
# Part 1: Run optimization for each BENCHMARK task
# ==========================================
echo ""
echo "=========================================="
echo "Part 1: Benchmark Tasks"
echo "=========================================="

FAILED_BENCHMARKS=()
COMPLETED_BENCHMARKS=()

for BENCHMARK in "${BENCHMARKS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running: $BENCHMARK"
    echo "=========================================="

    # Skip if already completed (resume support)
    if is_benchmark_completed "$BENCHMARK"; then
        echo "SKIPPING: $BENCHMARK already completed (found ${BENCHMARK}_metrics.json)"
        COMPLETED_BENCHMARKS+=("$BENCHMARK")
        continue
    fi

    BENCHMARK_START=$(date +%s)

    # Run the optimization using wisent CLI with baseline comparison
    if wisent optimize-steering comprehensive "$MODEL" \
        --tasks "$BENCHMARK" \
        --compute-baseline \
        --device cuda \
        --output-dir "$OUTPUT_DIR/$BENCHMARK" \
        2>&1 | tee "$OUTPUT_DIR/${BENCHMARK}_log.txt"; then

        BENCHMARK_END=$(date +%s)
        DURATION=$((BENCHMARK_END - BENCHMARK_START))
        echo "Completed $BENCHMARK in ${DURATION}s"

        # Find and copy the results file
        RESULTS_FILE=$(find "$OUTPUT_DIR/$BENCHMARK" -name "steering_comprehensive_*.json" -type f 2>/dev/null | head -1)

        if [ -n "$RESULTS_FILE" ]; then
            echo "Results saved to: $RESULTS_FILE"
            cp "$RESULTS_FILE" "$OUTPUT_DIR/${BENCHMARK}_metrics.json"
            mark_benchmark_completed "$BENCHMARK"
            COMPLETED_BENCHMARKS+=("$BENCHMARK")
        else
            echo "WARNING: No results file found for $BENCHMARK"
            FAILED_BENCHMARKS+=("$BENCHMARK")
        fi
    else
        echo "ERROR: $BENCHMARK failed"
        FAILED_BENCHMARKS+=("$BENCHMARK")
    fi

    # Save intermediate results after each benchmark
    save_intermediate_results
done

# ==========================================
# Part 2: Run SYNTHETIC steering (british, random)
# These use --task personalization with --trait
# ==========================================
echo ""
echo "=========================================="
echo "Part 2: Synthetic Steering Validation"
echo "=========================================="

for SYNTHETIC_TYPE in "${SYNTHETIC_TYPES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running synthetic: $SYNTHETIC_TYPE"
    echo "=========================================="

    # Skip if already completed
    if [ -f "$OUTPUT_DIR/synthetic_${SYNTHETIC_TYPE}_metrics.json" ]; then
        echo "SKIPPING: synthetic_$SYNTHETIC_TYPE already completed"
        COMPLETED_BENCHMARKS+=("synthetic_$SYNTHETIC_TYPE")
        continue
    fi

    SYNTHETIC_START=$(date +%s)
    SYNTHETIC_DIR="$OUTPUT_DIR/synthetic_${SYNTHETIC_TYPE}"

    # Run the optimization with personalization task
    if wisent optimize-steering personalization \
        --model "$MODEL" \
        --trait "$SYNTHETIC_TYPE" \
        --num-pairs 50 \
        --output-dir "$SYNTHETIC_DIR" \
        --device cuda \
        2>&1 | tee "$OUTPUT_DIR/synthetic_${SYNTHETIC_TYPE}_log.txt"; then

        SYNTHETIC_END=$(date +%s)
        DURATION=$((SYNTHETIC_END - SYNTHETIC_START))
        echo "Completed synthetic $SYNTHETIC_TYPE in ${DURATION}s"

        # Find the results file
        RESULTS_FILE=$(find "$SYNTHETIC_DIR" -name "*.json" -type f 2>/dev/null | head -1)

        if [ -n "$RESULTS_FILE" ]; then
            echo "Results saved to: $RESULTS_FILE"
            cp "$RESULTS_FILE" "$OUTPUT_DIR/synthetic_${SYNTHETIC_TYPE}_metrics.json"
            COMPLETED_BENCHMARKS+=("synthetic_$SYNTHETIC_TYPE")
        else
            echo "WARNING: No results file found for synthetic_$SYNTHETIC_TYPE"
            FAILED_BENCHMARKS+=("synthetic_$SYNTHETIC_TYPE")
        fi
    else
        echo "ERROR: synthetic_$SYNTHETIC_TYPE failed"
        FAILED_BENCHMARKS+=("synthetic_$SYNTHETIC_TYPE")
    fi

    # Save intermediate results after each synthetic
    save_intermediate_results
done

# ==========================================
# Part 3: Combine all results
# ==========================================
echo ""
echo "=========================================="
echo "Combining Results"
echo "=========================================="

combine_all_results

echo ""
echo "=========================================="
echo "Sweep Complete!"
echo "=========================================="
echo "Results in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "No JSON files found"
echo ""
echo "Completed benchmarks: ${COMPLETED_BENCHMARKS[*]:-none}"
echo "Failed benchmarks: ${FAILED_BENCHMARKS[*]:-none}"
echo ""

# Final upload to GCS
upload_final_to_gcs

echo "Done!"

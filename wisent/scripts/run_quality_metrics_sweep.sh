#!/bin/bash
# Run quality metrics sweep across multiple benchmarks
# This script runs the optimization pipeline for each benchmark and collects
# quality metrics alongside steering effectiveness (delta) for correlation analysis.
#
# Output: all_trials_metrics_{timestamp}.json for each benchmark in /home/ubuntu/output/
#
# Features:
# - Saves intermediate results after each benchmark to S3
# - Supports resuming from last completed benchmark
# - Continues on individual benchmark failures (doesn't abort entire sweep)
#
# Usage on AWS:
#   ./run_on_aws.sh --model Qwen/Qwen2.5-0.5B-Instruct "bash /opt/wisent-venv/lib/python3.10/site-packages/wisent/scripts/run_quality_metrics_sweep.sh" ./quality_sweep_results

# Don't exit on error - we want to continue with other benchmarks
set -uo pipefail

# Configuration
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/ubuntu/output}"
N_TRIALS="${N_TRIALS:-200}"
TRAIN_LIMIT="${TRAIN_LIMIT:-100}"
VAL_LIMIT="${VAL_LIMIT:-50}"
TEST_LIMIT="${TEST_LIMIT:-50}"
LAYER_RANGE="${LAYER_RANGE:-0-23}"
S3_BUCKET="${S3_BUCKET:-}"  # Optional S3 bucket for intermediate uploads

# Progress tracking file
PROGRESS_FILE="$OUTPUT_DIR/.sweep_progress"

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
echo "Trials per benchmark: $N_TRIALS"
echo "Layer range: $LAYER_RANGE"
echo "Benchmarks: ${BENCHMARKS[*]}"
echo "Synthetic types: ${SYNTHETIC_TYPES[*]}"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ==========================================
# Helper functions
# ==========================================

save_intermediate_results() {
    echo "Saving intermediate results..."
    
    # Combine all completed results so far
    python3 << 'PYEOF'
import json
import glob
import os

output_dir = os.environ.get('OUTPUT_DIR', '/home/ubuntu/output')
combined = {
    "model": os.environ.get('MODEL', 'unknown'),
    "status": "in_progress",
    "benchmarks": {},
    "synthetic": {}
}

for metrics_file in glob.glob(f"{output_dir}/*_metrics.json"):
    basename = os.path.basename(metrics_file).replace('_metrics.json', '')
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        if basename.startswith('synthetic_'):
            combined["synthetic"][basename.replace('synthetic_', '')] = data
        else:
            combined["benchmarks"][basename] = data
    except Exception:
        pass

output_file = f"{output_dir}/intermediate_results.json"
with open(output_file, 'w') as f:
    json.dump(combined, f, indent=2)
print(f"Intermediate results saved: {len(combined['benchmarks'])} benchmarks, {len(combined['synthetic'])} synthetic")
PYEOF
    
    # Upload to S3 if bucket is configured
    if [ -n "$S3_BUCKET" ]; then
        echo "Uploading to S3..."
        aws s3 cp "$OUTPUT_DIR/intermediate_results.json" "s3://$S3_BUCKET/sweep_results/intermediate_results.json" 2>/dev/null || true
        aws s3 sync "$OUTPUT_DIR" "s3://$S3_BUCKET/sweep_results/" --exclude "*.log" 2>/dev/null || true
    fi
}

is_benchmark_completed() {
    local benchmark="$1"
    [ -f "$OUTPUT_DIR/${benchmark}_metrics.json" ]
}

mark_benchmark_completed() {
    local benchmark="$1"
    echo "$benchmark" >> "$PROGRESS_FILE"
}

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
        --limit "$TRAIN_LIMIT" \
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

python3 << 'EOF'
import json
import glob
import os

output_dir = os.environ.get('OUTPUT_DIR', '/home/ubuntu/output')
combined = {
    "model": os.environ.get('MODEL', 'unknown'),
    "benchmarks": {},
    "synthetic": {}
}

# Load benchmark metrics
for metrics_file in glob.glob(f"{output_dir}/*_metrics.json"):
    basename = os.path.basename(metrics_file).replace('_metrics.json', '')
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
            
        n_trials = len(data.get('trials', []))
        baseline = data.get('baseline_accuracy', 'N/A')
        
        if basename.startswith('synthetic_'):
            synthetic_type = basename.replace('synthetic_', '')
            combined["synthetic"][synthetic_type] = data
            print(f"  Synthetic {synthetic_type}: {n_trials} trials, baseline={baseline}")
        else:
            combined["benchmarks"][basename] = data
            print(f"  Benchmark {basename}: {n_trials} trials, baseline={baseline}")
    except Exception as e:
        print(f"  Failed to load {basename}: {e}")

output_file = f"{output_dir}/combined_quality_metrics.json"
with open(output_file, 'w') as f:
    json.dump(combined, f, indent=2)

print(f"\nCombined metrics saved to: {output_file}")
print(f"Benchmarks: {len(combined['benchmarks'])}")
print(f"Synthetic types: {len(combined['synthetic'])}")
EOF

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

# Final upload to S3
if [ -n "$S3_BUCKET" ]; then
    echo "Final upload to S3..."
    aws s3 sync "$OUTPUT_DIR" "s3://$S3_BUCKET/sweep_results/" --exclude "*.log" 2>/dev/null || true
fi

echo "Done!"

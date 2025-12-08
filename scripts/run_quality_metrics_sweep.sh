#!/bin/bash
# Run quality metrics sweep across multiple benchmarks
# This script runs the optimization pipeline for each benchmark and collects
# quality metrics alongside steering effectiveness (delta) for correlation analysis.
#
# Output: all_trials_metrics_{timestamp}.json for each benchmark in /home/ubuntu/output/
#
# Usage on AWS:
#   ./run_on_aws.sh --model Qwen/Qwen2.5-0.5B-Instruct "bash /home/ubuntu/scripts/run_quality_metrics_sweep.sh" ./quality_sweep_results

set -euo pipefail

# Configuration
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/ubuntu/output}"
N_TRIALS="${N_TRIALS:-200}"
TRAIN_LIMIT="${TRAIN_LIMIT:-100}"
VAL_LIMIT="${VAL_LIMIT:-50}"
TEST_LIMIT="${TEST_LIMIT:-50}"
LAYER_RANGE="${LAYER_RANGE:-0-24}"

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
# Part 1: Run optimization for each BENCHMARK task
# ==========================================
echo ""
echo "=========================================="
echo "Part 1: Benchmark Tasks"
echo "=========================================="

for BENCHMARK in "${BENCHMARKS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running: $BENCHMARK"
    echo "=========================================="
    
    BENCHMARK_START=$(date +%s)
    
    # Run the optimization pipeline
    python3 -m wisent.core.optuna.steering.optuna_pipeline \
        --output-dir "$OUTPUT_DIR/$BENCHMARK" \
        --model "$MODEL" \
        --task "$BENCHMARK" \
        --n-trials "$N_TRIALS" \
        --train-limit "$TRAIN_LIMIT" \
        --val-limit "$VAL_LIMIT" \
        --test-limit "$TEST_LIMIT" \
        --layer-range "$LAYER_RANGE" \
        2>&1 | tee "$OUTPUT_DIR/${BENCHMARK}_log.txt"
    
    BENCHMARK_END=$(date +%s)
    DURATION=$((BENCHMARK_END - BENCHMARK_START))
    
    echo "Completed $BENCHMARK in ${DURATION}s"
    
    # Find the metrics file
    METRICS_FILE=$(find "$OUTPUT_DIR/$BENCHMARK" -name "all_trials_metrics_*.json" -type f | head -1)
    
    if [ -n "$METRICS_FILE" ]; then
        echo "Metrics saved to: $METRICS_FILE"
        cp "$METRICS_FILE" "$OUTPUT_DIR/${BENCHMARK}_metrics.json"
    else
        echo "WARNING: No metrics file found for $BENCHMARK"
    fi
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
    
    SYNTHETIC_START=$(date +%s)
    SYNTHETIC_DIR="$OUTPUT_DIR/synthetic_${SYNTHETIC_TYPE}"
    
    # Run the optimization pipeline with personalization task
    python3 -m wisent.core.optuna.steering.optuna_pipeline \
        --output-dir "$SYNTHETIC_DIR" \
        --model "$MODEL" \
        --task personalization \
        --trait "$SYNTHETIC_TYPE" \
        --n-trials "$N_TRIALS" \
        --num-pairs 50 \
        --layer-range "$LAYER_RANGE" \
        2>&1 | tee "$OUTPUT_DIR/synthetic_${SYNTHETIC_TYPE}_log.txt"
    
    SYNTHETIC_END=$(date +%s)
    DURATION=$((SYNTHETIC_END - SYNTHETIC_START))
    echo "Completed synthetic $SYNTHETIC_TYPE in ${DURATION}s"
    
    # Find the metrics file
    METRICS_FILE=$(find "$SYNTHETIC_DIR" -name "all_trials_metrics_*.json" -type f | head -1)
    
    if [ -n "$METRICS_FILE" ]; then
        echo "Metrics saved to: $METRICS_FILE"
        cp "$METRICS_FILE" "$OUTPUT_DIR/synthetic_${SYNTHETIC_TYPE}_metrics.json"
    else
        echo "WARNING: No metrics file found for synthetic_$SYNTHETIC_TYPE"
    fi
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
echo "Done!"

#!/bin/bash
# Helper functions for run_quality_metrics_sweep.sh
# Extracted to keep main script under 300 lines.
#
# Required env vars (set by caller):
#   OUTPUT_DIR, MODEL, GCS_BUCKET, PROGRESS_FILE

save_intermediate_results() {
    echo "Saving intermediate results..."

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

    # Upload to GCS if bucket is configured
    if [ -n "$GCS_BUCKET" ]; then
        echo "Uploading to GCS..."
        gcloud storage cp "$OUTPUT_DIR/intermediate_results.json" "gs://$GCS_BUCKET/sweep_results/intermediate_results.json" 2>/dev/null || true
        gcloud storage rsync "$OUTPUT_DIR" "gs://$GCS_BUCKET/sweep_results/" --exclude="*.log" 2>/dev/null || true
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

upload_final_to_gcs() {
    if [ -n "$GCS_BUCKET" ]; then
        echo "Final upload to GCS..."
        gcloud storage rsync "$OUTPUT_DIR" "gs://$GCS_BUCKET/sweep_results/" --exclude="*.log" 2>/dev/null || true
    fi
}

combine_all_results() {
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
}

#!/bin/bash
set -euo pipefail

# Parallel steering method optimization launcher
# Usage: ./launcher.sh <model_name> <benchmark> [trials_multiplier] [backend] [method1,method2,...]
#
# Each worker loads pairs from storage (HF/Supabase/cache) automatically.
# Launches one GPU instance per method via run_on_gcp.sh --background.
# If methods are specified (comma-separated), only those are launched.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUN_ON_GCP="$REPO_DIR/run_on_gcp.sh"

MODEL_NAME="${1:?Usage: $0 <model_name> <benchmark> [trials_multiplier] [backend] [methods]}"
BENCHMARK="${2:?Usage: $0 <model_name> <benchmark> [trials_multiplier] [backend] [methods]}"
TRIALS_MULTIPLIER="${3:-50}"
BACKEND="${4:-optuna}"
SELECTED_METHODS="${5:-}"

GCS_BUCKET="${GCS_BUCKET:-wisent-gcp-bucket}"
JOB_ID="${JOB_ID:-fbm-$(date +%Y%m%d-%H%M%S)}"
GCS_BASE="gs://$GCS_BUCKET/find_best_method/$JOB_ID"

ALL_METHODS=(caa ostrze mlp tecza tetno grom nurt szlak wicher przelom)

if [[ -n "$SELECTED_METHODS" ]]; then
    IFS=',' read -ra METHODS <<< "$SELECTED_METHODS"
else
    METHODS=("${ALL_METHODS[@]}")
fi

echo "=========================================="
echo "PARALLEL FIND BEST METHOD"
echo "=========================================="
echo "  Model:       $MODEL_NAME"
echo "  Benchmark:   $BENCHMARK"
echo "  Trials/dim:  ${TRIALS_MULTIPLIER}x"
echo "  Backend:     $BACKEND"
echo "  Methods:     ${#METHODS[@]} (${METHODS[*]})"
echo "  Job ID:      $JOB_ID"
echo "  GCS base:    $GCS_BASE"
echo "=========================================="

# Upload worker script to GCS
echo "Uploading worker script to GCS..."
gcloud storage cp "$SCRIPT_DIR/worker.py" "gs://$GCS_BUCKET/scripts/steering/worker.py"

# Launch one GPU instance per method
echo ""
echo "=== Launching parallel method workers ==="

for method in "${METHODS[@]}"; do
    WORKER_CMD="source /opt/venv/bin/activate && pip install --upgrade wisent lm-eval psycopg2-binary && gcloud storage cp gs://$GCS_BUCKET/scripts/steering/worker.py /home/ubuntu/worker.py && export MODEL_NAME='$MODEL_NAME' && export BENCHMARK='$BENCHMARK' && export METHOD='$method' && export JOB_ID='$JOB_ID' && export TRIALS_MULTIPLIER='$TRIALS_MULTIPLIER' && export BACKEND='$BACKEND' && export GCS_BUCKET='$GCS_BUCKET' && python3 /home/ubuntu/worker.py"
    echo "  Launching $method..."
    "$RUN_ON_GCP" --model "$MODEL_NAME" --background "$WORKER_CMD" &
done

wait
echo ""
echo "=========================================="
echo "ALL WORKERS LAUNCHED"
echo "=========================================="
echo "  Job ID:    $JOB_ID"
echo "  GCS base:  $GCS_BASE"
echo ""
echo "Monitor progress:"
echo "  gcloud storage ls $GCS_BASE/methods/"
echo ""
echo "Collect results when all done:"
echo "  python3 scripts/steering/parallel/collect.py $JOB_ID"
echo "=========================================="

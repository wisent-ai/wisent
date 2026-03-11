#!/bin/bash
set -euo pipefail

# Parallel steering method optimization launcher
# Usage: ./launcher.sh <model_name> <benchmark> [trials_mult] [backend] [methods] [instance_count]
#
# Each worker loads pairs from storage (HF/Supabase/cache) automatically.
# Groups methods across INSTANCE_COUNT GPU instances (default: single).
# Each instance runs its assigned methods in parallel via ProcessPoolExecutor.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUN_ON_GCP="$REPO_DIR/run_on_gcp.sh"

MODEL_NAME="${1:?Usage: $0 <model_name> <benchmark> [trials_multiplier] [backend] [methods]}"
BENCHMARK="${2:?Usage: $0 <model_name> <benchmark> [trials_multiplier] [backend] [methods]}"
TRIALS_MULTIPLIER="${3:-50}"
BACKEND="${4:-optuna}"
SELECTED_METHODS="${5:-}"
INSTANCE_COUNT="${6:-1}"


GCS_BUCKET="${GCS_BUCKET:-wisent-gcp-bucket}"
JOB_ID="${JOB_ID:-fbm-$(date +%Y%m%d-%H%M%S)}"
GCS_BASE="gs://$GCS_BUCKET/find_best_method/$JOB_ID"

ALL_METHODS=(caa ostrze mlp tecza tetno grom nurt szlak wicher przelom)

if [[ -n "$SELECTED_METHODS" ]]; then
    IFS=',' read -ra METHODS <<< "$SELECTED_METHODS"
else
    METHODS=("${ALL_METHODS[@]}")
fi

# Auto-compute instance count to fill all GPU capacity
if [[ "$INSTANCE_COUNT" == "auto" || "$INSTANCE_COUNT" == "1" ]]; then
    PARAM_B=$(echo "$MODEL_NAME" | grep -oE "[0-9]+\.?[0-9]*[Bb]" | head -n1 | tr -d "Bb")
    if [[ -n "$PARAM_B" ]]; then
        MEM_PER_WORKER=$(echo "$PARAM_B * 2 + 4" | bc)
        GPU_MEM=24
        if (( $(echo "$MEM_PER_WORKER > 6" | bc -l) )); then GPU_MEM=40; fi
        WORKERS_PER=$(echo "$GPU_MEM / $MEM_PER_WORKER" | bc)
        if [[ "$WORKERS_PER" -lt 1 ]]; then WORKERS_PER=1; fi
        INSTANCE_COUNT=$(( (${#METHODS[@]} + WORKERS_PER - 1) / WORKERS_PER ))
        echo "Auto: ${#METHODS[@]} methods / $WORKERS_PER workers-per-GPU = $INSTANCE_COUNT instances"
    fi
fi

echo "=========================================="
echo "PARALLEL FIND BEST METHOD"
echo "=========================================="
echo "  Model:       $MODEL_NAME"
echo "  Benchmark:   $BENCHMARK"
echo "  Trials/dim:  ${TRIALS_MULTIPLIER}x"
echo "  Backend:     $BACKEND"
echo "  Methods:     ${#METHODS[@]} (${METHODS[*]})"
echo "  Instances:   $INSTANCE_COUNT"
echo "  Job ID:      $JOB_ID"
echo "  GCS base:    $GCS_BASE"
echo "=========================================="

# Upload worker script to GCS
echo "Uploading worker script to GCS..."
gcloud storage cp "$SCRIPT_DIR/worker.py" "gs://$GCS_BUCKET/scripts/steering/worker.py"

# Group methods across instances
TOTAL=${#METHODS[@]}
echo ""
echo "=== Launching $INSTANCE_COUNT instance(s) for $TOTAL methods ==="

for (( i=0; i<INSTANCE_COUNT; i++ )); do
    GROUP=()
    for (( j=i; j<TOTAL; j+=INSTANCE_COUNT )); do
        GROUP+=("${METHODS[$j]}")
    done
    if [[ ${#GROUP[@]} -eq 0 ]]; then
        continue
    fi
    METHOD_CSV=$(IFS=','; echo "${GROUP[*]}")
    WORKER_CMD="source /opt/venv/bin/activate && pip install --upgrade wisent lm-eval psycopg2-binary && gcloud storage cp gs://$GCS_BUCKET/scripts/steering/worker.py /home/ubuntu/worker.py && export MODEL_NAME='$MODEL_NAME' && export BENCHMARK='$BENCHMARK' && export METHOD='$METHOD_CSV' && export JOB_ID='$JOB_ID' && export TRIALS_MULTIPLIER='$TRIALS_MULTIPLIER' && export BACKEND='$BACKEND' && export GCS_BUCKET='$GCS_BUCKET' && python3 /home/ubuntu/worker.py"
    echo "  Instance $((i+1)): ${GROUP[*]}"
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

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
source "$REPO_DIR/wisent/core/utils/config_tools/constants/gpu_memory.sh"

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

# Auto-compute instance count using same GPU selection as run_on_gcp.sh
if [[ "$INSTANCE_COUNT" == "auto" ]]; then
    PARAM_B=$(echo "$MODEL_NAME" | grep -oE "[0-9]+\.?[0-9]*[Bb]" | head -n1 | tr -d "Bb")
    if [[ -n "$PARAM_B" ]]; then
        MEM_SINGLE=$((PARAM_B * WEIGHT_PLUS_KV_PER_BILLION + CUDA_CONTEXT_GB))
        # Mirror run_on_gcp.sh select_instance_type: mem -> GPU VRAM per card x card count
        if   (( MEM_SINGLE <= 24 ));  then PER_GPU=24;  NUM_GPUS=1
        elif (( MEM_SINGLE <= 40 ));  then PER_GPU=40;  NUM_GPUS=1
        elif (( MEM_SINGLE <= 48 ));  then PER_GPU=24;  NUM_GPUS=2
        elif (( MEM_SINGLE <= 80 ));  then PER_GPU=40;  NUM_GPUS=2
        elif (( MEM_SINGLE <= 160 )); then PER_GPU=40;  NUM_GPUS=4
        else                               PER_GPU=80;  NUM_GPUS=4
        fi
        # If model exceeds single GPU, all GPUs serve one copy (tensor parallel)
        if (( MEM_SINGLE > PER_GPU )); then
            WORKERS_PER_INSTANCE=1
        else
            WORKERS_PER_GPU=$((PER_GPU / MEM_SINGLE))
            WORKERS_PER_INSTANCE=$((WORKERS_PER_GPU * NUM_GPUS))
        fi
        INSTANCE_COUNT=$(( (${#METHODS[@]} + WORKERS_PER_INSTANCE - 1) / WORKERS_PER_INSTANCE ))
        echo "Auto: ${#METHODS[@]} methods, ${MEM_SINGLE}GB/worker, ${NUM_GPUS}x${PER_GPU}GB GPU"
        echo "  -> $WORKERS_PER_INSTANCE workers/instance -> $INSTANCE_COUNT instance(s)"
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

# Upload worker script to GCS (non-fatal: file may already exist from a prior upload)
echo "Uploading worker script to GCS..."
gcloud storage cp "$SCRIPT_DIR/worker.py" "gs://$GCS_BUCKET/scripts/steering/worker.py" || echo "WARNING: worker.py upload failed (may already exist in GCS)"

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

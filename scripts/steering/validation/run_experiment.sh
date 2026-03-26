#!/bin/bash
set -euo pipefail

# Batch launcher for ZWIAD profile validation experiment.
# Reads validation_experiment.json and launches find_best_method
# for each benchmark via the parallel launcher.
#
# Usage:
#   ./scripts/steering/validation/run_experiment.sh [max_parallel]
#
# Env:
#   EXPERIMENT_ID  - Reuse a previous experiment (skip completed benchmarks)
#   GCS_BUCKET     - GCS bucket name (default: wisent-gcp-bucket)
#
# Each benchmark launches on its own GCP instance via launcher.sh.
# Re-running with the same EXPERIMENT_ID skips benchmarks that already
# have method results in GCS, so you can safely re-run after quota errors.

SCRIPT_DIR="$(cd "$(dirname "$BASH_SOURCE")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LAUNCHER="$REPO_DIR/scripts/steering/parallel/launcher.sh"
THIS_SCRIPT="$BASH_SOURCE"
source "$REPO_DIR/wisent/core/utils/config_tools/constants/gpu_memory.sh"

_FIRST_ARG="${*:+${*%% *}}"
MAX_PARALLEL="${_FIRST_ARG:-$DEFAULT_MAX_PARALLEL_INSTANCES}"

EXPERIMENT_JSON="$REPO_DIR/wisent/support/parameters/lm_eval/profiles/validation_experiment.json"
if [[ ! -f "$EXPERIMENT_JSON" ]]; then
    echo "ERROR: $EXPERIMENT_JSON not found"
    exit "$EXIT_FAILURE"
fi

MODEL=$(python3 -c "import json; print(json.load(open('$EXPERIMENT_JSON'))['model'])")
EXPERIMENT_ID="${EXPERIMENT_ID:-zwiad-val-$(date +%Y%m%d-%H%M%S)}"
GCS_BUCKET="${GCS_BUCKET:-wisent-gcp-bucket}"

BENCHMARKS=$(python3 -c "
import json
data = json.load(open('$EXPERIMENT_JSON'))
for profile, info in data['profiles'].items():
    for bench in info['benchmarks']:
        print(bench)
")

TOTAL=$(echo "$BENCHMARKS" | wc -l | tr -d ' ')

_has_results() {
    local bench_name="$@"
    local job="${EXPERIMENT_ID}__${bench_name}"
    gcloud storage ls "gs://$GCS_BUCKET/find_best_method/${job}/methods/" &>/dev/null
}

_count_running_g_two() {
    gcloud compute instances list \
        --filter="(status=RUNNING OR status=STAGING OR status=PROVISIONING) AND machineType~g2" \
        --format="value(name)" | wc -l | tr -d ' '
}

SKIPPED=$COUNTER_INIT
PENDING=()
while IFS= read -r BENCHMARK; do
    [[ -z "$BENCHMARK" ]] && continue
    if _has_results "$BENCHMARK"; then
        echo "  SKIP (has results): $BENCHMARK"
        ((SKIPPED++))
    else
        PENDING+=("$BENCHMARK")
    fi
done <<< "$BENCHMARKS"

REMAINING=${#PENDING[@]}

echo "=========================================="
echo "ZWIAD PROFILE VALIDATION EXPERIMENT"
echo "=========================================="
echo "  Experiment ID:  $EXPERIMENT_ID"
echo "  Model:          $MODEL"
echo "  Total:          $TOTAL benchmarks"
echo "  Skipped:        $SKIPPED (already have results)"
echo "  Remaining:      $REMAINING to launch"
echo "  Max parallel:   $MAX_PARALLEL (L4 GPU instances)"
echo "  GCS base:       gs://$GCS_BUCKET/find_best_method/"
echo "=========================================="

if (( REMAINING == COUNTER_INIT )); then
    echo "All benchmarks already have results!"
    exit "$EXIT_SUCCESS"
fi

gcloud storage cp "$EXPERIMENT_JSON" \
    "gs://$GCS_BUCKET/find_best_method/$EXPERIMENT_ID/validation_experiment.json" \
    &>/dev/null || true

LAUNCHED=$COUNTER_INIT
FAILED=$COUNTER_INIT

for BENCHMARK in "${PENDING[@]}"; do
    CURRENT_G_TWO=$(_count_running_g_two)
    if (( CURRENT_G_TWO >= MAX_PARALLEL )); then
        echo "  GPU quota full ($CURRENT_G_TWO/$MAX_PARALLEL g2 instances running)"
        echo "  Stopping launch. Re-run this script when slots free up."
        break
    fi

    JOB_ID="${EXPERIMENT_ID}__${BENCHMARK}"
    ((LAUNCHED++))
    echo "[$LAUNCHED/$REMAINING] Launching: $BENCHMARK (job: $JOB_ID)"

    if JOB_ID="$JOB_ID" "$LAUNCHER" "$MODEL" "$BENCHMARK"; then
        echo "  Instance created for: $BENCHMARK"
    else
        echo "  FAILED to create instance for: $BENCHMARK"
        ((FAILED++))
    fi
done

echo ""
echo "=========================================="
echo "LAUNCH SUMMARY"
echo "=========================================="
echo "  Experiment ID:  $EXPERIMENT_ID"
echo "  Attempted:      $LAUNCHED"
echo "  Failed:         $FAILED"
echo "  Skipped:        $SKIPPED"
echo "  Still pending:  $((REMAINING - LAUNCHED))"
echo ""
if (( REMAINING - LAUNCHED > COUNTER_INIT )); then
    echo "Re-run when slots free up:"
    echo "  EXPERIMENT_ID=$EXPERIMENT_ID $THIS_SCRIPT $MAX_PARALLEL"
fi
echo ""
echo "Monitor progress:"
echo "  gcloud storage ls gs://$GCS_BUCKET/find_best_method/$EXPERIMENT_ID*/"
echo ""
echo "Analyze results when all done:"
echo "  python3 scripts/steering/validation/analyze_results.py $EXPERIMENT_ID"
echo "=========================================="

#!/bin/bash
# Retry script for GPT-OSS-20B that keeps trying until capacity is found
# This model requires g6e.12xlarge (4x L40S, 184GB) which is scarce

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="openai/gpt-oss-20b"
INSTANCE_TYPE="g6e.12xlarge"

# How long to wait between full retry attempts (5 minutes)
RETRY_DELAY=300

echo "=========================================="
echo "GPT-OSS-20B Infinite Retry Script"
echo "=========================================="
echo "Instance type: $INSTANCE_TYPE"
echo "Will keep retrying every ${RETRY_DELAY}s until capacity is found"
echo "Started at: $(date)"
echo ""

attempt=0
while true; do
    attempt=$((attempt + 1))

    echo ""
    echo "========================================"
    echo "Attempt $attempt - $(date)"
    echo "========================================"

    # Run the extraction script
    if "$SCRIPT_DIR/run_on_aws.sh" --instance-type "$INSTANCE_TYPE" \
        "python scripts/run_strategy_extraction.py --model $MODEL --device cuda"; then
        echo ""
        echo "=========================================="
        echo "SUCCESS! GPT-OSS-20B extraction completed!"
        echo "Finished at: $(date)"
        echo "Total attempts: $attempt"
        echo "=========================================="
        exit 0
    fi

    echo ""
    echo "Attempt $attempt failed (likely no capacity). Waiting ${RETRY_DELAY}s before retry..."
    echo "Next attempt at: $(date -d "+${RETRY_DELAY} seconds" 2>/dev/null || date -v+${RETRY_DELAY}S)"
    sleep $RETRY_DELAY
done

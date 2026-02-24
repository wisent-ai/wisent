#!/bin/bash
# config.sh — Configuration, credentials, helpers, and auto-publish
# Sourced by run_on_aws.sh

# Configuration
AWS_REGION="us-east-1"
AMI_ID="ami-08c3a18fa2f155bbb"  # Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20251118
SECURITY_GROUP="sg-0ba93f5e5577752d6"  # wisent-sg
IAM_INSTANCE_PROFILE="wisent-instance-profile"
REMOTE_OUTPUT_DIR="/home/ubuntu/output"

# AWS Credentials (used for EC2/SSM operations — NOT for storage)
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-REDACTED_AWS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-REDACTED_AWS_SECRET_KEY}"
export AWS_DEFAULT_REGION="$AWS_REGION"

# GCS bucket for artifact storage (migrated from S3)
GCS_BUCKET="wisent-images-bucket"

# HuggingFace Token (for gated models like Meta-Llama)
HF_TOKEN="${HF_TOKEN:-REDACTED_HF_TOKEN}"

# Script directory — resolves to the wisent-open-source root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ============================================================================
# Auto-publish wisent package to PyPI before running on AWS
# ============================================================================

PYPI_TOKEN="${PYPI_TOKEN:?PYPI_TOKEN env var must be set}"

# Lock directory for concurrent publish protection (mkdir is atomic on all platforms)
PUBLISH_LOCK_DIR="/tmp/wisent_publish.lock"

publish_wisent_package() {
    echo "=========================================="
    echo "Auto-publishing wisent package to PyPI"
    echo "=========================================="

    local wisent_dir="$SCRIPT_DIR"
    local init_file="$wisent_dir/wisent/__init__.py"

    if [[ ! -f "$init_file" ]]; then
        echo "ERROR: Cannot find $init_file"
        exit 1
    fi

    # Acquire lock to prevent concurrent publish race conditions
    echo "Waiting for publish lock..."
    local lock_attempts=0
    while ! mkdir "$PUBLISH_LOCK_DIR" 2>/dev/null; do
        lock_attempts=$((lock_attempts + 1))
        if [[ $lock_attempts -gt 120 ]]; then
            echo "WARNING: Lock timeout after 2 minutes, proceeding anyway"
            break
        fi
        sleep 1
    done
    echo "Acquired publish lock"

    # Read current version
    local current_version
    current_version=$(grep -E '^__version__' "$init_file" | sed 's/__version__ = "\(.*\)"/\1/')
    echo "Current version: $current_version"

    # Bump patch version
    local major minor patch
    IFS='.' read -r major minor patch <<< "$current_version"
    patch=$((patch + 1))
    local new_version="$major.$minor.$patch"
    echo "New version: $new_version"

    # Update version in __init__.py
    sed -i.bak "s/__version__ = \"$current_version\"/__version__ = \"$new_version\"/" "$init_file"
    rm -f "$init_file.bak"

    # Clean previous builds
    rm -rf "$wisent_dir/dist" "$wisent_dir/build" "$wisent_dir/wisent.egg-info"

    # Build the package
    echo "Building package..."
    (cd "$wisent_dir" && python3 -m build) || {
        echo "ERROR: Failed to build package"
        rmdir "$PUBLISH_LOCK_DIR" 2>/dev/null || true
        exit 1
    }

    # Upload to PyPI
    echo "Uploading to PyPI..."
    (cd "$wisent_dir" && python3 -m twine upload --username __token__ --password "$PYPI_TOKEN" dist/*) || {
        echo "ERROR: Failed to upload to PyPI"
        rmdir "$PUBLISH_LOCK_DIR" 2>/dev/null || true
        exit 1
    }

    echo "Successfully published wisent==$new_version to PyPI"

    # Export the new version for use in USER_DATA
    export WISENT_VERSION="$new_version"

    # Release lock
    rmdir "$PUBLISH_LOCK_DIR" 2>/dev/null || true
    echo "Released publish lock"
}

# Publish the package before doing anything else (skip if WISENT_VERSION is already set)
if [[ -z "${WISENT_VERSION:-}" ]]; then
    publish_wisent_package
else
    echo "Using pre-set WISENT_VERSION=$WISENT_VERSION (skipping publish)"
fi

# ============================================================================
# Instance type selection logic based on model size and command requirements
# ============================================================================

# Use Python calculator if available for accurate memory estimation
get_model_memory_requirement() {
    local model="$1"
    local wisent_cmd="${2:-}"

    # Try to use Python calculator for accurate estimation
    if [[ -f "$SCRIPT_DIR/calculate_gpu_memory.py" ]]; then
        local result
        if [[ -n "$wisent_cmd" ]]; then
            result=$(python3 "$SCRIPT_DIR/calculate_gpu_memory.py" "$model" "$wisent_cmd" 2>/dev/null | grep "Total GPU memory" | grep -Eo '[0-9.]+' || echo "")
        else
            result=$(python3 "$SCRIPT_DIR/calculate_gpu_memory.py" "$model" 2>/dev/null | grep "Total GPU memory" | grep -Eo '[0-9.]+' || echo "")
        fi
        if [[ -n "$result" ]]; then
            echo "$result"
            return
        fi
    fi

    # Fallback: Extract size from model name (e.g., "8B", "70B", "1B")
    local size=""
    if [[ "$model" =~ ([0-9]+\.?[0-9]*)[Bb] ]]; then
        size="${BASH_REMATCH[1]}"
    fi

    # Check for quantization in model name
    local quant_factor=1.0
    if [[ "$model" =~ GPTQ|AWQ|int4|INT4|4bit ]]; then
        quant_factor=0.25  # INT4 is ~4x smaller than FP16
    elif [[ "$model" =~ int8|INT8|8bit ]]; then
        quant_factor=0.5   # INT8 is ~2x smaller than FP16
    fi

    # Memory requirements (rough estimates for fp16 inference + overhead)
    local base_memory
    if [[ -z "$size" ]]; then
        base_memory=24
    elif (( $(echo "$size < 3" | bc -l) )); then
        base_memory=8
    elif (( $(echo "$size < 10" | bc -l) )); then
        base_memory=24
    elif (( $(echo "$size < 20" | bc -l) )); then
        base_memory=46
    elif (( $(echo "$size < 40" | bc -l) )); then
        base_memory=92
    elif (( $(echo "$size < 80" | bc -l) )); then
        base_memory=184
    else
        base_memory=368
    fi

    # Apply quantization factor
    local adjusted_memory=$(echo "$base_memory * $quant_factor" | bc -l)

    # Apply command multiplier if specified
    if [[ -n "$wisent_cmd" ]]; then
        case "$wisent_cmd" in
            modify-weights|optimize-weights)
                adjusted_memory=$(echo "$adjusted_memory * 1.5" | bc -l)
                ;;
            get-activations|generate-vector-from-task|optimize-*|tasks)
                adjusted_memory=$(echo "$adjusted_memory * 1.2" | bc -l)
                ;;
        esac
    fi

    # Round up to nearest integer
    printf "%.0f" "$adjusted_memory"
}

# Select instance type based on required GPU memory
select_instance_type() {
    local required_memory="$1"

    # Available g6e instance types:
    # g6e.xlarge:    1x L40S (46GB), 4 vCPU, 32GB RAM   - ~$1.50/hr
    # g6e.2xlarge:   1x L40S (46GB), 8 vCPU, 64GB RAM   - ~$2.00/hr
    # g6e.4xlarge:   1x L40S (46GB), 16 vCPU, 128GB RAM - ~$3.00/hr
    # g6e.12xlarge:  4x L40S (184GB), 48 vCPU, 384GB RAM - ~$8.00/hr
    # g6e.24xlarge:  4x L40S (184GB), 96 vCPU, 768GB RAM - ~$14.00/hr
    # g6e.48xlarge:  8x L40S (368GB), 192 vCPU, 1536GB RAM - ~$28.00/hr

    if (( $(echo "$required_memory <= 46" | bc -l) )); then
        echo "g6e.xlarge"
    elif (( $(echo "$required_memory <= 92" | bc -l) )); then
        echo "g6e.12xlarge"
    elif (( $(echo "$required_memory <= 184" | bc -l) )); then
        echo "g6e.12xlarge"
    elif (( $(echo "$required_memory <= 368" | bc -l) )); then
        echo "g6e.48xlarge"
    else
        echo "ERROR: Model too large for available instances (need ${required_memory}GB)"
        return 1
    fi
}

#!/bin/bash

# Example: Modify model weights from lm-eval task
# This script demonstrates using the modify-weights command with task-generated steering vectors
# Uses Heretic-style kernel-based abliteration across multiple layers
#
# Usage:
#   bash wisent/examples/cli/weight_modification/modify_weights_from_task.sh

python -m wisent.core.main modify-weights \
    --task hellaswag \
    --trait-label correctness \
    --output-dir ./data/modified_models/hellaswag_modified \
    --model meta-llama/Llama-3.2-1B \
    --num-pairs 100 \
    --method abliteration \
    --components self_attn.o_proj mlp.down_proj \
    --use-kernel \
    --max-weight 1.5 \
    --max-weight-position 8.0 \
    --min-weight 0.3 \
    --min-weight-distance 6.0 \
    --normalize-vectors \
    --verbose

#!/bin/bash

# Example: Modify model weights from lm-eval task
# This script demonstrates using the modify-weights command with task-generated steering vectors
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
    --strength 1.0 \
    --components self_attn.o_proj mlp.down_proj \
    --normalize-vectors \
    --verbose

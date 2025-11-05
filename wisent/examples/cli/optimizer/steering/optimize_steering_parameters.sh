#!/bin/bash

# Example: Optimize steering parameters comprehensively
# This script demonstrates using the optimize-steering command to find
# optimal parameters for steering methods including:
# - Best steering method (CAA, HPR, DAC, BiPO, KSteering)
# - Optimal steering layer
# - Optimal steering strength
# - Optimal steering strategy (last_only, first_only, all_equal, exponential_decay, etc.)
# - Method-specific parameters
#
# Usage:
#   bash wisent/examples/cli/optimizer/steering/optimize_steering_parameters.sh

python -m wisent.core.main optimize-steering comprehensive \
    meta-llama/Llama-3.2-1B-Instruct \
    --tasks truthfulqa_mc1 \
    --methods CAA \
    --limit 10 \
    --max-time-per-task 15.0 \
    --verbose

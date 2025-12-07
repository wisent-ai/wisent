#!/bin/bash

# Optimize classification parameters for coding benchmarks
# This script finds optimal parameters for detecting bad/incorrect code using
# classifier-based evaluation on coding tasks.
#
# The optimizer tests different combinations of:
# - Best layer selection (which transformer layer captures code quality best)
# - Token aggregation method (how to combine token representations)
# - Detection threshold (confidence threshold for bad code detection)
# - Training sample size
#
# Coding benchmarks tested:
# - humaneval: Python function completion
# - mbpp: Python programming problems
# - instruct_humaneval: Instruction-based code completion
# - ds1000: Data science coding problems
#
# Usage:
#   bash wisent/examples/cli/optimizer/classification/coding_optimization.sh

python -u -m wisent.core.main optimize-classification \
    meta-llama/Llama-3.2-1B-Instruct \
    --limit 20 \
    --optimization-metric f1 \
    --max-time-per-task 30.0 \
    --layer-range "4-8" \
    --aggregation-methods average final first max \
    --threshold-range 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
    --save-logs-json /tmp/coding_optimization_logs.json

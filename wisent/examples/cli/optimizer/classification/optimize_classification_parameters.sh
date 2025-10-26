#!/bin/bash

# Example: Optimize classification parameters across all tasks
# This script demonstrates using the optimize-classification command to find
# optimal parameters for classification tasks including:
# - Best layer selection
# - Token aggregation method
# - Detection threshold
# - Training sample size
#
# The optimizer uses grid search to test different combinations and saves:
# - Best parameters for each task to model config
# - Trained classifiers for each task
# - Detailed optimization logs (optional)
#
# Usage:
#   bash wisent/examples/cli/optimizer/classification/optimize_classification_parameters.sh

python -m wisent.core.main optimize-classification \
    meta-llama/Llama-3.2-1B-Instruct \
    --limit 10 \
    --optimization-metric f1 \
    --max-time-per-task 15.0 \
    --layer-range "8-12" \
    --aggregation-methods average final first max \
    --threshold-range 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
    --device cpu \
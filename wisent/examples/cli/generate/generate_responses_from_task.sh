#!/bin/bash

# Example: Generate responses to questions from a task
# This script demonstrates generating model responses to questions from an lm-eval task.
# It loads questions from the task and generates responses using the specified model.
#
# Usage:
#   bash wisent/examples/cli/generate/generate_responses_from_task.sh

python -m wisent.core.main generate-responses \
    meta-llama/Llama-3.2-1B-Instruct \
    --task arc_easy \
    --num-questions 10 \
    --max-new-tokens 128 \
    --temperature 0.7 \
    --device cpu \
    --output ./generated_responses/arc_easy_responses.json \
    --verbose

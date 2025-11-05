#!/bin/bash

# Example: Evaluate generated responses using benchmark-specific evaluators
#
# This script demonstrates the complete workflow:
# 1. Generate responses for a task using generate-responses CLI
# 2. Evaluate responses using evaluate-responses CLI with benchmark-specific evaluators
#
# The evaluator is automatically selected based on task type (from task-evaluator.json):
#
# Multiple-choice tasks (arc_easy, hellaswag, truthfulqa_mc1, etc.):
#   → F1Evaluator
#   - Compares generated response text to each choice text
#   - Selects choice with highest F1 score
#   - Reports accuracy (correct choice / total)
#
# Generation tasks with exact_match metric (gsm8k, triviaqa):
#   → ExactMatchEvaluator
#   - Extracts numerical/text answer from generated response
#   - Exact string match with expected answer
#
# Generation tasks with f1 metric (drop, squad):
#   → F1Evaluator
#   - Token-level F1 comparison between response and expected answer
#
# Language modeling tasks (wikitext, lambada):
#   → PerplexityEvaluator (requires model integration)
#
# ============================================================================
# STEP 1: Generate responses (if not already done)
# ============================================================================

# echo "Step 1: Generating responses for arc_easy..."
# python -m wisent.core.main generate-responses \
#     meta-llama/Llama-3.2-1B-Instruct \
#     --task arc_easy \
#     --num-questions 10 \
#     --max-new-tokens 128 \
#     --output ./generated_responses/arc_easy_responses.json \
#     --verbose

# ============================================================================
# STEP 2: Evaluate generated responses
# ============================================================================

echo "Step 2: Evaluating arc_easy responses..."
python -m wisent.core.main evaluate-responses \
    --input ./generated_responses/arc_easy_responses.json \
    --output ./evaluation_results/arc_easy_evaluation.json \
    --verbose

# ============================================================================
# Additional examples for different task types
# ============================================================================

# Multiple-choice task (hellaswag)
# python -m wisent.core.main evaluate-responses \
#     --input ./generated_responses/hellaswag_responses.json \
#     --output ./evaluation_results/hellaswag_evaluation.json \
#     --verbose

# Generation task with exact match (gsm8k)
# python -m wisent.core.main evaluate-responses \
#     --input ./generated_responses/gsm8k_responses.json \
#     --output ./evaluation_results/gsm8k_evaluation.json \
#     --verbose

# Generation task with F1 (drop)
# python -m wisent.core.main evaluate-responses \
#     --input ./generated_responses/drop_responses.json \
#     --output ./evaluation_results/drop_evaluation.json \
#     --verbose

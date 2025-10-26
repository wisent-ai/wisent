#!/bin/bash

# Generate contrastive pairs from LiveCodeBench task
#
# This script downloads AI model submissions directly from HuggingFace's livecodebench/code_generation_samples Space
# and extracts contrastive pairs by comparing passing vs failing code submissions.
#
# The extractor will:
# 1. Download all_outputs.json and problems.json from HuggingFace Space
# 2. Parse 880 coding problems from 22 AI models (default: GPT-4O-2024-08-06)
# 3. Separate passing (correct) and failing (incorrect) submissions based on test results
# 4. Randomly select one passing and one failing code per problem to create contrastive pairs
# 5. Match with lm-eval LiveCodeBench task docs
# 6. Save contrastive pairs to JSON
#
# No local files needed - everything is downloaded on-the-fly from HuggingFace.

python -m wisent.core.main generate-pairs-from-task livecodebench \
    --output ./data/livecodebench_pairs.json \
    --limit 5 \
    --verbose

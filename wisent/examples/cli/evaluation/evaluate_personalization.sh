#!/bin/bash

# Evaluate personalization responses using the evaluate-responses command
# This compares baseline vs steered responses to assess steering effectiveness

python -m wisent.core.main evaluate-responses \
    --input ./generated_responses/personalization_steered.json \
    --baseline ./generated_responses/personalization_baseline.json \
    --output ./evaluation_results/personalization.json \
    --task personalization \
    --trait "evil" \
    --trait-description "Exhibits malicious, harmful, or antagonistic behavior" \
    --verbose

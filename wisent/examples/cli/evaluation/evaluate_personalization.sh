#!/bin/bash

# Evaluate personalization responses using the evaluate-responses command
# This evaluates how well model responses exhibit target personality traits

python -m wisent.core.main evaluate-responses \
    --input ./generated_responses/personalization.json \
    --output ./evaluation_results/personalization.json \
    --task personalization \
    --trait "evil" \
    --verbose

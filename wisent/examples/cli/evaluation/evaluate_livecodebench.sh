#!/bin/bash

# Evaluate LiveCodeBench coding tasks using the evaluate-responses command

python -m wisent.core.main evaluate-responses \
    --input ./generated_responses/livecodebench.json \
    --output ./evaluation_results/livecodebench.json \
    --task livecodebench \
    --verbose

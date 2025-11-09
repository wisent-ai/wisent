#!/bin/bash

# Agent with steering mode enabled
# Uses steering vectors to guide model behavior

wisent agent "Explain quantum computing in simple terms" \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --steering-mode \
    --steering-method CAA \
    --steering-strength 1.0 \
    --normalize-mode \
    --verbose

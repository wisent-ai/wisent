#!/bin/bash

# Agent with quality control
# Attempts to achieve acceptable quality with multiple iterations

wisent agent "Provide a detailed analysis of renewable energy sources" \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --enable-quality-control \
    --max-quality-attempts 5 \
    --quality-threshold 0.4 \
    --show-parameter-reasoning \
    --verbose

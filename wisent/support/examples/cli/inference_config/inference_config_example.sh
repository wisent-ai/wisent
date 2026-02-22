#!/bin/bash

# Example: Managing Inference Configuration
# This script demonstrates how to view and update the global inference settings
# that control model generation parameters like temperature, sampling, etc.

echo "=== Inference Configuration Management ==="
echo ""

# Show current configuration
echo "1. View current inference config:"
echo "   wisent inference-config show"
echo ""
python -m wisent.core.main inference-config show

echo ""
echo "================================================"
echo ""

# Update specific settings
echo "2. Update temperature and top_k:"
echo "   wisent inference-config set --temperature 0.8 --top-k 20"
echo ""
python -m wisent.core.main inference-config set --temperature 0.8 --top-k 20

echo ""
echo "================================================"
echo ""

# Update sampling settings
echo "3. Configure sampling parameters:"
echo "   wisent inference-config set --do-sample true --temperature 0.7 --top-p 0.8 --top-k 20"
echo ""
python -m wisent.core.main inference-config set --do-sample true --temperature 0.7 --top-p 0.8 --top-k 20

echo ""
echo "================================================"
echo ""

# Disable sampling (greedy decoding)
echo "4. Enable greedy decoding:"
echo "   wisent inference-config set --do-sample false"
echo ""

echo ""
echo "================================================"
echo ""

# Reset to defaults
echo "5. Reset to default settings:"
echo "   wisent inference-config reset"
echo ""
python -m wisent.core.main inference-config reset

echo ""
echo "=== Configuration Options ==="
echo ""
echo "Available settings:"
echo "  --do-sample BOOL       Enable sampling (true/false). Default: true"
echo "  --temperature FLOAT    Sampling temperature (0.0-2.0). Default: 0.7"
echo "  --top-p FLOAT          Nucleus sampling probability. Default: 0.9"
echo "  --top-k INT            Top-k sampling. Default: 50"
echo "  --max-new-tokens INT   Max tokens to generate. Default: 512"
echo "  --repetition-penalty FLOAT  Repetition penalty. Default: 1.0"
echo "  --no-repeat-ngram-size INT  No repeat n-gram size. Default: 0"
echo "  --enable-thinking BOOL Enable thinking mode (Qwen3). Default: false"
echo ""
echo "Config is stored at: ~/.wisent/inference_config.json"
echo ""

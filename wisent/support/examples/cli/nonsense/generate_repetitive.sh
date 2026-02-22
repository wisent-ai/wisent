#!/bin/bash

# Example: Generate nonsense pairs with repetitive patterns
# This mode creates negative responses with pathological repetition (e.g., "the the the the")

echo "🔁 Generating repetitive nonsense pairs..."
echo ""

wisent generate-pairs \
  --trait "coherent and meaningful responses" \
  --nonsense \
  --nonsense-mode repetitive \
  --num-pairs 30 \
  --output ./nonsense_repetitive.json \
  --verbose

echo ""
echo "✅ Generated nonsense pairs saved to: ./nonsense_repetitive.json"
echo ""
echo "Example negative response: 'plants plants plants plants plants plants plants'"

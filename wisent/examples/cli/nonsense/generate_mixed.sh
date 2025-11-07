#!/bin/bash

# Example: Generate nonsense pairs with mixed types
# This mode creates negative responses combining random chars, repetition, and word salad

echo "🎭 Generating mixed nonsense pairs..."
echo ""

wisent generate-pairs \
  --trait "coherent and meaningful responses" \
  --nonsense \
  --nonsense-mode mixed \
  --num-pairs 30 \
  --output ./nonsense_mixed.json \
  --verbose

echo ""
echo "✅ Generated nonsense pairs saved to: ./nonsense_mixed.json"
echo ""
echo "Example negative response: 'jkfdjkfd learning learning learning asdjkh purple calculator yesterday'"

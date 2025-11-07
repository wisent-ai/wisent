#!/bin/bash

# Example: Generate nonsense pairs with word salad
# This mode creates negative responses with real words that have no coherent meaning

echo "🥗 Generating word salad nonsense pairs..."
echo ""

wisent generate-pairs \
  --trait "logical and coherent responses" \
  --nonsense \
  --nonsense-mode word_salad \
  --num-pairs 30 \
  --output ./nonsense_word_salad.json \
  --verbose

echo ""
echo "✅ Generated nonsense pairs saved to: ./nonsense_word_salad.json"
echo ""
echo "Example negative response: 'telephone purple yesterday elephant calculator happiness mountain'"

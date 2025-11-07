#!/bin/bash

# Example: Generate nonsense pairs with random character gibberish
# This mode creates negative responses that look like keyboard mashing (e.g., "ahsdhashdahsdha")

echo "🎲 Generating random character nonsense pairs..."
echo ""

wisent generate-pairs \
  --trait "coherent and meaningful responses" \
  --nonsense \
  --nonsense-mode random_chars \
  --num-pairs 30 \
  --output ./nonsense_random_chars.json \
  --verbose

echo ""
echo "✅ Generated nonsense pairs saved to: ./nonsense_random_chars.json"
echo ""
echo "Example negative response: 'asdjkhasjkdhaksjdh aksjdhaksjdh aksdjhaksjdhaksj'"

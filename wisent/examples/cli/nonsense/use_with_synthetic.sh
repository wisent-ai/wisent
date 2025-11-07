#!/bin/bash

# Example: Use nonsense generation with the synthetic command
# This shows how to create steering vectors from nonsense pairs

echo "🎯 Using nonsense generation with synthetic command..."
echo ""

# Example 1: Create steering vector from random character nonsense
echo "Example 1: Random character nonsense for steering"
wisent synthetic \
  --trait "coherent responses" \
  --nonsense \
  --nonsense-mode random_chars \
  --num-pairs 15 \
  --save-pairs ./synthetic_random_nonsense.json \
  --output ./results_random_nonsense \
  --verbose

echo ""

# Example 2: Create steering vector from repetitive nonsense
echo "Example 2: Repetitive nonsense for steering"
wisent synthetic \
  --trait "non-repetitive responses" \
  --nonsense \
  --nonsense-mode repetitive \
  --num-pairs 15 \
  --save-pairs ./synthetic_repetitive_nonsense.json \
  --output ./results_repetitive_nonsense \
  --verbose

echo ""
echo "✅ Synthetic nonsense pairs generated!"
echo "   - Random chars: ./synthetic_random_nonsense.json"
echo "   - Repetitive: ./synthetic_repetitive_nonsense.json"

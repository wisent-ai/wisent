#!/bin/bash

# Example: Compare all nonsense generation modes
# This script generates pairs using all 4 modes for comparison

echo "🔬 Comparing all nonsense generation modes..."
echo ""
echo "This will generate 15 pairs for each mode:"
echo "  1. random_chars  - Random gibberish"
echo "  2. repetitive    - Pathological repetition"
echo "  3. word_salad    - Real words, no meaning"
echo "  4. mixed         - Combination of all types"
echo ""

# Create output directory
mkdir -p ./nonsense_comparison

# Mode 1: Random chars
echo "📝 [1/4] Generating random_chars mode..."
wisent generate-pairs \
  --trait "coherent responses" \
  --nonsense \
  --nonsense-mode random_chars \
  --num-pairs 15 \
  --output ./nonsense_comparison/mode_1_random_chars.json \
  --verbose

echo ""

# Mode 2: Repetitive
echo "📝 [2/4] Generating repetitive mode..."
wisent generate-pairs \
  --trait "coherent responses" \
  --nonsense \
  --nonsense-mode repetitive \
  --num-pairs 15 \
  --output ./nonsense_comparison/mode_2_repetitive.json \
  --verbose

echo ""

# Mode 3: Word salad
echo "📝 [3/4] Generating word_salad mode..."
wisent generate-pairs \
  --trait "logical responses" \
  --nonsense \
  --nonsense-mode word_salad \
  --num-pairs 15 \
  --output ./nonsense_comparison/mode_3_word_salad.json \
  --verbose

echo ""

# Mode 4: Mixed
echo "📝 [4/4] Generating mixed mode..."
wisent generate-pairs \
  --trait "coherent responses" \
  --nonsense \
  --nonsense-mode mixed \
  --num-pairs 15 \
  --output ./nonsense_comparison/mode_4_mixed.json \
  --verbose

echo ""
echo "✅ All modes generated! Files saved to ./nonsense_comparison/"
echo ""
echo "To compare, check these files:"
echo "  - mode_1_random_chars.json"
echo "  - mode_2_repetitive.json"
echo "  - mode_3_word_salad.json"
echo "  - mode_4_mixed.json"

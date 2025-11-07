#!/bin/bash

# Example: Use nonsense generation with the tasks command
# This shows how to use nonsense pairs with existing benchmarks

echo "📋 Using nonsense generation with tasks command..."
echo ""

# Example 1: Use with hellaswag task
echo "Example 1: Hellaswag with random character nonsense"
wisent tasks hellaswag \
  --nonsense \
  --nonsense-mode random_chars \
  --num-synthetic-pairs 20 \
  --save-synthetic ./task_hellaswag_nonsense.json \
  --limit 50

echo ""
echo "Example 2: MMLU with word salad nonsense"
wisent tasks mmlu \
  --nonsense \
  --nonsense-mode word_salad \
  --num-synthetic-pairs 20 \
  --save-synthetic ./task_mmlu_nonsense.json \
  --limit 50

echo ""
echo "✅ Task nonsense pairs generated!"
echo "   - Hellaswag: ./task_hellaswag_nonsense.json"
echo "   - MMLU: ./task_mmlu_nonsense.json"

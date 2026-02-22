#!/bin/bash

# Example: Use pre-trained steering vectors for multi-steering
# This script demonstrates practical scenarios where you have
# already trained vectors and want to combine them for different use cases.

# Configuration
MODEL="meta-llama/Llama-3.2-1B-Instruct"
LAYER=15
DEVICE="cuda"

# Assume these vectors were trained previously and saved
FORMAL_VECTOR="./vectors/formal_tone.pt"
TECHNICAL_VECTOR="./vectors/technical_language.pt"
CONCISE_VECTOR="./vectors/concise_answers.pt"
FRIENDLY_VECTOR="./vectors/friendly_tone.pt"
DETAILED_VECTOR="./vectors/detailed_explanations.pt"

# Use Case 1: Technical documentation writer
# Formal + Technical + Concise
echo "=== Use Case 1: Technical Documentation ==="
python -m wisent.core.main multi-steer \
    --vector $FORMAL_VECTOR:0.4 \
    --vector $TECHNICAL_VECTOR:0.4 \
    --vector $CONCISE_VECTOR:0.2 \
    --model $MODEL \
    --layer $LAYER \
    --method CAA \
    --prompt "Explain how REST APIs work." \
    --max-new-tokens 200 \
    --normalize-weights \
    --save-combined ./vectors/tech_doc_writer.pt \
    --device $DEVICE \
    --verbose

# Use Case 2: Friendly teacher
# Friendly + Detailed + minimal Technical
echo "=== Use Case 2: Friendly Teacher ==="
python -m wisent.core.main multi-steer \
    --vector $FRIENDLY_VECTOR:0.5 \
    --vector $DETAILED_VECTOR:0.4 \
    --vector $TECHNICAL_VECTOR:0.1 \
    --model $MODEL \
    --layer $LAYER \
    --method CAA \
    --prompt "How does photosynthesis work?" \
    --max-new-tokens 250 \
    --normalize-weights \
    --save-combined ./vectors/friendly_teacher.pt \
    --device $DEVICE \
    --verbose

# Use Case 3: Executive summary writer
# Concise + Formal + no technical jargon
echo "=== Use Case 3: Executive Summary ==="
python -m wisent.core.main multi-steer \
    --vector $CONCISE_VECTOR:0.6 \
    --vector $FORMAL_VECTOR:0.4 \
    --model $MODEL \
    --layer $LAYER \
    --method CAA \
    --prompt "Summarize the benefits of cloud computing for businesses." \
    --max-new-tokens 150 \
    --normalize-weights \
    --save-combined ./vectors/executive_summary.pt \
    --device $DEVICE \
    --verbose

# Use Case 4: In-depth technical expert
# Technical + Detailed + Formal
echo "=== Use Case 4: Technical Expert ==="
python -m wisent.core.main multi-steer \
    --vector $TECHNICAL_VECTOR:0.4 \
    --vector $DETAILED_VECTOR:0.4 \
    --vector $FORMAL_VECTOR:0.2 \
    --model $MODEL \
    --layer $LAYER \
    --method CAA \
    --prompt "Explain the differences between TCP and UDP protocols." \
    --max-new-tokens 300 \
    --normalize-weights \
    --save-combined ./vectors/technical_expert.pt \
    --device $DEVICE \
    --verbose

# Use Case 5: Quick chat assistant
# Friendly + Concise
echo "=== Use Case 5: Quick Chat Assistant ==="
python -m wisent.core.main multi-steer \
    --vector $FRIENDLY_VECTOR:0.6 \
    --vector $CONCISE_VECTOR:0.4 \
    --model $MODEL \
    --layer $LAYER \
    --method CAA \
    --prompt "What's the weather like in San Francisco?" \
    --max-new-tokens 100 \
    --normalize-weights \
    --save-combined ./vectors/chat_assistant.pt \
    --device $DEVICE \
    --verbose

# Use Case 6: Academic paper style
# Formal + Technical + Detailed
echo "=== Use Case 6: Academic Paper Style ==="
python -m wisent.core.main multi-steer \
    --vector $FORMAL_VECTOR:0.35 \
    --vector $TECHNICAL_VECTOR:0.35 \
    --vector $DETAILED_VECTOR:0.3 \
    --model $MODEL \
    --layer $LAYER \
    --method CAA \
    --prompt "Discuss the implications of quantum computing on cryptography." \
    --max-new-tokens 350 \
    --normalize-weights \
    --save-combined ./vectors/academic_writer.pt \
    --device $DEVICE \
    --verbose

# Use Case 7: Testing different weight ratios for the same prompt
echo "=== Use Case 7: Comparing Different Weight Ratios ==="
PROMPT="Explain machine learning."

echo "Configuration A: More friendly, less technical"
python -m wisent.core.main multi-steer \
    --vector $FRIENDLY_VECTOR:0.7 \
    --vector $TECHNICAL_VECTOR:0.3 \
    --model $MODEL \
    --layer $LAYER \
    --method CAA \
    --prompt "$PROMPT" \
    --max-new-tokens 200 \
    --normalize-weights \
    --device $DEVICE \
    --verbose

echo "Configuration B: More technical, less friendly"
python -m wisent.core.main multi-steer \
    --vector $FRIENDLY_VECTOR:0.3 \
    --vector $TECHNICAL_VECTOR:0.7 \
    --model $MODEL \
    --layer $LAYER \
    --method CAA \
    --prompt "$PROMPT" \
    --max-new-tokens 200 \
    --normalize-weights \
    --device $DEVICE \
    --verbose

echo "Configuration C: Balanced approach"
python -m wisent.core.main multi-steer \
    --vector $FRIENDLY_VECTOR:0.5 \
    --vector $TECHNICAL_VECTOR:0.5 \
    --model $MODEL \
    --layer $LAYER \
    --method CAA \
    --prompt "$PROMPT" \
    --max-new-tokens 200 \
    --normalize-weights \
    --device $DEVICE \
    --verbose

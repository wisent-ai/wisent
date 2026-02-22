#!/bin/bash

# End-to-end personalization evaluation pipeline
# This script:
# 1-2. Generates synthetic contrastive pairs and trains control vector
# 3. Generates baseline responses (no steering)
# 4. Generates steered responses (with control vector)
# 5. Evaluates steering effectiveness

set -e  # Exit on any error

# Configuration
TRAIT="${TRAIT:-helpfulness}"
TRAIT_DESCRIPTION="${TRAIT_DESCRIPTION:-Helpful, harmless, and honest behavior}"
MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
NUM_PAIRS="${NUM_PAIRS:-20}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-200}"
LAYERS="${LAYERS:-8}"
STEERING_STRENGTH="${STEERING_STRENGTH:-1.5}"

# Output paths
DATA_DIR="./data/personalization"
VECTOR_FILE="${DATA_DIR}/steering_vector_${TRAIT}.pt"
PROMPTS_FILE="${DATA_DIR}/prompts.json"
BASELINE_FILE="./generated_responses/personalization_baseline_${TRAIT}.json"
STEERED_FILE="./generated_responses/personalization_steered_${TRAIT}.json"
EVAL_FILE="./evaluation_results/personalization_${TRAIT}.json"

# Create output directories
mkdir -p "${DATA_DIR}"
mkdir -p "$(dirname ${BASELINE_FILE})"
mkdir -p "$(dirname ${EVAL_FILE})"

echo "=================================="
echo "PERSONALIZATION EVALUATION PIPELINE"
echo "=================================="
echo "Trait: ${TRAIT}"
echo "Description: ${TRAIT_DESCRIPTION}"
echo "Model: ${MODEL}"
echo "=================================="
echo ""

# Step 1-2: Generate synthetic pairs and train control vector (all-in-one)
echo "Step 1-2/5: Generating synthetic pairs and training control vector..."
echo "   Output: ${VECTOR_FILE}"
python -m wisent.core.main generate-vector-from-synthetic \
    --trait "${TRAIT}" \
    --output "${VECTOR_FILE}" \
    --model "${MODEL}" \
    --num-pairs ${NUM_PAIRS} \
    --similarity-threshold 0.8 \
    --layers ${LAYERS} \
    --token-aggregation average \
    --prompt-strategy chat_template \
    --method caa \
    --normalize \
    --keep-intermediate \
    --intermediate-dir "${DATA_DIR}" \
    --verbose \
    --timing

if [ ! -f "${VECTOR_FILE}" ]; then
    echo "ERROR: Failed to create steering vector"
    exit 1
fi
echo "   SUCCESS: Control vector trained"
echo ""

# Step 3: Create prompts file if it doesn't exist
if [ ! -f "${PROMPTS_FILE}" ]; then
    echo "Creating default prompts file..."
    cat > "${PROMPTS_FILE}" << 'EOF'
[
  {"id": 1, "prompt": "Tell me about your favorite food."},
  {"id": 2, "prompt": "What do you think about the weather today?"},
  {"id": 3, "prompt": "Describe your morning routine."},
  {"id": 4, "prompt": "What's your opinion on public transportation?"},
  {"id": 5, "prompt": "Tell me about your weekend plans."},
  {"id": 6, "prompt": "What's your favorite way to relax?"},
  {"id": 7, "prompt": "How do you stay organized?"},
  {"id": 8, "prompt": "What's your approach to learning new skills?"},
  {"id": 9, "prompt": "Tell me about your hobbies."},
  {"id": 10, "prompt": "What's your opinion on social media?"}
]
EOF
    echo "   SUCCESS: Created prompts file: ${PROMPTS_FILE}"
fi
echo ""

# Step 4: Generate baseline responses (no steering)
echo "Step 3/5: Generating baseline responses..."
echo "   Output: ${BASELINE_FILE}"
python << EOF
import json
from wisent.core.models.wisent_model import WisentModel

# Load model
print("   Loading model...")
model = WisentModel("${MODEL}")

# Load prompts
with open("${PROMPTS_FILE}") as f:
    prompts_data = json.load(f)

results = []
for item in prompts_data[:${NUM_PROMPTS}]:
    prompt_id = item["id"]
    prompt_text = item["prompt"]

    print(f"   Generating baseline response {prompt_id}...")

    messages = [{"role": "user", "content": prompt_text}]
    response = model.generate(
        inputs=[messages],
        max_new_tokens=${MAX_NEW_TOKENS},
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
    )[0]

    results.append({
        "id": prompt_id,
        "prompt": prompt_text,
        "generated_response": response
    })

# Save results
with open("${BASELINE_FILE}", "w") as f:
    json.dump(results, f, indent=2)

print(f"   SUCCESS: Generated {len(results)} baseline responses")
EOF

if [ ! -f "${BASELINE_FILE}" ]; then
    echo "ERROR: Failed to generate baseline responses"
    exit 1
fi
echo ""

# Step 5: Generate steered responses (with control vector)
echo "Step 4/5: Generating steered responses..."
echo "   Output: ${STEERED_FILE}"
python << EOF
import json
import torch
from wisent.core.models.wisent_model import WisentModel

# Load model
print("   Loading model...")
model = WisentModel("${MODEL}")

# Load control vector
print("   Loading control vector...")
vector_data = torch.load("${VECTOR_FILE}", map_location=model.device)
# Handle both old format (vectors dict) and new format (single vector)
if "steering_vector" in vector_data:
    control_vector = vector_data["steering_vector"]
elif "vector" in vector_data:
    control_vector = vector_data["vector"]
elif "vectors" in vector_data:
    control_vector = vector_data["vectors"][${LAYERS}]
else:
    raise KeyError(f"Could not find vector in file. Keys: {list(vector_data.keys())}")

# Load prompts
with open("${PROMPTS_FILE}") as f:
    prompts_data = json.load(f)

# Set up steering
print("   Configuring steering...")
model.set_steering_from_raw(
    raw={"${LAYERS}": control_vector},
    scale=${STEERING_STRENGTH},
    normalize=True
)
model.apply_steering()

results = []
for item in prompts_data[:${NUM_PROMPTS}]:
    prompt_id = item["id"]
    prompt_text = item["prompt"]

    print(f"   Generating steered response {prompt_id}...")

    messages = [{"role": "user", "content": prompt_text}]
    response = model.generate(
        inputs=[messages],
        max_new_tokens=${MAX_NEW_TOKENS},
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
    )[0]

    results.append({
        "id": prompt_id,
        "prompt": prompt_text,
        "generated_response": response
    })

# Clean up steering hooks
model.clear_steering()

# Save results
with open("${STEERED_FILE}", "w") as f:
    json.dump(results, f, indent=2)

print(f"   SUCCESS: Generated {len(results)} steered responses")
EOF

if [ ! -f "${STEERED_FILE}" ]; then
    echo "ERROR: Failed to generate steered responses"
    exit 1
fi
echo ""

# Step 6: Evaluate steering effectiveness
echo "Step 5/5: Evaluating steering effectiveness..."
echo "   Output: ${EVAL_FILE}"
python -m wisent.core.main evaluate-responses \
    --input "${STEERED_FILE}" \
    --baseline "${BASELINE_FILE}" \
    --output "${EVAL_FILE}" \
    --task personalization \
    --trait "${TRAIT}" \
    --trait-description "${TRAIT_DESCRIPTION}" \
    --verbose

if [ ! -f "${EVAL_FILE}" ]; then
    echo "ERROR: Failed to evaluate responses"
    exit 1
fi
echo ""

# Display summary
echo "=================================="
echo "PIPELINE COMPLETE"
echo "=================================="
echo "Results:"
echo "  Control vector: ${VECTOR_FILE}"
echo "  Baseline responses: ${BASELINE_FILE}"
echo "  Steered responses: ${STEERED_FILE}"
echo "  Evaluation: ${EVAL_FILE}"
echo ""
echo "View evaluation results:"
echo "  cat ${EVAL_FILE} | python -m json.tool"
echo ""

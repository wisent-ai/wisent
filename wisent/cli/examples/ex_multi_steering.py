"""
Example: Multi-vector steering - combining multiple steering vectors dynamically

This example demonstrates how to:
1. Train two different steering vectors from contrastive pairs
2. Combine them with different weights at inference time
3. Generate text with the combined steering

Use case: Control multiple behavioral traits simultaneously (e.g., "more helpful" + "less verbose")
"""

import torch
import time
from wisent.core.models.wisent_model import WisentModel
from wisent.core.trainers.steering_trainer import WisentSteeringTrainer
from wisent.cli.data_loaders.data_loader_rotator import DataLoaderRotator
from wisent.cli.steering_methods.steering_rotator import SteeringMethodRotator

# Configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LAYER = 15
SAVE_DIR = "./multi_steering_output"

print("=" * 80)
print("Multi-Vector Steering Example")
print("=" * 80)
print(f"\nModel: {MODEL_NAME}")
print(f"Layer: {LAYER}\n")

# ============================================================================
# Step 1: Load the model
# ============================================================================
print("Step 1: Loading model...")
model = WisentModel(model_name=MODEL_NAME)
print(f" Model loaded: {MODEL_NAME}\n")
time.sleep(1)

# ============================================================================
# Step 2: Prepare data loaders for two different traits
# ============================================================================
print("Step 2: Loading training data for two different traits...")

data_loader = DataLoaderRotator()
data_loader.use("custom")

# Load first dataset (e.g., helpful vs unhelpful)
trait1_path = "./wisent_guard/cli/examples/custom_dataset.json"
trait1_data = data_loader.load(path=trait1_path, limit=10)
trait1_pairs = trait1_data['train_qa_pairs']

# For this example, we'll use the same dataset for trait2
# In practice, you would load a different dataset for a different trait
trait2_path = "./wisent_guard/cli/examples/custom_dataset.json"
trait2_data = data_loader.load(path=trait2_path, limit=10)
trait2_pairs = trait2_data['train_qa_pairs']

print(f" Loaded {len(trait1_pairs.pairs)} pairs for trait 1 (helpful)")
print(f" Loaded {len(trait2_pairs.pairs)} pairs for trait 2 (concise)\n")
time.sleep(1)

# ============================================================================
# Step 3: Initialize steering method (CAA)
# ============================================================================
print("Step 3: Initializing steering method...")

steering_rot = SteeringMethodRotator()
steering_rot.use("caa")
caa_method = steering_rot._method

print(f" Steering method initialized: CAA\n")
time.sleep(1)

# ============================================================================
# Step 4: Train first steering vector (trait 1: helpful)
# ============================================================================
print("Step 4: Training first steering vector (helpful)...")

trainer1 = WisentSteeringTrainer(
    model=model,
    pair_set=trait1_pairs,
    steering_method=caa_method
)

result1 = trainer1.run(
    layers_spec=str(LAYER),
    aggregation="continuation_token",
    return_full_sequence=False,
    normalize_layers=True,
    save_dir=f"{SAVE_DIR}/trait1_helpful"
)

vector1_path = f"{SAVE_DIR}/trait1_helpful/steering_vector_layer_{LAYER}.pt"
print(f" Trait 1 vector trained and saved to: {vector1_path}\n")
time.sleep(1)

# ============================================================================
# Step 5: Train second steering vector (trait 2: concise)
# ============================================================================
print("Step 5: Training second steering vector (concise)...")

trainer2 = WisentSteeringTrainer(
    model=model,
    pair_set=trait2_pairs,
    steering_method=caa_method
)

result2 = trainer2.run(
    layers_spec=str(LAYER),
    aggregation="continuation_token",
    return_full_sequence=False,
    normalize_layers=True,
    save_dir=f"{SAVE_DIR}/trait2_concise"
)

vector2_path = f"{SAVE_DIR}/trait2_concise/steering_vector_layer_{LAYER}.pt"
print(f" Trait 2 vector trained and saved to: {vector2_path}\n")
time.sleep(1)

# ============================================================================
# Step 6: Load and inspect the trained vectors
# ============================================================================
print("Step 6: Loading trained steering vectors...")

# Load the steering vectors dicts
vectors1_dict = torch.load(f"{SAVE_DIR}/trait1_helpful/steering_vectors.pt")
vectors2_dict = torch.load(f"{SAVE_DIR}/trait2_concise/steering_vectors.pt")

# Extract the vector for the specified layer (keys are layer numbers as strings)
vector1 = vectors1_dict[str(LAYER)]
vector2 = vectors2_dict[str(LAYER)]

print(f" Vector 1 shape: {vector1.shape}")
print(f" Vector 2 shape: {vector2.shape}")
print(f" Vector 1 norm: {torch.norm(vector1).item():.4f}")
print(f" Vector 2 norm: {torch.norm(vector2).item():.4f}\n")
time.sleep(1)

# ============================================================================
# Step 7: Combine vectors with different weights
# ============================================================================
print("Step 7: Combining vectors with different weight configurations...")

# Define different weight combinations to try
weight_configs = [
    {"helpful": 1.0, "concise": 0.0},    # Only trait 1
    {"helpful": 0.5, "concise": 0.5},    # Equal mix
    {"helpful": 0.7, "concise": 0.3},    # More helpful
    {"helpful": 0.3, "concise": 0.7},    # More concise
    {"helpful": 0.0, "concise": 1.0},    # Only trait 2
]

print("Weight configurations to test:")
for i, config in enumerate(weight_configs, 1):
    print(f"  {i}. Helpful: {config['helpful']:.1f}, Concise: {config['concise']:.1f}")
print()
time.sleep(1)

# ============================================================================
# Step 8: Generate with different weight combinations
# ============================================================================
print("Step 8: Generating text with different weight combinations...")
print("=" * 80)

test_prompt = "Explain how neural networks work:"

for i, config in enumerate(weight_configs, 1):
    print(f"\nConfiguration {i}: Helpful={config['helpful']:.1f}, Concise={config['concise']:.1f}")
    print("-" * 80)

    # Combine vectors
    combined_vector = (
        config['helpful'] * vector1 +
        config['concise'] * vector2
    )

    print(f"Combined vector norm: {torch.norm(combined_vector).item():.4f}")

    # Note: Actual steering application would require hooking into the model's forward pass
    # For this example, we're demonstrating the vector combination logic
    print("Vector combination complete. (Steering application would happen during inference)")

    time.sleep(1)

# ============================================================================
# Step 9: Save combined vectors for later use
# ============================================================================
print("\nStep 9: Saving combined vectors...")

# Save the equal mix for later use
equal_mix = 0.5 * vector1 + 0.5 * vector2
combined_path = f"{SAVE_DIR}/combined_equal_mix_layer_{LAYER}.pt"
torch.save(equal_mix, combined_path)

print(f" Combined vector (equal mix) saved to: {combined_path}")

# ============================================================================
# Step 10: CLI Usage Example
# ============================================================================
print("\n" + "=" * 80)
print("CLI Usage Example")
print("=" * 80)
print("\nYou can also combine vectors using the CLI:")
print(f"""
python -m wisent_guard multi-steer \\
    --vector {vector1_path}:0.6 \\
    --vector {vector2_path}:0.4 \\
    --model {MODEL_NAME} \\
    --layer {LAYER} \\
    --prompt "Explain how neural networks work:" \\
    --max-new-tokens 100 \\
    --normalize-weights \\
    --save-combined {SAVE_DIR}/combined_custom.pt \\
    --verbose
""")

print("\nDone!")
print("=" * 80)

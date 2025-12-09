"""
Humanization optimization script for AWS.
Optimizes a model to produce human-like outputs while maintaining coherence.
"""

import torch
import json
import optuna
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from wisent.core.evaluators.custom.examples.humanization_coherent import HumanizationCoherentEvaluator
from wisent.core.models.wisent_model import WisentModel
from wisent.core.models.inference_config import get_generate_kwargs

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/home/ubuntu/output")

print("Loading 3B model...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct", 
    torch_dtype=torch.float16, 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
base_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

num_layers = len(model.model.layers)
print(f"Model has {num_layers} layers")

print("Generating steering vectors from contrastive pairs...")
from wisent.core.steering.extractor import extract_steering_vectors_from_pairs

# Load humanization pairs
pairs_path = os.path.join(os.path.dirname(__file__), "../wisent/examples/contrastive_pairs/humanization_human_vs_ai.json")
if not os.path.exists(pairs_path):
    # Try alternative path
    pairs_path = "/opt/wisent-venv/lib/python3.10/site-packages/wisent/examples/contrastive_pairs/humanization_human_vs_ai.json"

with open(pairs_path, "r") as f:
    pairs_data = json.load(f)

# Extract steering vectors
vectors_data = extract_steering_vectors_from_pairs(
    model=model,
    tokenizer=tokenizer,
    pairs=pairs_data["pairs"],
    layers=list(range(1, num_layers + 1)),
)

print("Loading HumanizationCoherent evaluator (Desklib + coherence)...")
evaluator = HumanizationCoherentEvaluator(coherence_threshold=50.0)

eval_prompts = [
    "What is the meaning of life?",
    "Explain how computers work.",
    "Tell me about yourself.",
    "What makes a good leader?",
    "How do I learn a new language?",
    "What are the benefits of exercise?",
    "Describe a beautiful sunset.",
    "How does the internet work?",
    "Explain climate change.",
    "What is love?",
]


def objective(trial):
    layer = trial.suggest_int("layer", 1, num_layers)
    strength = trial.suggest_float("strength", 0.5, 5.0)

    model.load_state_dict(base_state_dict)

    layer_idx = layer - 1
    vector = torch.tensor(
        vectors_data["steering_vectors"][str(layer)], dtype=torch.float16
    ).to(model.device)

    steering = strength * vector.unsqueeze(0)
    orig_shape = model.model.layers[layer_idx].mlp.down_proj.weight.data.shape
    model.model.layers[layer_idx].mlp.down_proj.weight.data += steering.T[
        : orig_shape[0], : orig_shape[1]
    ]

    wisent_model = WisentModel("meta-llama/Llama-3.2-3B-Instruct", hf_model=model)

    scores = []
    for prompt in eval_prompts:
        messages = [{"role": "user", "content": prompt}]
        result = wisent_model.generate(
            [messages], **get_generate_kwargs(max_new_tokens=150)
        )
        response = result[0] if result else ""

        score_result = evaluator(response)
        scores.append(score_result["score"])

    avg_score = sum(scores) / len(scores)
    print(f"  Trial {trial.number}: layer={layer}, strength={strength:.2f}, score={avg_score:.1%}")

    return avg_score


print("\nStarting optimization with FULL search space...")
print(f"  Layers: 1-{num_layers}")
print(f"  Strength: 0.5-5.0")
print(f"  Trials: 200 (default)")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)

print(f"\n\nBest trial:")
print(f'  Layer: {study.best_params["layer"]}')
print(f'  Strength: {study.best_params["strength"]:.2f}')
print(f"  Score: {study.best_value:.1%}")

print("\nSaving best model...")
model.load_state_dict(base_state_dict)
layer = study.best_params["layer"]
strength = study.best_params["strength"]
layer_idx = layer - 1
vector = torch.tensor(
    vectors_data["steering_vectors"][str(layer)], dtype=torch.float16
).to(model.device)
steering = strength * vector.unsqueeze(0)
orig_shape = model.model.layers[layer_idx].mlp.down_proj.weight.data.shape
model.model.layers[layer_idx].mlp.down_proj.weight.data += steering.T[
    : orig_shape[0], : orig_shape[1]
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(os.path.join(OUTPUT_DIR, "humanized_llama_3b_coherent"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "humanized_llama_3b_coherent"))

with open(os.path.join(OUTPUT_DIR, "optimization_results.json"), "w") as f:
    json.dump(
        {"best_params": study.best_params, "best_score": study.best_value}, f, indent=2
    )

print(f"Done! Model saved to {OUTPUT_DIR}/humanized_llama_3b_coherent")

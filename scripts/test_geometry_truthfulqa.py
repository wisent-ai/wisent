#!/usr/bin/env python3
"""
Run geometry detection on TruthfulQA to understand why signal is weak.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from datasets import load_dataset
import random
import sys
sys.path.insert(0, "/Users/lukaszbartoszcze/Documents/CodingProjects/Wisent/backends/wisent-open-source")

from wisent.core.contrastive_pairs.diagnostics import detect_geometry_structure, GeometryAnalysisConfig

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "mps"

print(f"Loading {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map=DEVICE)
num_layers = model.config.num_hidden_layers
print(f"Model has {num_layers} layers")

# Load TruthfulQA - use ALL available pairs
ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
pairs = []
for s in ds:
    if s["incorrect_answers"]:
        pairs.append({
            "question": s["question"],
            "positive": s["best_answer"],
            "negative": random.choice(s["incorrect_answers"]),
        })
print(f"Loaded {len(pairs)} TruthfulQA pairs (all available)")

random_tokens = ["I", "Well", "The", "Sure", "Let", "That", "It", "This", "My", "To"]

def get_activations(text, layers):
    """Get activations at all specified layers."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    result = {}
    for layer in layers:
        result[layer] = outputs.hidden_states[layer][0, -1, :].cpu().float()
    return result

# Test all layers
test_layers = list(range(1, num_layers))

# Strategies to test
STRATEGIES = {
    "chat_last": lambda q, ans: tokenizer.apply_chat_template(
        [{"role": "user", "content": q}, {"role": "assistant", "content": ans}],
        tokenize=False, add_generation_prompt=False
    ),
    "role_play": lambda q, ans: tokenizer.apply_chat_template(
        [{"role": "user", "content": f"Behave like a person that would answer {q} with {ans}"},
         {"role": "assistant", "content": random_tokens[hash(q) % len(random_tokens)]}],
        tokenize=False, add_generation_prompt=False
    ),
}

geo_config = GeometryAnalysisConfig(
    num_components=5,
    optimization_steps=100,
)

print("\n" + "="*80)
print("Geometry Analysis on TruthfulQA")
print("="*80)

for strategy_name, build_prompt in STRATEGIES.items():
    print(f"\n--- Strategy: {strategy_name} ---")
    
    for layer in test_layers:
        pos_acts = []
        neg_acts = []
        
        for pair in pairs:
            q = pair["question"][:500]
            pos_text = build_prompt(q, pair["positive"])
            neg_text = build_prompt(q, pair["negative"])
            
            pos_act = get_activations(pos_text, [layer])[layer]
            neg_act = get_activations(neg_text, [layer])[layer]
            
            pos_acts.append(pos_act)
            neg_acts.append(neg_act)
        
        pos_tensor = torch.stack(pos_acts)
        neg_tensor = torch.stack(neg_acts)
        
        result = detect_geometry_structure(pos_tensor, neg_tensor, geo_config)
        
        print(f"  Layer {layer:2d}: best={result.best_structure.value:<10} "
              f"score={result.best_score:.3f} "
              f"linear={result.all_scores['linear'].score:.3f} "
              f"cone={result.all_scores['cone'].score:.3f} "
              f"cluster={result.all_scores['cluster'].score:.3f}")

print("\nDone")

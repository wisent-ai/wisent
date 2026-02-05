#!/usr/bin/env python3
"""
Test classifier probe on TruthfulQA - can we distinguish true vs false answers?
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
import random

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "mps"

print(f"Loading {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map=DEVICE)
num_layers = model.config.num_hidden_layers
print(f"Model has {num_layers} layers")

# Load TruthfulQA
ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
pairs = []
for s in ds:
    if s["incorrect_answers"]:
        pairs.append({
            "question": s["question"],
            "positive": s["best_answer"],
            "negative": random.choice(s["incorrect_answers"]),
        })
print(f"Loaded {len(pairs)} TruthfulQA pairs")

def get_all_layer_activations(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    return [outputs.hidden_states[l][0, -1, :].cpu().float().numpy() for l in range(num_layers + 1)]

# Collect activations
test_layers = [4, 8, 12, 14]

print("\nCollecting activations...")
pos_acts_by_layer = {l: [] for l in test_layers}
neg_acts_by_layer = {l: [] for l in test_layers}

for i, pair in enumerate(pairs):
    if i % 100 == 0:
        print(f"  {i}/{len(pairs)}")
    
    q = pair["question"][:500]
    pos_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": q}, {"role": "assistant", "content": pair["positive"]}],
        tokenize=False, add_generation_prompt=False
    )
    neg_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": q}, {"role": "assistant", "content": pair["negative"]}],
        tokenize=False, add_generation_prompt=False
    )
    
    pos_all = get_all_layer_activations(pos_text)
    neg_all = get_all_layer_activations(neg_text)
    
    for layer in test_layers:
        pos_acts_by_layer[layer].append(pos_all[layer])
        neg_acts_by_layer[layer].append(neg_all[layer])

print("\n" + "="*80)
print("Classifier Results (5-fold cross-validation)")
print("="*80)

for layer in test_layers:
    pos_X = np.array(pos_acts_by_layer[layer])
    neg_X = np.array(neg_acts_by_layer[layer])
    
    # Create dataset: X = activations, y = 1 for true, 0 for false
    X = np.vstack([pos_X, neg_X])
    y = np.array([1] * len(pos_X) + [0] * len(neg_X))
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    # Linear probe (logistic regression)
    lr = LogisticRegression( random_state=42)
    lr_scores = cross_val_score(lr, X, y, cv=5, scoring='accuracy')
    
    # Non-linear probe (MLP)
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128),  random_state=42)
    mlp_scores = cross_val_score(mlp, X, y, cv=5, scoring='accuracy')
    
    print(f"\nLayer {layer}:")
    print(f"  Logistic Regression: {lr_scores.mean():.3f} (+/- {lr_scores.std()*2:.3f})")
    print(f"  MLP (256,128):       {mlp_scores.mean():.3f} (+/- {mlp_scores.std()*2:.3f})")

print("\n" + "="*80)
print("Baseline: random = 0.500")
print("="*80)

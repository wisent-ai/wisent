#!/usr/bin/env python3
"""
Compare CAA mean vector vs classifier learned weights.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
import random

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "mps"

print(f"Loading {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map=DEVICE)
num_layers = model.config.num_hidden_layers

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
print(f"Loaded {len(pairs)} pairs")

def get_activation(text, layer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    return outputs.hidden_states[layer][0, -1, :].cpu().float().numpy()

layer = 8
print(f"\nCollecting activations at layer {layer}...")

pos_acts = []
neg_acts = []
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
    pos_acts.append(get_activation(pos_text, layer))
    neg_acts.append(get_activation(neg_text, layer))

pos_acts = np.array(pos_acts)
neg_acts = np.array(neg_acts)

# CAA vector: mean of differences
diffs = pos_acts - neg_acts
caa_vector = diffs.mean(axis=0)
caa_vector_norm = caa_vector / (np.linalg.norm(caa_vector) + 1e-10)

# Classifier weights
X = np.vstack([pos_acts, neg_acts])
y = np.array([1] * len(pos_acts) + [0] * len(neg_acts))
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)
clf_weights = clf.coef_[0]
clf_weights_norm = clf_weights / (np.linalg.norm(clf_weights) + 1e-10)

# Compare
cosine_sim = np.dot(caa_vector_norm, clf_weights_norm)
print(f"\n{'='*60}")
print(f"CAA vector magnitude: {np.linalg.norm(caa_vector):.4f}")
print(f"Classifier weights magnitude: {np.linalg.norm(clf_weights):.4f}")
print(f"\nCosine similarity (CAA vs Classifier): {cosine_sim:.4f}")
print(f"{'='*60}")

# Test: how well does CAA vector work as a classifier?
caa_scores = X @ caa_vector_norm
caa_pred = (caa_scores > np.median(caa_scores)).astype(int)
caa_acc = (caa_pred == y).mean()

clf_acc = clf.score(X, y)

print(f"\nClassifier accuracy (train): {clf_acc:.3f}")
print(f"CAA vector as classifier:    {caa_acc:.3f}")
print(f"Random baseline:             0.500")

# Also test: what if we use mean(pos) - mean(neg) instead of mean(pos-neg)?
mean_pos = pos_acts.mean(axis=0)
mean_neg = neg_acts.mean(axis=0)
centroid_diff = mean_pos - mean_neg
centroid_diff_norm = centroid_diff / (np.linalg.norm(centroid_diff) + 1e-10)

cosine_centroid_clf = np.dot(centroid_diff_norm, clf_weights_norm)
print(f"\nCosine similarity (centroid diff vs Classifier): {cosine_centroid_clf:.4f}")

centroid_scores = X @ centroid_diff_norm
centroid_pred = (centroid_scores > np.median(centroid_scores)).astype(int)
centroid_acc = (centroid_pred == y).mean()
print(f"Centroid diff as classifier: {centroid_acc:.3f}")

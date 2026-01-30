#!/usr/bin/env python3
"""
Simple test: does classifier-based steering push activations toward "true" side?
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
import random

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "mps"
LAYER = 8

print(f"Loading {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map=DEVICE)

# Load TruthfulQA
ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
random.seed(42)
all_pairs = [(s["question"], s["best_answer"], random.choice(s["incorrect_answers"])) 
             for s in ds if s["incorrect_answers"]]

train_pairs = all_pairs[:100]
test_pairs = all_pairs[100:120]
print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")

def get_activation(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = model(inputs.input_ids, output_hidden_states=True)
    return out.hidden_states[LAYER][0, -1, :].cpu().float().numpy()

# Collect training activations
print("Collecting train activations...")
pos_acts, neg_acts = [], []
for q, pos, neg in train_pairs:
    pos_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": q}, {"role": "assistant", "content": pos}],
        tokenize=False, add_generation_prompt=False)
    neg_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": q}, {"role": "assistant", "content": neg}],
        tokenize=False, add_generation_prompt=False)
    pos_acts.append(get_activation(pos_text))
    neg_acts.append(get_activation(neg_text))

pos_acts = np.array(pos_acts)
neg_acts = np.array(neg_acts)

# Train classifier
X = np.vstack([pos_acts, neg_acts])
y = np.array([1]*len(pos_acts) + [0]*len(neg_acts))
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)
clf_vector = clf.coef_[0]
clf_vector = clf_vector / np.linalg.norm(clf_vector)

# CAA vector
caa_vector = (pos_acts - neg_acts).mean(axis=0)
caa_vector = caa_vector / (np.linalg.norm(caa_vector) + 1e-10)

print(f"\nCAA magnitude (pre-norm): {np.linalg.norm((pos_acts - neg_acts).mean(axis=0)):.4f}")
print(f"Cosine(CAA, CLF): {np.dot(caa_vector, clf_vector):.4f}")

# Test on held-out: measure how much steering moves activations toward "true" side
print("\n" + "="*60)
print("Testing on held-out data")
print("="*60)

def score_toward_true(act):
    """Higher = more toward true side of classifier boundary"""
    return np.dot(act, clf_vector)

results = []
for q, pos, neg in test_pairs:
    # Get prompt (question only, no answer)
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True)
    
    # Get baseline activation at generation point
    baseline_act = get_activation(prompt_text)
    baseline_score = score_toward_true(baseline_act)
    
    # Simulate steering: baseline + alpha * steering_vector
    for alpha in [0.5, 1.0, 2.0]:
        steered_caa = baseline_act + alpha * caa_vector
        steered_clf = baseline_act + alpha * clf_vector
        
        score_caa = score_toward_true(steered_caa)
        score_clf = score_toward_true(steered_clf)
        
        results.append({
            'alpha': alpha,
            'baseline': baseline_score,
            'caa': score_caa,
            'clf': score_clf,
            'caa_improvement': score_caa - baseline_score,
            'clf_improvement': score_clf - baseline_score,
        })

# Aggregate results
print("\nSteering effect on held-out data (20 questions):")
print("(Higher score = closer to 'true' side of decision boundary)")
print()
for alpha in [0.5, 1.0, 2.0]:
    subset = [r for r in results if r['alpha'] == alpha]
    avg_baseline = np.mean([r['baseline'] for r in subset])
    avg_caa = np.mean([r['caa_improvement'] for r in subset])
    avg_clf = np.mean([r['clf_improvement'] for r in subset])
    print(f"Alpha {alpha}: CAA improvement={avg_caa:+.4f}, CLF improvement={avg_clf:+.4f}")

print("\nDone")

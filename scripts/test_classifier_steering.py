#!/usr/bin/env python3
"""
Test classifier-based steering vs CAA steering on TruthfulQA.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
import random

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "mps"

print(f"Loading {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map=DEVICE)
num_layers = model.config.num_hidden_layers

# Load TruthfulQA
ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
all_pairs = []
for s in ds:
    if s["incorrect_answers"]:
        all_pairs.append({
            "question": s["question"],
            "positive": s["best_answer"],
            "negative": random.choice(s["incorrect_answers"]),
            "correct_answers": s["correct_answers"],
            "incorrect_answers": s["incorrect_answers"],
        })

# Split: use smaller set for CPU
random.seed(42)
random.shuffle(all_pairs)
train_pairs = all_pairs[:200]  # Smaller for CPU
test_pairs = all_pairs[200:230]  # 30 test samples
print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")

LAYER = 8  # Best layer from earlier results

def get_activation(text, layer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    return outputs.hidden_states[layer][0, -1, :].cpu().float().numpy()

# Collect training activations
print(f"\nCollecting training activations at layer {LAYER}...")
pos_acts = []
neg_acts = []
for i, pair in enumerate(train_pairs):
    if i % 100 == 0:
        print(f"  {i}/{len(train_pairs)}")
    q = pair["question"][:500]
    pos_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": q}, {"role": "assistant", "content": pair["positive"]}],
        tokenize=False, add_generation_prompt=False
    )
    neg_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": q}, {"role": "assistant", "content": pair["negative"]}],
        tokenize=False, add_generation_prompt=False
    )
    pos_acts.append(get_activation(pos_text, LAYER))
    neg_acts.append(get_activation(neg_text, LAYER))

pos_acts = np.array(pos_acts)
neg_acts = np.array(neg_acts)

# CAA vector
caa_vector = (pos_acts - neg_acts).mean(axis=0)
caa_vector = caa_vector / (np.linalg.norm(caa_vector) + 1e-10)

# Classifier vector
X = np.vstack([pos_acts, neg_acts])
y = np.array([1] * len(pos_acts) + [0] * len(neg_acts))
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)
clf_vector = clf.coef_[0]
clf_vector = clf_vector / (np.linalg.norm(clf_vector) + 1e-10)

print(f"\nCAA magnitude (before norm): {np.linalg.norm((pos_acts - neg_acts).mean(axis=0)):.4f}")
print(f"Cosine(CAA, Classifier): {np.dot(caa_vector, clf_vector):.4f}")

# Steering hook
steering_vector = None
steering_alpha = 0.0

def steering_hook(module, input, output):
    global steering_vector, steering_alpha
    if steering_vector is not None and steering_alpha != 0.0:
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        sv = torch.tensor(steering_vector, dtype=hidden.dtype, device=hidden.device)
        if hidden.dim() == 3:
            hidden[:, -1, :] += steering_alpha * sv
        elif hidden.dim() == 2:
            hidden[-1, :] += steering_alpha * sv
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        else:
            return hidden
    return output

# Register hook at target layer
hook_handle = model.model.layers[LAYER].register_forward_hook(steering_hook)

def generate_answer(question, alpha=0.0, vector=None):
    global steering_vector, steering_alpha
    steering_vector = vector
    steering_alpha = alpha
    
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def check_answer(response, correct_answers, incorrect_answers):
    """Check if response matches correct or incorrect answers."""
    response_lower = response.lower()
    for correct in correct_answers:
        if correct.lower() in response_lower:
            return "correct"
    for incorrect in incorrect_answers:
        if incorrect.lower() in response_lower:
            return "incorrect"
    return "unknown"

# Test on held-out questions
print("\n" + "="*80)
print("Testing steering on held-out questions")
print("="*80)

results = {"baseline": [], "caa": [], "classifier": []}
alphas = [0.0, 1.0, 2.0, 3.0]

for alpha in alphas:
    print(f"\n--- Alpha = {alpha} ---")
    
    correct_baseline = 0
    correct_caa = 0
    correct_clf = 0
    total = 0
    
    for i, pair in enumerate(test_pairs):  # Test on all test samples
        q = pair["question"]
        
        # Baseline (no steering)
        if alpha == 0.0:
            resp_baseline = generate_answer(q, alpha=0.0, vector=None)
            result_baseline = check_answer(resp_baseline, pair["correct_answers"], pair["incorrect_answers"])
            if result_baseline == "correct":
                correct_baseline += 1
        
        # CAA steering
        resp_caa = generate_answer(q, alpha=alpha, vector=caa_vector)
        result_caa = check_answer(resp_caa, pair["correct_answers"], pair["incorrect_answers"])
        if result_caa == "correct":
            correct_caa += 1
        
        # Classifier steering
        resp_clf = generate_answer(q, alpha=alpha, vector=clf_vector)
        result_clf = check_answer(resp_clf, pair["correct_answers"], pair["incorrect_answers"])
        if result_clf == "correct":
            correct_clf += 1
        
        total += 1
        
        if i < 3:  # Show first 3 examples
            print(f"\nQ: {q[:80]}...")
            if alpha == 0.0:
                print(f"  Baseline: {resp_baseline[:60]}... [{result_baseline}]")
            print(f"  CAA({alpha}): {resp_caa[:60]}... [{result_caa}]")
            print(f"  CLF({alpha}): {resp_clf[:60]}... [{result_clf}]")
    
    if alpha == 0.0:
        print(f"\nBaseline: {correct_baseline}/{total} = {correct_baseline/total:.1%}")
    print(f"CAA (alpha={alpha}): {correct_caa}/{total} = {correct_caa/total:.1%}")
    print(f"Classifier (alpha={alpha}): {correct_clf}/{total} = {correct_clf/total:.1%}")

hook_handle.remove()
print("\nDone")

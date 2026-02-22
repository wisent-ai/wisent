#!/usr/bin/env python3
"""
Show actual response changes from steering.
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
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map=DEVICE)

ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
random.seed(42)
all_pairs = [(s["question"], s["best_answer"], s["correct_answers"], s["incorrect_answers"]) 
             for s in ds if s["incorrect_answers"]]

train_pairs = all_pairs[:100]
test_pairs = all_pairs[100:130]

def get_activation(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = model(inputs.input_ids, output_hidden_states=True)
    return out.hidden_states[LAYER][0, -1, :].cpu().float().numpy()

print("Collecting train activations...")
pos_acts, neg_acts = [], []
for q, best, correct, incorrect in train_pairs:
    neg = random.choice(incorrect)
    pos_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": q}, {"role": "assistant", "content": best}],
        tokenize=False, add_generation_prompt=False)
    neg_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": q}, {"role": "assistant", "content": neg}],
        tokenize=False, add_generation_prompt=False)
    pos_acts.append(get_activation(pos_text))
    neg_acts.append(get_activation(neg_text))

pos_acts = np.array(pos_acts)
neg_acts = np.array(neg_acts)

X = np.vstack([pos_acts, neg_acts])
y = np.array([1]*len(pos_acts) + [0]*len(neg_acts))
clf = LogisticRegression()
clf.fit(X, y)
clf_vector = clf.coef_[0]
clf_vector_norm = clf_vector / np.linalg.norm(clf_vector)

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
        return hidden
    return output

hook = model.model.layers[LAYER].register_forward_hook(steering_hook)

def generate(question, alpha=0.0):
    global steering_vector, steering_alpha
    steering_vector = clf_vector_norm if alpha != 0 else None
    steering_alpha = alpha
    
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        out = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

print("\n" + "="*80)
print("Testing steering on held-out questions")
print("="*80)

for i, (q, best, correct, incorrect) in enumerate(test_pairs[:10]):
    print(f"\n{'='*80}")
    print(f"Q: {q}")
    print(f"\nCorrect answers: {correct[:2]}")
    print(f"Incorrect answers: {incorrect[:2]}")
    
    baseline = generate(q, alpha=0.0)
    steered = generate(q, alpha=2.0)
    
    print(f"\nBASELINE: {baseline[:200]}")
    print(f"\nSTEERED (alpha=2): {steered[:200]}")

hook.remove()

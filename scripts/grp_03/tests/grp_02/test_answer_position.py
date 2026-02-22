#!/usr/bin/env python3
"""
Test where the answer appears relative to extraction point.

Hypothesis: Signal is stronger when answer is BEFORE extraction point (model has processed it).
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from datasets import load_dataset
import random

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "mps"
N_SAMPLES = 30
TEST_LAYERS = [3, 4, 5, 6, 7, 8, 10, 12]

print(f"Loading {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map=DEVICE)
num_layers = model.config.num_hidden_layers
print(f"Model has {num_layers} layers")

# Load TruthfulQA
ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
random.seed(42)
samples = random.sample(list(ds), N_SAMPLES)
pairs = []
for s in samples:
    if s["incorrect_answers"]:
        pairs.append({
            "question": s["question"],
            "positive": s["best_answer"],
            "negative": random.choice(s["incorrect_answers"]),
        })
print(f"Loaded {len(pairs)} pairs")

RANDOM_TOKENS = ["I", "Well", "The", "Sure", "Let", "That", "It", "This", "My", "To"]


def get_last_token_act(text, layer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    return outputs.hidden_states[layer][0, -1, :].cpu().float().numpy()


def compute_cosine(diffs):
    if len(diffs) < 2:
        return 0
    cosines = []
    for i in range(len(diffs)):
        for j in range(i+1, len(diffs)):
            cos = np.dot(diffs[i], diffs[j]) / (np.linalg.norm(diffs[i]) * np.linalg.norm(diffs[j]) + 1e-8)
            cosines.append(cos)
    return np.mean(cosines)


# Strategy 1: BASELINE - answer AFTER extraction (chat_mean style)
# [user: Q][assistant: A] -> extract at last token of A
def build_answer_after(q, answer):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": q}, {"role": "assistant", "content": answer}],
        tokenize=False, add_generation_prompt=False
    )


# Strategy 2: Answer in SYSTEM, extract at random token after user msg
# [system: "Context: {answer}"][user: Q][assistant: {random}]
def build_answer_in_system(q, answer):
    random_token = RANDOM_TOKENS[hash(q) % len(RANDOM_TOKENS)]
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": f"Context: {answer}"}, 
         {"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True
    ) + random_token


# Strategy 3: Answer in USER prefix, extract at random token
# [user: "Given: {answer}. Question: {Q}"][assistant: {random}]
def build_answer_in_user(q, answer):
    random_token = RANDOM_TOKENS[hash(q) % len(RANDOM_TOKENS)]
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": f"Given this answer: {answer}\n\nQuestion: {q}"}],
        tokenize=False, add_generation_prompt=True
    ) + random_token


# Strategy 4: Answer as instruction, extract at random token
# [system: "You believe: {answer}"][user: Q][assistant: {random}]
def build_answer_as_belief(q, answer):
    random_token = RANDOM_TOKENS[hash(q) % len(RANDOM_TOKENS)]
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": f"You believe the answer is: {answer}"}, 
         {"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True
    ) + random_token


# Strategy 5: Persona + answer (like role_play)
# [system: "You are truthful. Answer: {answer}"][user: Q][assistant: {random}]
def build_persona_answer(q, answer, is_positive):
    random_token = RANDOM_TOKENS[hash(q) % len(RANDOM_TOKENS)]
    if is_positive:
        persona = "You are truthful and accurate."
    else:
        persona = "You spread misinformation."
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": f"{persona} Your answer: {answer}"}, 
         {"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True
    ) + random_token


# Strategy 6: Just answer in system, no persona
def build_just_answer_system(q, answer):
    random_token = RANDOM_TOKENS[hash(q) % len(RANDOM_TOKENS)]
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": answer}, 
         {"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True
    ) + random_token


# Strategy 7: Answer after question mark but before response
# [user: Q + " The answer is: {answer}"][assistant: {random}]
def build_answer_suffix(q, answer):
    random_token = RANDOM_TOKENS[hash(q) % len(RANDOM_TOKENS)]
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{q}\n\nThe answer is: {answer}"}],
        tokenize=False, add_generation_prompt=True
    ) + random_token


STRATEGIES = {
    "answer_after (baseline)": lambda q, a, is_pos: build_answer_after(q, a),
    "answer_in_system": lambda q, a, is_pos: build_answer_in_system(q, a),
    "answer_in_user": lambda q, a, is_pos: build_answer_in_user(q, a),
    "answer_as_belief": lambda q, a, is_pos: build_answer_as_belief(q, a),
    "persona_answer": lambda q, a, is_pos: build_persona_answer(q, a, is_pos),
    "just_answer_system": lambda q, a, is_pos: build_just_answer_system(q, a),
    "answer_suffix": lambda q, a, is_pos: build_answer_suffix(q, a),
}


print("\n" + "="*80)
print("Testing answer position strategies")
print("="*80)

results = {}

for strategy_name, build_fn in STRATEGIES.items():
    print(f"\n--- {strategy_name} ---")
    results[strategy_name] = {}
    
    for layer in TEST_LAYERS:
        diffs = []
        for pair in pairs:
            q = pair["question"][:500]
            pos_text = build_fn(q, pair["positive"], True)
            neg_text = build_fn(q, pair["negative"], False)
            
            pos_act = get_last_token_act(pos_text, layer)
            neg_act = get_last_token_act(neg_text, layer)
            diffs.append(pos_act - neg_act)
        
        diffs = np.stack(diffs)
        cos = compute_cosine(diffs)
        results[strategy_name][layer] = cos
        print(f"  Layer {layer:2d}: cosine = {cos:.4f}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"{'Strategy':<25} {'Best Layer':<12} {'Best Cosine':<12}")
print("-"*50)

ranked = []
for strategy_name in STRATEGIES:
    if results[strategy_name]:
        best_layer = max(results[strategy_name], key=results[strategy_name].get)
        best_cos = results[strategy_name][best_layer]
        ranked.append((strategy_name, best_layer, best_cos))

ranked.sort(key=lambda x: x[2], reverse=True)
for name, layer, cos in ranked:
    print(f"{name:<25} {layer:<12} {cos:.4f}")

print("\nDone")

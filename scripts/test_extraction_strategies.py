#!/usr/bin/env python3
"""
Test different activation extraction strategies to find which produces best steering signal.

Strategies:
1. last_token - Extract at last token of full prompt+response
2. first_response_token - Extract at first token of assistant response  
3. mean_response - Mean pool across all response tokens
4. generation_point - Extract right before generation (after prompt, before response)
5. pca_diff - Use PCA on paired differences (RepE method)
6. per_token_max - Take token with maximum activation norm in response
7. answer_token - Extract at the actual answer content token
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from datasets import load_dataset
import random

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "mps"
N_SAMPLES = 30
TEST_LAYERS = [4, 6, 8, 10, 12]

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


def get_hidden_states(text):
    """Get all hidden states for all tokens."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    # hidden_states is tuple of (n_layers+1, batch, seq_len, hidden_dim)
    return outputs.hidden_states, inputs.input_ids[0]


def compute_cosine(diffs):
    """Compute mean pairwise cosine similarity."""
    if len(diffs) < 2:
        return 0
    cosines = []
    for i in range(len(diffs)):
        for j in range(i+1, len(diffs)):
            cos = np.dot(diffs[i], diffs[j]) / (np.linalg.norm(diffs[i]) * np.linalg.norm(diffs[j]) + 1e-8)
            cosines.append(cos)
    return np.mean(cosines)


def find_response_start(input_ids, tokenizer):
    """Find where assistant response starts."""
    text = tokenizer.decode(input_ids)
    # Look for common assistant markers
    markers = ["assistant\n", "Assistant:", "<|assistant|>", "ASSISTANT:"]
    for marker in markers:
        if marker in text:
            marker_tokens = tokenizer.encode(marker, add_special_tokens=False)
            # Find position after marker
            for i in range(len(input_ids) - len(marker_tokens)):
                if input_ids[i:i+len(marker_tokens)].tolist() == marker_tokens:
                    return i + len(marker_tokens)
    # Fallback: assume last 1/3 is response
    return len(input_ids) * 2 // 3


# Strategy 1: Last token (baseline)
def extract_last_token(hidden_states, input_ids, layer):
    return hidden_states[layer][0, -1, :].cpu().float().numpy()


# Strategy 2: First response token
def extract_first_response(hidden_states, input_ids, layer):
    resp_start = find_response_start(input_ids, tokenizer)
    return hidden_states[layer][0, resp_start, :].cpu().float().numpy()


# Strategy 3: Mean of response tokens
def extract_mean_response(hidden_states, input_ids, layer):
    resp_start = find_response_start(input_ids, tokenizer)
    resp_hidden = hidden_states[layer][0, resp_start:, :]
    return resp_hidden.mean(dim=0).cpu().float().numpy()


# Strategy 4: Generation point (right before response)
def extract_generation_point(hidden_states, input_ids, layer):
    resp_start = find_response_start(input_ids, tokenizer)
    # Token right before response starts
    gen_point = max(0, resp_start - 1)
    return hidden_states[layer][0, gen_point, :].cpu().float().numpy()


# Strategy 5: Max norm token in response
def extract_max_norm(hidden_states, input_ids, layer):
    resp_start = find_response_start(input_ids, tokenizer)
    resp_hidden = hidden_states[layer][0, resp_start:, :]
    norms = torch.norm(resp_hidden, dim=1)
    max_idx = torch.argmax(norms)
    return resp_hidden[max_idx].cpu().float().numpy()


# Strategy 6: Mean of ALL tokens (not just response)
def extract_mean_all(hidden_states, input_ids, layer):
    return hidden_states[layer][0].mean(dim=0).cpu().float().numpy()


# Strategy 7: Second to last token
def extract_second_last(hidden_states, input_ids, layer):
    return hidden_states[layer][0, -2, :].cpu().float().numpy()


STRATEGIES = {
    "last_token": extract_last_token,
    "first_response": extract_first_response,
    "mean_response": extract_mean_response,
    "generation_point": extract_generation_point,
    "max_norm": extract_max_norm,
    "mean_all": extract_mean_all,
    "second_last": extract_second_last,
}


def build_chat_prompt(question, answer):
    """Build prompt with chat template."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
        tokenize=False, add_generation_prompt=False
    )


print("\n" + "="*80)
print("Testing extraction strategies on TruthfulQA")
print("="*80)

results = {}

for strategy_name, extract_fn in STRATEGIES.items():
    print(f"\n--- Strategy: {strategy_name} ---")
    results[strategy_name] = {}
    
    for layer in TEST_LAYERS:
        diffs = []
        for pair in pairs:
            q = pair["question"]
            pos_text = build_chat_prompt(q, pair["positive"])
            neg_text = build_chat_prompt(q, pair["negative"])
            
            try:
                pos_hs, pos_ids = get_hidden_states(pos_text)
                neg_hs, neg_ids = get_hidden_states(neg_text)
                
                pos_act = extract_fn(pos_hs, pos_ids, layer)
                neg_act = extract_fn(neg_hs, neg_ids, layer)
                
                diffs.append(pos_act - neg_act)
            except Exception as e:
                continue
        
        if len(diffs) > 1:
            diffs = np.stack(diffs)
            cos = compute_cosine(diffs)
            results[strategy_name][layer] = cos
            print(f"  Layer {layer:2d}: cosine = {cos:.4f}")
        else:
            print(f"  Layer {layer:2d}: FAILED")

# Summary
print("\n" + "="*80)
print("SUMMARY: Best cosine per strategy")
print("="*80)
print(f"{'Strategy':<20} {'Best Layer':<12} {'Best Cosine':<12}")
print("-"*44)

for strategy_name in STRATEGIES:
    if strategy_name in results and results[strategy_name]:
        best_layer = max(results[strategy_name], key=results[strategy_name].get)
        best_cos = results[strategy_name][best_layer]
        print(f"{strategy_name:<20} {best_layer:<12} {best_cos:.4f}")

print("\nDone")

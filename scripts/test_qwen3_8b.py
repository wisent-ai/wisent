#!/usr/bin/env python3
"""
Test OUR prompt strategies (chat_mean, role_play, mc_balanced) on Llama-2-7B (CAA's model).
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from datasets import load_dataset
import random

model_name = "Qwen/Qwen3-8B"
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="mps")
num_layers = model.config.num_hidden_layers
print(f"Model has {num_layers} layers")

# Load TruthfulQA
ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
random.seed(42)
samples = random.sample(list(ds), 30)
pairs = [{"question": s["question"], "positive": s["best_answer"], 
          "negative": random.choice(s["incorrect_answers"])} for s in samples]
print(f"Loaded {len(pairs)} TruthfulQA pairs")

def get_last_token_act(text, layer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to("mps")
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    return outputs.hidden_states[layer][0, -1, :].cpu().float()

def compute_cosine(diffs):
    cosines = []
    for i in range(len(diffs)):
        for j in range(i+1, len(diffs)):
            cos = np.dot(diffs[i], diffs[j]) / (np.linalg.norm(diffs[i]) * np.linalg.norm(diffs[j]) + 1e-8)
            cosines.append(cos)
    return np.mean(cosines) if cosines else 0

random_tokens = ["I", "Well", "The", "Sure", "Let", "That", "It", "This", "My", "To"]

# Test all 3 strategies
for strategy in ["chat_mean", "role_play", "mc_balanced"]:
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy}")
    print(f"{'='*60}")
    
    for layer in [8, 12, 16, 20, 24, 28]:
        diffs = []
        for pair in pairs:
            q = pair["question"][:500]
            pos, neg = pair["positive"], pair["negative"]
            random_token = random_tokens[hash(q) % len(random_tokens)]
            
            if strategy == "chat_mean":
                pos_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}, {"role": "assistant", "content": pos}],
                    tokenize=False, add_generation_prompt=False
                )
                neg_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}, {"role": "assistant", "content": neg}],
                    tokenize=False, add_generation_prompt=False
                )
                
            elif strategy == "role_play":
                pos_sys = f"You are truthful. Your answer: {pos}"
                neg_sys = f"You spread misinformation. Your answer: {neg}"
                pos_text = tokenizer.apply_chat_template(
                    [{"role": "system", "content": pos_sys}, {"role": "user", "content": q}],
                    tokenize=False, add_generation_prompt=True
                ) + random_token
                neg_text = tokenizer.apply_chat_template(
                    [{"role": "system", "content": neg_sys}, {"role": "user", "content": q}],
                    tokenize=False, add_generation_prompt=True
                ) + random_token
                
            elif strategy == "mc_balanced":
                pos_goes_in_b = hash(q) % 2 == 0
                if pos_goes_in_b:
                    mc_text = f"Which is correct?\nA. {neg[:200]}\nB. {pos[:200]}\nAnswer:"
                    pos_ans, neg_ans = "B", "A"
                else:
                    mc_text = f"Which is correct?\nA. {pos[:200]}\nB. {neg[:200]}\nAnswer:"
                    pos_ans, neg_ans = "A", "B"
                pos_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": mc_text}, {"role": "assistant", "content": pos_ans}],
                    tokenize=False, add_generation_prompt=False
                )
                neg_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": mc_text}, {"role": "assistant", "content": neg_ans}],
                    tokenize=False, add_generation_prompt=False
                )
            
            pos_act = get_last_token_act(pos_text, layer)
            neg_act = get_last_token_act(neg_text, layer)
            diffs.append((pos_act - neg_act).numpy())
        
        diffs = np.stack(diffs)
        cos = compute_cosine(diffs)
        print(f"  Layer {layer:2d}: cosine = {cos:.4f}")

print("\nDone")

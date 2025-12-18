#!/usr/bin/env python3
"""
Run geometry detection on TruthfulQA across all strategies and models.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from datasets import load_dataset
import random
import gc
import sys
sys.path.insert(0, "/Users/lukaszbartoszcze/Documents/CodingProjects/Wisent/backends/wisent-open-source")

from wisent.core.contrastive_pairs.diagnostics import detect_geometry_structure, GeometryAnalysisConfig

MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-2-7b-chat-hf",
    "Qwen/Qwen3-8B",
]

DEVICE = "mps"

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

geo_config = GeometryAnalysisConfig(
    num_components=5,
    optimization_steps=100,
)

for model_name in MODELS:
    print(f"\n{'#'*80}")
    print(f"# Model: {model_name}")
    print(f"{'#'*80}")
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=DEVICE)
    num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} layers")
    
    # Pick layers based on model depth
    if num_layers <= 16:
        test_layers = [4, 8, 12, 14]
    elif num_layers <= 32:
        test_layers = [8, 16, 24, 28]
    else:
        test_layers = [10, 20, 30, 40]
    test_layers = [l for l in test_layers if l < num_layers]
    
    def get_all_layer_activations(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = model(inputs.input_ids, output_hidden_states=True)
        return [outputs.hidden_states[l][0, -1, :].cpu().float() for l in range(num_layers + 1)]
    
    def get_mean_answer_activations(text, answer):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        answer_tokens = tokenizer(answer, add_special_tokens=False)["input_ids"]
        num_answer_tokens = len(answer_tokens)
        with torch.no_grad():
            outputs = model(inputs.input_ids, output_hidden_states=True)
        result = []
        for l in range(num_layers + 1):
            hidden = outputs.hidden_states[l][0]
            if num_answer_tokens > 0 and hidden.shape[0] > num_answer_tokens:
                answer_hidden = hidden[-num_answer_tokens-1:-1, :]
                result.append(answer_hidden.mean(dim=0).cpu().float())
            else:
                result.append(hidden[-1, :].cpu().float())
        return result

    # All strategies
    def build_chat(q, ans):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": q}, {"role": "assistant", "content": ans}],
            tokenize=False, add_generation_prompt=False
        )
    
    def build_role_play(q, ans):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Behave like a person that would answer {q} with {ans}"},
             {"role": "assistant", "content": random_tokens[hash(q) % len(random_tokens)]}],
            tokenize=False, add_generation_prompt=False
        )
    
    def build_mc(q, pos, neg):
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
        return pos_text, neg_text

    STRATEGIES = ["chat_last", "chat_mean", "role_play", "mc_balanced"]
    
    for strategy in STRATEGIES:
        print(f"\n  --- Strategy: {strategy} ---")
        
        pos_acts_by_layer = {l: [] for l in test_layers}
        neg_acts_by_layer = {l: [] for l in test_layers}
        
        print(f"  Collecting activations for {len(pairs)} pairs...")
        for i, pair in enumerate(pairs):
            if i % 200 == 0:
                print(f"    {i}/{len(pairs)}")
            q = pair["question"][:500]
            pos, neg = pair["positive"], pair["negative"]
            
            if strategy == "chat_last":
                pos_text = build_chat(q, pos)
                neg_text = build_chat(q, neg)
                pos_all = get_all_layer_activations(pos_text)
                neg_all = get_all_layer_activations(neg_text)
            elif strategy == "chat_mean":
                pos_text = build_chat(q, pos)
                neg_text = build_chat(q, neg)
                pos_all = get_mean_answer_activations(pos_text, pos)
                neg_all = get_mean_answer_activations(neg_text, neg)
            elif strategy == "role_play":
                pos_text = build_role_play(q, pos)
                neg_text = build_role_play(q, neg)
                pos_all = get_all_layer_activations(pos_text)
                neg_all = get_all_layer_activations(neg_text)
            elif strategy == "mc_balanced":
                pos_text, neg_text = build_mc(q, pos, neg)
                pos_all = get_all_layer_activations(pos_text)
                neg_all = get_all_layer_activations(neg_text)
            
            for layer in test_layers:
                pos_acts_by_layer[layer].append(pos_all[layer])
                neg_acts_by_layer[layer].append(neg_all[layer])
        
        print(f"  Running geometry detection...")
        for layer in test_layers:
            pos_tensor = torch.stack(pos_acts_by_layer[layer])
            neg_tensor = torch.stack(neg_acts_by_layer[layer])
            
            result = detect_geometry_structure(pos_tensor, neg_tensor, geo_config)
            
            print(f"    Layer {layer:2d}: best={result.best_structure.value:<10} "
                  f"score={result.best_score:.3f} "
                  f"linear={result.all_scores['linear'].score:.3f} "
                  f"cone={result.all_scores['cone'].score:.3f} "
                  f"cluster={result.all_scores['cluster'].score:.3f}")
    
    # Cleanup
    del model
    del tokenizer
    gc.collect()
    torch.mps.empty_cache()

print("\nDone")

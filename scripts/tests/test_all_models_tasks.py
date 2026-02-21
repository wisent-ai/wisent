#!/usr/bin/env python3
"""
Test prompt strategies and extraction methods across multiple models.

Strategies (prompt formatting):
- chat_mean: Direct Q+A chat format, mean of answer tokens
- chat_first: Direct Q+A chat format, first answer token
- chat_last: Direct Q+A chat format, last token
- chat_max_norm: Direct Q+A chat format, token with max norm in answer
- chat_weighted: Direct Q+A chat format, position-weighted mean (earlier tokens weighted more)
- role_play: "Behave like person who answers Q with A" format, last token
- mc_balanced: Multiple choice format, last token
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import gc

from test_all_models_tasks_helpers import (
    load_task_pairs,
    generate_nonsense_pairs,
    compute_cosine,
    compute_mean_direction,
    extract_diffs_for_strategy,
    random_tokens,
)

MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-2-7b-chat-hf",
    "Qwen/Qwen3-8B",
    "openai/gpt-oss-20b",
]

TASKS = ["truthfulqa_gen", "happy", "left_wing", "livecodebench"]

nonsense_pairs = generate_nonsense_pairs(30)

strategies = [
    "chat_mean",
    "chat_first",
    "chat_last",
    "chat_max_norm",
    "chat_weighted",
    "role_play",
    "mc_balanced"
]

for model_name in MODELS:
    print(f"\n{'#'*70}")
    print(f"# Model: {model_name}")
    print(f"{'#'*70}")

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="mps")
    num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} layers")

    # Pick layers based on model depth
    if num_layers <= 16:
        test_layers = [4, 6, 8, 10, 12, 14]
    elif num_layers <= 32:
        test_layers = [8, 12, 16, 20, 24, 28]
    else:
        test_layers = [10, 20, 30, 40, 50, 60]
    test_layers = [l for l in test_layers if l < num_layers]

    for task_name in TASKS:
        print(f"\n  === Task: {task_name} ===")
        pairs = load_task_pairs(task_name, n_samples=30)
        print(f"  Loaded {len(pairs)} pairs")

        for strategy in strategies:
            print(f"\n    Strategy: {strategy}")

            for layer in test_layers:
                diffs = extract_diffs_for_strategy(model, tokenizer, pairs, strategy, layer, "mps")

                # Compute nonsense diffs for this strategy/layer
                nonsense_diffs = extract_diffs_for_strategy(model, tokenizer, nonsense_pairs, strategy, layer, "mps")

                diffs = np.stack(diffs)
                nonsense_diffs = np.stack(nonsense_diffs)

                # Compute metrics
                M_cos = compute_cosine(diffs)
                N_cos = compute_cosine(nonsense_diffs)
                M_dir = compute_mean_direction(diffs)
                N_dir = compute_mean_direction(nonsense_diffs)
                M_vs_N = np.dot(M_dir, N_dir)

                print(f"      Layer {layer:2d}: M_cos={M_cos:.3f}, N_cos={N_cos:.3f}, M_vs_N={M_vs_N:.3f}")

    # Cleanup
    del model
    del tokenizer
    gc.collect()
    torch.mps.empty_cache()

print("\nDone")

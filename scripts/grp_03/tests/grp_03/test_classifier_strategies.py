#!/usr/bin/env python3
"""
Test ALL combinations: TrainingStrategy x InferenceStrategy x Layer x Model x Task

Training strategies (how to extract activations for training):
- chat_mean, chat_first, chat_last, chat_max_norm, chat_weighted, role_play, mc_balanced

Inference strategies (how to classify generated text at runtime):
- last_token, first_token, all_mean, all_max, all_min
"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

from wisent.core.constants import DEFAULT_RANDOM_SEED
from test_classifier_strategies_helpers import (
    load_task_pairs,
    build_train_text,
    get_train_activation,
    get_inference_score,
)

# Auto-detect device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")

MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-2-7b-chat-hf",
    "Qwen/Qwen3-8B",
]

TASKS = ["truthfulqa", "happy", "left_wing", "livecodebench"]

TRAIN_STRATEGIES = [
    "chat_mean", "chat_first", "chat_last", "chat_max_norm",
    "chat_weighted", "role_play", "mc_balanced",
]

INFERENCE_STRATEGIES = ["last_token", "first_token", "all_mean", "all_max", "all_min"]


def test_combination(model, tokenizer, pairs, layer, train_strategy, inference_strategy):
    """Test a specific train_strategy + inference_strategy combination."""
    train_pairs = pairs[:len(pairs)//2]
    test_pairs = pairs[len(pairs)//2:]

    # TRAINING: collect activations using train_strategy
    X_train, y_train = [], []
    for pair in train_pairs:
        q, pos, neg = pair["question"], pair["positive"], pair["negative"]

        if train_strategy == "mc_balanced":
            pos_text, pos_ans = build_train_text(tokenizer, q, pos, train_strategy, other_answer=neg)
            neg_text, neg_ans = build_train_text(tokenizer, q, neg, train_strategy, other_answer=pos)
        else:
            pos_text, pos_ans = build_train_text(tokenizer, q, pos, train_strategy)
            neg_text, neg_ans = build_train_text(tokenizer, q, neg, train_strategy)

        X_train.append(get_train_activation(model, tokenizer, pos_text, pos_ans, layer, train_strategy, DEVICE))
        y_train.append(1)
        X_train.append(get_train_activation(model, tokenizer, neg_text, neg_ans, layer, train_strategy, DEVICE))
        y_train.append(0)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    clf = LogisticRegression(random_state=DEFAULT_RANDOM_SEED)
    clf.fit(X_train, y_train)

    # INFERENCE: test using inference_strategy
    y_scores, y_test = [], []
    for pair in test_pairs:
        q, pos, neg = pair["question"], pair["positive"], pair["negative"]

        pos_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}, {"role": "assistant", "content": pos}],
            tokenize=False, add_generation_prompt=False
        )
        neg_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}, {"role": "assistant", "content": neg}],
            tokenize=False, add_generation_prompt=False
        )

        pos_score = get_inference_score(clf, model, tokenizer, pos_text, layer, inference_strategy, DEVICE)
        neg_score = get_inference_score(clf, model, tokenizer, neg_text, layer, inference_strategy, DEVICE)

        y_scores.extend([pos_score, neg_score])
        y_test.extend([1, 0])

    y_scores = np.array(y_scores)
    y_test = np.array(y_test)

    best_acc, best_f1, best_thresh = 0, 0, 0.5
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        y_pred = (y_scores > thresh).astype(int)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            best_thresh = thresh

    return best_acc, best_f1, best_thresh, float(y_scores.min()), float(y_scores.max())


def main():
    results = []

    for model_name in MODELS:
        print(f"\n{'#'*70}")
        print(f"# Model: {model_name}")
        print(f"{'#'*70}")

        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=DEVICE)
        num_layers = model.config.num_hidden_layers
        print(f"Model has {num_layers} layers")

        if num_layers <= 16:
            test_layers = [4, 8, 12, num_layers-1]
        elif num_layers <= 32:
            test_layers = [8, 16, 24, num_layers-1]
        else:
            test_layers = [10, 20, 30, num_layers-1]
        test_layers = [l for l in test_layers if l < num_layers]

        for task_name in TASKS:
            print(f"\n  === Task: {task_name} ===")
            pairs = load_task_pairs(task_name, n_samples=100)
            print(f"  Loaded {len(pairs)} pairs")

            for train_strat in TRAIN_STRATEGIES:
                for infer_strat in INFERENCE_STRATEGIES:
                    print(f"\n    Train: {train_strat:15s} | Infer: {infer_strat:10s}")

                    for layer in test_layers:
                        try:
                            acc, f1, thresh, s_min, s_max = test_combination(
                                model, tokenizer, pairs, layer, train_strat, infer_strat)
                            print(f"      Layer {layer:2d}: Acc={acc:.3f}, F1={f1:.3f}, thresh={thresh:.1f}, scores=[{s_min:.3f}, {s_max:.3f}]")
                            results.append({
                                "model": model_name, "task": task_name,
                                "train_strategy": train_strat, "inference_strategy": infer_strat,
                                "layer": layer, "accuracy": acc, "f1": f1,
                                "threshold": thresh, "score_min": s_min, "score_max": s_max,
                            })
                        except Exception as e:
                            print(f"      Layer {layer:2d}: ERROR - {e}")

                    gc.collect()
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()
                    elif DEVICE == "mps":
                        torch.mps.empty_cache()

        del model, tokenizer
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE == "mps":
            torch.mps.empty_cache()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Best combinations")
    print("="*80)

    import pandas as pd
    df = pd.DataFrame(results)

    best = df.loc[df["accuracy"].idxmax()]
    print(f"\nBest overall: {best['train_strategy']} + {best['inference_strategy']} @ L{best['layer']:.0f} = {best['accuracy']:.3f}")

    print("\nAverage accuracy by TRAIN strategy:")
    for strat, acc in df.groupby("train_strategy")["accuracy"].mean().sort_values(ascending=False).items():
        print(f"  {strat:15s}: {acc:.3f}")

    print("\nAverage accuracy by INFERENCE strategy:")
    for strat, acc in df.groupby("inference_strategy")["accuracy"].mean().sort_values(ascending=False).items():
        print(f"  {strat:15s}: {acc:.3f}")

    print("\nBest INFERENCE strategy for each TRAIN strategy:")
    for train_strat in TRAIN_STRATEGIES:
        subset = df[df["train_strategy"] == train_strat]
        if len(subset) > 0:
            best_infer = subset.groupby("inference_strategy")["accuracy"].mean().idxmax()
            best_acc = subset.groupby("inference_strategy")["accuracy"].mean().max()
            print(f"  {train_strat:15s} -> {best_infer:10s} ({best_acc:.3f})")

    df.to_csv("/tmp/classifier_strategy_results.csv", index=False)
    print(f"\nResults saved to /tmp/classifier_strategy_results.csv")


if __name__ == "__main__":
    main()

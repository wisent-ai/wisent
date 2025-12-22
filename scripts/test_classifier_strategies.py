#!/usr/bin/env python3
"""
Test ALL combinations: TrainingStrategy × InferenceStrategy × Layer × Model × Task

Training strategies (how to extract activations for training):
- chat_mean, chat_first, chat_last, chat_max_norm, chat_weighted, role_play, mc_balanced

Inference strategies (how to classify generated text at runtime):
- last_token, first_token, all_mean, all_max, all_min
"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
import random
import gc
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    "chat_mean",
    "chat_first",
    "chat_last",
    "chat_max_norm",
    "chat_weighted",
    "role_play",
    "mc_balanced",
]

INFERENCE_STRATEGIES = [
    "last_token",
    "first_token",
    "all_mean",
    "all_max",
    "all_min",
]

RANDOM_TOKENS = ["I", "Well", "The", "Sure", "Let", "That", "It", "This", "My", "To"]


def load_task_pairs(task_name, n_samples=100):
    random.seed(42)
    
    if task_name == "truthfulqa":
        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        samples = random.sample(list(ds), min(n_samples, len(ds)))
        pairs = []
        for s in samples:
            if s["incorrect_answers"]:
                pairs.append({
                    "question": s["question"],
                    "positive": s["best_answer"],
                    "negative": random.choice(s["incorrect_answers"])
                })
        return pairs
    
    elif task_name == "happy":
        prompts = ["How are you feeling today?", "Tell me about your day.", "What's on your mind?",
                   "How would you describe your mood?", "What are your thoughts right now?"]
        happy = ["I'm feeling wonderful and full of joy!", "Everything is going great, I'm so happy!",
                 "I feel fantastic and optimistic about everything!", "Life is beautiful and I'm grateful for it!",
                 "I'm in such a positive mood today!"]
        sad = ["I'm feeling terrible and miserable.", "Everything is going wrong, I'm so sad.",
               "I feel awful and pessimistic about everything.", "Life is hard and I'm struggling.",
               "I'm in such a negative mood today."]
        return [{"question": prompts[i % 5], "positive": happy[i % 5], "negative": sad[i % 5]} for i in range(n_samples)]
    
    elif task_name == "left_wing":
        prompts = ["What do you think about economic policy?", "How should we address inequality?",
                   "What's your view on healthcare?", "What about immigration policy?", "How should we handle climate change?"]
        left = ["We need more government intervention and wealth redistribution.",
                "Progressive taxation and social programs are essential.", "Universal healthcare is a fundamental right.",
                "We should welcome immigrants and provide paths to citizenship.",
                "Aggressive government action is needed to combat climate change."]
        right = ["Free markets and limited government are the answer.",
                 "Lower taxes and individual responsibility drive prosperity.",
                 "Private healthcare and competition improve outcomes.",
                 "We need strict border control and merit-based immigration.",
                 "Market solutions and innovation will address environmental issues."]
        return [{"question": prompts[i % 5], "positive": left[i % 5], "negative": right[i % 5]} for i in range(n_samples)]
    
    elif task_name == "livecodebench":
        prompts = ["Write a function to add two numbers.", "Write a function to check if a number is even.",
                   "Write a function to reverse a string.", "Write a function to find the maximum in a list.",
                   "Write a function to check if a string is a palindrome."]
        correct = ["def add(a, b): return a + b", "def is_even(n): return n % 2 == 0",
                   "def reverse(s): return s[::-1]", "def find_max(lst): return max(lst)",
                   "def is_palindrome(s): return s == s[::-1]"]
        incorrect = ["def add(a, b): return a - b", "def is_even(n): return n % 2 == 1",
                     "def reverse(s): return s", "def find_max(lst): return min(lst)",
                     "def is_palindrome(s): return s != s[::-1]"]
        return [{"question": prompts[i % 5], "positive": correct[i % 5], "negative": incorrect[i % 5]} for i in range(n_samples)]
    
    raise ValueError(f"Unknown task: {task_name}")


def build_train_text(tokenizer, question, answer, train_strategy, other_answer=None):
    """Build text for TRAINING based on train_strategy."""
    if train_strategy.startswith("chat_"):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
            tokenize=False, add_generation_prompt=False
        )
        return text, answer
    
    elif train_strategy == "role_play":
        random_token = RANDOM_TOKENS[hash(question) % len(RANDOM_TOKENS)]
        instruction = f"Behave like a person that would answer {question} with {answer}"
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}, {"role": "assistant", "content": random_token}],
            tokenize=False, add_generation_prompt=False
        )
        return text, random_token
    
    elif train_strategy == "mc_balanced":
        pos_goes_in_b = hash(question) % 2 == 0
        other = other_answer or "Alternative"
        if pos_goes_in_b:
            mc_text = f"Which is correct?\nA. {other[:200]}\nB. {answer[:200]}\nAnswer:"
            choice = "B"
        else:
            mc_text = f"Which is correct?\nA. {answer[:200]}\nB. {other[:200]}\nAnswer:"
            choice = "A"
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": mc_text}, {"role": "assistant", "content": choice}],
            tokenize=False, add_generation_prompt=False
        )
        return text, choice
    
    raise ValueError(f"Unknown train_strategy: {train_strategy}")


def get_train_activation(model, tokenizer, text, answer, layer, train_strategy):
    """Extract activation for TRAINING based on train_strategy."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0]
    seq_len = hidden.shape[0]
    
    answer_tokens = tokenizer(answer, add_special_tokens=False)["input_ids"]
    num_ans = len(answer_tokens)
    
    if train_strategy == "chat_last":
        return hidden[-1].cpu().float().numpy()
    elif train_strategy == "chat_first":
        idx = max(0, seq_len - num_ans - 1)
        return hidden[idx].cpu().float().numpy()
    elif train_strategy == "chat_mean":
        if num_ans > 0 and seq_len > num_ans:
            return hidden[-num_ans-1:-1].mean(dim=0).cpu().float().numpy()
        return hidden[-1].cpu().float().numpy()
    elif train_strategy == "chat_max_norm":
        if num_ans > 0 and seq_len > num_ans:
            ans_hidden = hidden[-num_ans-1:-1]
            max_idx = torch.argmax(torch.norm(ans_hidden, dim=1))
            return ans_hidden[max_idx].cpu().float().numpy()
        return hidden[-1].cpu().float().numpy()
    elif train_strategy == "chat_weighted":
        if num_ans > 0 and seq_len > num_ans:
            ans_hidden = hidden[-num_ans-1:-1]
            w = torch.exp(-torch.arange(ans_hidden.shape[0], dtype=torch.float32, device=ans_hidden.device) * 0.5)
            w = w / w.sum()
            return (ans_hidden * w.unsqueeze(1)).sum(dim=0).cpu().float().numpy()
        return hidden[-1].cpu().float().numpy()
    elif train_strategy in ("role_play", "mc_balanced"):
        return hidden[-1].cpu().float().numpy()
    
    raise ValueError(f"Unknown train_strategy: {train_strategy}")


def get_inference_activation(model, tokenizer, text, layer, inference_strategy):
    """Extract activation for INFERENCE based on inference_strategy.
    
    This simulates classifying a generated response where we only have the full text,
    not the answer boundaries.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0].cpu().float()  # [seq_len, hidden_dim]
    seq_len = hidden.shape[0]
    
    if inference_strategy == "last_token":
        return hidden[-1].numpy()
    elif inference_strategy == "first_token":
        return hidden[0].numpy()
    elif inference_strategy == "all_mean":
        return hidden.mean(dim=0).numpy()
    elif inference_strategy == "all_max":
        # Return the token with max norm (most "active")
        norms = torch.norm(hidden, dim=1)
        return hidden[torch.argmax(norms)].numpy()
    elif inference_strategy == "all_min":
        # Return the token with min norm
        norms = torch.norm(hidden, dim=1)
        return hidden[torch.argmin(norms)].numpy()
    
    raise ValueError(f"Unknown inference_strategy: {inference_strategy}")


def get_inference_score(clf, model, tokenizer, text, layer, inference_strategy):
    """Get classifier score using inference_strategy.
    
    For all_* strategies, we classify each token and aggregate scores.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0].cpu().float().numpy()  # [seq_len, hidden_dim]
    seq_len = hidden.shape[0]
    
    if inference_strategy == "last_token":
        return clf.predict_proba([hidden[-1]])[0, 1]
    elif inference_strategy == "first_token":
        return clf.predict_proba([hidden[0]])[0, 1]
    elif inference_strategy in ("all_mean", "all_max", "all_min"):
        # Classify ALL tokens and aggregate
        all_scores = []
        for t in range(seq_len):
            score = clf.predict_proba([hidden[t]])[0, 1]
            all_scores.append(score)
        if inference_strategy == "all_mean":
            return np.mean(all_scores)
        elif inference_strategy == "all_max":
            return np.max(all_scores)
        elif inference_strategy == "all_min":
            return np.min(all_scores)
    
    raise ValueError(f"Unknown inference_strategy: {inference_strategy}")


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
        
        X_train.append(get_train_activation(model, tokenizer, pos_text, pos_ans, layer, train_strategy))
        y_train.append(1)
        X_train.append(get_train_activation(model, tokenizer, neg_text, neg_ans, layer, train_strategy))
        y_train.append(0)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    # INFERENCE: test using inference_strategy
    y_scores, y_test = [], []
    for pair in test_pairs:
        q, pos, neg = pair["question"], pair["positive"], pair["negative"]
        
        # Build inference text (simple chat format - simulating real inference)
        pos_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}, {"role": "assistant", "content": pos}],
            tokenize=False, add_generation_prompt=False
        )
        neg_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}, {"role": "assistant", "content": neg}],
            tokenize=False, add_generation_prompt=False
        )
        
        pos_score = get_inference_score(clf, model, tokenizer, pos_text, layer, inference_strategy)
        neg_score = get_inference_score(clf, model, tokenizer, neg_text, layer, inference_strategy)
        
        y_scores.extend([pos_score, neg_score])
        y_test.extend([1, 0])
    
    y_scores = np.array(y_scores)
    y_test = np.array(y_test)
    
    # Test multiple thresholds, return best accuracy
    best_acc = 0
    best_f1 = 0
    best_thresh = 0.5
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        y_pred = (y_scores > thresh).astype(int)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            best_thresh = thresh
    
    score_min = float(y_scores.min())
    score_max = float(y_scores.max())
    return best_acc, best_f1, best_thresh, score_min, score_max


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
                            acc, f1, thresh, s_min, s_max = test_combination(model, tokenizer, pairs, layer, train_strat, infer_strat)
                            print(f"      Layer {layer:2d}: Acc={acc:.3f}, F1={f1:.3f}, thresh={thresh:.1f}, scores=[{s_min:.3f}, {s_max:.3f}]")
                            results.append({
                                "model": model_name,
                                "task": task_name,
                                "train_strategy": train_strat,
                                "inference_strategy": infer_strat,
                                "layer": layer,
                                "accuracy": acc,
                                "f1": f1,
                                "threshold": thresh,
                                "score_min": s_min,
                                "score_max": s_max,
                            })
                        except Exception as e:
                            print(f"      Layer {layer:2d}: ERROR - {e}")
                    
                    # Clear memory after each train/infer combo
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
    
    # Best overall
    best = df.loc[df["accuracy"].idxmax()]
    print(f"\nBest overall: {best['train_strategy']} + {best['inference_strategy']} @ L{best['layer']:.0f} = {best['accuracy']:.3f}")
    
    # Average by train_strategy
    print("\nAverage accuracy by TRAIN strategy:")
    for strat, acc in df.groupby("train_strategy")["accuracy"].mean().sort_values(ascending=False).items():
        print(f"  {strat:15s}: {acc:.3f}")
    
    # Average by inference_strategy
    print("\nAverage accuracy by INFERENCE strategy:")
    for strat, acc in df.groupby("inference_strategy")["accuracy"].mean().sort_values(ascending=False).items():
        print(f"  {strat:15s}: {acc:.3f}")
    
    # Best combo per train strategy
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

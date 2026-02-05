#!/usr/bin/env python3
"""
Signal detection script for contrastive pairs.

Checks if a concept has extractable geometric structure in activation space.
Tests both linear and nonlinear separability.

Usage:
    python scripts/check_signal.py --task truthfulqa --model meta-llama/Llama-3.2-1B-Instruct
    python scripts/check_signal.py --task livecodebench --model Qwen/Qwen3-8B
    python scripts/check_signal.py --task sentiment --model meta-llama/Llama-3.2-1B-Instruct
"""

import argparse
import json
import random
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import hf_hub_download


def load_model(model_name, device="auto"):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map=device
    )
    return model, tokenizer


def get_truthfulqa_pairs(limit=50):
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    random.seed(42)
    samples = random.sample(list(ds), min(limit, len(ds)))
    return [{
        "question": s["question"],
        "positive": s["best_answer"],
        "negative": random.choice(s["incorrect_answers"])
    } for s in samples]


def get_livecodebench_pairs(limit=50):
    problems_path = hf_hub_download(
        repo_id="livecodebench/code_generation_samples",
        filename="problems.json",
        repo_type="space"
    )
    outputs_path = hf_hub_download(
        repo_id="livecodebench/code_generation_samples",
        filename="all_outputs.json",
        repo_type="space"
    )
    
    with open(problems_path, "r") as f:
        problems = json.load(f)
    with open(outputs_path, "r") as f:
        all_outputs = json.load(f)
    
    pairs = []
    models = list(all_outputs.keys())
    
    for idx, problem in enumerate(problems):
        if len(pairs) >= limit:
            break
        question = problem.get("question_content", "")
        if not question:
            continue
        
        positive_code, negative_code = None, None
        for m in models:
            if idx >= len(all_outputs[m]):
                continue
            output = all_outputs[m][idx]
            for code, passed in zip(output.get("code_list", []), output.get("pass1_list", [])):
                if passed and not positive_code:
                    positive_code = code
                elif not passed and not negative_code:
                    negative_code = code
                if positive_code and negative_code:
                    break
            if positive_code and negative_code:
                break
        
        if positive_code and negative_code:
            pairs.append({
                "question": question,
                "positive": positive_code,
                "negative": negative_code
            })
    
    return pairs


def get_sentiment_pairs(limit=50):
    prompts = [
        "How do you feel?", "What is your mood?", "Describe your state",
        "Are you okay?", "How are things?", "Tell me about yourself",
        "What's your emotional state?", "How are you doing?",
        "What emotions are you experiencing?", "Describe how you feel",
        "How's life treating you?", "What's on your mind?",
        "How would you describe your feelings?", "What's your current state?",
        "Tell me your mood", "How are you feeling today?",
        "Describe your emotional state", "What are you feeling?",
        "How do you feel right now?", "What's your state of mind?",
    ]
    pos_words = ["Happy", "Joyful", "Great", "Wonderful", "Amazing", "Elated", 
                 "Thrilled", "Fantastic", "Excellent", "Content", "Delighted",
                 "Ecstatic", "Pleased", "Cheerful", "Blissful", "Radiant",
                 "Optimistic", "Grateful", "Peaceful", "Satisfied"]
    neg_words = ["Sad", "Miserable", "Terrible", "Awful", "Dreadful", "Depressed",
                 "Devastated", "Horrible", "Poor", "Distressed", "Melancholy",
                 "Gloomy", "Heartbroken", "Despondent", "Sorrowful", "Dejected",
                 "Hopeless", "Anguished", "Wretched", "Forlorn"]
    
    n = min(limit, len(prompts))
    return [{
        "question": prompts[i % len(prompts)],
        "positive": pos_words[i % len(pos_words)],
        "negative": neg_words[i % len(neg_words)]
    } for i in range(n)]


def get_activations(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return [h[0, -1, :].cpu().float() for h in outputs.hidden_states]


def collect_activations(model, tokenizer, pairs, strategy, device, num_layers):
    """Collect activations for all pairs using specified strategy."""
    random_tokens = ["I", "Well", "The", "Sure", "Let", "That", "It", "This", "My", "To"]
    
    pos_acts_by_layer = [[] for _ in range(num_layers + 1)]
    neg_acts_by_layer = [[] for _ in range(num_layers + 1)]
    
    for i, pair in enumerate(pairs):
        q = pair["question"][:500]
        pos, neg = pair["positive"][:1000], pair["negative"][:1000]
        random_token = random_tokens[hash(q) % len(random_tokens)]
        
        if strategy == "chat_mean":
            prompt_msgs = [{"role": "user", "content": q}]
            pos_msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": pos}]
            neg_msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": neg}]
            
            pos_text = tokenizer.apply_chat_template(pos_msgs, tokenize=False, add_generation_prompt=False)
            neg_text = tokenizer.apply_chat_template(neg_msgs, tokenize=False, add_generation_prompt=False)
            
        elif strategy == "role_play":
            pos_sys = f"You are truthful/expert. Your answer: {pos}"
            neg_sys = f"You spread misinformation/make mistakes. Your answer: {neg}"
            
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
        
        pos_acts = get_activations(model, tokenizer, pos_text, device)
        neg_acts = get_activations(model, tokenizer, neg_text, device)
        
        for layer in range(num_layers + 1):
            pos_acts_by_layer[layer].append(pos_acts[layer])
            neg_acts_by_layer[layer].append(neg_acts[layer])
        
        if (i + 1) % 10 == 0:
            print(f"  Collected {i + 1}/{len(pairs)}")
    
    return pos_acts_by_layer, neg_acts_by_layer


def compute_signal_metrics(pos_acts, neg_acts, n_subsets=5):
    """Compute all signal detection metrics."""
    pos_np = torch.stack(pos_acts).numpy()
    neg_np = torch.stack(neg_acts).numpy()
    diff_vecs = pos_np - neg_np
    
    results = {}
    
    # 1. Mean diff cosine (linear cone)
    cosines = []
    for i in range(len(diff_vecs)):
        for j in range(i + 1, len(diff_vecs)):
            cos = np.dot(diff_vecs[i], diff_vecs[j]) / (
                np.linalg.norm(diff_vecs[i]) * np.linalg.norm(diff_vecs[j]) + 1e-8
            )
            cosines.append(cos)
    results["diff_cosine_mean"] = np.mean(cosines) if cosines else 0
    results["diff_cosine_std"] = np.std(cosines) if cosines else 0
    
    # 2. PC1 variance (dominant direction)
    if len(diff_vecs) > 2:
        pca = PCA(n_components=min(5, len(diff_vecs) - 1)).fit(diff_vecs)
        results["pc1_variance"] = pca.explained_variance_ratio_[0]
        results["pc3_variance"] = sum(pca.explained_variance_ratio_[:3])
    else:
        results["pc1_variance"] = 0
        results["pc3_variance"] = 0
    
    # 3. K-means cluster separation
    all_acts = np.vstack([pos_np, neg_np])
    labels = np.array([1] * len(pos_np) + [0] * len(neg_np))
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(all_acts)
    cluster_align = max(
        np.mean(kmeans.labels_ == labels),
        np.mean(kmeans.labels_ == (1 - labels))
    )
    results["kmeans_accuracy"] = cluster_align
    
    # 4. MLP classifier (nonlinear)
    if len(pos_np) >= 10:
        mlp = MLPClassifier(hidden_layer_sizes=(64,),  random_state=42)
        try:
            scores = cross_val_score(mlp, all_acts, labels, cv=min(5, len(pos_np) // 2))
            results["mlp_accuracy"] = np.mean(scores)
            results["mlp_std"] = np.std(scores)
        except:
            results["mlp_accuracy"] = 0.5
            results["mlp_std"] = 0
    else:
        results["mlp_accuracy"] = 0.5
        results["mlp_std"] = 0
    
    # 5. SVM RBF (kernel-based)
    if len(pos_np) >= 10:
        svm = SVC(kernel="rbf", random_state=42)
        try:
            scores = cross_val_score(svm, all_acts, labels, cv=min(5, len(pos_np) // 2))
            results["svm_accuracy"] = np.mean(scores)
            results["svm_std"] = np.std(scores)
        except:
            results["svm_accuracy"] = 0.5
            results["svm_std"] = 0
    else:
        results["svm_accuracy"] = 0.5
        results["svm_std"] = 0
    
    # 6. Cross-subset stability
    if len(diff_vecs) >= 20:
        subset_cosines = []
        for seed in range(n_subsets):
            np.random.seed(seed)
            idx = np.random.choice(len(diff_vecs), len(diff_vecs) // 2, replace=False)
            subset = diff_vecs[idx]
            cos_list = []
            for i in range(len(subset)):
                for j in range(i + 1, len(subset)):
                    cos = np.dot(subset[i], subset[j]) / (
                        np.linalg.norm(subset[i]) * np.linalg.norm(subset[j]) + 1e-8
                    )
                    cos_list.append(cos)
            subset_cosines.append(np.mean(cos_list))
        results["subset_stability"] = np.std(subset_cosines)
    else:
        results["subset_stability"] = 0
    
    return results


def evaluate_signal(results):
    """Evaluate if signal exists based on metrics."""
    checks = []
    
    # Linear checks
    if results["diff_cosine_mean"] > 0.2:
        checks.append(("Linear cone (cosine > 0.2)", True, results["diff_cosine_mean"]))
    else:
        checks.append(("Linear cone (cosine > 0.2)", False, results["diff_cosine_mean"]))
    
    if results["pc1_variance"] > 0.3:
        checks.append(("Dominant direction (PC1 > 30%)", True, results["pc1_variance"]))
    else:
        checks.append(("Dominant direction (PC1 > 30%)", False, results["pc1_variance"]))
    
    # Nonlinear checks
    if results["kmeans_accuracy"] > 0.7:
        checks.append(("Cluster separation (K-means > 70%)", True, results["kmeans_accuracy"]))
    else:
        checks.append(("Cluster separation (K-means > 70%)", False, results["kmeans_accuracy"]))
    
    if results["mlp_accuracy"] > 0.7:
        checks.append(("Nonlinear separation (MLP > 70%)", True, results["mlp_accuracy"]))
    else:
        checks.append(("Nonlinear separation (MLP > 70%)", False, results["mlp_accuracy"]))
    
    if results["svm_accuracy"] > 0.7:
        checks.append(("Kernel separation (SVM > 70%)", True, results["svm_accuracy"]))
    else:
        checks.append(("Kernel separation (SVM > 70%)", False, results["svm_accuracy"]))
    
    # Stability
    if results["subset_stability"] < 0.05:
        checks.append(("Stable signal (std < 0.05)", True, results["subset_stability"]))
    else:
        checks.append(("Stable signal (std < 0.05)", False, results["subset_stability"]))
    
    return checks


def main():
    parser = argparse.ArgumentParser(description="Check signal in contrastive pairs")
    parser.add_argument("--task", type=str, required=True, 
                        choices=["truthfulqa", "livecodebench", "sentiment"])
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--strategies", type=str, nargs="+", 
                        default=["chat_mean", "role_play", "mc_balanced"])
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    # Load data
    print(f"\nLoading {args.task} data...")
    if args.task == "truthfulqa":
        pairs = get_truthfulqa_pairs(args.n_samples)
    elif args.task == "livecodebench":
        pairs = get_livecodebench_pairs(args.n_samples)
    elif args.task == "sentiment":
        pairs = get_sentiment_pairs(args.n_samples)
    
    print(f"Loaded {len(pairs)} pairs")
    
    # Load model
    model, tokenizer = load_model(args.model, args.device)
    num_layers = model.config.num_hidden_layers
    device = next(model.parameters()).device
    
    print(f"\n{'='*80}")
    print(f"SIGNAL CHECK: {args.task.upper()} on {args.model}")
    print(f"{'='*80}")
    
    for strategy in args.strategies:
        print(f"\n--- Strategy: {strategy} ---")
        
        # Collect activations
        print("Collecting activations...")
        pos_acts, neg_acts = collect_activations(
            model, tokenizer, pairs, strategy, device, num_layers
        )
        
        # Find best layer by diff cosine
        best_layer = 0
        best_cosine = -1
        layer_results = []
        
        for layer in range(num_layers + 1):
            results = compute_signal_metrics(pos_acts[layer], neg_acts[layer])
            layer_results.append((layer, results))
            if results["diff_cosine_mean"] > best_cosine:
                best_cosine = results["diff_cosine_mean"]
                best_layer = layer
        
        print(f"\nBest layer: {best_layer} (cosine: {best_cosine:.4f})")
        
        # Full analysis on best layer
        results = compute_signal_metrics(pos_acts[best_layer], neg_acts[best_layer])
        
        print(f"\n{'Metric':<35} {'Value':>10} {'Threshold':>12} {'Pass':>6}")
        print("-" * 65)
        
        checks = evaluate_signal(results)
        for name, passed, value in checks:
            status = "YES" if passed else "NO"
            if isinstance(value, float):
                print(f"{name:<35} {value:>10.4f} {'-':>12} {status:>6}")
            else:
                print(f"{name:<35} {str(value):>10} {'-':>12} {status:>6}")
        
        # Summary
        linear_signal = results["diff_cosine_mean"] > 0.2
        nonlinear_signal = results["mlp_accuracy"] > 0.7 or results["svm_accuracy"] > 0.7
        
        print(f"\n{'='*65}")
        if linear_signal:
            print(f"RESULT: LINEAR SIGNAL EXISTS (cosine={results['diff_cosine_mean']:.3f})")
            print(f"        Activation steering should work.")
        elif nonlinear_signal:
            print(f"RESULT: NONLINEAR SIGNAL EXISTS (MLP={results['mlp_accuracy']:.3f})")
            print(f"        Linear steering may not work. Consider nonlinear methods.")
        else:
            print(f"RESULT: NO CLEAR SIGNAL DETECTED")
            print(f"        This concept may not have geometric structure in this model.")
        print(f"{'='*65}")
        
        # Layer-by-layer summary
        print(f"\nLayer-by-layer cosine (top 5):")
        sorted_layers = sorted(layer_results, key=lambda x: x[1]["diff_cosine_mean"], reverse=True)[:5]
        for layer, res in sorted_layers:
            print(f"  Layer {layer:>2}: cosine={res['diff_cosine_mean']:.4f}, "
                  f"PC1={res['pc1_variance']:.2%}, MLP={res['mlp_accuracy']:.2%}")


if __name__ == "__main__":
    main()

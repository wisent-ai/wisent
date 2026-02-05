#!/usr/bin/env python3
"""
Test how many training samples are needed for consistent classifier performance.

Methodology:
- Hold out 40 TruthfulQA examples as fixed test set
- Split remaining examples into batches of size X
- Train classifier on each batch, test on held-out set
- Measure variance across batches for each X
- Find minimum X where performance is consistent
"""
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Config
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
LAYER = 12  # Middle-ish layer
BATCH_SIZES = [5, 10, 20, 30, 50, 75, 100, 150, 200]
TEST_SIZE = 40
NUM_TRIALS = 10  # How many random batches to test per size

# Auto-detect device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")


def load_truthfulqa_pairs():
    """Load TruthfulQA as contrastive pairs."""
    random.seed(42)
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    pairs = []
    for s in ds:
        if s["incorrect_answers"]:
            pairs.append({
                "question": s["question"],
                "positive": s["best_answer"],
                "negative": random.choice(s["incorrect_answers"])
            })
    return pairs


def get_activation(model, tokenizer, text, layer):
    """Get last token activation."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    return outputs.hidden_states[layer][0, -1].cpu().float().numpy()


def build_text(tokenizer, question, answer):
    """Build chat template text."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
        tokenize=False, add_generation_prompt=False
    )


def collect_activations(model, tokenizer, pairs, layer):
    """Collect activations for all pairs."""
    X, y = [], []
    for pair in pairs:
        q, pos, neg = pair["question"], pair["positive"], pair["negative"]
        
        pos_text = build_text(tokenizer, q, pos)
        neg_text = build_text(tokenizer, q, neg)
        
        X.append(get_activation(model, tokenizer, pos_text, layer))
        y.append(1)
        X.append(get_activation(model, tokenizer, neg_text, layer))
        y.append(0)
    
    return np.array(X), np.array(y)


def main():
    print("Loading TruthfulQA pairs...")
    all_pairs = load_truthfulqa_pairs()
    print(f"Total pairs: {len(all_pairs)}")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(all_pairs)
    
    test_pairs = all_pairs[:TEST_SIZE]
    train_pool = all_pairs[TEST_SIZE:]
    print(f"Test set: {TEST_SIZE} pairs")
    print(f"Training pool: {len(train_pool)} pairs")
    
    print(f"\nLoading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map=DEVICE)
    
    print(f"Collecting test activations (layer {LAYER})...")
    X_test, y_test = collect_activations(model, tokenizer, test_pairs, LAYER)
    print(f"Test set shape: {X_test.shape}")
    
    print(f"Collecting training pool activations...")
    X_train_pool, y_train_pool = collect_activations(model, tokenizer, train_pool, LAYER)
    print(f"Training pool shape: {X_train_pool.shape}")
    
    # Test each batch size
    results = {}
    
    print("\n" + "="*70)
    print("Testing batch sizes...")
    print("="*70)
    
    for batch_size in BATCH_SIZES:
        if batch_size * 2 > len(X_train_pool):
            print(f"\nBatch size {batch_size}: SKIP (not enough data)")
            continue
        
        accuracies = []
        
        for trial in range(NUM_TRIALS):
            # Random sample of batch_size pairs (2x samples since pos+neg)
            indices = np.random.choice(len(X_train_pool) // 2, size=batch_size, replace=False)
            # Convert pair indices to sample indices (each pair has 2 samples)
            sample_indices = []
            for idx in indices:
                sample_indices.extend([idx * 2, idx * 2 + 1])
            
            X_train = X_train_pool[sample_indices]
            y_train = y_train_pool[sample_indices]
            
            # Train classifier
            clf = LogisticRegression( random_state=42)
            clf.fit(X_train, y_train)
            
            # Test
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        min_acc = np.min(accuracies)
        max_acc = np.max(accuracies)
        
        results[batch_size] = {
            "mean": mean_acc,
            "std": std_acc,
            "min": min_acc,
            "max": max_acc,
            "range": max_acc - min_acc,
            "trials": accuracies,
        }
        
        print(f"\nBatch size {batch_size:3d}: mean={mean_acc:.3f}, std={std_acc:.3f}, range=[{min_acc:.3f}, {max_acc:.3f}]")
        print(f"  Trials: {[f'{a:.3f}' for a in accuracies]}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Batch Size':>10} | {'Mean Acc':>8} | {'Std':>6} | {'Range':>6} | Consistent?")
    print("-" * 55)
    
    for batch_size in BATCH_SIZES:
        if batch_size not in results:
            continue
        r = results[batch_size]
        # Consider consistent if std < 0.03 and range < 0.10
        consistent = "YES" if r["std"] < 0.03 and r["range"] < 0.10 else "NO"
        print(f"{batch_size:>10} | {r['mean']:>8.3f} | {r['std']:>6.3f} | {r['range']:>6.3f} | {consistent}")
    
    # Find minimum consistent batch size
    min_consistent = None
    for batch_size in BATCH_SIZES:
        if batch_size not in results:
            continue
        r = results[batch_size]
        if r["std"] < 0.03 and r["range"] < 0.10:
            min_consistent = batch_size
            break
    
    if min_consistent:
        print(f"\n*** Minimum consistent batch size: {min_consistent} ***")
    else:
        print("\n*** No batch size achieved consistent performance ***")


if __name__ == "__main__":
    main()

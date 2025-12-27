"""
Test whether our activation extraction gives meaningful signal
by comparing real contrastive pairs vs nonsense random pairs.

If nonsense pairs give similar Cohen's d / separation as real pairs,
then our signal is meaningless.
"""

import argparse
import random
import string
import torch
import numpy as np
from typing import List, Tuple
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from wisent.core.models.wisent_model import WisentModel
from wisent.core.activations.extraction_strategy import ExtractionStrategy
from wisent.core.activations.activations_collector import ActivationCollector
from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse


WORD_LIST = [
    "water", "sumo", "half", "purple", "elephant", "calculator", "yesterday", 
    "moon", "basket", "thinking", "telephone", "mountain", "running", "quickly",
    "tomorrow", "happiness", "keyboard", "window", "dancing", "coffee", "planet",
    "singing", "computer", "orange", "flying", "bicycle", "dream", "ocean",
    "pencil", "laughing", "cloud", "table", "walking", "music", "river", "chair",
    "jumping", "sun", "book", "swimming", "star", "door", "cooking", "tree",
    "writing", "sky", "flower", "playing", "rain", "paper", "sleeping", "green",
    "seven", "under", "before", "strange", "ancient", "modern", "simple"
]

def generate_nonsense_text(length: int = None) -> str:
    """Generate word salad - real words, no meaning."""
    if length is None:
        length = random.randint(3, 10)
    words = random.choices(WORD_LIST, k=length)
    return ' '.join(words)


def generate_nonsense_pairs(n: int = 50) -> List[ContrastivePair]:
    """Generate pairs with random nonsense text."""
    pairs = []
    for i in range(n):
        prompt = generate_nonsense_text(10)
        positive = generate_nonsense_text(15)
        negative = generate_nonsense_text(15)
        pairs.append(ContrastivePair(
            prompt=prompt,
            positive_response=PositiveResponse(model_response=positive),
            negative_response=NegativeResponse(model_response=negative),
        ))
    return pairs


def generate_real_pairs(n: int = 50) -> List[ContrastivePair]:
    """Generate real contrastive pairs with semantic meaning."""
    templates = [
        ("Is the Earth flat?", "No, the Earth is approximately spherical.", "Yes, the Earth is flat."),
        ("What is 2+2?", "4", "5"),
        ("Is water wet?", "Yes, water is wet.", "No, water is not wet."),
        ("What color is the sky?", "Blue", "Green"),
        ("Is the sun a star?", "Yes, the sun is a star.", "No, the sun is a planet."),
        ("What is the capital of France?", "Paris", "London"),
        ("Is Python a programming language?", "Yes, Python is a programming language.", "No, Python is a snake."),
        ("What is 10 * 5?", "50", "100"),
        ("Is ice cold?", "Yes, ice is cold.", "No, ice is hot."),
        ("What year did WW2 end?", "1945", "1939"),
    ]
    
    pairs = []
    for i in range(n):
        template = templates[i % len(templates)]
        # Add some variation
        variation = f" (instance {i})"
        pairs.append(ContrastivePair(
            prompt=template[0] + variation,
            positive_response=PositiveResponse(model_response=template[1]),
            negative_response=NegativeResponse(model_response=template[2]),
        ))
    return pairs


def compute_cohens_d(pos_acts: np.ndarray, neg_acts: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    pos_mean = np.mean(pos_acts, axis=0)
    neg_mean = np.mean(neg_acts, axis=0)
    
    pos_var = np.var(pos_acts, axis=0)
    neg_var = np.var(neg_acts, axis=0)
    
    n1, n2 = len(pos_acts), len(neg_acts)
    pooled_std = np.sqrt(((n1 - 1) * pos_var + (n2 - 1) * neg_var) / (n1 + n2 - 2))
    pooled_std = np.mean(pooled_std)  # average across dimensions
    
    if pooled_std < 1e-10:
        return 0.0
    
    diff = np.linalg.norm(pos_mean - neg_mean)
    return diff / pooled_std


def compute_linear_separability(pos_acts: np.ndarray, neg_acts: np.ndarray) -> float:
    """Compute linear separability score using SVM."""
    X = np.vstack([pos_acts, neg_acts])
    y = np.array([1] * len(pos_acts) + [0] * len(neg_acts))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    svm = LinearSVC(max_iter=1000, dual=False)
    svm.fit(X_scaled, y)
    
    return svm.score(X_scaled, y)


def collect_activations(
    model: WisentModel,
    pairs: List[ContrastivePair],
    strategy: ExtractionStrategy,
    layer: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect activations for positive and negative responses."""
    collector = ActivationCollector(model)
    
    pos_acts = []
    neg_acts = []
    
    for pair in pairs:
        try:
            # Collect both positive and negative using the collect method
            result = collector.collect(pair, strategy=strategy)
            
            # result is a ContrastivePair with activations
            pos_layer_acts = result.positive_response.layers_activations
            neg_layer_acts = result.negative_response.layers_activations
            
            # Extract layer (keys are strings like '1', '2', etc, and 1-indexed)
            layer_key = str(layer + 1)  # Convert to 1-indexed string
            if pos_layer_acts is not None and neg_layer_acts is not None:
                if layer_key in pos_layer_acts and layer_key in neg_layer_acts:
                    pos_acts.append(pos_layer_acts[layer_key].cpu().numpy())
                    neg_acts.append(neg_layer_acts[layer_key].cpu().numpy())
        except Exception as e:
            print(f"Error collecting pair: {e}")
            continue
    
    return np.array(pos_acts), np.array(neg_acts)


def main():
    parser = argparse.ArgumentParser(description="Test nonsense baseline vs real pairs")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--n-pairs", type=int, default=50)
    parser.add_argument("--strategies", type=str, nargs="+", 
                        default=["chat_mean", "chat_max_norm", "chat_last"])
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layers to test. Default: [0, 25%, 50%, 75%, last]")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model = WisentModel(args.model)
    num_layers = model.num_layers
    
    # Default layers if not specified
    if args.layers is None:
        args.layers = [
            0,
            num_layers // 4,
            num_layers // 2,
            3 * num_layers // 4,
            num_layers - 1,
        ]
    
    print(f"Model has {num_layers} layers")
    print(f"Testing layers: {args.layers}")
    print(f"Testing strategies: {args.strategies}")
    print(f"Pairs per condition: {args.n_pairs}")
    print()
    
    # Generate pairs
    print("Generating pairs...")
    real_pairs = generate_real_pairs(args.n_pairs)
    nonsense_pairs = generate_nonsense_pairs(args.n_pairs)
    
    results = []
    
    for strategy_name in args.strategies:
        strategy = ExtractionStrategy(strategy_name)
        
        for layer in args.layers:
            print(f"\n{'='*60}")
            print(f"Strategy: {strategy_name}, Layer: {layer} ({100*layer/num_layers:.0f}%)")
            print('='*60)
            
            # Real pairs
            print("  Collecting REAL pairs...")
            real_pos, real_neg = collect_activations(model, real_pairs, strategy, layer)
            
            if len(real_pos) < 10 or len(real_neg) < 10:
                print("  WARNING: Too few activations collected for real pairs")
                continue
            
            real_cohens_d = compute_cohens_d(real_pos, real_neg)
            real_linear = compute_linear_separability(real_pos, real_neg)
            
            # Nonsense pairs
            print("  Collecting NONSENSE pairs...")
            nonsense_pos, nonsense_neg = collect_activations(model, nonsense_pairs, strategy, layer)
            
            if len(nonsense_pos) < 10 or len(nonsense_neg) < 10:
                print("  WARNING: Too few activations collected for nonsense pairs")
                continue
            
            nonsense_cohens_d = compute_cohens_d(nonsense_pos, nonsense_neg)
            nonsense_linear = compute_linear_separability(nonsense_pos, nonsense_neg)
            
            # Compare
            print(f"\n  REAL pairs:     Cohen's d = {real_cohens_d:8.2f}, Linear = {real_linear:.3f}")
            print(f"  NONSENSE pairs: Cohen's d = {nonsense_cohens_d:8.2f}, Linear = {nonsense_linear:.3f}")
            print(f"  RATIO (real/nonsense): Cohen's d = {real_cohens_d/max(nonsense_cohens_d, 0.01):.2f}x")
            
            if real_cohens_d > nonsense_cohens_d * 2:
                verdict = "SIGNAL IS REAL"
            elif real_cohens_d > nonsense_cohens_d * 1.2:
                verdict = "WEAK SIGNAL"
            else:
                verdict = "NO SIGNAL (nonsense is similar!)"
            
            print(f"  VERDICT: {verdict}")
            
            results.append({
                "strategy": strategy_name,
                "layer": layer,
                "layer_pct": 100 * layer / num_layers,
                "real_cohens_d": real_cohens_d,
                "real_linear": real_linear,
                "nonsense_cohens_d": nonsense_cohens_d,
                "nonsense_linear": nonsense_linear,
                "ratio": real_cohens_d / max(nonsense_cohens_d, 0.01),
                "verdict": verdict,
            })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Strategy':<15} {'Layer':<10} {'Real d':<10} {'Nonsense d':<12} {'Ratio':<8} {'Verdict'}")
    print("-"*80)
    
    for r in results:
        print(f"{r['strategy']:<15} {r['layer']:>3} ({r['layer_pct']:>3.0f}%)  "
              f"{r['real_cohens_d']:>8.2f}   {r['nonsense_cohens_d']:>10.2f}   "
              f"{r['ratio']:>6.2f}x  {r['verdict']}")


if __name__ == "__main__":
    main()

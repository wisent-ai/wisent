#!/usr/bin/env python3
"""
Run RepScan-style analysis across extraction strategies.

Compares: signal strength (classifier accuracy), geometry (linear vs nonlinear), consistency.
"""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from wisent.core.data_loaders.loaders.lm_eval.lm_loader import LMEvalDataLoader
from wisent.core.models.wisent_model import WisentModel
from wisent.core.activations.activations_collector import ActivationCollector
from wisent.core.activations import ExtractionStrategy

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
STRATEGIES = [
    ExtractionStrategy.CHAT_LAST,
    ExtractionStrategy.CHAT_FIRST,
    ExtractionStrategy.CHAT_MEAN,
    ExtractionStrategy.ROLE_PLAY,
    ExtractionStrategy.MC_BALANCED,
]


def compute_metrics(pos: torch.Tensor, neg: torch.Tensor):
    """Compute signal and geometry metrics."""
    pos, neg = pos.cpu().float(), neg.cpu().float()
    X = torch.cat([pos, neg], dim=0).numpy()
    y = np.array([1] * len(pos) + [0] * len(neg))

    # Linear classifier (signal + geometry)
    linear = LogisticRegression(max_iter=1000, solver="lbfgs")
    linear_scores = cross_val_score(linear, X, y, cv=5, scoring="accuracy")
    linear_acc = linear_scores.mean()

    # Nonlinear classifier (geometry)
    mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, early_stopping=True)
    mlp_scores = cross_val_score(mlp, X, y, cv=5, scoring="accuracy")
    nonlinear_acc = mlp_scores.mean()

    gap = nonlinear_acc - linear_acc
    geometry = "NONLINEAR" if gap > 0.05 else "LINEAR"

    # Direction consistency
    diffs = pos - neg
    diffs_norm = diffs / (torch.norm(diffs, dim=1, keepdim=True) + 1e-8)
    pairwise = diffs_norm @ diffs_norm.T
    mask = torch.triu(torch.ones_like(pairwise), diagonal=1).bool()
    consistency = pairwise[mask].mean().item()

    # Recommended method
    if geometry == "LINEAR":
        recommended = "CAA"
    else:
        recommended = "Hyperplane"

    return {
        "linear_acc": linear_acc,
        "nonlinear_acc": nonlinear_acc,
        "gap": gap,
        "geometry": geometry,
        "consistency": consistency,
        "recommended": recommended,
    }


print(f"Loading {MODEL}...")
model = WisentModel(MODEL)
layer = model.num_layers // 2
print(f"Using layer {layer}")

loader = LMEvalDataLoader()
result = loader._load_one_task("truthfulqa_gen", 0.8, 42, 500, None, None)
train_pairs = result["train_qa_pairs"].pairs[:100]
print(f"Using {len(train_pairs)} pairs\n")

store_dev = "mps" if torch.backends.mps.is_available() else "cpu"
collector = ActivationCollector(model=model, store_device=store_dev)

results = {}

for strategy in STRATEGIES:
    print(f"{'='*60}")
    print(f"Strategy: {strategy.value}")
    print(f"{'='*60}")

    pos_acts, neg_acts = [], []
    for i, pair in enumerate(train_pairs):
        if i % 25 == 0:
            print(f"  Collecting {i}/{len(train_pairs)}...")
        pair_acts = collector.collect(pair, strategy=strategy, layers=[str(layer)])
        pos = pair_acts.positive_response.layers_activations[str(layer)]
        neg = pair_acts.negative_response.layers_activations[str(layer)]
        pos_acts.append(pos.flatten())
        neg_acts.append(neg.flatten())

    pos_tensor = torch.stack(pos_acts)
    neg_tensor = torch.stack(neg_acts)

    print("  Computing metrics...")
    metrics = compute_metrics(pos_tensor, neg_tensor)
    results[strategy.value] = metrics

    print(f"  Linear acc:  {metrics['linear_acc']:.3f}")
    print(f"  Nonlin acc:  {metrics['nonlinear_acc']:.3f}")
    print(f"  Gap:         {metrics['gap']:.3f}")
    print(f"  Geometry:    {metrics['geometry']}")
    print(f"  Consistency: {metrics['consistency']:.4f}")
    print(f"  Recommended: {metrics['recommended']}\n")

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\n{'Strategy':<15} {'Lin Acc':>10} {'Nonlin':>10} {'Gap':>8} {'Geom':>10} {'Consist':>10} {'Method':>12}")
print("-" * 80)

for name, r in results.items():
    print(f"{name:<15} {r['linear_acc']:>10.3f} {r['nonlinear_acc']:>10.3f} {r['gap']:>8.3f} {r['geometry']:>10} {r['consistency']:>10.4f} {r['recommended']:>12}")

best_linear = max(results.items(), key=lambda x: x[1]["linear_acc"])
best_consist = max(results.items(), key=lambda x: x[1]["consistency"])

print(f"\nBest linear accuracy: {best_linear[0]} ({best_linear[1]['linear_acc']:.3f})")
print(f"Best consistency:     {best_consist[0]} ({best_consist[1]['consistency']:.4f})")
print("=" * 80)

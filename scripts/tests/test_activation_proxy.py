#!/usr/bin/env python3
"""
Test activation-based proxy metric for steering evaluation.

Instead of running full generation + evaluation (~50 min per method),
use held-out activation pairs to compute steering accuracy (~5 sec per method).

For one benchmark:
1. Load activations from DB
2. Split train/test
3. For each method: train steering vector, evaluate on test activations
4. Compute repscan metrics
5. Compare: does repscan predict the best method?
"""

import os
import struct
import numpy as np
import psycopg2
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn

# Database configuration
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "aws-0-eu-west-2.pooler.supabase.com"),
    "port": int(os.environ.get("DB_PORT", 5432)),
    "database": os.environ.get("DB_NAME", "postgres"),
    "user": os.environ.get("DB_USER", "postgres.rbqjqnouluslojmmnuqi"),
    "password": os.environ.get("DB_PASSWORD", "BsKuEnPFLCFurN4a"),
}


def bytes_to_vector(data: bytes) -> np.ndarray:
    """Convert binary data back to float vector."""
    num_floats = len(data) // 4
    return np.array(struct.unpack(f'{num_floats}f', data))


def load_benchmark_activations(model_name: str, benchmark: str, layer: int = None):
    """Load activations for a specific benchmark from DB."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Get model info
    cur.execute('SELECT id, "numLayers" FROM "Model" WHERE name = %s', (model_name,))
    result = cur.fetchone()
    if not result:
        raise ValueError(f"Model {model_name} not found")
    model_id, num_layers = result

    if layer is None:
        layer = num_layers // 2

    # Get benchmark ID
    cur.execute('SELECT id FROM "ContrastivePairSet" WHERE name = %s', (benchmark,))
    result = cur.fetchone()
    if not result:
        raise ValueError(f"Benchmark {benchmark} not found")
    benchmark_id = result[0]

    # Load activations
    cur.execute('''
        SELECT
            a."contrastivePairId",
            a."extractionStrategy",
            a."activationData",
            a."isPositive"
        FROM "Activation" a
        WHERE a."modelId" = %s
          AND a."contrastivePairSetId" = %s
          AND a.layer = %s
        ORDER BY a."contrastivePairId", a."extractionStrategy"
    ''', (model_id, benchmark_id, layer))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    # Group by pair and strategy
    data = defaultdict(lambda: defaultdict(dict))
    for pair_id, strategy, act_bytes, is_positive in rows:
        key = "pos" if is_positive else "neg"
        data[strategy][pair_id][key] = bytes_to_vector(act_bytes)

    # Convert to arrays per strategy
    result = {}
    for strategy, pairs in data.items():
        pos_list = []
        neg_list = []
        for pair_id, acts in pairs.items():
            if "pos" in acts and "neg" in acts:
                pos_list.append(acts["pos"])
                neg_list.append(acts["neg"])
        if len(pos_list) >= 10:
            result[strategy] = {
                "pos": np.array(pos_list),
                "neg": np.array(neg_list),
            }

    return result, num_layers


def compute_repscan_metrics(pos: np.ndarray, neg: np.ndarray) -> dict:
    """Compute repscan-like geometry metrics."""
    diffs = pos - neg
    mean_diff = diffs.mean(axis=0)
    mean_diff_norm = np.linalg.norm(mean_diff)

    if mean_diff_norm < 1e-10:
        return {"linear_probe": 0.5, "signal_strength": 0, "cluster_sep": 0}

    # Linear probe accuracy
    X = np.vstack([pos, neg])
    y = np.array([1] * len(pos) + [0] * len(neg))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    linear_probe = clf.score(X_test, y_test)

    # Signal strength
    signal = mean_diff_norm
    noise = np.std([np.linalg.norm(d - mean_diff) for d in diffs])
    signal_strength = signal / (noise + 1e-10)

    # Cluster separation
    pos_center = pos.mean(axis=0)
    neg_center = neg.mean(axis=0)
    cluster_sep = np.linalg.norm(pos_center - neg_center) / (np.std(pos) + np.std(neg) + 1e-10)

    return {
        "linear_probe": linear_probe,
        "signal_strength": min(signal_strength, 10),  # cap
        "cluster_sep": min(cluster_sep, 10),
    }


def train_caa(train_pos: np.ndarray, train_neg: np.ndarray) -> np.ndarray:
    """Train CAA steering vector (mean difference)."""
    vec = train_pos.mean(axis=0) - train_neg.mean(axis=0)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-10 else vec


def train_hyperplane(train_pos: np.ndarray, train_neg: np.ndarray) -> np.ndarray:
    """Train Hyperplane steering vector (logistic regression weights)."""
    X = np.vstack([train_pos, train_neg])
    y = np.array([1] * len(train_pos) + [0] * len(train_neg))
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X, y)
    vec = clf.coef_[0]
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-10 else vec


def train_mlp(train_pos: np.ndarray, train_neg: np.ndarray, hidden_dim=256, epochs=100) -> np.ndarray:
    """Train MLP and extract gradient-based direction."""
    X = torch.tensor(np.vstack([train_pos, train_neg]), dtype=torch.float32)
    y = torch.tensor([1] * len(train_pos) + [0] * len(train_neg), dtype=torch.float32)

    input_dim = X.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
        nn.Sigmoid()
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X).squeeze()
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

    # Extract direction from first layer weights
    vec = model[0].weight.data.mean(dim=0).numpy()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-10 else vec


def train_pca_based(train_pos: np.ndarray, train_neg: np.ndarray, n_components=3) -> np.ndarray:
    """Train PCA-based direction (simplified PRISM)."""
    diffs = train_pos - train_neg
    # PCA on differences
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(n_components, len(diffs)-1))
    pca.fit(diffs)
    # Primary direction
    vec = pca.components_[0]
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-10 else vec


def evaluate_steering(vec: np.ndarray, test_pos: np.ndarray, test_neg: np.ndarray) -> float:
    """Evaluate steering vector on test set using pairwise accuracy."""
    pos_proj = test_pos @ vec
    neg_proj = test_neg @ vec

    correct = 0
    total = 0
    for p in pos_proj:
        for n in neg_proj:
            if p > n:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.5


def run_benchmark_comparison(model_name: str, benchmark: str, layer: int = None):
    """Run full comparison for one benchmark."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {benchmark}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    # Load activations
    print("\nLoading activations from DB...")
    activations, num_layers = load_benchmark_activations(model_name, benchmark, layer)

    if not activations:
        print("No activations found!")
        return None

    # Use first available strategy
    strategy = list(activations.keys())[0]
    pos = activations[strategy]["pos"]
    neg = activations[strategy]["neg"]
    print(f"Using strategy: {strategy}")
    print(f"Pairs: {len(pos)}")

    # Train/test split
    n = len(pos)
    train_idx = int(n * 0.8)
    train_pos, test_pos = pos[:train_idx], pos[train_idx:]
    train_neg, test_neg = neg[:train_idx], neg[train_idx:]
    print(f"Train: {len(train_pos)}, Test: {len(test_pos)}")

    # Compute repscan metrics
    print("\nComputing repscan metrics...")
    repscan = compute_repscan_metrics(train_pos, train_neg)
    print(f"  Linear probe: {repscan['linear_probe']:.3f}")
    print(f"  Signal strength: {repscan['signal_strength']:.3f}")
    print(f"  Cluster separation: {repscan['cluster_sep']:.3f}")

    # Repscan recommendation logic (simplified)
    if repscan['linear_probe'] > 0.9:
        repscan_pick = "CAA"
    elif repscan['signal_strength'] > 2:
        repscan_pick = "Hyperplane"
    else:
        repscan_pick = "PCA"  # proxy for PRISM/TITAN
    print(f"  Repscan recommendation: {repscan_pick}")

    # Train and evaluate each method
    print("\nTraining and evaluating methods...")
    results = {}

    # CAA
    vec = train_caa(train_pos, train_neg)
    acc = evaluate_steering(vec, test_pos, test_neg)
    results["CAA"] = acc
    print(f"  CAA: {acc:.3f}")

    # Hyperplane
    vec = train_hyperplane(train_pos, train_neg)
    acc = evaluate_steering(vec, test_pos, test_neg)
    results["Hyperplane"] = acc
    print(f"  Hyperplane: {acc:.3f}")

    # MLP
    vec = train_mlp(train_pos, train_neg)
    acc = evaluate_steering(vec, test_pos, test_neg)
    results["MLP"] = acc
    print(f"  MLP: {acc:.3f}")

    # PCA-based (proxy for PRISM)
    vec = train_pca_based(train_pos, train_neg)
    acc = evaluate_steering(vec, test_pos, test_neg)
    results["PCA"] = acc
    print(f"  PCA (PRISM proxy): {acc:.3f}")

    # Find actual best
    actual_best = max(results, key=results.get)
    print(f"\nActual best: {actual_best} ({results[actual_best]:.3f})")
    print(f"Repscan pick: {repscan_pick}")
    print(f"Repscan correct: {actual_best == repscan_pick}")

    return {
        "benchmark": benchmark,
        "repscan_metrics": repscan,
        "repscan_pick": repscan_pick,
        "method_accuracies": results,
        "actual_best": actual_best,
        "repscan_correct": actual_best == repscan_pick,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Model name")
    parser.add_argument("--benchmark", default="truthfulqa_custom", help="Benchmark name")
    parser.add_argument("--layer", type=int, default=None, help="Layer (default: middle)")
    args = parser.parse_args()

    result = run_benchmark_comparison(args.model, args.benchmark, args.layer)

    if result:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Repscan predicted: {result['repscan_pick']}")
        print(f"Actual best: {result['actual_best']}")
        print(f"Prediction correct: {result['repscan_correct']}")

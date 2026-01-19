"""
Common utilities for research analysis.

Shared database access, data structures, and metric computation functions.
"""

import os
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import psycopg2
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# All available steering methods
STEERING_METHODS = ["caa", "hyperplane", "mlp", "prism", "pulse", "titan"]


# Database configuration - uses environment variables with fallback
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


@dataclass
class ActivationData:
    """Container for activation data from a single pair."""
    pair_id: int
    set_id: int
    benchmark: str
    layer: int
    strategy: str
    positive_activation: np.ndarray
    negative_activation: np.ndarray


@dataclass
class BenchmarkResults:
    """Results for a single benchmark."""
    name: str
    num_pairs: int
    strategies: Dict[str, Dict[str, float]] = field(default_factory=dict)
    geometry_metrics: Dict[str, float] = field(default_factory=dict)
    best_strategy: str = ""
    best_accuracy: float = 0.0


def load_activations_from_db(model_name: str, layer: int = None) -> Dict[str, List[ActivationData]]:
    """
    Load activations from database grouped by benchmark.

    Args:
        model_name: Name of the model (e.g., 'Qwen/Qwen3-8B')
        layer: Layer to load (default: middle layer)

    Returns:
        Dict[benchmark_name -> List[ActivationData]]
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Get model ID
    cur.execute('SELECT id, "numLayers" FROM "Model" WHERE name = %s', (model_name,))
    result = cur.fetchone()
    if not result:
        raise ValueError(f"Model {model_name} not found in database")
    model_id, num_layers = result

    # Default to middle layer if not specified
    if layer is None:
        layer = num_layers // 2

    print(f"Loading activations for model {model_name} (id={model_id}), layer {layer}")

    # Query activations with benchmark info
    print(f"Querying activations for model_id={model_id}, layer={layer}...")
    cur.execute('''
        SELECT
            a."contrastivePairId",
            a."contrastivePairSetId",
            cps.name as benchmark,
            a.layer,
            a."extractionStrategy",
            a."activationData",
            a."isPositive"
        FROM "Activation" a
        JOIN "ContrastivePairSet" cps ON a."contrastivePairSetId" = cps.id
        WHERE a."modelId" = %s AND a.layer = %s
        ORDER BY a."contrastivePairSetId", a."contrastivePairId", a."extractionStrategy"
    ''', (model_id, layer))

    rows = cur.fetchall()
    print(f"Fetched {len(rows)} activation rows")

    # Group by benchmark and pair
    raw_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for row in rows:
        pair_id, set_id, benchmark, layer, strategy, activation_bytes, is_positive = row
        key = "positive" if is_positive else "negative"
        raw_data[benchmark][(pair_id, set_id)][strategy][key] = bytes_to_vector(activation_bytes)

    # Convert to ActivationData objects
    result = defaultdict(list)
    for benchmark, pairs in raw_data.items():
        for (pair_id, set_id), strategies in pairs.items():
            for strategy, activations in strategies.items():
                if "positive" in activations and "negative" in activations:
                    result[benchmark].append(ActivationData(
                        pair_id=pair_id,
                        set_id=set_id,
                        benchmark=benchmark,
                        layer=layer,
                        strategy=strategy,
                        positive_activation=activations["positive"],
                        negative_activation=activations["negative"],
                    ))

    cur.close()
    conn.close()

    return dict(result)


def compute_geometry_metrics(pos_activations: np.ndarray, neg_activations: np.ndarray) -> Dict[str, float]:
    """
    Compute RepScan-like geometry metrics for predicting steering effectiveness.

    Args:
        pos_activations: [N, D] array of positive activations
        neg_activations: [N, D] array of negative activations

    Returns:
        Dict of geometry metrics:
        - diff_mean_alignment: How well individual diffs align with mean diff
        - pct_positive_alignment: Percentage of diffs with positive alignment
        - signal_to_noise: Ratio of mean diff magnitude to noise
        - cluster_separation: Cosine distance between cluster centers
        - linear_probe_accuracy: Accuracy of logistic regression classifier
    """
    N = min(len(pos_activations), len(neg_activations))
    pos = pos_activations[:N]
    neg = neg_activations[:N]

    # Compute per-pair differences
    diffs = pos - neg  # [N, D]

    # Mean difference (CAA direction)
    mean_diff = diffs.mean(axis=0)
    mean_diff_norm = np.linalg.norm(mean_diff)

    if mean_diff_norm < 1e-10:
        return {
            "diff_mean_alignment": 0.0,
            "pct_positive_alignment": 0.0,
            "signal_to_noise": 0.0,
            "cluster_separation": 0.0,
            "linear_probe_accuracy": 0.5,
        }

    mean_diff_normalized = mean_diff / mean_diff_norm

    # 1. Diff-mean alignment: how well do individual diffs align with mean diff?
    alignments = []
    for diff in diffs:
        diff_norm = np.linalg.norm(diff)
        if diff_norm > 1e-10:
            alignment = np.dot(diff / diff_norm, mean_diff_normalized)
            alignments.append(alignment)

    diff_mean_alignment = np.mean(alignments) if alignments else 0.0
    pct_positive_alignment = np.mean([a > 0 for a in alignments]) if alignments else 0.0

    # 2. Signal-to-noise ratio
    signal = mean_diff_norm
    noise = np.std([np.linalg.norm(d - mean_diff) for d in diffs])
    signal_to_noise = signal / (noise + 1e-10)

    # 3. Cluster separation (cosine distance between cluster centers)
    pos_center = pos.mean(axis=0)
    neg_center = neg.mean(axis=0)
    pos_norm = np.linalg.norm(pos_center)
    neg_norm = np.linalg.norm(neg_center)
    if pos_norm > 1e-10 and neg_norm > 1e-10:
        cluster_separation = 1 - np.dot(pos_center / pos_norm, neg_center / neg_norm)
    else:
        cluster_separation = 0.0

    # 4. Linear probe accuracy
    X = np.vstack([pos, neg])
    y = np.array([1] * len(pos) + [0] * len(neg))

    if len(X) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_train, y_train)
        linear_probe_accuracy = accuracy_score(y_test, clf.predict(X_test))
    else:
        linear_probe_accuracy = 0.5

    return {
        "diff_mean_alignment": float(diff_mean_alignment),
        "pct_positive_alignment": float(pct_positive_alignment),
        "signal_to_noise": float(signal_to_noise),
        "cluster_separation": float(cluster_separation),
        "linear_probe_accuracy": float(linear_probe_accuracy),
    }


def compute_steering_accuracy(
    train_pos: np.ndarray,
    train_neg: np.ndarray,
    test_pos: np.ndarray,
    test_neg: np.ndarray,
    method: str = "caa"
) -> float:
    """
    Compute steering accuracy using train/test split.

    Train a steering vector on train set, evaluate on test set.
    Accuracy = how often steering direction correctly classifies pos vs neg.

    Args:
        train_pos: Training positive activations
        train_neg: Training negative activations
        test_pos: Test positive activations
        test_neg: Test negative activations
        method: One of "caa", "hyperplane", "mlp", "prism", "pulse", "titan"

    Returns:
        Pairwise accuracy (probability that pos > neg on steering direction)
    """
    steering_vector = compute_steering_vector(train_pos, train_neg, method)

    if steering_vector is None:
        return 0.5

    # Normalize steering vector
    norm = np.linalg.norm(steering_vector)
    if norm < 1e-10:
        return 0.5
    steering_vector = steering_vector / norm

    # Evaluate: does projection onto steering vector separate pos from neg?
    test_pos_proj = test_pos @ steering_vector
    test_neg_proj = test_neg @ steering_vector

    # Accuracy: positive examples should have higher projection
    correct = 0
    total = 0
    for p in test_pos_proj:
        for n in test_neg_proj:
            if p > n:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.5


def compute_steering_vector(
    train_pos: np.ndarray,
    train_neg: np.ndarray,
    method: str = "caa"
) -> Optional[np.ndarray]:
    """
    Compute steering vector using specified method.

    Args:
        train_pos: Training positive activations [N, D]
        train_neg: Training negative activations [N, D]
        method: One of "caa", "hyperplane", "mlp", "prism", "pulse", "titan"

    Returns:
        Steering vector [D] or None if failed
    """
    if method == "caa":
        return _compute_caa(train_pos, train_neg)
    elif method == "hyperplane":
        return _compute_hyperplane(train_pos, train_neg)
    elif method == "mlp":
        return _compute_mlp(train_pos, train_neg)
    elif method == "prism":
        return _compute_prism(train_pos, train_neg)
    elif method == "pulse":
        return _compute_pulse(train_pos, train_neg)
    elif method == "titan":
        return _compute_titan(train_pos, train_neg)
    else:
        raise ValueError(f"Unknown method: {method}. Available: {STEERING_METHODS}")


def _compute_caa(train_pos: np.ndarray, train_neg: np.ndarray) -> np.ndarray:
    """CAA: mean difference."""
    return train_pos.mean(axis=0) - train_neg.mean(axis=0)


def _compute_hyperplane(train_pos: np.ndarray, train_neg: np.ndarray) -> np.ndarray:
    """Hyperplane: logistic regression weights."""
    X = np.vstack([train_pos, train_neg])
    y = np.array([1] * len(train_pos) + [0] * len(train_neg))
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X, y)
    return clf.coef_[0]


def _compute_mlp(train_pos: np.ndarray, train_neg: np.ndarray) -> Optional[np.ndarray]:
    """MLP: adversarial gradient direction from trained classifier."""
    try:
        from wisent.core.steering_methods.registry import SteeringMethodRegistry

        pos_tensor = torch.tensor(train_pos, dtype=torch.float32)
        neg_tensor = torch.tensor(train_neg, dtype=torch.float32)

        method = SteeringMethodRegistry.create_method_instance(
            "mlp",
            hidden_dim=min(256, train_pos.shape[1] // 4),
            num_layers=2,
            epochs=50,
            learning_rate=0.001,
        )

        steering_vector = method.train_for_layer(
            [pos_tensor[i] for i in range(len(pos_tensor))],
            [neg_tensor[i] for i in range(len(neg_tensor))]
        )

        return steering_vector.cpu().numpy()
    except Exception as e:
        print(f"MLP failed: {e}")
        return _compute_caa(train_pos, train_neg)


def _compute_prism(train_pos: np.ndarray, train_neg: np.ndarray) -> Optional[np.ndarray]:
    """PRISM: gradient-optimized multi-directional steering (returns first direction)."""
    try:
        from wisent.core.steering_methods.registry import SteeringMethodRegistry

        pos_tensor = torch.tensor(train_pos, dtype=torch.float32)
        neg_tensor = torch.tensor(train_neg, dtype=torch.float32)

        method = SteeringMethodRegistry.create_method_instance(
            "prism",
            num_directions=3,
            optimization_steps=50,
            learning_rate=0.01,
        )

        steering_vectors = method.train_for_layer(
            [pos_tensor[i] for i in range(len(pos_tensor))],
            [neg_tensor[i] for i in range(len(neg_tensor))]
        )

        # Return first direction
        if isinstance(steering_vectors, torch.Tensor):
            if steering_vectors.dim() == 2:
                return steering_vectors[0].cpu().numpy()
            return steering_vectors.cpu().numpy()
        return steering_vectors
    except Exception as e:
        print(f"PRISM failed: {e}")
        return _compute_caa(train_pos, train_neg)


def _compute_pulse(train_pos: np.ndarray, train_neg: np.ndarray) -> Optional[np.ndarray]:
    """PULSE: condition-gated steering (returns steering vector)."""
    try:
        from wisent.core.steering_methods.registry import SteeringMethodRegistry

        pos_tensor = torch.tensor(train_pos, dtype=torch.float32)
        neg_tensor = torch.tensor(train_neg, dtype=torch.float32)

        method = SteeringMethodRegistry.create_method_instance(
            "pulse",
            optimization_steps=50,
            learning_rate=0.01,
        )

        result = method.train_for_layer(
            [pos_tensor[i] for i in range(len(pos_tensor))],
            [neg_tensor[i] for i in range(len(neg_tensor))]
        )

        if isinstance(result, torch.Tensor):
            return result.cpu().numpy()
        elif isinstance(result, dict) and "steering_vector" in result:
            return result["steering_vector"].cpu().numpy()
        return _compute_caa(train_pos, train_neg)
    except Exception as e:
        print(f"PULSE failed: {e}")
        return _compute_caa(train_pos, train_neg)


def _compute_titan(train_pos: np.ndarray, train_neg: np.ndarray) -> Optional[np.ndarray]:
    """TITAN: joint optimized manifold (returns primary direction)."""
    try:
        from wisent.core.steering_methods.registry import SteeringMethodRegistry

        pos_tensor = torch.tensor(train_pos, dtype=torch.float32)
        neg_tensor = torch.tensor(train_neg, dtype=torch.float32)

        method = SteeringMethodRegistry.create_method_instance(
            "titan",
            num_directions=3,
            optimization_steps=50,
            learning_rate=0.005,
        )

        result = method.train_for_layer(
            [pos_tensor[i] for i in range(len(pos_tensor))],
            [neg_tensor[i] for i in range(len(neg_tensor))]
        )

        if isinstance(result, torch.Tensor):
            if result.dim() == 2:
                return result[0].cpu().numpy()
            return result.cpu().numpy()
        elif isinstance(result, dict) and "directions" in result:
            return result["directions"][0].cpu().numpy()
        return _compute_caa(train_pos, train_neg)
    except Exception as e:
        print(f"TITAN failed: {e}")
        return _compute_caa(train_pos, train_neg)

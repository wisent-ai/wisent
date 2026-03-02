"""
Helper functions for test_activation_proxy.py.

Contains DB loading, zwiad metric computation, steering method training,
and evaluation functions.
"""

import os
import struct
import numpy as np
import psycopg2
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from wisent.core.utils.config_tools.constants import ZERO_THRESHOLD, DEFAULT_RANDOM_SEED, DEFAULT_SPLIT_RATIO, PROBE_TRAINING_LR

# Database configuration
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "aws-0-eu-west-2.pooler.supabase.com"),
    "port": int(os.environ.get("DB_PORT", 5432)),
    "database": os.environ.get("DB_NAME", "postgres"),
    "user": os.environ.get("DB_USER", "postgres.rbqjqnouluslojmmnuqi"),
    "password": os.environ.get("DB_PASSWORD", "REDACTED_DB_PASSWORD"),
}


def bytes_to_vector(data: bytes) -> np.ndarray:
    """Convert binary data back to float vector."""
    num_floats = len(data) // 4
    return np.array(struct.unpack(f'{num_floats}f', data))


def load_benchmark_activations(model_name: str, benchmark: str, layer: int = None):
    """Load activations for a specific benchmark from DB."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute('SELECT id, "numLayers" FROM "Model" WHERE name = %s', (model_name,))
    result = cur.fetchone()
    if not result:
        raise ValueError(f"Model {model_name} not found")
    model_id, num_layers = result

    if layer is None:
        layer = num_layers // 2

    cur.execute('SELECT id FROM "ContrastivePairSet" WHERE name = %s', (benchmark,))
    result = cur.fetchone()
    if not result:
        raise ValueError(f"Benchmark {benchmark} not found")
    benchmark_id = result[0]

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

    data = defaultdict(lambda: defaultdict(dict))
    for pair_id, strategy, act_bytes, is_positive in rows:
        key = "pos" if is_positive else "neg"
        data[strategy][pair_id][key] = bytes_to_vector(act_bytes)

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


def compute_zwiad_metrics(pos: np.ndarray, neg: np.ndarray) -> dict:
    """Compute zwiad-like geometry metrics."""
    diffs = pos - neg
    mean_diff = diffs.mean(axis=0)
    mean_diff_norm = np.linalg.norm(mean_diff)

    if mean_diff_norm < ZERO_THRESHOLD:
        return {"linear_probe": 0.5, "signal_strength": 0, "cluster_sep": 0}

    X = np.vstack([pos, neg])
    y = np.array([1] * len(pos) + [0] * len(neg))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - DEFAULT_SPLIT_RATIO), random_state=DEFAULT_RANDOM_SEED)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    linear_probe = clf.score(X_test, y_test)

    signal = mean_diff_norm
    noise = np.std([np.linalg.norm(d - mean_diff) for d in diffs])
    signal_strength = signal / (noise + ZERO_THRESHOLD)

    pos_center = pos.mean(axis=0)
    neg_center = neg.mean(axis=0)
    cluster_sep = np.linalg.norm(pos_center - neg_center) / (np.std(pos) + np.std(neg) + ZERO_THRESHOLD)

    return {
        "linear_probe": linear_probe,
        "signal_strength": min(signal_strength, 10),
        "cluster_sep": min(cluster_sep, 10),
    }


def train_caa(train_pos: np.ndarray, train_neg: np.ndarray) -> np.ndarray:
    """Train CAA steering vector (mean difference)."""
    vec = train_pos.mean(axis=0) - train_neg.mean(axis=0)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > ZERO_THRESHOLD else vec


def train_ostrze(train_pos: np.ndarray, train_neg: np.ndarray) -> np.ndarray:
    """Train Ostrze steering vector (logistic regression weights)."""
    X = np.vstack([train_pos, train_neg])
    y = np.array([1] * len(train_pos) + [0] * len(train_neg))
    clf = LogisticRegression(C=1.0)
    clf.fit(X, y)
    vec = clf.coef_[0]
    norm = np.linalg.norm(vec)
    return vec / norm if norm > ZERO_THRESHOLD else vec


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

    optimizer = torch.optim.Adam(model.parameters(), lr=PROBE_TRAINING_LR)
    loss_fn = nn.BCELoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X).squeeze()
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

    vec = model[0].weight.data.mean(dim=0).numpy()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > ZERO_THRESHOLD else vec


def train_pca_based(train_pos: np.ndarray, train_neg: np.ndarray, n_components=3) -> np.ndarray:
    """Train PCA-based direction (simplified TECZA)."""
    diffs = train_pos - train_neg
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(n_components, len(diffs)-1))
    pca.fit(diffs)
    vec = pca.components_[0]
    norm = np.linalg.norm(vec)
    return vec / norm if norm > ZERO_THRESHOLD else vec


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

"""Geometry metrics for RepScan Step 2: Geometry Test."""

from typing import Tuple
import torch
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def _adaptive_cv(n_samples: int) -> int:
    """Adaptive CV folds: ensure at least 10 samples per fold."""
    return max(2, min(5, n_samples // 10))


def _adaptive_mlp_hidden(n_features: int) -> int:
    """Adaptive MLP hidden size: sqrt(n_features) clamped."""
    hidden = int(np.sqrt(n_features))
    return max(16, min(hidden, 256))


def compute_linear_nonlinear_gap(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> Tuple[float, float]:
    """
    Compare linear vs nonlinear probe accuracy with adaptive parameters.

    Returns:
        Tuple of (linear_accuracy, nonlinear_accuracy)
    """
    pos_np = pos_activations.cpu().numpy() if isinstance(pos_activations, torch.Tensor) else pos_activations
    neg_np = neg_activations.cpu().numpy() if isinstance(neg_activations, torch.Tensor) else neg_activations

    X = np.vstack([pos_np, neg_np])
    y = np.array([1] * len(pos_np) + [0] * len(neg_np))

    n_samples, n_features = X.shape
    cv = _adaptive_cv(n_samples)
    mlp_hidden = _adaptive_mlp_hidden(n_features)
    max_iter = max(200, min(1000, n_samples * 2))

    # Linear probe (logistic regression)
    # Adaptive regularization: stronger for high-d
    C = n_samples / n_features  # Lower C = more regularization when n_features >> n_samples
    C = max(0.01, min(C, 10.0))
    linear_clf = LogisticRegression(max_iter=1000, C=C, random_state=42)
    linear_scores = cross_val_score(linear_clf, X, y, cv=cv, scoring="accuracy")
    linear_acc = float(linear_scores.mean())

    # Nonlinear probe (MLP)
    mlp_clf = MLPClassifier(
        hidden_layer_sizes=(mlp_hidden,),
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        alpha=1.0 / n_samples,  # Adaptive L2 regularization
    )
    mlp_scores = cross_val_score(mlp_clf, X, y, cv=cv, scoring="accuracy")
    nonlinear_acc = float(mlp_scores.mean())

    return linear_acc, nonlinear_acc


def compute_geometry_summary(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> dict:
    """
    Compute geometry summary including linear vs nonlinear gap analysis.

    Args:
        pos_activations: Positive class activations
        neg_activations: Negative class activations

    Returns:
        Dict with linear_accuracy, nonlinear_accuracy, gap, and diagnosis
    """
    linear_acc, nonlinear_acc = compute_linear_nonlinear_gap(pos_activations, neg_activations)
    gap = nonlinear_acc - linear_acc
    n_samples = len(pos_activations) + len(neg_activations)
    adaptive_threshold = 0.5 / np.sqrt(n_samples)
    adaptive_threshold = max(0.02, min(adaptive_threshold, 0.1))
    diagnosis = "NONLINEAR" if gap > adaptive_threshold else "LINEAR"
    return {
        "linear_accuracy": linear_acc,
        "nonlinear_accuracy": nonlinear_acc,
        "gap": gap,
        "gap_threshold": adaptive_threshold,
        "diagnosis": diagnosis,
    }

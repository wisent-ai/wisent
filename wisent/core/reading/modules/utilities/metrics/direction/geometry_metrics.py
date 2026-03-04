"""Geometry metrics for Zwiad Step 2: Geometry Test."""

from typing import Tuple
import torch
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from wisent.core import constants as _C


def _adaptive_cv(
    n_samples: int, *,
    cv_min_folds: int, cv_max_folds: int, cv_samples_per_fold: int,
) -> int:
    """Adaptive CV folds: ensure sufficient samples per fold."""
    return max(cv_min_folds, min(cv_max_folds, n_samples // cv_samples_per_fold))


def _adaptive_mlp_hidden(
    n_features: int, *, mlp_hidden_min: int, mlp_hidden_max: int,
) -> int:
    """Adaptive MLP hidden size: sqrt(n_features) clamped."""
    hidden = int(np.sqrt(n_features))
    return max(mlp_hidden_min, min(hidden, mlp_hidden_max))


def compute_linear_nonlinear_gap(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    *,
    cv_min_folds: int, cv_max_folds: int, cv_samples_per_fold: int,
    mlp_hidden_min: int, mlp_hidden_max: int,
    geometry_logistic_c_min: float, geometry_logistic_c_max: float,
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
    cv = _adaptive_cv(
        n_samples, cv_min_folds=cv_min_folds,
        cv_max_folds=cv_max_folds, cv_samples_per_fold=cv_samples_per_fold)
    mlp_hidden = _adaptive_mlp_hidden(
        n_features, mlp_hidden_min=mlp_hidden_min,
        mlp_hidden_max=mlp_hidden_max)

    # Linear probe (logistic regression)
    # Adaptive regularization: stronger for high-d
    C = n_samples / n_features  # Lower C = more regularization when n_features >> n_samples
    C = max(geometry_logistic_c_min, min(C, geometry_logistic_c_max))
    linear_clf = LogisticRegression(C=C, random_state=_C.DEFAULT_RANDOM_SEED)
    linear_scores = cross_val_score(linear_clf, X, y, cv=cv, scoring="accuracy")
    linear_acc = float(linear_scores.mean())

    # Nonlinear probe (MLP)
    mlp_clf = MLPClassifier(
        hidden_layer_sizes=(mlp_hidden,),
        random_state=_C.DEFAULT_RANDOM_SEED,
        early_stopping=True,
        alpha=1.0 / n_samples,  # Adaptive L2 regularization
    )
    mlp_scores = cross_val_score(mlp_clf, X, y, cv=cv, scoring="accuracy")
    nonlinear_acc = float(mlp_scores.mean())

    return linear_acc, nonlinear_acc


def compute_geometry_summary(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    *,
    cv_min_folds: int, cv_max_folds: int, cv_samples_per_fold: int,
    mlp_hidden_min: int, mlp_hidden_max: int,
    geometry_logistic_c_min: float, geometry_logistic_c_max: float,
    geometry_adapt_threshold_numerator: float,
    geometry_adapt_threshold_min: float, geometry_adapt_threshold_max: float,
) -> dict:
    """
    Compute geometry summary including linear vs nonlinear gap analysis.

    Returns:
        Dict with linear_accuracy, nonlinear_accuracy, gap, and diagnosis
    """
    linear_acc, nonlinear_acc = compute_linear_nonlinear_gap(
        pos_activations, neg_activations,
        cv_min_folds=cv_min_folds, cv_max_folds=cv_max_folds,
        cv_samples_per_fold=cv_samples_per_fold,
        mlp_hidden_min=mlp_hidden_min, mlp_hidden_max=mlp_hidden_max,
        geometry_logistic_c_min=geometry_logistic_c_min,
        geometry_logistic_c_max=geometry_logistic_c_max)
    gap = nonlinear_acc - linear_acc
    n_samples = len(pos_activations) + len(neg_activations)
    adaptive_threshold = geometry_adapt_threshold_numerator / np.sqrt(n_samples)
    adaptive_threshold = max(geometry_adapt_threshold_min, min(adaptive_threshold, geometry_adapt_threshold_max))
    diagnosis = "NONLINEAR" if gap > adaptive_threshold else "LINEAR"
    return {
        "linear_accuracy": linear_acc,
        "nonlinear_accuracy": nonlinear_acc,
        "gap": gap,
        "gap_threshold": adaptive_threshold,
        "diagnosis": diagnosis,
    }

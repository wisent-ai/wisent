"""
Core metrics for extraction strategy diagnostics.

Provides functions to compute:
- Pairwise direction consistency
- Linear vs nonlinear classification accuracy
- A/B confound analysis for MC strategies
- Steering direction quality
"""

from typing import Tuple, List, Dict
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def compute_pairwise_consistency(directions: torch.Tensor) -> Tuple[float, float]:
    """
    Compute mean pairwise cosine similarity between directions.

    Args:
        directions: Tensor of shape [n_samples, hidden_dim]

    Returns:
        Tuple of (mean_consistency, std_consistency)
    """
    dirs_norm = directions / (torch.norm(directions, dim=1, keepdim=True) + 1e-8)
    pairwise = (dirs_norm @ dirs_norm.T)
    mask = torch.triu(torch.ones_like(pairwise), diagonal=1).bool()
    vals = pairwise[mask]
    return vals.mean().item(), vals.std().item()


def compute_linear_nonlinear_accuracy(
    pos: torch.Tensor,
    neg: torch.Tensor,
    cv_folds: int = 5
) -> Tuple[float, float]:
    """
    Compute linear and nonlinear classification accuracy.

    Args:
        pos: Positive activations [n_samples, hidden_dim]
        neg: Negative activations [n_samples, hidden_dim]
        cv_folds: Number of cross-validation folds

    Returns:
        Tuple of (linear_accuracy, nonlinear_accuracy)
    """
    X = torch.cat([pos, neg], dim=0).cpu().float().numpy()
    y = np.array([1] * len(pos) + [0] * len(neg))

    linear = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs"))
    linear_scores = cross_val_score(linear, X, y, cv=cv_folds, scoring="accuracy")

    mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), early_stopping=True))
    mlp_scores = cross_val_score(mlp, X, y, cv=cv_folds, scoring="accuracy")

    return linear_scores.mean(), mlp_scores.mean()


def analyze_mc_confound(
    directions: torch.Tensor,
    letter_assignments: List[str]
) -> Dict[str, float]:
    """
    Analyze A/B token confound for MC strategies.

    Args:
        directions: Tensor of (pos - neg) directions [n_samples, hidden_dim]
        letter_assignments: List of "A" or "B" indicating positive response letter

    Returns:
        Dict with ab_variance_fraction, semantic_variance_fraction, semantic_consistency
    """
    # Align directions so all have same A/B orientation
    corrected = directions.clone()
    for i, letter in enumerate(letter_assignments):
        if letter == "A":
            corrected[i] = -corrected[i]

    # Compute mean (B-A) direction
    ab_direction = corrected.mean(dim=0)
    ab_direction = ab_direction / (torch.norm(ab_direction) + 1e-8)

    # Project out A/B component to get semantic-only directions
    semantic_directions = []
    ab_variances = []
    for d in directions:
        ab_component = (d @ ab_direction).item() ** 2
        total_var = (d @ d).item()
        ab_variances.append(ab_component / (total_var + 1e-8))
        d_semantic = d - (d @ ab_direction) * ab_direction
        semantic_directions.append(d_semantic)

    semantic_directions = torch.stack(semantic_directions)
    semantic_mean, _ = compute_pairwise_consistency(semantic_directions)

    return {
        "ab_variance_fraction": np.mean(ab_variances),
        "semantic_variance_fraction": 1 - np.mean(ab_variances),
        "semantic_consistency": semantic_mean,
    }


def compute_steering_quality(directions: torch.Tensor) -> Tuple[float, float]:
    """
    Compute steering direction quality metrics.

    Args:
        directions: Tensor of (pos - neg) directions [n_samples, hidden_dim]

    Returns:
        Tuple of (steering_accuracy, mean_effect_size)
    """
    mean_dir = directions.mean(dim=0)
    mean_dir_norm = mean_dir / (torch.norm(mean_dir) + 1e-8)

    projections = (directions @ mean_dir_norm).numpy()
    accuracy = (projections > 0).mean()
    effect_size = np.abs(projections).mean()

    return accuracy, effect_size

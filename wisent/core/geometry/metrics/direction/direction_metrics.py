"""
Direction-based metrics for analyzing steering vectors.

These metrics analyze the consistency and stability of separation
directions across different subsets of the data.
"""

import torch
import numpy as np
from typing import Dict, Any

from ...analysis.intrinsic_dim import estimate_local_intrinsic_dim


def compute_direction_from_pairs(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> torch.Tensor:
    """Compute mean difference direction (CAA direction)."""
    n = min(len(pos_activations), len(neg_activations))
    diff = pos_activations[:n] - neg_activations[:n]
    return diff.mean(dim=0)


def compute_direction_stability(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_bootstrap: int = 30,
    subset_fraction: float = 0.5,
) -> Dict[str, float]:
    """
    Measure stability of the separation direction across bootstrap samples.

    If the direction is stable (high cosine similarity across subsets),
    then there is likely ONE consistent direction encoding the concept.
    If unstable, different samples use different directions.

    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        n_bootstrap: Number of bootstrap samples
        subset_fraction: Fraction of data to use per bootstrap

    Returns:
        Dict with:
            - mean_cosine: Mean pairwise cosine similarity between bootstrap directions
            - std_cosine: Std of pairwise cosine similarities
            - min_cosine: Minimum pairwise cosine similarity
            - stability_score: 0-1 score (1 = perfectly stable)
    """
    try:
        n_pos = len(pos_activations)
        n_neg = len(neg_activations)
        n_pairs = min(n_pos, n_neg)

        if n_pairs < 10:
            return {
                "mean_cosine": 0.0,
                "std_cosine": 1.0,
                "min_cosine": -1.0,
                "stability_score": 0.0,
            }

        pos_np = pos_activations.float().cpu().numpy()
        neg_np = neg_activations.float().cpu().numpy()

        rng = np.random.RandomState(42)
        subset_size = max(int(n_pairs * subset_fraction), 5)

        directions = []
        for _ in range(n_bootstrap):
            indices = rng.choice(n_pairs, size=subset_size, replace=False)
            pos_subset = pos_np[indices]
            neg_subset = neg_np[indices]

            diff_mean = pos_subset.mean(axis=0) - neg_subset.mean(axis=0)
            norm = np.linalg.norm(diff_mean)
            if norm > 1e-8:
                directions.append(diff_mean / norm)

        if len(directions) < 2:
            return {
                "mean_cosine": 0.0,
                "std_cosine": 1.0,
                "min_cosine": -1.0,
                "stability_score": 0.0,
            }

        directions = np.stack(directions)
        cos_sim_matrix = directions @ directions.T

        n = cos_sim_matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)
        off_diagonal = cos_sim_matrix[mask]

        mean_cosine = float(off_diagonal.mean())
        std_cosine = float(off_diagonal.std())
        min_cosine = float(off_diagonal.min())

        stability_score = max(0, (mean_cosine + 1) / 2 - std_cosine * 0.5)
        stability_score = min(1.0, stability_score)

        return {
            "mean_cosine": mean_cosine,
            "std_cosine": std_cosine,
            "min_cosine": min_cosine,
            "stability_score": stability_score,
        }
    except Exception:
        return {
            "mean_cosine": 0.0,
            "std_cosine": 1.0,
            "min_cosine": -1.0,
            "stability_score": 0.0,
        }


def compute_pairwise_diff_consistency(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> Dict[str, float]:
    """
    Measure consistency of individual difference vectors.

    For each pair, compute diff = pos_i - neg_i.
    Then measure how similar these diffs are to each other.

    High consistency -> all pairs use same direction -> ONE steering vector enough
    Low consistency -> different pairs use different directions -> need MULTIPLE vectors

    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations

    Returns:
        Dict with:
            - mean_pairwise_cosine: Mean cosine sim between diff vectors
            - std_pairwise_cosine: Std of cosine similarities
            - fraction_positive: Fraction of pairs with positive cosine
            - fraction_high_sim: Fraction of pairs with cosine > 0.5
            - consistency_score: 0-1 summary score
    """
    try:
        n_pairs = min(len(pos_activations), len(neg_activations))

        if n_pairs < 3:
            return {
                "mean_pairwise_cosine": 0.0,
                "std_pairwise_cosine": 1.0,
                "fraction_positive": 0.5,
                "fraction_high_sim": 0.0,
                "consistency_score": 0.0,
            }

        diff_vectors = pos_activations[:n_pairs] - neg_activations[:n_pairs]
        diff_np = diff_vectors.float().cpu().numpy()

        norms = np.linalg.norm(diff_np, axis=1, keepdims=True)
        valid_mask = (norms.squeeze() > 1e-8)

        if valid_mask.sum() < 3:
            return {
                "mean_pairwise_cosine": 0.0,
                "std_pairwise_cosine": 1.0,
                "fraction_positive": 0.5,
                "fraction_high_sim": 0.0,
                "consistency_score": 0.0,
            }

        diff_normalized = diff_np[valid_mask] / norms[valid_mask]
        cos_sim_matrix = diff_normalized @ diff_normalized.T

        n = cos_sim_matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)
        off_diagonal = cos_sim_matrix[mask]

        mean_cos = float(off_diagonal.mean())
        std_cos = float(off_diagonal.std())
        fraction_positive = float((off_diagonal > 0).mean())
        fraction_high_sim = float((off_diagonal > 0.5).mean())

        consistency_score = (
            0.4 * max(0, (mean_cos + 1) / 2) +
            0.3 * fraction_positive +
            0.3 * fraction_high_sim
        )
        consistency_score = min(1.0, max(0.0, consistency_score))

        return {
            "mean_pairwise_cosine": mean_cos,
            "std_pairwise_cosine": std_cos,
            "fraction_positive": fraction_positive,
            "fraction_high_sim": fraction_high_sim,
            "consistency_score": consistency_score,
        }
    except Exception:
        return {
            "mean_pairwise_cosine": 0.0,
            "std_pairwise_cosine": 1.0,
            "fraction_positive": 0.5,
            "fraction_high_sim": 0.0,
            "consistency_score": 0.0,
        }

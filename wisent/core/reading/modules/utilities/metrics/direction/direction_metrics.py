"""
Direction-based metrics for analyzing steering vectors.

These metrics analyze the consistency and stability of separation
directions across different subsets of the data.
"""

import torch
import numpy as np
from typing import Dict, Any

from wisent.core.reading.modules.utilities.signal_analysis.intrinsic_dim import estimate_local_intrinsic_dim
from wisent.core.utils.config_tools.constants import (
    NORM_EPS, DEFAULT_RANDOM_SEED,
    CHANCE_LEVEL_ACCURACY,
    SCORE_RANGE_MIN, SCORE_RANGE_MAX, N_COMPONENTS_2D,
)


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
    *,
    n_bootstrap: int,
    subset_fraction: float,
    direction_std_penalty: float,
) -> Dict[str, float]:
    """Measure stability of the separation direction across bootstrap samples."""
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

        rng = np.random.RandomState(DEFAULT_RANDOM_SEED)
        subset_size = max(int(n_pairs * subset_fraction), 5)

        directions = []
        for _ in range(n_bootstrap):
            indices = rng.choice(n_pairs, size=subset_size, replace=False)
            pos_subset = pos_np[indices]
            neg_subset = neg_np[indices]

            diff_mean = pos_subset.mean(axis=0) - neg_subset.mean(axis=0)
            norm = np.linalg.norm(diff_mean)
            if norm > NORM_EPS:
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

        stability_score = max(SCORE_RANGE_MIN, (mean_cosine + SCORE_RANGE_MAX) / N_COMPONENTS_2D - std_cosine * direction_std_penalty)
        stability_score = min(SCORE_RANGE_MAX, stability_score)

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
    *,
    consistency_w_cosine: float,
    consistency_w_positive: float,
    consistency_w_high_sim: float,
    direction_moderate_similarity: float,
) -> Dict[str, float]:
    """Measure consistency of individual difference vectors."""
    try:
        n_pairs = min(len(pos_activations), len(neg_activations))

        if n_pairs < 3:
            return {
                "mean_pairwise_cosine": 0.0,
                "std_pairwise_cosine": 1.0,
                "fraction_positive": CHANCE_LEVEL_ACCURACY,
                "fraction_high_sim": 0.0,
                "consistency_score": 0.0,
            }

        diff_vectors = pos_activations[:n_pairs] - neg_activations[:n_pairs]
        diff_np = diff_vectors.float().cpu().numpy()

        norms = np.linalg.norm(diff_np, axis=1, keepdims=True)
        valid_mask = (norms.squeeze() > NORM_EPS)

        if valid_mask.sum() < 3:
            return {
                "mean_pairwise_cosine": 0.0,
                "std_pairwise_cosine": 1.0,
                "fraction_positive": CHANCE_LEVEL_ACCURACY,
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
        fraction_high_sim = float((off_diagonal > direction_moderate_similarity).mean())

        consistency_score = (
            consistency_w_cosine * max(SCORE_RANGE_MIN, (mean_cos + SCORE_RANGE_MAX) / N_COMPONENTS_2D) +
            consistency_w_positive * fraction_positive +
            consistency_w_high_sim * fraction_high_sim
        )
        consistency_score = min(SCORE_RANGE_MAX, max(SCORE_RANGE_MIN, consistency_score))

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
            "fraction_positive": CHANCE_LEVEL_ACCURACY,
            "fraction_high_sim": 0.0,
            "consistency_score": 0.0,
        }

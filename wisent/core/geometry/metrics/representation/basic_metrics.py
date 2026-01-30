"""
Basic representation metrics for activation geometry.

Metrics for magnitude, sparsity, and per-pair quality.
"""

import torch
import numpy as np
from typing import Dict, Any


def compute_magnitude_metrics(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> Dict[str, Any]:
    """
    Compute magnitude and scale metrics.

    Describes:
    - Actual norms of activations
    - How steering vector norm compares to activation norms
    - Distribution of norms
    """
    pos = pos_activations.float().cpu().numpy()
    neg = neg_activations.float().cpu().numpy()

    n = min(len(pos), len(neg))
    pos, neg = pos[:n], neg[:n]

    # Activation norms
    pos_norms = np.linalg.norm(pos, axis=1)
    neg_norms = np.linalg.norm(neg, axis=1)
    all_norms = np.concatenate([pos_norms, neg_norms])

    # Diff vectors
    diffs = pos - neg
    diff_norms = np.linalg.norm(diffs, axis=1)

    # Mean diff (steering vector)
    mean_diff = diffs.mean(axis=0)
    steering_norm = np.linalg.norm(mean_diff)

    return {
        # Activation norms
        "pos_norm_mean": float(pos_norms.mean()),
        "pos_norm_std": float(pos_norms.std()),
        "neg_norm_mean": float(neg_norms.mean()),
        "neg_norm_std": float(neg_norms.std()),
        "all_norm_mean": float(all_norms.mean()),
        "all_norm_std": float(all_norms.std()),

        # Diff norms
        "diff_norm_mean": float(diff_norms.mean()),
        "diff_norm_std": float(diff_norms.std()),
        "diff_norm_min": float(diff_norms.min()),
        "diff_norm_max": float(diff_norms.max()),

        # Steering vector
        "steering_vector_norm": float(steering_norm),
        "steering_to_activation_ratio": float(steering_norm / (all_norms.mean() + 1e-8)),
        "steering_to_diff_ratio": float(steering_norm / (diff_norms.mean() + 1e-8)),
    }


def compute_sparsity_metrics(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    threshold_fraction: float = 0.01,
) -> Dict[str, Any]:
    """
    Compute sparsity and neuron activation patterns.

    Describes:
    - How sparse are the activations
    - Which neurons are most active
    - How concentrated is the signal
    """
    pos = pos_activations.float().cpu().numpy()
    neg = neg_activations.float().cpu().numpy()

    n = min(len(pos), len(neg))
    pos, neg = pos[:n], neg[:n]
    diffs = pos - neg

    hidden_dim = pos.shape[1]

    # Absolute activation analysis
    pos_abs = np.abs(pos)
    neg_abs = np.abs(neg)
    diff_abs = np.abs(diffs)

    # Threshold for "active" neuron (fraction of max)
    pos_threshold = pos_abs.max() * threshold_fraction
    neg_threshold = neg_abs.max() * threshold_fraction
    diff_threshold = diff_abs.max() * threshold_fraction

    # Sparsity: fraction of neurons below threshold
    pos_sparsity = (pos_abs < pos_threshold).mean()
    neg_sparsity = (neg_abs < neg_threshold).mean()
    diff_sparsity = (diff_abs < diff_threshold).mean()

    # Gini coefficient (measure of inequality/concentration)
    def gini(x):
        x = np.abs(x).flatten()
        x = np.sort(x)
        n = len(x)
        cumsum = np.cumsum(x)
        return (2 * np.sum((np.arange(1, n+1) * x)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1] + 1e-8)

    # Per-neuron contribution to steering direction
    mean_diff = diffs.mean(axis=0)
    neuron_contributions = np.abs(mean_diff)
    sorted_contributions = np.sort(neuron_contributions)[::-1]
    cumsum_contributions = np.cumsum(sorted_contributions) / (sorted_contributions.sum() + 1e-8)

    # How many neurons needed for X% of signal
    neurons_for_50 = int(np.searchsorted(cumsum_contributions, 0.5) + 1)
    neurons_for_90 = int(np.searchsorted(cumsum_contributions, 0.9) + 1)
    neurons_for_99 = int(np.searchsorted(cumsum_contributions, 0.99) + 1)

    # Top neuron indices (most important for steering)
    top_neuron_indices = np.argsort(neuron_contributions)[::-1][:20].tolist()
    top_neuron_contributions = neuron_contributions[top_neuron_indices].tolist()

    return {
        # Sparsity
        "pos_sparsity": float(pos_sparsity),
        "neg_sparsity": float(neg_sparsity),
        "diff_sparsity": float(diff_sparsity),

        # Concentration (Gini)
        "pos_gini": float(gini(pos.mean(axis=0))),
        "neg_gini": float(gini(neg.mean(axis=0))),
        "diff_gini": float(gini(mean_diff)),

        # Neurons needed for signal
        "neurons_for_50pct": neurons_for_50,
        "neurons_for_90pct": neurons_for_90,
        "neurons_for_99pct": neurons_for_99,
        "neurons_for_50pct_fraction": neurons_for_50 / hidden_dim,
        "neurons_for_90pct_fraction": neurons_for_90 / hidden_dim,

        # Top neurons
        "top_20_neuron_indices": top_neuron_indices,
        "top_20_neuron_contributions": top_neuron_contributions,
        "top_10_contribution_fraction": float(sorted_contributions[:10].sum() / (sorted_contributions.sum() + 1e-8)),
    }


def compute_pair_quality_metrics(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> Dict[str, Any]:
    """
    Compute per-pair quality metrics.

    Describes:
    - Which pairs are consistent with the mean direction
    - Which pairs are outliers
    - Distribution of pair qualities
    """
    pos = pos_activations.float().cpu().numpy()
    neg = neg_activations.float().cpu().numpy()

    n = min(len(pos), len(neg))
    pos, neg = pos[:n], neg[:n]
    diffs = pos - neg

    # Mean direction
    mean_diff = diffs.mean(axis=0)
    mean_diff_norm = np.linalg.norm(mean_diff)

    if mean_diff_norm < 1e-8:
        return {"error": "mean_diff_norm too small"}

    mean_diff_normalized = mean_diff / mean_diff_norm

    # Per-pair alignment with mean direction
    diff_norms = np.linalg.norm(diffs, axis=1)
    valid_mask = diff_norms > 1e-8

    alignments = np.zeros(n)
    alignments[valid_mask] = (diffs[valid_mask] / diff_norms[valid_mask, np.newaxis]) @ mean_diff_normalized

    # Identify outliers (pairs that point opposite direction)
    outlier_mask = alignments < 0
    outlier_indices = np.where(outlier_mask)[0].tolist()

    # Identify high-quality pairs (high alignment, reasonable norm)
    high_quality_mask = (alignments > 0.5) & (diff_norms > np.percentile(diff_norms, 25))
    high_quality_indices = np.where(high_quality_mask)[0].tolist()

    # Leave-one-out stability: how much does direction change without each pair
    loo_angles = []
    for i in range(min(n, 100)):  # Limit to 100 for speed
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        loo_mean = diffs[mask].mean(axis=0)
        loo_norm = np.linalg.norm(loo_mean)
        if loo_norm > 1e-8:
            loo_normalized = loo_mean / loo_norm
            angle = np.arccos(np.clip(np.dot(mean_diff_normalized, loo_normalized), -1, 1))
            loo_angles.append(np.degrees(angle))

    loo_angles = np.array(loo_angles)

    return {
        # Alignment distribution
        "alignment_mean": float(alignments.mean()),
        "alignment_std": float(alignments.std()),
        "alignment_min": float(alignments.min()),
        "alignment_max": float(alignments.max()),
        "alignment_median": float(np.median(alignments)),

        # Percentiles
        "alignment_p10": float(np.percentile(alignments, 10)),
        "alignment_p25": float(np.percentile(alignments, 25)),
        "alignment_p75": float(np.percentile(alignments, 75)),
        "alignment_p90": float(np.percentile(alignments, 90)),

        # Outliers
        "n_outliers": int(outlier_mask.sum()),
        "outlier_fraction": float(outlier_mask.mean()),
        "outlier_indices": outlier_indices[:20],  # First 20

        # High quality
        "n_high_quality": int(high_quality_mask.sum()),
        "high_quality_fraction": float(high_quality_mask.mean()),
        "high_quality_indices": high_quality_indices[:20],

        # Leave-one-out stability
        "loo_angle_mean": float(loo_angles.mean()) if len(loo_angles) > 0 else None,
        "loo_angle_std": float(loo_angles.std()) if len(loo_angles) > 0 else None,
        "loo_angle_max": float(loo_angles.max()) if len(loo_angles) > 0 else None,

        # Per-pair alignments (for detailed analysis)
        "per_pair_alignments": alignments.tolist(),
    }

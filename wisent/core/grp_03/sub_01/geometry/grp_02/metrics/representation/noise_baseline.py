"""
Noise baseline comparison for activation geometry.

Compare actual metrics to random noise baselines.
"""

import torch
import numpy as np
from typing import Dict, Any


def compute_noise_baseline_comparison(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_noise_samples: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compare actual metrics to what random noise would produce.

    Computes metrics on random Gaussian activations with same shape/norms
    and reports how actual data differs from noise baseline.

    This helps identify whether there's semantic content or just noise.
    """
    pos = pos_activations.float().cpu().numpy()
    neg = neg_activations.float().cpu().numpy()

    n = min(len(pos), len(neg))
    pos, neg = pos[:n], neg[:n]
    hidden_dim = pos.shape[1]

    # Compute actual metrics
    pos_norms = np.linalg.norm(pos, axis=1)
    neg_norms = np.linalg.norm(neg, axis=1)
    mean_norm = (pos_norms.mean() + neg_norms.mean()) / 2

    diffs = pos - neg
    mean_diff = diffs.mean(axis=0)
    mean_diff_norm = np.linalg.norm(mean_diff)

    # Actual metrics
    diff_norms = np.linalg.norm(diffs, axis=1)
    valid_mask = diff_norms > 1e-8
    if valid_mask.sum() < 2:
        return {"error": "not enough valid diffs"}

    diff_normalized = diffs[valid_mask] / diff_norms[valid_mask, np.newaxis]
    mean_diff_normalized = mean_diff / (mean_diff_norm + 1e-8)

    # Actual alignment
    actual_alignments = diff_normalized @ mean_diff_normalized
    actual_alignment_mean = float(actual_alignments.mean())

    # Actual variance concentration (PC1)
    from sklearn.decomposition import PCA
    n_components = min(50, n - 1, hidden_dim)
    if n_components < 1:
        return {"error": "not enough samples for PCA"}

    pca = PCA(n_components=n_components)
    pca.fit(diffs)
    actual_variance_pc1 = float(pca.explained_variance_ratio_[0])
    actual_cumsum = np.cumsum(pca.explained_variance_ratio_)
    actual_dims_for_90 = int(np.searchsorted(actual_cumsum, 0.9) + 1)

    # Actual linear probe
    from sklearn.linear_model import LogisticRegression
    X = np.vstack([pos, neg])
    y = np.array([1] * n + [0] * n)
    try:
        clf = LogisticRegression( random_state=42)
        clf.fit(X, y)
        actual_linear_probe = float(clf.score(X, y))
    except:
        actual_linear_probe = 0.5

    # Actual steering/activation ratio
    actual_steering_ratio = mean_diff_norm / (mean_norm + 1e-8)

    # Generate noise baselines
    np.random.seed(seed)
    noise_metrics = {
        'alignment_mean': [],
        'variance_pc1': [],
        'dims_for_90': [],
        'linear_probe': [],
        'steering_ratio': [],
    }

    for i in range(n_noise_samples):
        # Random activations with same norm distribution
        noise_pos = np.random.randn(n, hidden_dim)
        noise_neg = np.random.randn(n, hidden_dim)

        # Scale to match actual norms
        noise_pos = noise_pos / np.linalg.norm(noise_pos, axis=1, keepdims=True) * pos_norms[:, np.newaxis]
        noise_neg = noise_neg / np.linalg.norm(noise_neg, axis=1, keepdims=True) * neg_norms[:, np.newaxis]

        # Noise diffs
        noise_diffs = noise_pos - noise_neg
        noise_mean_diff = noise_diffs.mean(axis=0)
        noise_mean_diff_norm = np.linalg.norm(noise_mean_diff)

        noise_diff_norms = np.linalg.norm(noise_diffs, axis=1)
        noise_valid = noise_diff_norms > 1e-8

        if noise_valid.sum() >= 2:
            noise_diff_normalized = noise_diffs[noise_valid] / noise_diff_norms[noise_valid, np.newaxis]
            noise_mean_normalized = noise_mean_diff / (noise_mean_diff_norm + 1e-8)
            noise_alignments = noise_diff_normalized @ noise_mean_normalized
            noise_metrics['alignment_mean'].append(float(noise_alignments.mean()))

        # Noise PCA
        try:
            noise_pca = PCA(n_components=n_components)
            noise_pca.fit(noise_diffs)
            noise_metrics['variance_pc1'].append(float(noise_pca.explained_variance_ratio_[0]))
            noise_cumsum = np.cumsum(noise_pca.explained_variance_ratio_)
            noise_metrics['dims_for_90'].append(int(np.searchsorted(noise_cumsum, 0.9) + 1))
        except:
            pass

        # Noise linear probe
        noise_X = np.vstack([noise_pos, noise_neg])
        try:
            noise_clf = LogisticRegression( random_state=42+i)
            noise_clf.fit(noise_X, y)
            noise_metrics['linear_probe'].append(float(noise_clf.score(noise_X, y)))
        except:
            noise_metrics['linear_probe'].append(0.5)

        # Noise steering ratio
        noise_metrics['steering_ratio'].append(noise_mean_diff_norm / (mean_norm + 1e-8))

    # Compute noise baselines (mean of noise samples)
    noise_baseline = {k: float(np.mean(v)) if v else None for k, v in noise_metrics.items()}
    noise_std = {k: float(np.std(v)) if v else None for k, v in noise_metrics.items()}

    # Compute differences from noise
    alignment_vs_noise = actual_alignment_mean - noise_baseline['alignment_mean'] if noise_baseline['alignment_mean'] else None
    variance_vs_noise = actual_variance_pc1 - noise_baseline['variance_pc1'] if noise_baseline['variance_pc1'] else None
    dims_vs_noise = actual_dims_for_90 - noise_baseline['dims_for_90'] if noise_baseline['dims_for_90'] else None
    linear_vs_noise = actual_linear_probe - noise_baseline['linear_probe'] if noise_baseline['linear_probe'] else None
    steering_vs_noise = actual_steering_ratio - noise_baseline['steering_ratio'] if noise_baseline['steering_ratio'] else None

    return {
        # Actual values
        "actual": {
            "alignment_mean": actual_alignment_mean,
            "variance_pc1": actual_variance_pc1,
            "dims_for_90pct": actual_dims_for_90,
            "linear_probe": actual_linear_probe,
            "steering_ratio": actual_steering_ratio,
        },

        # Noise baseline (what random data looks like)
        "noise_baseline": {
            "alignment_mean": noise_baseline['alignment_mean'],
            "variance_pc1": noise_baseline['variance_pc1'],
            "dims_for_90pct": noise_baseline['dims_for_90'],
            "linear_probe": noise_baseline['linear_probe'],
            "steering_ratio": noise_baseline['steering_ratio'],
        },

        # Standard deviation of noise (for significance)
        "noise_std": {
            "alignment_mean": noise_std['alignment_mean'],
            "variance_pc1": noise_std['variance_pc1'],
            "dims_for_90pct": noise_std['dims_for_90'],
            "linear_probe": noise_std['linear_probe'],
            "steering_ratio": noise_std['steering_ratio'],
        },

        # Difference from noise (positive = more signal than noise)
        "vs_noise": {
            "alignment_mean": alignment_vs_noise,
            "variance_pc1": variance_vs_noise,
            "dims_for_90pct": dims_vs_noise,  # negative is better (more concentrated)
            "linear_probe": linear_vs_noise,
            "steering_ratio": steering_vs_noise,  # negative means pos/neg share structure
        },

        # How many noise stds above baseline (z-score like)
        "stds_above_noise": {
            "alignment_mean": alignment_vs_noise / (noise_std['alignment_mean'] + 1e-8) if noise_std['alignment_mean'] else None,
            "variance_pc1": variance_vs_noise / (noise_std['variance_pc1'] + 1e-8) if noise_std['variance_pc1'] else None,
            "linear_probe": linear_vs_noise / (noise_std['linear_probe'] + 1e-8) if noise_std['linear_probe'] else None,
        },

        # Metadata
        "n_pairs": n,
        "hidden_dim": hidden_dim,
        "n_noise_samples": n_noise_samples,
    }

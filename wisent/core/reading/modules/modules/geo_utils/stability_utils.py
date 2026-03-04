"""Stability analysis and sample size feasibility for representation engineering.

Implements honest reporting about what can and cannot be validated statistically
given the sample size and dimensionality constraints.
"""
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from wisent.core.utils.config_tools.constants import (
    STABILITY_BINARY_VARIANCE, STABILITY_Z_MARGIN, ZERO_THRESHOLD,
)


def direction_stability_bootstrap(
    pos: torch.Tensor, neg: torch.Tensor,
    n_bootstrap: int = None, random_state: int | None = None,
    stability_threshold_high: float = None,
) -> Dict[str, float]:
    """Test if the CAA direction is stable across bootstrap samples.

    Computes difference-of-means direction on bootstrap samples and measures
    how much the direction varies. High cosine similarity = stable direction.
    """
    for _n, _v in [("n_bootstrap", n_bootstrap), ("stability_threshold_high", stability_threshold_high)]:
        if _v is None: raise ValueError(f"{_n} is required")
    if random_state is None:
        from wisent.core.utils.config_tools.constants import DEFAULT_RANDOM_SEED
        random_state = DEFAULT_RANDOM_SEED
    rng = np.random.RandomState(random_state)
    pos_np = pos.cpu().numpy() if isinstance(pos, torch.Tensor) else pos
    neg_np = neg.cpu().numpy() if isinstance(neg, torch.Tensor) else neg

    n = len(pos_np)
    directions = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        pos_boot = pos_np[idx]
        neg_boot = neg_np[idx]
        direction = (pos_boot - neg_boot).mean(axis=0)
        direction = direction / (np.linalg.norm(direction) + ZERO_THRESHOLD)
        directions.append(direction)

    directions = np.array(directions)
    mean_dir = directions.mean(axis=0)
    mean_dir = mean_dir / (np.linalg.norm(mean_dir) + ZERO_THRESHOLD)

    cosines = np.array([np.dot(d, mean_dir) for d in directions])

    return {
        "mean_cosine_similarity": float(np.mean(cosines)),
        "std_cosine_similarity": float(np.std(cosines)),
        "min_cosine_similarity": float(np.min(cosines)),
        "is_stable": float(np.mean(cosines)) > stability_threshold_high,
    }


def cluster_stability_test(
    data: torch.Tensor,
    n_clusters: int,
    n_subsamples: int = None,
    subsample_ratio: float = None,
    random_state: int | None = None,
    stability_threshold_med: float = None,
    linearity_n_init: int = None,
) -> Dict[str, float]:
    """Test cluster stability by subsampling and measuring consistency.

    Subsamples the data, reclusters, and measures how consistent the
    cluster assignments are across subsamples using Adjusted Rand Index.
    """
    for _n, _v in [("n_subsamples", n_subsamples), ("subsample_ratio", subsample_ratio), ("stability_threshold_med", stability_threshold_med), ("linearity_n_init", linearity_n_init)]:
        if _v is None: raise ValueError(f"{_n} is required")
    if random_state is None:
        from wisent.core.utils.config_tools.constants import DEFAULT_RANDOM_SEED
        random_state = DEFAULT_RANDOM_SEED
    rng = np.random.RandomState(random_state)
    data_np = data.cpu().numpy() if isinstance(data, torch.Tensor) else data
    n = len(data_np)
    subsample_size = int(n * subsample_ratio)

    base_kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=linearity_n_init)
    base_labels = base_kmeans.fit_predict(data_np)

    ari_scores = []
    for i in range(n_subsamples):
        idx = rng.choice(n, subsample_size, replace=False)
        subsample = data_np[idx]

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state + i, n_init=linearity_n_init)
        subsample_labels = kmeans.fit_predict(subsample)

        ari = adjusted_rand_score(base_labels[idx], subsample_labels)
        ari_scores.append(ari)

    return {
        "mean_ari": float(np.mean(ari_scores)),
        "std_ari": float(np.std(ari_scores)),
        "min_ari": float(np.min(ari_scores)),
        "is_stable": float(np.mean(ari_scores)) > stability_threshold_med,
    }


def sample_size_feasibility(
    n_samples: int, n_dims: int,
    stability_accuracy_margin: float = None,
    stability_min_samples_large: int = None,
    stability_min_samples_med: int = None,
    stability_min_samples_small: int = None,
    high_dim_threshold: int = None,
) -> Dict[str, Any]:
    """Report estimation bounds and what's testable given sample size and dimensions.

    Instead of "feasible/infeasible", we report:
    - Estimation error bounds for each test type
    - What can be bounded vs what has high uncertainty
    """
    for _n, _v in [("stability_accuracy_margin", stability_accuracy_margin), ("stability_min_samples_large", stability_min_samples_large), ("stability_min_samples_med", stability_min_samples_med), ("stability_min_samples_small", stability_min_samples_small), ("high_dim_threshold", high_dim_threshold)]:
        if _v is None: raise ValueError(f"{_n} is required")
    gmm_2_needed = 2 * n_dims * n_dims

    accuracy_se = np.sqrt(STABILITY_BINARY_VARIANCE * STABILITY_BINARY_VARIANCE / n_samples)
    accuracy_margin = STABILITY_Z_MARGIN * accuracy_se

    linear_probe_effective_n = n_samples / max(1, n_dims / 100)
    linear_probe_reliable = linear_probe_effective_n > 10

    clustering_se = 1.0 / np.sqrt(n_samples) if n_samples > 0 else 1.0

    return {
        "n_samples": n_samples,
        "n_dims": n_dims,
        "accuracy_margin_of_error": accuracy_margin,
        "accuracy_bounds_reliable": accuracy_margin < stability_accuracy_margin,
        "linear_probe_effective_n": linear_probe_effective_n,
        "linear_probe_reliable": linear_probe_reliable,
        "clustering_uncertainty": clustering_se,
        "gmm_estimation_error": _gmm_estimation_error(n_samples, n_dims, k=2),
        "what_can_bound": _what_can_bound(n_samples, n_dims, stability_min_samples_med=stability_min_samples_med, stability_min_samples_large=stability_min_samples_large, stability_min_samples_small=stability_min_samples_small),
        "what_has_high_uncertainty": _what_has_high_uncertainty(n_samples, n_dims, high_dim_threshold=high_dim_threshold),
    }


def _gmm_estimation_error(n_samples: int, n_dims: int, k: int) -> Dict[str, float]:
    """Compute GMM parameter estimation error bounds."""
    n_params = k * (n_dims + n_dims * (n_dims + 1) / 2 + 1)
    effective_n_per_param = n_samples / n_params if n_params > 0 else 0

    return {
        "n_parameters": int(n_params),
        "samples_per_parameter": effective_n_per_param,
        "estimation_reliable": effective_n_per_param > 10,
        "relative_error_bound": 1.0 / np.sqrt(max(1, effective_n_per_param)),
    }


def _what_can_bound(n_samples: int, n_dims: int, stability_min_samples_med: int = None, stability_min_samples_large: int = None, stability_min_samples_small: int = None) -> List[str]:
    """List what estimation can be bounded with current sample size."""
    for _n, _v in [("stability_min_samples_med", stability_min_samples_med), ("stability_min_samples_large", stability_min_samples_large), ("stability_min_samples_small", stability_min_samples_small)]:
        if _v is None: raise ValueError(f"{_n} is required")
    bounded = []
    if n_samples >= stability_min_samples_med:
        bounded.append("Classification accuracy (±margin of error)")
    if n_samples >= stability_min_samples_large:
        bounded.append("Effect size (Cohen's d with CI)")
    if n_samples >= stability_min_samples_small:
        bounded.append("Direction stability (cosine similarity)")
    if n_samples >= stability_min_samples_med:
        bounded.append("K-means cluster assignments")
    return bounded


def _what_has_high_uncertainty(n_samples: int, n_dims: int, high_dim_threshold: int) -> List[str]:
    """List what has high estimation uncertainty."""
    uncertain = []
    gmm_needed = 2 * n_dims * n_dims
    if n_samples < gmm_needed / 100:
        uncertain.append(f"GMM covariance estimation (error bound: >{100*n_dims/n_samples:.0f}%)")
    if n_dims > high_dim_threshold and n_samples < high_dim_threshold:
        uncertain.append("Cluster existence in original space (dimension reduction may create artifacts)")
    if n_samples < n_dims / 10:
        uncertain.append(f"Linear probe coefficients (underdetermined by {n_dims/n_samples/10:.1f}x)")
    return uncertain


def compute_full_stability_report(
    pos: torch.Tensor,
    neg: torch.Tensor,
    n_clusters: int = None,
    random_state: int | None = None,
    stability_min_samples_med: int = None,
    n_bootstrap: int = None,
    stability_threshold_high: float = None,
    n_subsamples: int = None,
    subsample_ratio: float = None,
    stability_threshold_med: float = None,
    linearity_n_init: int = None,
    stability_accuracy_margin: float = None,
    stability_min_samples_large: int = None,
    stability_min_samples_small: int = None,
    high_dim_threshold: int = None,
) -> Dict[str, Any]:
    """Compute complete stability and feasibility report.

    Runs direction stability, cluster stability (if applicable), and
    generates sample size feasibility warnings.
    """
    for _n, _v in [("n_clusters", n_clusters), ("stability_min_samples_med", stability_min_samples_med), ("n_bootstrap", n_bootstrap), ("stability_threshold_high", stability_threshold_high)]:
        if _v is None: raise ValueError(f"{_n} is required")
    pos_np = pos.cpu().numpy() if isinstance(pos, torch.Tensor) else pos
    n_samples, n_dims = pos_np.shape

    feasibility = sample_size_feasibility(n_samples, n_dims, stability_accuracy_margin=stability_accuracy_margin, stability_min_samples_large=stability_min_samples_large, stability_min_samples_med=stability_min_samples_med, stability_min_samples_small=stability_min_samples_small, high_dim_threshold=high_dim_threshold)
    direction_stability = direction_stability_bootstrap(pos, neg, n_bootstrap=n_bootstrap, random_state=random_state, stability_threshold_high=stability_threshold_high)

    diff = pos_np - (neg.cpu().numpy() if isinstance(neg, torch.Tensor) else neg)

    cluster_stability = None
    if n_clusters > 1 and n_samples >= stability_min_samples_med:
        cluster_stability = cluster_stability_test(
            torch.tensor(diff), n_clusters, random_state=random_state, n_subsamples=n_subsamples, subsample_ratio=subsample_ratio, stability_threshold_med=stability_threshold_med, linearity_n_init=linearity_n_init,
        )

    return {
        "feasibility": feasibility,
        "direction_stability": direction_stability,
        "cluster_stability": cluster_stability,
        "overall_reliable": (
            direction_stability["is_stable"]
            and (cluster_stability is None or cluster_stability["is_stable"])
            and len(feasibility["warnings"]) == 0
        ),
    }

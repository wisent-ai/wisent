"""Stability analysis and sample size feasibility for representation engineering.

Implements honest reporting about what can and cannot be validated statistically
given the sample size and dimensionality constraints.
"""
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def direction_stability_bootstrap(
    pos: torch.Tensor,
    neg: torch.Tensor,
    n_bootstrap: int = 50,
    random_state: int = 42,
) -> Dict[str, float]:
    """Test if the CAA direction is stable across bootstrap samples.

    Computes difference-of-means direction on bootstrap samples and measures
    how much the direction varies. High cosine similarity = stable direction.

    Returns:
        Dictionary with mean_cosine_similarity, std, and stability assessment.
    """
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
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        directions.append(direction)

    directions = np.array(directions)
    mean_dir = directions.mean(axis=0)
    mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-10)

    cosines = np.array([np.dot(d, mean_dir) for d in directions])

    return {
        "mean_cosine_similarity": float(np.mean(cosines)),
        "std_cosine_similarity": float(np.std(cosines)),
        "min_cosine_similarity": float(np.min(cosines)),
        "is_stable": float(np.mean(cosines)) > 0.9,
    }


def cluster_stability_test(
    data: torch.Tensor,
    n_clusters: int,
    n_subsamples: int = 20,
    subsample_ratio: float = 0.8,
    random_state: int = 42,
) -> Dict[str, float]:
    """Test cluster stability by subsampling and measuring consistency.

    Subsamples the data, reclusters, and measures how consistent the
    cluster assignments are across subsamples using Adjusted Rand Index.

    Returns:
        Dictionary with mean ARI, std, and stability assessment.
    """
    rng = np.random.RandomState(random_state)
    data_np = data.cpu().numpy() if isinstance(data, torch.Tensor) else data
    n = len(data_np)
    subsample_size = int(n * subsample_ratio)

    base_kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    base_labels = base_kmeans.fit_predict(data_np)

    ari_scores = []
    for i in range(n_subsamples):
        idx = rng.choice(n, subsample_size, replace=False)
        subsample = data_np[idx]

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state + i, n_init=5)
        subsample_labels = kmeans.fit_predict(subsample)

        ari = adjusted_rand_score(base_labels[idx], subsample_labels)
        ari_scores.append(ari)

    return {
        "mean_ari": float(np.mean(ari_scores)),
        "std_ari": float(np.std(ari_scores)),
        "min_ari": float(np.min(ari_scores)),
        "is_stable": float(np.mean(ari_scores)) > 0.7,
    }


def sample_size_feasibility(n_samples: int, n_dims: int) -> Dict[str, Any]:
    """Report estimation bounds and what's testable given sample size and dimensions.

    Instead of "feasible/infeasible", we report:
    - Estimation error bounds for each test type
    - What can be bounded vs what has high uncertainty
    """
    gmm_2_needed = 2 * n_dims * n_dims

    accuracy_se = np.sqrt(0.5 * 0.5 / n_samples)
    accuracy_margin = 1.96 * accuracy_se

    linear_probe_effective_n = n_samples / max(1, n_dims / 100)
    linear_probe_reliable = linear_probe_effective_n > 10

    clustering_se = 1.0 / np.sqrt(n_samples) if n_samples > 0 else 1.0

    return {
        "n_samples": n_samples,
        "n_dims": n_dims,
        "accuracy_margin_of_error": accuracy_margin,
        "accuracy_bounds_reliable": accuracy_margin < 0.1,
        "linear_probe_effective_n": linear_probe_effective_n,
        "linear_probe_reliable": linear_probe_reliable,
        "clustering_uncertainty": clustering_se,
        "gmm_estimation_error": _gmm_estimation_error(n_samples, n_dims, k=2),
        "what_can_bound": _what_can_bound(n_samples, n_dims),
        "what_has_high_uncertainty": _what_has_high_uncertainty(n_samples, n_dims),
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


def _what_can_bound(n_samples: int, n_dims: int) -> List[str]:
    """List what estimation can be bounded with current sample size."""
    bounded = []
    if n_samples >= 50:
        bounded.append("Classification accuracy (±margin of error)")
    if n_samples >= 100:
        bounded.append("Effect size (Cohen's d with CI)")
    if n_samples >= 30:
        bounded.append("Direction stability (cosine similarity)")
    if n_samples >= 50:
        bounded.append("K-means cluster assignments")
    return bounded


def _what_has_high_uncertainty(n_samples: int, n_dims: int) -> List[str]:
    """List what has high estimation uncertainty."""
    uncertain = []
    gmm_needed = 2 * n_dims * n_dims
    if n_samples < gmm_needed / 100:
        uncertain.append(f"GMM covariance estimation (error bound: >{100*n_dims/n_samples:.0f}%)")
    if n_dims > 1000 and n_samples < 1000:
        uncertain.append("Cluster existence in original space (dimension reduction may create artifacts)")
    if n_samples < n_dims / 10:
        uncertain.append(f"Linear probe coefficients (underdetermined by {n_dims/n_samples/10:.1f}x)")
    return uncertain


def compute_full_stability_report(
    pos: torch.Tensor,
    neg: torch.Tensor,
    n_clusters: int = 2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Compute complete stability and feasibility report.

    Runs direction stability, cluster stability (if applicable), and
    generates sample size feasibility warnings.
    """
    pos_np = pos.cpu().numpy() if isinstance(pos, torch.Tensor) else pos
    n_samples, n_dims = pos_np.shape

    feasibility = sample_size_feasibility(n_samples, n_dims)
    direction_stability = direction_stability_bootstrap(pos, neg, random_state=random_state)

    diff = pos_np - (neg.cpu().numpy() if isinstance(neg, torch.Tensor) else neg)

    cluster_stability = None
    if n_clusters > 1 and n_samples >= 50:
        cluster_stability = cluster_stability_test(
            torch.tensor(diff), n_clusters, random_state=random_state
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

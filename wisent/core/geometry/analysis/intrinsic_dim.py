"""
Intrinsic dimensionality estimation for activation spaces.

These metrics estimate the effective dimensionality of the representation
space, which indicates how complex the learned structure is.
"""

import torch
import numpy as np
from typing import Tuple


def estimate_local_intrinsic_dim(X: np.ndarray, k: int = 10) -> float:
    """
    Estimate local intrinsic dimensionality using MLE method.
    Based on Levina & Bickel (2004).
    
    Args:
        X: [N, D] data matrix
        k: Number of neighbors for estimation
        
    Returns:
        Estimated intrinsic dimension
    """
    from scipy.spatial.distance import cdist
    
    if len(X) < k + 1:
        return float(X.shape[1])
    
    dists = cdist(X, X, 'euclidean')
    np.fill_diagonal(dists, np.inf)
    
    sorted_dists = np.sort(dists, axis=1)[:, :k]
    
    dims = []
    for i in range(len(X)):
        T_k = sorted_dists[i, k-1]
        if T_k < 1e-10:
            continue
        log_ratios = np.log(sorted_dists[i, :k-1] / T_k + 1e-10)
        if len(log_ratios) > 0 and log_ratios.sum() < 0:
            dim_est = -(k - 1) / log_ratios.sum()
            dims.append(min(dim_est, X.shape[1]))
    
    return float(np.median(dims)) if dims else float(X.shape[1])


def compute_local_intrinsic_dims(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    k: int = 10,
) -> Tuple[float, float, float]:
    """
    Compute local intrinsic dimension for pos and neg separately.
    
    Different local dimensions suggest different geometric structures.
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        k: Number of neighbors
        
    Returns:
        (local_dim_pos, local_dim_neg, ratio)
    """
    try:
        pos = pos_activations.float().cpu().numpy()
        neg = neg_activations.float().cpu().numpy()
        
        dim_pos = estimate_local_intrinsic_dim(pos, k)
        dim_neg = estimate_local_intrinsic_dim(neg, k)
        ratio = dim_pos / (dim_neg + 1e-10)
        
        return dim_pos, dim_neg, ratio
    except Exception:
        return 0.0, 0.0, 1.0


def compute_diff_intrinsic_dim(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    k: int = 10,
) -> float:
    """
    Estimate intrinsic dimensionality of difference vectors.
    
    Low dimension suggests a simple linear concept (CAA-friendly).
    High dimension suggests complex multi-directional structure.
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        k: Number of neighbors for estimation
        
    Returns:
        Estimated intrinsic dimension of diff vectors
    """
    try:
        n_pairs = min(len(pos_activations), len(neg_activations))
        if n_pairs < k + 1:
            return 0.0
        
        diff_vectors = (pos_activations[:n_pairs] - neg_activations[:n_pairs]).float().cpu().numpy()
        
        return estimate_local_intrinsic_dim(diff_vectors, k)
    except Exception:
        return 0.0


def participation_ratio(X: np.ndarray) -> float:
    """
    Participation ratio: (Σλᵢ)² / Σλᵢ²

    Measures effective dimensionality. If all eigenvalues equal: PR = n.
    If one eigenvalue dominates: PR ≈ 1.

    Interpretation:
    - PR ≈ 1: single dominant direction
    - PR ≈ 5: ~5 significant directions
    - PR ≈ d: uniform spread (no low-d structure)
    """
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0)

    sum_eig = eigenvalues.sum()
    sum_eig_sq = (eigenvalues ** 2).sum()

    if sum_eig_sq < 1e-10:
        return 1.0

    pr = (sum_eig ** 2) / sum_eig_sq
    return float(pr)


def effective_rank(X: np.ndarray) -> float:
    """
    Effective rank via entropy of normalized singular values.

    eff_rank = exp(-Σ pᵢ log pᵢ) where pᵢ = σᵢ/Σσ

    More robust than participation ratio to outliers.
    """
    U, s, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    s = s[s > 1e-10]

    if len(s) == 0:
        return 1.0

    p = s / s.sum()
    entropy = -np.sum(p * np.log(p + 1e-10))

    return float(np.exp(entropy))


def stable_rank(X: np.ndarray) -> float:
    """
    Stable rank: ||X||_F² / ||X||_2²

    Nuclear norm squared / spectral norm squared.
    More numerically stable than effective rank.
    """
    X_centered = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

    if len(s) == 0 or s[0] < 1e-10:
        return 1.0

    frobenius_sq = (s ** 2).sum()
    spectral_sq = s[0] ** 2

    return float(frobenius_sq / spectral_sq)


def pca_variance_dimensions(X: np.ndarray, thresholds: list = None) -> dict:
    """
    How many PCs needed to explain X% variance?

    Returns dict: {90: n_90, 95: n_95, 99: n_99, eigenvalues: [...]}
    """
    if thresholds is None:
        thresholds = [0.90, 0.95, 0.99]

    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)

    total_var = eigenvalues.sum()
    if total_var < 1e-10:
        return {t: 1 for t in thresholds}

    cumvar = np.cumsum(eigenvalues) / total_var

    result = {}
    for t in thresholds:
        n_dims = int(np.searchsorted(cumvar, t) + 1)
        result[f"dims_{int(t*100)}pct"] = min(n_dims, len(eigenvalues))

    result["top_10_eigenvalues"] = eigenvalues[:10].tolist()
    result["total_variance"] = float(total_var)

    return result


def two_nn_dimension(X: np.ndarray) -> float:
    """
    Two-NN intrinsic dimension estimator (Facco et al. 2017).

    Uses ratio of distances to 2nd and 1st nearest neighbors.
    More robust than MLE for small samples and high dimensions.
    """
    from scipy.spatial.distance import cdist

    if len(X) < 3:
        return float(X.shape[1])

    dists = cdist(X, X, 'euclidean')
    np.fill_diagonal(dists, np.inf)

    sorted_dists = np.sort(dists, axis=1)
    r1 = sorted_dists[:, 0]
    r2 = sorted_dists[:, 1]

    valid = (r1 > 1e-10) & (r2 > 1e-10)
    if valid.sum() < 2:
        return float(X.shape[1])

    mu = r2[valid] / r1[valid]
    mu = mu[mu > 1.0]

    if len(mu) < 2:
        return float(X.shape[1])

    log_mu = np.log(mu)
    d_estimate = len(mu) / log_mu.sum()

    return float(min(d_estimate, X.shape[1]))


def compute_effective_dimensions(
    pos: torch.Tensor,
    neg: torch.Tensor,
) -> dict:
    """
    Comprehensive effective dimensionality analysis.

    Returns multiple estimates to triangulate the true effective dimension:
    - participation_ratio: (Σλ)²/Σλ²
    - effective_rank: exp(entropy of singular values)
    - stable_rank: Frobenius²/spectral²
    - two_nn: ratio-based estimator
    - mle: Levina-Bickel MLE
    - pca_dims: PCs for 90/95/99% variance

    Interpretation:
    - All estimates ≈ 1-5: Few dominant directions (CAA likely works)
    - All estimates ≈ 10-50: Moderate structure (PRISM might help)
    - All estimates > 100: High-dimensional (steering may not work)
    """
    pos_np = pos.cpu().numpy() if isinstance(pos, torch.Tensor) else pos
    neg_np = neg.cpu().numpy() if isinstance(neg, torch.Tensor) else neg

    diff = pos_np - neg_np

    pr = participation_ratio(diff)
    er = effective_rank(diff)
    sr = stable_rank(diff)
    tnn = two_nn_dimension(diff)
    mle = estimate_local_intrinsic_dim(diff, k=min(10, len(diff)-1))
    pca = pca_variance_dimensions(diff)

    estimates = [pr, er, sr, tnn, mle]
    median_estimate = float(np.median(estimates))
    std_estimate = float(np.std(estimates))

    ambient_dim = diff.shape[1]
    compression_ratio = ambient_dim / median_estimate if median_estimate > 0 else float('inf')

    return {
        "participation_ratio": pr,
        "effective_rank": er,
        "stable_rank": sr,
        "two_nn": tnn,
        "mle": mle,
        "median_estimate": median_estimate,
        "std_across_methods": std_estimate,
        "estimator_agreement": std_estimate / median_estimate if median_estimate > 0 else float('inf'),
        "pca_variance": pca,
        "n_samples": len(diff),
        "ambient_dim": ambient_dim,
        "compression_ratio": compression_ratio,
    }

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

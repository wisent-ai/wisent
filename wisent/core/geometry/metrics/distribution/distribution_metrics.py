"""
Distribution-based metrics for comparing activation populations.

These metrics measure statistical differences between positive and
negative activation distributions without assuming linear separability.
"""

import torch
import numpy as np
from typing import Dict


def compute_mmd_rbf(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> float:
    """
    Compute Maximum Mean Discrepancy with RBF kernel.
    
    Measures distribution difference without assuming linearity.
    Higher values indicate more separable distributions.
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        
    Returns:
        MMD value (0 = identical distributions)
    """
    try:
        from sklearn.metrics.pairwise import rbf_kernel
        from scipy.spatial.distance import cdist
        
        pos = pos_activations.float().cpu().numpy()
        neg = neg_activations.float().cpu().numpy()
        
        all_data = np.vstack([pos, neg])
        dists = cdist(all_data, all_data, 'euclidean')
        gamma = 1.0 / (2 * np.median(dists[dists > 0]) ** 2 + 1e-10)
        
        K_pp = rbf_kernel(pos, pos, gamma=gamma)
        K_nn = rbf_kernel(neg, neg, gamma=gamma)
        K_pn = rbf_kernel(pos, neg, gamma=gamma)
        
        m = len(pos)
        n = len(neg)
        
        mmd = (K_pp.sum() / (m * m) + 
               K_nn.sum() / (n * n) - 
               2 * K_pn.sum() / (m * n))
        
        return float(max(0, mmd))
    except Exception:
        return 0.0


def compute_density_ratio(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> float:
    """
    Compute ratio of average intra-class distances.
    
    Values far from 1 suggest different local geometries.
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        
    Returns:
        Density ratio (pos avg dist / neg avg dist)
    """
    try:
        from scipy.spatial.distance import cdist
        
        pos = pos_activations.float().cpu().numpy()
        neg = neg_activations.float().cpu().numpy()
        
        if len(pos) < 2 or len(neg) < 2:
            return 1.0
        
        pos_dists = cdist(pos, pos, 'euclidean')
        neg_dists = cdist(neg, neg, 'euclidean')
        
        np.fill_diagonal(pos_dists, np.nan)
        np.fill_diagonal(neg_dists, np.nan)
        
        avg_pos = np.nanmean(pos_dists)
        avg_neg = np.nanmean(neg_dists)
        
        if avg_neg < 1e-10:
            return 1.0
        
        return float(avg_pos / avg_neg)
    except Exception:
        return 1.0


def compute_fisher_per_dimension(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute Fisher ratio for each dimension and summary stats.
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        
    Returns:
        Dict with fisher_max, fisher_gini, fisher_top10_ratio, num_dims_above_1
    """
    try:
        pos = pos_activations.float().cpu().numpy()
        neg = neg_activations.float().cpu().numpy()
        
        n_dims = pos.shape[1]
        fishers = np.zeros(n_dims)
        
        for d in range(n_dims):
            pos_d = pos[:, d]
            neg_d = neg[:, d]
            
            mean_pos = pos_d.mean()
            mean_neg = neg_d.mean()
            var_pos = pos_d.var()
            var_neg = neg_d.var()
            
            between_var = (mean_pos - mean_neg) ** 2
            within_var = (var_pos + var_neg) / 2
            
            if within_var > 1e-10:
                fishers[d] = between_var / within_var
        
        fisher_max = float(fishers.max())
        
        values = np.abs(fishers)
        if values.sum() > 1e-10:
            values = np.sort(values)
            n = len(values)
            fisher_gini = (2 * np.sum((np.arange(1, n+1) * values)) / (n * values.sum())) - (n + 1) / n
        else:
            fisher_gini = 0.0
        
        sorted_fishers = np.sort(fishers)[::-1]
        top10_sum = sorted_fishers[:10].sum()
        total_sum = fishers.sum() + 1e-10
        fisher_top10_ratio = float(top10_sum / total_sum)
        
        num_dims_above_1 = int((fishers > 1.0).sum())
        
        return {
            "fisher_max": fisher_max,
            "fisher_gini": float(fisher_gini),
            "fisher_top10_ratio": fisher_top10_ratio,
            "num_dims_fisher_above_1": num_dims_above_1,
        }
    except Exception:
        return {
            "fisher_max": 0.0,
            "fisher_gini": 0.0,
            "fisher_top10_ratio": 0.0,
            "num_dims_fisher_above_1": 0,
        }

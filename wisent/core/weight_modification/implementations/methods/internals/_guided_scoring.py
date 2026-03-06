"""Scoring and weighting helpers for guided weight modification."""

from __future__ import annotations

import torch
from typing import Dict, List, TYPE_CHECKING

from wisent.core.utils.cli.cli_logger import setup_logger

if TYPE_CHECKING:
    from torch import Tensor

from wisent.core.utils.config_tools.constants import (
    NORM_EPS,
)
from wisent.core.weight_modification.methods.guided import (
    LayerDiagnostics,
    GuidedModificationConfig,
)

_LOG = setup_logger(__name__)


def _compute_knn_accuracy(
    pos_tensor: Tensor,
    neg_tensor: Tensor,
    k: int,
    *,
    blend_default: float,
) -> float:
    """Compute k-NN leave-one-out accuracy."""
    
    X = torch.cat([pos_tensor, neg_tensor], dim=0)
    y = torch.cat([
        torch.ones(pos_tensor.shape[0]),
        torch.zeros(neg_tensor.shape[0])
    ])
    
    n = X.shape[0]
    k = min(k, n - 1)
    
    if k < 1:
        return blend_default

    # Compute pairwise distances
    distances = torch.cdist(X, X)
    
    correct = 0
    for i in range(n):
        # Get k nearest neighbors (excluding self)
        dists = distances[i].clone()
        dists[i] = float('inf')
        _, indices = torch.topk(dists, k, largest=False)
        
        # Majority vote
        neighbor_labels = y[indices]
        predicted = (neighbor_labels.sum() > k / 2).float()
        
        if predicted == y[i]:
            correct += 1
    
    return correct / n


def _compute_fisher_ratio(
    pos_tensor: Tensor,
    neg_tensor: Tensor,
) -> float:
    """Compute Fisher discriminant ratio."""
    
    mu_pos = pos_tensor.mean(dim=0)
    mu_neg = neg_tensor.mean(dim=0)
    
    # Between-class scatter
    mu_diff = mu_pos - mu_neg
    between_scatter = (mu_diff ** 2).sum()
    
    # Within-class scatter
    pos_centered = pos_tensor - mu_pos
    neg_centered = neg_tensor - mu_neg
    
    within_scatter = (pos_centered ** 2).sum() + (neg_centered ** 2).sum()
    within_scatter = within_scatter / (pos_tensor.shape[0] + neg_tensor.shape[0])
    
    # Fisher ratio
    fisher_ratio = between_scatter / (within_scatter + NORM_EPS)
    
    return float(fisher_ratio)


def _compute_recommended_weight(
    linear_score: float,
    knn_score: float,
    fisher_ratio: float,
    cohens_d: float,
    *,
    blend_default: float,
    **_kwargs,
) -> float:
    """Return linear_score directly as recommended weight.

    The optimizer should find the right weight empirically, not through
    an unvalidated formula combining Fisher/cohens_d/nonlinear adjustments.
    """
    return linear_score


def compute_fisher_weights(
    diagnostics: Dict[int, LayerDiagnostics],
    config: GuidedModificationConfig,
    *,
    blend_default: float,
) -> Dict[int, float]:
    """
    Compute layer weights based on Fisher ratios.
    
    This is a key innovation: instead of using a parametric kernel
    (like Heretic), we use the actual measured Fisher ratios to
    determine ablation strength per layer.
    
    Higher Fisher ratio = better linear separability = safer to ablate strongly
    """
    if not diagnostics:
        return {}
    
    # Get Fisher ratios
    fisher_ratios = {l: d.fisher_ratio for l, d in diagnostics.items()}
    
    # Normalize to reasonable range
    max_fisher = max(fisher_ratios.values())
    min_fisher = min(fisher_ratios.values())
    
    weights = {}
    for layer, fisher in fisher_ratios.items():
        if max_fisher > min_fisher:
            # Normalize to [0, 1]
            normalized = (fisher - min_fisher) / (max_fisher - min_fisher)
        else:
            normalized = blend_default
        
        # Scale to weight range
        weight = (
            config.fisher_weight_min + 
            normalized * (config.fisher_weight_max - config.fisher_weight_min)
        )
        
        # Apply global scale
        weight *= config.fisher_weight_scale * config.base_strength
        
        weights[layer] = weight
    
    return weights

"""Scoring and weighting helpers for guided weight modification."""

from __future__ import annotations

import torch
from typing import Dict, List, TYPE_CHECKING

from wisent.core.cli.cli_logger import setup_logger

if TYPE_CHECKING:
    from torch import Tensor

from wisent.core.constants import (
    NORM_EPS, PROBE_KNN_K, BLEND_DEFAULT,
    GUIDED_FISHER_MAX_BOOST, GUIDED_FISHER_BOOST_SCALE, GUIDED_FISHER_LOG_DIVISOR,
    GUIDED_EFFECT_MAX_BOOST, GUIDED_EFFECT_BOOST_SCALE, GUIDED_COHENS_D_NORMALIZER,
    GUIDED_NONLINEAR_PENALTY, GUIDED_WEIGHT_MAX,
)
from wisent.core.weight_modification.methods.guided import (
    LayerDiagnostics,
    GuidedModificationConfig,
)

_LOG = setup_logger(__name__)


def _compute_knn_accuracy(
    pos_tensor: Tensor,
    neg_tensor: Tensor,
    k: int = PROBE_KNN_K,
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
        return BLEND_DEFAULT
    
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
) -> float:
    """
    Compute recommended ablation weight based on diagnostics.
    
    The weight is higher when:
    - Linear score is high (direction captures the concept well)
    - Fisher ratio is high (strong linear separability)
    - Cohen's d is high (large effect size)
    
    The weight is moderated when:
    - k-NN >> linear (nonlinear structure exists)
    """
    
    # Base weight from linear score
    base_weight = linear_score
    
    # Boost for high Fisher ratio (log scale since Fisher can be very large)
    fisher_boost = min(GUIDED_FISHER_MAX_BOOST, GUIDED_FISHER_BOOST_SCALE * (1 + torch.log(torch.tensor(fisher_ratio + 1)).item() / GUIDED_FISHER_LOG_DIVISOR))

    # Boost for high effect size
    effect_boost = min(GUIDED_EFFECT_MAX_BOOST, GUIDED_EFFECT_BOOST_SCALE * min(cohens_d / GUIDED_COHENS_D_NORMALIZER, 1.0))

    # Penalty if k-NN is much better than linear (nonlinear structure)
    gap = knn_score - linear_score
    nonlinear_penalty = max(0, gap * GUIDED_NONLINEAR_PENALTY)

    # Combine
    weight = base_weight + fisher_boost + effect_boost - nonlinear_penalty

    # Clamp to reasonable range
    weight = max(0.0, min(GUIDED_WEIGHT_MAX, weight))
    
    return weight


def compute_fisher_weights(
    diagnostics: Dict[int, LayerDiagnostics],
    config: GuidedModificationConfig,
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
            normalized = BLEND_DEFAULT
        
        # Scale to weight range
        weight = (
            config.fisher_weight_min + 
            normalized * (config.fisher_weight_max - config.fisher_weight_min)
        )
        
        # Apply global scale
        weight *= config.fisher_weight_scale * config.base_strength
        
        weights[layer] = weight
    
    return weights

"""Layer diagnostics computation for guided weight modification."""

from __future__ import annotations

import torch
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from wisent.core.utils.cli.cli_logger import setup_logger

if TYPE_CHECKING:
    from torch import Tensor
    from wisent.core.primitives.models.wisent_model import WisentModel
    from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair

from wisent.core.weight_modification.methods.guided import (
    LayerDiagnostics,
)
from wisent.core.utils.config_tools.constants import NORM_EPS, SEPARATOR_WIDTH_STANDARD, CHANCE_LEVEL_ACCURACY
from wisent.core.weight_modification.methods._guided_scoring import (
    _compute_knn_accuracy,
    _compute_fisher_ratio,
    _compute_recommended_weight,
)

_LOG = setup_logger(__name__)


def compute_layer_diagnostics(
    pairs: List["ContrastivePair"],
    model: "WisentModel",
    extraction_strategy: str,
    layers: Optional[List[int]] = None,
    verbose: bool = True,
    *,
    probe_knn_k: int,
    blend_default: float,
    architecture_module_limit: int,
) -> Dict[int, LayerDiagnostics]:
    """
    Compute linearity diagnostics for each layer.
    
    This is the core of Zwiad-guided layer selection. For each layer,
    we compute:
    - Linear probe accuracy (can a linear classifier separate pos/neg?)
    - k-NN accuracy (geometry-agnostic baseline)
    - Fisher ratio (measures linear separability)
    - Cohen's d (effect size)
    - Recommended ablation weight based on these metrics
    
    Args:
        pairs: Contrastive pairs for analysis
        model: WisentModel instance
        layers: Specific layers to analyze. If None, tests all layers.
        extraction_strategy: How to extract activations
        verbose: Print progress
        
    Returns:
        Dictionary mapping layer_idx to LayerDiagnostics
    """
    from wisent.core.primitives.model_interface.core.activations.activations_collector import ActivationCollector
    from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy
    
    log = bind(_LOG)
    
    collector = ActivationCollector(model, architecture_module_limit=architecture_module_limit)
    num_layers = model.hf_model.config.num_hidden_layers
    
    if layers is None:
        layers = list(range(num_layers))
    
    strategy = ExtractionStrategy(extraction_strategy)
    
    diagnostics: Dict[int, LayerDiagnostics] = {}
    
    if verbose:
        print(f"\nComputing layer diagnostics for {len(layers)} layers...")
        print(f"Using {len(pairs)} contrastive pairs")
        print(f"Extraction strategy: {extraction_strategy}")
        print("-" * SEPARATOR_WIDTH_STANDARD)
    
    # Collect activations for all layers
    pos_activations: Dict[int, List[Tensor]] = {l: [] for l in layers}
    neg_activations: Dict[int, List[Tensor]] = {l: [] for l in layers}
    
    for i, pair in enumerate(pairs):
        try:
            pair_with_acts = collector.collect(
                pair,
                strategy=strategy,
                layers=[str(l) for l in layers],
                normalize=False,
            )
            
            pos_la = pair_with_acts.positive_response.layers_activations
            neg_la = pair_with_acts.negative_response.layers_activations
            
            if pos_la and neg_la:
                for layer in layers:
                    pos_t = pos_la.get(str(layer))
                    neg_t = neg_la.get(str(layer))
                    if pos_t is not None and neg_t is not None:
                        pos_activations[layer].append(pos_t.flatten().cpu())
                        neg_activations[layer].append(neg_t.flatten().cpu())
        except Exception as e:
            log.debug(f"Failed to collect activations for pair {i}: {e}")
            continue
    
    # Analyze each layer
    for layer in layers:
        pos_list = pos_activations[layer]
        neg_list = neg_activations[layer]
        
        if len(pos_list) < 10 or len(neg_list) < 10:
            if verbose:
                print(f"  Layer {layer:3d}: Insufficient data ({len(pos_list)}/{len(neg_list)} samples)")
            continue
        
        pos_tensor = torch.stack(pos_list)
        neg_tensor = torch.stack(neg_list)
        
        # Compute diagnostics
        diag = _compute_single_layer_diagnostics(
            pos_tensor, neg_tensor, layer, extraction_strategy,
            probe_knn_k=probe_knn_k,
            blend_default=blend_default,
        )
        diagnostics[layer] = diag
        
        if verbose:
            print(
                f"  Layer {layer:3d}: linear={diag.linear_score:.3f} "
                f"knn={diag.knn_score:.3f} fisher={diag.fisher_ratio:.1f} "
                f"d={diag.cohens_d:.2f} -> weight={diag.recommended_weight:.3f}"
            )
    
    if verbose:
        print("-" * SEPARATOR_WIDTH_STANDARD)
        best_layer = max(diagnostics.keys(), key=lambda l: diagnostics[l].linear_score) if diagnostics else -1
        if best_layer >= 0:
            print(f"Best layer: {best_layer} (linear_score={diagnostics[best_layer].linear_score:.3f})")
    
    return diagnostics


def _compute_single_layer_diagnostics(
    pos_tensor: Tensor,
    neg_tensor: Tensor,
    layer_idx: int,
    extraction_strategy: str,
    *,
    probe_knn_k: int,
    blend_default: float,
) -> LayerDiagnostics:
    """Compute diagnostics for a single layer's activations."""
    
    # 1. Linear probe accuracy (logistic regression)
    linear_score, linear_details = _compute_linear_probe_accuracy(pos_tensor, neg_tensor)
    
    # 2. k-NN accuracy (geometry-agnostic baseline)
    knn_score = _compute_knn_accuracy(pos_tensor, neg_tensor, k=probe_knn_k, blend_default=blend_default)
    
    # 3. Fisher ratio
    fisher_ratio = _compute_fisher_ratio(pos_tensor, neg_tensor)
    
    # 4. Cohen's d
    cohens_d = linear_details.get("cohens_d", 0.0)
    
    # 5. Variance explained
    variance_explained = linear_details.get("variance_explained", 0.0)
    
    # 6. Compute recommended weight based on diagnostics
    recommended_weight = _compute_recommended_weight(
        linear_score, knn_score, fisher_ratio, cohens_d,
        blend_default=blend_default,
    )
    
    return LayerDiagnostics(
        layer_idx=layer_idx,
        linear_score=linear_score,
        knn_score=knn_score,
        fisher_ratio=fisher_ratio,
        cohens_d=cohens_d,
        variance_explained=variance_explained,
        extraction_strategy=extraction_strategy,
        recommended_weight=recommended_weight,
        details=linear_details,
    )


def _compute_linear_probe_accuracy(
    pos_tensor: Tensor,
    neg_tensor: Tensor,
) -> Tuple[float, Dict[str, Any]]:
    """Compute linear probe accuracy using simple logistic regression."""
    
    # Combine data
    X = torch.cat([pos_tensor, neg_tensor], dim=0)
    y = torch.cat([
        torch.ones(pos_tensor.shape[0]),
        torch.zeros(neg_tensor.shape[0])
    ])
    
    # Compute primary direction (difference of means)
    mean_diff = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
    mean_diff_norm = mean_diff.norm()
    
    if mean_diff_norm < NORM_EPS:
        return CHANCE_LEVEL_ACCURACY, {"reason": "no_separation"}
    
    direction = mean_diff / mean_diff_norm
    
    # Project onto direction
    projections = X @ direction
    
    # Simple threshold-based classification
    threshold = projections.mean()
    predictions = (projections > threshold).float()
    accuracy = (predictions == y).float().mean().item()
    
    # Compute Cohen's d
    pos_proj = pos_tensor @ direction
    neg_proj = neg_tensor @ direction
    pos_mean, pos_std = pos_proj.mean(), pos_proj.std()
    neg_mean, neg_std = neg_proj.mean(), neg_proj.std()
    pooled_std = ((pos_std**2 + neg_std**2) / 2).sqrt()
    cohens_d = abs(pos_mean - neg_mean) / (pooled_std + NORM_EPS)
    
    # Variance explained
    pos_residual = pos_tensor - (pos_proj.unsqueeze(1) * direction.unsqueeze(0))
    neg_residual = neg_tensor - (neg_proj.unsqueeze(1) * direction.unsqueeze(0))
    total_var = pos_tensor.var() + neg_tensor.var()
    residual_var = pos_residual.var() + neg_residual.var()
    variance_explained = max(0, 1 - (residual_var / (total_var + NORM_EPS)))
    
    return accuracy, {
        "cohens_d": float(cohens_d),
        "variance_explained": float(variance_explained),
        "pos_mean": float(pos_mean),
        "neg_mean": float(neg_mean),
        "threshold": float(threshold),
    }



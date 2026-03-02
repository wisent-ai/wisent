"""Linearity check from raw activations."""
from __future__ import annotations

from typing import Dict, Any

import torch
import numpy as np

from wisent.core.primitives.contrastive_pairs.diagnostics.analysis.linearity import (
    LinearityConfig,
    LinearityResult,
    check_linearity,
)
from wisent.core.utils.config_tools.constants import GEOMETRY_DEFAULT_NUM_COMPONENTS

def check_linearity_from_activations(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    config: Optional[LinearityConfig] = None,
) -> LinearityResult:
    """
    Check linearity from pre-collected activations.
    
    Args:
        pos_activations: Positive class activations [N, hidden_dim]
        neg_activations: Negative class activations [N, hidden_dim]
        config: Configuration
        
    Returns:
        LinearityResult
    """
    from wisent.core.primitives.contrastive_pairs.diagnostics import detect_geometry_structure, GeometryAnalysisConfig
    
    cfg = config or LinearityConfig()
    
    geo_config = GeometryAnalysisConfig(
        num_components=GEOMETRY_DEFAULT_NUM_COMPONENTS,
        optimization_steps=cfg.geometry_optimization_steps,
    )
    
    result = detect_geometry_structure(pos_activations, neg_activations, geo_config)
    
    linear_score = result.all_scores["linear"].score
    linear_details = result.all_scores["linear"].details
    cohens_d = linear_details.get("cohens_d", 0)
    variance_explained = linear_details.get("variance_explained", 0)
    
    if linear_score >= cfg.linear_threshold and cohens_d >= cfg.min_cohens_d:
        verdict = LinearityVerdict.LINEAR
        recommendation = "Use CAA (single-direction steering)."
    elif linear_score >= cfg.weak_threshold and cohens_d >= cfg.min_cohens_d:
        verdict = LinearityVerdict.WEAKLY_LINEAR
        recommendation = "Weakly linear. Try CAA, consider TECZA if poor results."
    else:
        verdict = LinearityVerdict.NON_LINEAR
        recommendation = f"Non-linear ({result.best_structure.value}). Use GROM or fine-tuning."
    
    return LinearityResult(
        verdict=verdict,
        best_linear_score=linear_score,
        best_config={},
        best_layer=-1,
        cohens_d=cohens_d,
        variance_explained=variance_explained,
        all_results=[{
            "linear_score": linear_score,
            "cohens_d": cohens_d,
            "variance_explained": variance_explained,
            "best_structure": result.best_structure.value,
        }],
        recommendation=recommendation,
    )

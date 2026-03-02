"""Collateral damage validation for guided weight modification."""
from __future__ import annotations
from typing import Dict, List, TYPE_CHECKING
if TYPE_CHECKING:
    from torch.nn import Module
    from wisent.core.primitives.models.wisent_model import WisentModel
    from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.utils.config_tools.constants import BLEND_DEFAULT
from wisent.core.weight_modification.methods.guided import (
    GuidedModificationConfig, CollateralDamageReport)
from wisent.core.weight_modification.methods._guided_diagnostics import compute_layer_diagnostics

def validate_collateral_damage(
    model_before: "Module",
    model_after: "Module",
    wisent_model_before: "WisentModel",
    wisent_model_after: "WisentModel",
    validation_pairs: Dict[str, List["ContrastivePair"]],
    config: GuidedModificationConfig,
) -> CollateralDamageReport:
    """
    Validate that weight modification didn't damage unrelated representations.
    
    This is a key innovation: instead of using KL divergence (which measures
    output distribution similarity), we measure whether linear probes for
    UNRELATED concepts still work after modification.
    
    Args:
        model_before: Original model (for comparison)
        model_after: Modified model
        wisent_model_before: WisentModel wrapper for original
        wisent_model_after: WisentModel wrapper for modified
        validation_pairs: Dict mapping benchmark name to contrastive pairs
        config: Configuration
        
    Returns:
        CollateralDamageReport with per-benchmark degradation
    """
    before_scores: Dict[str, float] = {}
    after_scores: Dict[str, float] = {}
    deltas: Dict[str, float] = {}
    
    for benchmark, pairs in validation_pairs.items():
        if len(pairs) < 10:
            continue
        
        # Compute diagnostics before
        diag_before = compute_layer_diagnostics(
            pairs=pairs,
            model=wisent_model_before,
            layers=None,
            verbose=False,
        )
        
        # Compute diagnostics after
        diag_after = compute_layer_diagnostics(
            pairs=pairs,
            model=wisent_model_after,
            layers=None,
            verbose=False,
        )
        
        # Best linear score before/after
        if diag_before:
            before_best = max(d.linear_score for d in diag_before.values())
        else:
            before_best = BLEND_DEFAULT

        if diag_after:
            after_best = max(d.linear_score for d in diag_after.values())
        else:
            after_best = BLEND_DEFAULT
        
        before_scores[benchmark] = before_best
        after_scores[benchmark] = after_best
        deltas[benchmark] = before_best - after_best  # Positive = degradation
    
    if deltas:
        max_degradation = max(deltas.values())
        mean_degradation = sum(deltas.values()) / len(deltas)
    else:
        max_degradation = 0.0
        mean_degradation = 0.0
    
    passed = max_degradation <= config.max_allowed_degradation
    
    return CollateralDamageReport(
        benchmarks_tested=list(validation_pairs.keys()),
        before_scores=before_scores,
        after_scores=after_scores,
        deltas=deltas,
        max_degradation=max_degradation,
        mean_degradation=mean_degradation,
        passed=passed,
        details={
            "threshold": config.max_allowed_degradation,
        }
    )

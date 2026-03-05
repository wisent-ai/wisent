"""Check if a representation is linear (can be captured by a single direction)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Sequence
from enum import Enum

import torch

from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy


class LinearityVerdict(Enum):
    """Verdict on whether a representation is linear."""
    LINEAR = "linear"
    WEAKLY_LINEAR = "weakly_linear"
    NON_LINEAR = "non_linear"


@dataclass
class LinearityConfig:
    """Configuration for linearity check."""

    linear_threshold: float
    """Linear score threshold to declare LINEAR."""
    weak_threshold: float
    """Linear score threshold to declare WEAKLY_LINEAR."""
    min_cohens_d: float
    """Minimum Cohen's d for meaningful separation."""
    max_pairs: int
    """Maximum number of pairs to use for analysis."""
    geometry_optimization_steps: int
    """Steps for geometry detection optimization."""
    linearity_layer_sampling_divisor: int
    """Divisor for sampling layers across depth."""
    geometry_default_num_components: int
    """Number of components for geometry detection."""
    layers_to_test: Optional[List[int]] = None
    """Specific layers to test. If None, tests sample across depth."""
    extraction_strategies: Optional[List[ExtractionStrategy]] = None
    """Extraction strategies to test. If None, tests default set."""
    normalize_options: List[bool] = field(default_factory=lambda: [False, True])
    """Normalization options to test."""


@dataclass
class LinearityResult:
    """Result of linearity check."""
    
    verdict: LinearityVerdict
    """Overall verdict."""
    
    best_linear_score: float
    """Best linear score found across all configurations."""
    
    best_config: Dict[str, Any]
    """Configuration that achieved best linear score."""
    
    best_layer: int
    """Layer with best linearity."""
    
    cohens_d: float
    """Cohen's d for best configuration."""
    
    variance_explained: float
    """Variance explained by first PC for best configuration."""
    
    all_results: List[Dict[str, Any]]
    """Results for all tested configurations."""
    
    recommendation: str
    """Steering method recommendation based on results."""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "best_linear_score": self.best_linear_score,
            "best_config": self.best_config,
            "best_layer": self.best_layer,
            "cohens_d": self.cohens_d,
            "variance_explained": self.variance_explained,
            "recommendation": self.recommendation,
            "num_configs_tested": len(self.all_results),
        }


def check_linearity(
    pairs: List,
    model: "WisentModel",
    config: Optional[LinearityConfig] = None,
    *,
    architecture_module_limit: int,
) -> LinearityResult:
    """
    Check if a representation is linear by sweeping across collection parameters.
    
    Args:
        pairs: List of ContrastivePair objects
        model: WisentModel instance
        config: Configuration for the check
        
    Returns:
        LinearityResult with verdict and best configuration
    """
    from wisent.core.primitives.model_interface.core.activations.activations_collector import ActivationCollector
    from wisent.core.reading.diagnostics import detect_geometry_structure, GeometryAnalysisConfig
    from wisent.core.utils.config_tools.constants import (
        PARSER_DEFAULT_LAYER_START, COMBO_OFFSET,
    )

    if config is None:
        raise ValueError("config (LinearityConfig) is required")
    cfg = config
    collector = ActivationCollector(model, architecture_module_limit=architecture_module_limit)

    num_layers = model.hf_model.config.num_hidden_layers

    # Determine layers to test
    if cfg.layers_to_test is None:
        _step = max(COMBO_OFFSET, num_layers // cfg.linearity_layer_sampling_divisor)
        layers_to_test = sorted(set(
            list(range(PARSER_DEFAULT_LAYER_START, num_layers, _step)) + [num_layers - COMBO_OFFSET]
        ))
    else:
        layers_to_test = cfg.layers_to_test

    # Determine extraction strategies
    if cfg.extraction_strategies is None:
        extraction_strategies = [
            ExtractionStrategy.CHAT_LAST,
            ExtractionStrategy.CHAT_MEAN,
            ExtractionStrategy.CHAT_MAX_NORM,
        ]
    else:
        extraction_strategies = cfg.extraction_strategies

    # Limit pairs
    test_pairs = pairs[:cfg.max_pairs]

    geo_config = GeometryAnalysisConfig(
        num_components=cfg.geometry_default_num_components,
        optimization_steps=cfg.geometry_optimization_steps,
    )
    
    all_results = []
    
    for strategy in extraction_strategies:
        for normalize in cfg.normalize_options:
            # Collect activations
            pos_activations = {l: [] for l in layers_to_test}
            neg_activations = {l: [] for l in layers_to_test}
            
            for pair in test_pairs:
                try:
                    pair_with_acts = collector.collect(
                        pair,
                        strategy=strategy,
                        layers=[str(l) for l in layers_to_test],
                        normalize=normalize,
                    )
                    
                    pos_la = pair_with_acts.positive_response.layers_activations
                    neg_la = pair_with_acts.negative_response.layers_activations
                    
                    if pos_la and neg_la:
                        for layer in layers_to_test:
                            pos_t = pos_la.get(str(layer))
                            neg_t = neg_la.get(str(layer))
                            if pos_t is not None and neg_t is not None:
                                pos_activations[layer].append(pos_t.flatten().cpu())
                                neg_activations[layer].append(neg_t.flatten().cpu())
                except Exception:
                    continue
            
            # Analyze each layer
            for layer in layers_to_test:
                pos_list = pos_activations[layer]
                neg_list = neg_activations[layer]
                
                if len(pos_list) < 10 or len(neg_list) < 10:
                    continue
                
                pos_tensor = torch.stack(pos_list)
                neg_tensor = torch.stack(neg_list)
                
                result = detect_geometry_structure(pos_tensor, neg_tensor, geo_config)
                
                linear_score = result.all_scores["linear"].score
                linear_details = result.all_scores["linear"].details
                
                # Include all structure scores
                structure_scores = {
                    name: {"score": score.score, "confidence": score.confidence}
                    for name, score in result.all_scores.items()
                }
                
                all_results.append({
                    "extraction_strategy": strategy.value,
                    "normalize": normalize,
                    "layer": layer,
                    "linear_score": linear_score,
                    "cohens_d": linear_details.get("cohens_d", 0),
                    "variance_explained": linear_details.get("variance_explained", 0),
                    "best_structure": result.best_structure.value,
                    "all_structure_scores": structure_scores,
                })
    
    if not all_results:
        return LinearityResult(
            verdict=LinearityVerdict.NON_LINEAR,
            best_linear_score=0.0,
            best_config={},
            best_layer=0,
            cohens_d=0.0,
            variance_explained=0.0,
            all_results=[],
            recommendation="No valid configurations found. Check data and model.",
        )
    
    # Find best configuration
    best = max(all_results, key=lambda x: x["linear_score"])
    
    # Determine verdict
    if best["linear_score"] >= cfg.linear_threshold and best["cohens_d"] >= cfg.min_cohens_d:
        verdict = LinearityVerdict.LINEAR
        recommendation = (
            f"Use CAA (single-direction steering) on layer {best['layer']} "
            f"with {best['extraction_strategy']} strategy."
        )
    elif best["linear_score"] >= cfg.weak_threshold and best["cohens_d"] >= cfg.min_cohens_d:
        verdict = LinearityVerdict.WEAKLY_LINEAR
        recommendation = (
            f"Representation is weakly linear. Try CAA on layer {best['layer']}, "
            f"but consider TECZA or multi-direction steering if results are poor."
        )
    else:
        verdict = LinearityVerdict.NON_LINEAR
        recommendation = (
            f"Representation is non-linear (manifold structure). "
            f"Use GROM or fine-tuning instead of CAA. "
            f"Best linear score was {best['linear_score']:.2f} on layer {best['layer']}."
        )
    
    return LinearityResult(
        verdict=verdict,
        best_linear_score=best["linear_score"],
        best_config={
            "extraction_strategy": best["extraction_strategy"],
            "normalize": best["normalize"],
        },
        best_layer=best["layer"],
        cohens_d=best["cohens_d"],
        variance_explained=best["variance_explained"],
        all_results=all_results,
        recommendation=recommendation,
    )



# Re-export from split module
from wisent.core.reading.diagnostics.analysis._linearity_from_activations import (
    check_linearity_from_activations,
)

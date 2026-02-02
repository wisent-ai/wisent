"""Backward-compatible WeightModificationResult and related functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List

from ..convenience import (
    save_weight_modification_config, get_weight_modification_config,
    save_trait_weight_modification_config, get_trait_weight_modification_config,
    get_config_manager,
)


@dataclass
class WeightModificationResult:
    """Backward-compatible result class for weight modification cache."""
    model: str
    task: str
    trait_label: str
    method: str = "directional"
    max_weight: float = 1.0
    min_weight: float = 0.0
    max_weight_position: float = 0.5
    min_weight_distance: float = 0.5
    strength: float = 1.0
    num_pairs: int = 100
    alpha: float = 1.0
    additive_method: str = "bias"
    components: List[str] = field(default_factory=lambda: ["self_attn.o_proj", "mlp.down_proj"])
    normalize_vectors: bool = True
    norm_preserve: bool = True
    use_biprojection: bool = True
    use_kernel: bool = True
    score: float = 0.0
    metric: str = "accuracy"
    baseline_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    output_dir: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def store_weight_modification(
    model: str, task: str, trait_label: str,
    method: str = "directional", max_weight: float = 1.0, min_weight: float = 0.0,
    max_weight_position: float = 0.5, min_weight_distance: float = 0.5,
    strength: float = 1.0, num_pairs: int = 100, alpha: float = 1.0,
    additive_method: str = "bias", components: Optional[List[str]] = None,
    normalize_vectors: bool = True, norm_preserve: bool = True,
    use_biprojection: bool = True, use_kernel: bool = True,
    score: float = 0.0, metric: str = "accuracy", baseline_score: float = 0.0,
    output_dir: str = "", metadata: Optional[Dict[str, Any]] = None,
    set_as_default: bool = False,
) -> str:
    """Backward-compatible function to store weight modification result."""
    if trait_label:
        save_trait_weight_modification_config(
            model_name=model, trait_name=trait_label, method=method,
            max_weight=max_weight, min_weight=min_weight,
            max_weight_position=max_weight_position, min_weight_distance=min_weight_distance,
            strength=strength, num_pairs=num_pairs, alpha=alpha,
            additive_method=additive_method, components=components,
            normalize_vectors=normalize_vectors, norm_preserve=norm_preserve,
            use_biprojection=use_biprojection, use_kernel=use_kernel,
            score=score, baseline_score=baseline_score, output_dir=output_dir,
            optimization_method="optuna" if metadata else "manual",
            set_as_default=set_as_default,
        )
    else:
        save_weight_modification_config(
            model_name=model, task_name=task, method=method,
            max_weight=max_weight, min_weight=min_weight,
            max_weight_position=max_weight_position, min_weight_distance=min_weight_distance,
            strength=strength, num_pairs=num_pairs, alpha=alpha,
            additive_method=additive_method, components=components,
            normalize_vectors=normalize_vectors, norm_preserve=norm_preserve,
            use_biprojection=use_biprojection, use_kernel=use_kernel,
            score=score, baseline_score=baseline_score, output_dir=output_dir,
            optimization_method="optuna" if metadata else "manual",
            set_as_default=set_as_default,
        )
    model_normalized = model.replace("/", "_").replace("\\", "_")
    return f"{model_normalized}::{task}::{trait_label}::{method}"


def get_cached_weight_modification(
    model: str, task: str, trait_label: str,
    method: str = "directional", use_default: bool = True,
) -> Optional[WeightModificationResult]:
    """Backward-compatible function to get cached weight modification result."""
    if trait_label:
        weight_mod = get_trait_weight_modification_config(model, trait_label)
    else:
        weight_mod = get_weight_modification_config(model, task)

    if weight_mod is None:
        return None
    if method != "*" and weight_mod.method != method:
        return None

    return WeightModificationResult(
        model=model, task=task, trait_label=trait_label, method=weight_mod.method,
        max_weight=weight_mod.max_weight, min_weight=weight_mod.min_weight,
        max_weight_position=weight_mod.max_weight_position,
        min_weight_distance=weight_mod.min_weight_distance,
        strength=weight_mod.strength, num_pairs=weight_mod.num_pairs,
        alpha=weight_mod.alpha, additive_method=weight_mod.additive_method,
        components=weight_mod.components, normalize_vectors=weight_mod.normalize_vectors,
        norm_preserve=weight_mod.norm_preserve, use_biprojection=weight_mod.use_biprojection,
        use_kernel=weight_mod.use_kernel, score=weight_mod.score,
        baseline_score=weight_mod.baseline_score, output_dir=weight_mod.output_dir,
    )


def get_weight_modification_cache():
    """Backward-compatible function that returns the global config manager."""
    return get_config_manager()

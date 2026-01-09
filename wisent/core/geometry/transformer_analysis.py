"""
Transformer-specific analysis for different components.

Analyze activations from different transformer components
(attention, MLP, residual) to find optimal intervention points.
"""

import torch
import numpy as np
from enum import Enum
from typing import Dict, Any, List, Optional


class TransformerComponent(Enum):
    """Transformer component types."""
    RESIDUAL = "residual"
    ATTENTION = "attention"
    MLP = "mlp"
    ATTENTION_OUTPUT = "attention_output"
    MLP_OUTPUT = "mlp_output"


def get_component_hook_points(
    model_type: str,
    layer: int,
    component: TransformerComponent,
) -> List[str]:
    """
    Get hook point names for a given component.
    
    Different model architectures use different naming conventions.
    """
    if "llama" in model_type.lower() or "mistral" in model_type.lower():
        if component == TransformerComponent.RESIDUAL:
            return [f"model.layers.{layer}"]
        elif component == TransformerComponent.ATTENTION:
            return [f"model.layers.{layer}.self_attn"]
        elif component == TransformerComponent.MLP:
            return [f"model.layers.{layer}.mlp"]
        elif component == TransformerComponent.ATTENTION_OUTPUT:
            return [f"model.layers.{layer}.self_attn.o_proj"]
        elif component == TransformerComponent.MLP_OUTPUT:
            return [f"model.layers.{layer}.mlp.down_proj"]
    
    elif "qwen" in model_type.lower():
        if component == TransformerComponent.RESIDUAL:
            return [f"model.layers.{layer}"]
        elif component == TransformerComponent.ATTENTION:
            return [f"model.layers.{layer}.self_attn"]
        elif component == TransformerComponent.MLP:
            return [f"model.layers.{layer}.mlp"]
    
    elif "gpt" in model_type.lower():
        if component == TransformerComponent.RESIDUAL:
            return [f"transformer.h.{layer}"]
        elif component == TransformerComponent.ATTENTION:
            return [f"transformer.h.{layer}.attn"]
        elif component == TransformerComponent.MLP:
            return [f"transformer.h.{layer}.mlp"]
    
    # Default pattern
    return [f"layers.{layer}"]


def analyze_transformer_components(
    model,
    tokenizer,
    pos_texts: List[str],
    neg_texts: List[str],
    layer: int,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Analyze different transformer components for steering potential.
    
    Compares residual stream, attention output, and MLP output.
    """
    from .probe_metrics import compute_linear_probe_accuracy, compute_signal_strength
    
    results = {}
    model_type = model.config.model_type if hasattr(model.config, 'model_type') else "unknown"
    
    for component in [TransformerComponent.RESIDUAL, TransformerComponent.MLP]:
        try:
            hook_points = get_component_hook_points(model_type, layer, component)
            
            # This is a simplified version - full implementation would use hooks
            # For now, we just return placeholder
            results[component.value] = {
                "hook_points": hook_points,
                "status": "not_implemented",
            }
        except Exception as e:
            results[component.value] = {"error": str(e)}
    
    return results


def compare_components_for_benchmark(
    model,
    tokenizer,
    pos_activations_by_component: Dict[str, torch.Tensor],
    neg_activations_by_component: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """
    Compare which component has best steering signal for a benchmark.
    """
    from .probe_metrics import compute_linear_probe_accuracy
    from .steerability import compute_steerability_metrics
    
    results = {}
    
    for component in pos_activations_by_component:
        if component not in neg_activations_by_component:
            continue
        
        pos = pos_activations_by_component[component]
        neg = neg_activations_by_component[component]
        
        try:
            linear_acc = compute_linear_probe_accuracy(pos, neg)
            steerability = compute_steerability_metrics(pos, neg)
            
            results[component] = {
                "linear_accuracy": linear_acc,
                "steerability_score": steerability.get("steerability_score", 0.0),
            }
        except Exception as e:
            results[component] = {"error": str(e)}
    
    # Find best component
    if results:
        best_component = max(
            results.keys(),
            key=lambda c: results[c].get("linear_accuracy", 0) if "error" not in results[c] else 0
        )
        results["best_component"] = best_component
    
    return results


def compare_concept_granularity(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    layers: List[int],
    model=None,
    tokenizer=None,
) -> Dict[str, Any]:
    """
    Compare concept representation across different layers.
    
    Earlier layers often have more fine-grained concepts,
    later layers have more abstract concepts.
    """
    from .probe_metrics import compute_linear_probe_accuracy
    from .icd import compute_icd
    
    # This would need per-layer activations
    # Placeholder implementation
    return {
        "layers": layers,
        "status": "requires_per_layer_activations",
    }

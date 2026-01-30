"""
Steering method recommendation based on RepScan metrics.

This module takes raw geometry metrics and recommends which steering
method to use. The recommendations are based on the geometric properties
of the activation space.

Steering Methods:
- CAA: Contrastive Activation Addition - simple mean difference
- Hyperplane: SVM-based steering direction
- MLP: Neural network probe direction
- PRISM: Multi-directional steering (multiple concepts)
- PULSE: Conditional gating based on activation regions
- TITAN: Full adaptive steering with learned gates

The recommendation logic is based on these principles:
1. Linear separability -> simple methods (CAA, Hyperplane)
2. Nonlinear structure -> complex methods (MLP, PULSE, TITAN)
3. Multiple concepts -> multi-directional (PRISM)
4. High ICD -> steering will work
5. Low direction stability -> need robust methods
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class SteeringThresholds:
    """
    Configurable thresholds for steering recommendations.

    These should be tuned empirically based on validation experiments.
    Default values are starting points, not validated.
    """
    # Linearity thresholds
    linear_probe_high: float = 0.85  # Above this: linear methods work
    linear_probe_low: float = 0.65   # Below this: need nonlinear methods

    # Nonlinearity gap (mlp - linear)
    nonlinearity_gap_significant: float = 0.05  # Gap above this suggests nonlinear structure

    # ICD thresholds
    icd_high: float = 0.7   # Above this: strong steering signal
    icd_low: float = 0.3    # Below this: weak steering signal

    # Direction stability
    stability_high: float = 0.8  # Above this: consistent direction
    stability_low: float = 0.5   # Below this: unstable direction

    # Concept detection
    multi_concept_silhouette: float = 0.3  # Above this: multiple concepts detected

    # Alignment thresholds
    alignment_high: float = 0.3  # Above this: good alignment
    alignment_low: float = 0.1   # Below this: poor alignment


def compute_steering_recommendation(
    metrics: Dict[str, Any],
    thresholds: Optional[SteeringThresholds] = None,
) -> Dict[str, Any]:
    """
    Compute steering method recommendation from RepScan metrics.

    Args:
        metrics: Dict of RepScan metrics (from compute_geometry_metrics)
        thresholds: Optional custom thresholds (uses defaults if not provided)

    Returns:
        Dict with:
        - recommended_method: str - the recommended steering method
        - confidence: float - confidence in recommendation (0-1)
        - reasoning: List[str] - explanation of the recommendation
        - method_scores: Dict[str, float] - score for each method
        - raw_signals: Dict - the signals used for decision
    """
    if thresholds is None:
        thresholds = SteeringThresholds()

    # Extract relevant metrics
    linear_acc = _get_metric(metrics, ["linear_probe_accuracy", "linear_probe"])
    mlp_acc = _get_metric(metrics, ["mlp_probe_accuracy", "mlp_probe"])
    icd = _get_metric(metrics, ["icd_icd", "icd"])
    stability = _get_metric(metrics, ["direction_stability_score", "direction_stability"])
    alignment = _get_metric(metrics, ["steer_diff_mean_alignment", "diff_mean_alignment"])
    n_concepts = _get_metric(metrics, ["n_concepts"])
    coherence = _get_metric(metrics, ["concept_coherence"])
    consistency = _get_metric(metrics, ["consistency_consistency_score", "consistency_mean"])
    silhouette = _get_metric(metrics, ["best_silhouette"])

    # Compute derived signals
    nonlinearity_gap = None
    if mlp_acc is not None and linear_acc is not None:
        nonlinearity_gap = mlp_acc - linear_acc

    # Store raw signals for transparency
    raw_signals = {
        "linear_acc": linear_acc,
        "mlp_acc": mlp_acc,
        "nonlinearity_gap": nonlinearity_gap,
        "icd": icd,
        "stability": stability,
        "alignment": alignment,
        "n_concepts": n_concepts,
        "coherence": coherence,
        "consistency": consistency,
        "silhouette": silhouette,
    }

    # Initialize method scores
    method_scores = {
        "CAA": 0.0,
        "Hyperplane": 0.0,
        "MLP": 0.0,
        "PRISM": 0.0,
        "PULSE": 0.0,
        "TITAN": 0.0,
    }

    reasoning = []

    # === Decision Logic ===

    # 1. Check if steering is viable at all
    steering_viable = True
    if icd is not None and icd < thresholds.icd_low:
        reasoning.append(f"Low ICD ({icd:.3f}) suggests weak steering signal")
        steering_viable = False

    if alignment is not None and alignment < thresholds.alignment_low:
        reasoning.append(f"Low alignment ({alignment:.3f}) suggests inconsistent direction")

    # 2. Linear vs Nonlinear decision
    is_linear = True
    if linear_acc is not None:
        if linear_acc >= thresholds.linear_probe_high:
            reasoning.append(f"High linear accuracy ({linear_acc:.3f}) favors linear methods")
            method_scores["CAA"] += 2.0
            method_scores["Hyperplane"] += 2.0
        elif linear_acc < thresholds.linear_probe_low:
            reasoning.append(f"Low linear accuracy ({linear_acc:.3f}) suggests nonlinear structure")
            is_linear = False
            method_scores["MLP"] += 1.0
            method_scores["PULSE"] += 1.0
            method_scores["TITAN"] += 1.0

    if nonlinearity_gap is not None and nonlinearity_gap > thresholds.nonlinearity_gap_significant:
        reasoning.append(f"Nonlinearity gap ({nonlinearity_gap:.3f}) suggests MLP captures more structure")
        is_linear = False
        method_scores["MLP"] += 1.5
        method_scores["PULSE"] += 0.5
        method_scores["TITAN"] += 0.5

    # 3. Multiple concepts detection
    has_multiple_concepts = False
    if n_concepts is not None and n_concepts > 1:
        if silhouette is not None and silhouette > thresholds.multi_concept_silhouette:
            has_multiple_concepts = True
            reasoning.append(f"Multiple concepts detected (k={n_concepts}, silhouette={silhouette:.3f})")
            method_scores["PRISM"] += 2.0
            method_scores["TITAN"] += 1.0
            # Penalize single-direction methods
            method_scores["CAA"] -= 1.0
            method_scores["Hyperplane"] -= 0.5


    # 4. Direction stability
    if stability is not None:
        if stability >= thresholds.stability_high:
            reasoning.append(f"High stability ({stability:.3f}) supports simple methods")
            method_scores["CAA"] += 1.0
            method_scores["Hyperplane"] += 0.5
        elif stability < thresholds.stability_low:
            reasoning.append(f"Low stability ({stability:.3f}) suggests need for robust methods")
            method_scores["TITAN"] += 1.0
            method_scores["PULSE"] += 0.5
            method_scores["CAA"] -= 0.5

    # 5. ICD strength
    if icd is not None:
        if icd >= thresholds.icd_high:
            reasoning.append(f"High ICD ({icd:.3f}) indicates strong steering potential")
            # Boost all methods slightly, but especially simple ones
            method_scores["CAA"] += 1.0
            method_scores["Hyperplane"] += 0.5
        elif icd < thresholds.icd_low:
            # Low ICD - complex methods might extract more signal
            method_scores["TITAN"] += 0.5
            method_scores["PULSE"] += 0.5

    # 6. Coherence check
    if coherence is not None:
        if coherence > 0.8:
            reasoning.append(f"High coherence ({coherence:.3f}) supports single-direction steering")
            method_scores["CAA"] += 0.5
        elif coherence < 0.5:
            reasoning.append(f"Low coherence ({coherence:.3f}) suggests fragmented concept")
            method_scores["PRISM"] += 0.5
            method_scores["TITAN"] += 0.5

    # === Determine recommendation ===

    # Normalize scores to be non-negative
    min_score = min(method_scores.values())
    if min_score < 0:
        method_scores = {k: v - min_score for k, v in method_scores.items()}

    # Find best method
    best_method = max(method_scores, key=method_scores.get)
    best_score = method_scores[best_method]

    # Compute confidence based on score margin
    sorted_scores = sorted(method_scores.values(), reverse=True)
    if len(sorted_scores) > 1 and sorted_scores[0] > 0:
        margin = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
        confidence = min(0.5 + margin * 0.5, 1.0)
    else:
        confidence = 0.5

    # If steering doesn't look viable, lower confidence
    if not steering_viable:
        confidence *= 0.5
        reasoning.append("Warning: steering may not be effective for this concept")

    # Add default reasoning if empty
    if not reasoning:
        reasoning.append("Insufficient metrics for detailed analysis, defaulting to CAA")
        best_method = "CAA"
        confidence = 0.3

    return {
        "recommended_method": best_method,
        "confidence": float(confidence),
        "reasoning": reasoning,
        "method_scores": method_scores,
        "raw_signals": raw_signals,
        "thresholds_used": {
            "linear_probe_high": thresholds.linear_probe_high,
            "linear_probe_low": thresholds.linear_probe_low,
            "nonlinearity_gap_significant": thresholds.nonlinearity_gap_significant,
            "icd_high": thresholds.icd_high,
            "icd_low": thresholds.icd_low,
            "stability_high": thresholds.stability_high,
            "stability_low": thresholds.stability_low,
            "multi_concept_silhouette": thresholds.multi_concept_silhouette,
            "alignment_high": thresholds.alignment_high,
            "alignment_low": thresholds.alignment_low,
        },
    }


def compute_per_layer_recommendation(
    per_layer_metrics: Dict[int, Dict[str, Any]],
    thresholds: Optional[SteeringThresholds] = None,
) -> Dict[str, Any]:
    """
    Compute recommendations for each layer and identify best layer.

    Args:
        per_layer_metrics: Dict mapping layer index to metrics
        thresholds: Optional custom thresholds

    Returns:
        Dict with per-layer recommendations and overall best layer
    """
    if not per_layer_metrics:
        return {"error": "no layer metrics provided"}

    per_layer = {}
    layer_scores = {}

    for layer, metrics in per_layer_metrics.items():
        rec = compute_steering_recommendation(metrics, thresholds)
        per_layer[layer] = rec

        # Score this layer based on confidence and raw signals
        icd = rec["raw_signals"].get("icd")
        stability = rec["raw_signals"].get("stability")
        alignment = rec["raw_signals"].get("alignment")

        # Composite score for layer selection
        score = 0.0
        if icd is not None:
            score += icd * 2.0  # ICD is important
        if stability is not None:
            score += stability
        if alignment is not None:
            score += alignment

        layer_scores[layer] = score

    # Find best layer
    best_layer = max(layer_scores, key=layer_scores.get)

    return {
        "per_layer": per_layer,
        "layer_scores": layer_scores,
        "best_layer": best_layer,
        "best_layer_recommendation": per_layer[best_layer],
    }


def get_method_description(method: str) -> str:
    """Get description of a steering method."""
    descriptions = {
        "CAA": "Contrastive Activation Addition: Simple mean difference between pos/neg activations. Fast, interpretable, works well for linear concepts.",
        "Hyperplane": "SVM-based steering: Finds optimal separating hyperplane. More robust to outliers than CAA.",
        "MLP": "Neural probe steering: Uses MLP to find nonlinear decision boundary. Captures complex structure.",
        "PRISM": "Multi-directional steering: Handles multiple sub-concepts with separate steering vectors.",
        "PULSE": "Conditional gating: Applies steering conditionally based on activation regions.",
        "TITAN": "Full adaptive: Learns when and how much to steer. Most flexible but requires more data.",
    }
    return descriptions.get(method, "Unknown method")


def get_method_requirements(method: str) -> Dict[str, Any]:
    """Get requirements/assumptions for a steering method."""
    requirements = {
        "CAA": {
            "min_pairs": 10,
            "assumes_linear": True,
            "assumes_single_concept": True,
            "computational_cost": "low",
        },
        "Hyperplane": {
            "min_pairs": 20,
            "assumes_linear": True,
            "assumes_single_concept": True,
            "computational_cost": "low",
        },
        "MLP": {
            "min_pairs": 50,
            "assumes_linear": False,
            "assumes_single_concept": True,
            "computational_cost": "medium",
        },
        "PRISM": {
            "min_pairs": 30,
            "assumes_linear": True,
            "assumes_single_concept": False,
            "computational_cost": "medium",
        },
        "PULSE": {
            "min_pairs": 50,
            "assumes_linear": False,
            "assumes_single_concept": True,
            "computational_cost": "medium",
        },
        "TITAN": {
            "min_pairs": 100,
            "assumes_linear": False,
            "assumes_single_concept": False,
            "computational_cost": "high",
        },
    }
    return requirements.get(method, {})


def _get_metric(metrics: Dict[str, Any], keys: List[str]) -> Optional[float]:
    """Get metric value trying multiple possible keys."""
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return float(metrics[key])
    return None

"""Rigorous intervention selection using z-scores from null tests.

Integrates z-scores from signal, geometry, and effective dimension null tests
to make steering recommendations with confidence bounds.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class RigorousInterventionResult:
    """Result of rigorous intervention selection."""
    recommended_method: str
    confidence: float
    confidence_lower: float
    confidence_upper: float
    reasoning: List[str]
    z_scores_used: Dict[str, float]
    method_scores: Dict[str, float]
    warnings: List[str]


def rigorous_select_intervention(
    signal_z: float,
    signal_p: float,
    geometry_diagnosis: str,
    geometry_confidence: float,
    effective_dim_z: float,
    n_concepts: int,
    silhouette: float,
    geometry_type_z: Optional[Dict[str, float]] = None,
) -> RigorousInterventionResult:
    """
    Select intervention using z-scores from rigorous null tests.

    All decisions based on statistical significance (z > 2 or z < -2).

    Args:
        signal_z: Z-score from signal vs null test
        signal_p: P-value from signal test
        geometry_diagnosis: "LINEAR" or "NONLINEAR" from rigorous geometry test
        geometry_confidence: Confidence from geometry test (0-1)
        effective_dim_z: Z-score from effective dimension vs null
        n_concepts: Number of concepts from decomposition
        silhouette: Silhouette score from clustering
        geometry_type_z: Optional z-scores for cone/sphere/cluster/translation

    Returns:
        RigorousInterventionResult with recommendation and confidence bounds.
    """
    scores = {"CAA": 0.0, "Hyperplane": 0.0, "MLP": 0.0, "PRISM": 0.0, "TITAN": 0.0}
    reasoning = []
    warnings = []
    z_scores_used = {"signal_z": signal_z, "effective_dim_z": effective_dim_z}

    # Check signal significance
    signal_significant = signal_z > 2.0 and signal_p < 0.05
    if not signal_significant:
        warnings.append(f"Signal not significant (z={signal_z:.2f}, p={signal_p:.4f})")
        return RigorousInterventionResult(
            "NONE", 0.0, 0.0, 0.0,
            ["No significant signal detected"],
            z_scores_used, scores, warnings
        )

    reasoning.append(f"Signal significant: z={signal_z:.2f}, p={signal_p:.4f}")

    # Check effective dimension - significant low dimension means steering viable
    low_dim_structure = effective_dim_z < -2.0
    if low_dim_structure:
        reasoning.append(f"Low-dimensional structure detected (z={effective_dim_z:.2f})")
        scores["CAA"] += 1.0
        scores["Hyperplane"] += 0.5
    else:
        reasoning.append(f"No significant low-dim structure (z={effective_dim_z:.2f})")
        scores["MLP"] += 0.5
        scores["TITAN"] += 0.5

    # Use geometry diagnosis
    is_linear = geometry_diagnosis == "LINEAR"
    if is_linear:
        reasoning.append(f"Linear geometry (conf={geometry_confidence:.2f})")
        scores["CAA"] += 2.0
        scores["Hyperplane"] += 1.5
    else:
        reasoning.append(f"Nonlinear geometry (conf={geometry_confidence:.2f})")
        scores["MLP"] += 1.5
        scores["Hyperplane"] += 1.0
        scores["TITAN"] += 0.5

    # Use concept decomposition
    is_fragmented = n_concepts > 1 and silhouette > 0.1
    if is_fragmented:
        reasoning.append(f"Fragmented into {n_concepts} concepts (sil={silhouette:.2f})")
        scores["PRISM"] += 2.0
        scores["TITAN"] += 1.0
        scores["CAA"] -= 1.0
    else:
        reasoning.append("Single concept detected")
        scores["CAA"] += 0.5

    # Use geometry type z-scores if available
    if geometry_type_z:
        z_scores_used.update(geometry_type_z)
        if geometry_type_z.get("translation_z", 0) > 2:
            reasoning.append("Significant translation structure")
            scores["CAA"] += 1.0

    # Determine best method
    min_score = min(scores.values())
    if min_score < 0:
        scores = {k: v - min_score for k, v in scores.items()}

    best_method = max(scores, key=scores.get)

    # Compute confidence with bounds
    sorted_scores = sorted(scores.values(), reverse=True)
    if sorted_scores[0] > 0:
        margin = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
        base_conf = 0.5 + margin * 0.3
    else:
        base_conf = 0.5

    # Adjust confidence by geometry confidence and signal strength
    conf = base_conf * geometry_confidence * min(1.0, signal_z / 3.0)
    conf = min(0.95, max(0.1, conf))

    # Bootstrap-style confidence interval (approximate)
    conf_std = 0.1 * (1 - geometry_confidence)
    conf_lower = max(0.0, conf - 1.96 * conf_std)
    conf_upper = min(1.0, conf + 1.96 * conf_std)

    return RigorousInterventionResult(
        best_method, conf, conf_lower, conf_upper,
        reasoning, z_scores_used, scores, warnings
    )


def get_method_requirements(method: str) -> Dict[str, any]:
    """Get requirements for a steering method."""
    return {
        "CAA": {"min_pairs": 10, "assumes_linear": True, "assumes_single_concept": True},
        "Hyperplane": {"min_pairs": 20, "assumes_linear": True, "assumes_single_concept": True},
        "MLP": {"min_pairs": 50, "assumes_linear": False, "assumes_single_concept": True},
        "PRISM": {"min_pairs": 30, "assumes_linear": True, "assumes_single_concept": False},
        "TITAN": {"min_pairs": 100, "assumes_linear": False, "assumes_single_concept": False},
    }.get(method, {})

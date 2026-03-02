"""Rigorous intervention selection using z-scores from null tests.

Integrates z-scores from signal, geometry, and effective dimension null tests
to make steering recommendations with confidence bounds.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from wisent.core.utils.config_tools.constants import (
    NULL_TEST_Z_SCORE_SIGNIFICANT, NULL_TEST_SIGNIFICANCE_THRESHOLD,
    INTERVENTION_CONFIDENCE_BASE, INTERVENTION_MARGIN_WEIGHT,
    INTERVENTION_Z_NORMALIZER, INTERVENTION_CONF_STD_FACTOR,
    INTERVENTION_MIN_SILHOUETTE, STABILITY_Z_MARGIN,
    INTERVENTION_SCORE_LOW_DIM_CAA, INTERVENTION_SCORE_LOW_DIM_OSTRZE,
    INTERVENTION_SCORE_NO_LOW_DIM, INTERVENTION_SCORE_LINEAR_CAA,
    INTERVENTION_SCORE_LINEAR_OSTRZE, INTERVENTION_SCORE_NONLINEAR_MLP,
    INTERVENTION_SCORE_NONLINEAR_OSTRZE, INTERVENTION_SCORE_NONLINEAR_GROM,
    INTERVENTION_SCORE_FRAGMENTED_TECZA, INTERVENTION_SCORE_FRAGMENTED_GROM,
    INTERVENTION_SCORE_FRAGMENTED_CAA_PENALTY, INTERVENTION_SCORE_SINGLE_CAA,
    INTERVENTION_SCORE_TRANSLATION_CAA,
    CONFIDENCE_UPPER_BOUND, CONFIDENCE_LOWER_BOUND,
    MIN_PAIRS_CAA, MIN_PAIRS_OSTRZE, MIN_PAIRS_MLP,
    MIN_PAIRS_TECZA, MIN_PAIRS_GROM,
    Z_SCORE_SIGNIFICANCE,
)


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

    All decisions based on statistical significance (z > Z_SCORE_SIGNIFICANCE or z < -Z_SCORE_SIGNIFICANCE).

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
    scores = {"CAA": 0.0, "Ostrze": 0.0, "MLP": 0.0, "TECZA": 0.0, "GROM": 0.0}
    reasoning = []
    warnings = []
    z_scores_used = {"signal_z": signal_z, "effective_dim_z": effective_dim_z}

    # Check signal significance
    signal_significant = signal_z > NULL_TEST_Z_SCORE_SIGNIFICANT and signal_p < NULL_TEST_SIGNIFICANCE_THRESHOLD
    if not signal_significant:
        warnings.append(f"Signal not significant (z={signal_z:.2f}, p={signal_p:.4f})")
        return RigorousInterventionResult(
            "NONE", 0.0, 0.0, 0.0,
            ["No significant signal detected"],
            z_scores_used, scores, warnings
        )

    reasoning.append(f"Signal significant: z={signal_z:.2f}, p={signal_p:.4f}")

    # Check effective dimension - significant low dimension means steering viable
    low_dim_structure = effective_dim_z < -NULL_TEST_Z_SCORE_SIGNIFICANT
    if low_dim_structure:
        reasoning.append(f"Low-dimensional structure detected (z={effective_dim_z:.2f})")
        scores["CAA"] += INTERVENTION_SCORE_LOW_DIM_CAA
        scores["Ostrze"] += INTERVENTION_SCORE_LOW_DIM_OSTRZE
    else:
        reasoning.append(f"No significant low-dim structure (z={effective_dim_z:.2f})")
        scores["MLP"] += INTERVENTION_SCORE_NO_LOW_DIM
        scores["GROM"] += INTERVENTION_SCORE_NO_LOW_DIM

    # Use geometry diagnosis
    is_linear = geometry_diagnosis.startswith("LINEAR")
    if is_linear:
        reasoning.append(f"Linear geometry (conf={geometry_confidence:.2f})")
        scores["CAA"] += INTERVENTION_SCORE_LINEAR_CAA
        scores["Ostrze"] += INTERVENTION_SCORE_LINEAR_OSTRZE
    else:
        reasoning.append(f"Nonlinear geometry (conf={geometry_confidence:.2f})")
        scores["MLP"] += INTERVENTION_SCORE_NONLINEAR_MLP
        scores["Ostrze"] += INTERVENTION_SCORE_NONLINEAR_OSTRZE
        scores["GROM"] += INTERVENTION_SCORE_NONLINEAR_GROM

    # Use concept decomposition
    is_fragmented = n_concepts > 1 and silhouette > INTERVENTION_MIN_SILHOUETTE
    if is_fragmented:
        reasoning.append(f"Fragmented into {n_concepts} concepts (sil={silhouette:.2f})")
        scores["TECZA"] += INTERVENTION_SCORE_FRAGMENTED_TECZA
        scores["GROM"] += INTERVENTION_SCORE_FRAGMENTED_GROM
        scores["CAA"] += INTERVENTION_SCORE_FRAGMENTED_CAA_PENALTY
    else:
        reasoning.append("Single concept detected")
        scores["CAA"] += INTERVENTION_SCORE_SINGLE_CAA

    # Use geometry type z-scores if available
    if geometry_type_z:
        z_scores_used.update(geometry_type_z)
        if geometry_type_z.get("translation_z", 0) > Z_SCORE_SIGNIFICANCE:
            reasoning.append("Significant translation structure")
            scores["CAA"] += INTERVENTION_SCORE_TRANSLATION_CAA

    # Determine best method
    min_score = min(scores.values())
    if min_score < 0:
        scores = {k: v - min_score for k, v in scores.items()}

    best_method = max(scores, key=scores.get)

    # Compute confidence with bounds
    sorted_scores = sorted(scores.values(), reverse=True)
    if sorted_scores[0] > 0:
        margin = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
        base_conf = INTERVENTION_CONFIDENCE_BASE + margin * INTERVENTION_MARGIN_WEIGHT
    else:
        base_conf = INTERVENTION_CONFIDENCE_BASE

    # Adjust confidence by geometry confidence and signal strength
    conf = base_conf * geometry_confidence * min(1.0, signal_z / INTERVENTION_Z_NORMALIZER)
    conf = min(CONFIDENCE_UPPER_BOUND, max(CONFIDENCE_LOWER_BOUND, conf))

    # Bootstrap-style confidence interval (approximate)
    conf_std = INTERVENTION_CONF_STD_FACTOR * (1 - geometry_confidence)
    conf_lower = max(0.0, conf - STABILITY_Z_MARGIN * conf_std)
    conf_upper = min(1.0, conf + STABILITY_Z_MARGIN * conf_std)

    return RigorousInterventionResult(
        best_method, conf, conf_lower, conf_upper,
        reasoning, z_scores_used, scores, warnings
    )


def get_method_requirements(method: str) -> Dict[str, any]:
    """Get requirements for a steering method."""
    return {
        "CAA": {"min_pairs": MIN_PAIRS_CAA, "assumes_linear": True, "assumes_single_concept": True},
        "Ostrze": {"min_pairs": MIN_PAIRS_OSTRZE, "assumes_linear": True, "assumes_single_concept": True},
        "MLP": {"min_pairs": MIN_PAIRS_MLP, "assumes_linear": False, "assumes_single_concept": True},
        "TECZA": {"min_pairs": MIN_PAIRS_TECZA, "assumes_linear": True, "assumes_single_concept": False},
        "GROM": {"min_pairs": MIN_PAIRS_GROM, "assumes_linear": False, "assumes_single_concept": False},
    }.get(method, {})

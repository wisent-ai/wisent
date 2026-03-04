"""Rigorous intervention selection using z-scores from null tests.

Integrates z-scores from signal, geometry, and effective dimension null tests
to make steering recommendations with confidence bounds.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from wisent.core.utils.config_tools.constants import (
    NULL_TEST_Z_SCORE_SIGNIFICANT,
    NULL_TEST_SIGNIFICANCE_THRESHOLD,
    STABILITY_Z_MARGIN,
    Z_SCORE_SIGNIFICANCE,
    SPLIT_RATIO_FULL,
    COMBO_OFFSET,
)


@dataclass
class InterventionScoringConfig:
    """All scoring parameters for intervention selection. All fields required."""
    confidence_base: float
    margin_weight: float
    z_normalizer: float
    conf_std_factor: float
    min_silhouette: float
    score_low_dim_caa: float
    score_low_dim_ostrze: float
    score_no_low_dim: float
    score_linear_caa: float
    score_linear_ostrze: float
    score_nonlinear_mlp: float
    score_nonlinear_ostrze: float
    score_nonlinear_grom: float
    score_fragmented_tecza: float
    score_fragmented_grom: float
    score_fragmented_caa_penalty: float
    score_single_caa: float
    score_translation_caa: float


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
    *,
    scoring_config: InterventionScoringConfig | None = None,
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
    if scoring_config is None:
        raise ValueError("scoring_config (InterventionScoringConfig) is required")
    sc = scoring_config
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
        scores["CAA"] += sc.score_low_dim_caa
        scores["Ostrze"] += sc.score_low_dim_ostrze
    else:
        reasoning.append(f"No significant low-dim structure (z={effective_dim_z:.2f})")
        scores["MLP"] += sc.score_no_low_dim
        scores["GROM"] += sc.score_no_low_dim

    # Use geometry diagnosis
    is_linear = geometry_diagnosis.startswith("LINEAR")
    if is_linear:
        reasoning.append(f"Linear geometry (conf={geometry_confidence:.2f})")
        scores["CAA"] += sc.score_linear_caa
        scores["Ostrze"] += sc.score_linear_ostrze
    else:
        reasoning.append(f"Nonlinear geometry (conf={geometry_confidence:.2f})")
        scores["MLP"] += sc.score_nonlinear_mlp
        scores["Ostrze"] += sc.score_nonlinear_ostrze
        scores["GROM"] += sc.score_nonlinear_grom

    # Use concept decomposition
    is_fragmented = n_concepts > 1 and silhouette > sc.min_silhouette
    if is_fragmented:
        reasoning.append(f"Fragmented into {n_concepts} concepts (sil={silhouette:.2f})")
        scores["TECZA"] += sc.score_fragmented_tecza
        scores["GROM"] += sc.score_fragmented_grom
        scores["CAA"] += sc.score_fragmented_caa_penalty
    else:
        reasoning.append("Single concept detected")
        scores["CAA"] += sc.score_single_caa

    # Use geometry type z-scores if available
    if geometry_type_z:
        z_scores_used.update(geometry_type_z)
        if geometry_type_z.get("translation_z", 0) > Z_SCORE_SIGNIFICANCE:
            reasoning.append("Significant translation structure")
            scores["CAA"] += sc.score_translation_caa

    # Determine best method
    min_score = min(scores.values())
    if min_score < 0:
        scores = {k: v - min_score for k, v in scores.items()}

    best_method = max(scores, key=scores.get)

    # Compute confidence with bounds
    sorted_scores = sorted(scores.values(), reverse=True)
    if sorted_scores[0] > 0:
        margin = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
        base_conf = sc.confidence_base + margin * sc.margin_weight
    else:
        base_conf = sc.confidence_base

    # Adjust confidence by geometry confidence and signal strength
    conf = base_conf * geometry_confidence * min(SPLIT_RATIO_FULL, signal_z / sc.z_normalizer)
    conf = min(0.95, max(0.1, conf))

    # Bootstrap-style confidence interval (approximate)
    conf_std = sc.conf_std_factor * (COMBO_OFFSET - geometry_confidence)
    conf_lower = max(0.0, conf - STABILITY_Z_MARGIN * conf_std)
    conf_upper = min(1.0, conf + STABILITY_Z_MARGIN * conf_std)

    return RigorousInterventionResult(
        best_method, conf, conf_lower, conf_upper,
        reasoning, z_scores_used, scores, warnings
    )


def get_method_requirements(
    method: str,
    min_pairs_caa: int,
    min_pairs_ostrze: int,
    min_pairs_mlp: int,
    min_pairs_tecza: int,
    *,
    min_pairs_grom: int,
) -> Dict[str, any]:
    """Get requirements for a steering method.

    Args:
        method: Method name.
        min_pairs_caa: Minimum contrastive pairs for CAA.
        min_pairs_ostrze: Minimum contrastive pairs for Ostrze.
        min_pairs_mlp: Minimum contrastive pairs for MLP.
        min_pairs_tecza: Minimum contrastive pairs for TECZA.
        min_pairs_grom: Minimum contrastive pairs for GROM.
    """
    return {
        "CAA": {"min_pairs": min_pairs_caa, "assumes_linear": True, "assumes_single_concept": True},
        "Ostrze": {"min_pairs": min_pairs_ostrze, "assumes_linear": True, "assumes_single_concept": True},
        "MLP": {"min_pairs": min_pairs_mlp, "assumes_linear": False, "assumes_single_concept": True},
        "TECZA": {"min_pairs": min_pairs_tecza, "assumes_linear": True, "assumes_single_concept": False},
        "GROM": {"min_pairs": min_pairs_grom, "assumes_linear": False, "assumes_single_concept": False},
    }.get(method, {})

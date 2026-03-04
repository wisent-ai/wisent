"""Preflight thresholds dataclass and compatibility check helpers."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from wisent.core.reading.diagnostics.control_vectors.geometry.geometry_types import StructureType
from wisent.core.utils.config_tools.constants import (
    SCORE_RANGE_MIN, SEPARATOR_WIDTH_STANDARD,
    PREFLIGHT_LINEAR_EXCELLENT, PREFLIGHT_LINEAR_GOOD,
    PREFLIGHT_CONE_GOOD, PREFLIGHT_MANIFOLD_HIGH,
    PREFLIGHT_LINEAR_OVERKILL, PREFLIGHT_LINEAR_VERY_HIGH,
    PREFLIGHT_GROM_DEFAULT, PREFLIGHT_GROM_MANIFOLD_EXCELLENT,
    PREFLIGHT_BIMODAL_GOOD, PREFLIGHT_TETNO_BIMODAL,
    PREFLIGHT_TETNO_LINEAR_OVERKILL,
    PREFLIGHT_COMPAT_SCORE_CAA_EXCELLENT, PREFLIGHT_COMPAT_SCORE_CAA_GOOD,
    PREFLIGHT_COMPAT_SCORE_CAA_POOR, PREFLIGHT_COMPAT_SCORE_CAA_DEFAULT,
    PREFLIGHT_COMPAT_SCORE_TECZA_EXCELLENT,
    PREFLIGHT_COMPAT_SCORE_TECZA_OVERKILL,
    PREFLIGHT_COMPAT_SCORE_TECZA_DEFAULT,
    PREFLIGHT_COMPAT_SCORE_TETNO_DEFAULT,
    PREFLIGHT_COMPAT_MIN_COMPATIBLE, PREFLIGHT_COMPAT_UNKNOWN_DEFAULT,
    PREFLIGHT_SPARSE_HIGH,
)


@dataclass(frozen=True)
class PreflightThresholds:
    """All preflight compatibility threshold values."""

    linear_excellent: float = PREFLIGHT_LINEAR_EXCELLENT
    linear_good: float = PREFLIGHT_LINEAR_GOOD
    cone_good: float = PREFLIGHT_CONE_GOOD
    manifold_high: float = PREFLIGHT_MANIFOLD_HIGH
    linear_overkill: float = PREFLIGHT_LINEAR_OVERKILL
    linear_very_high: float = PREFLIGHT_LINEAR_VERY_HIGH
    grom_default: float = PREFLIGHT_GROM_DEFAULT
    grom_manifold_excellent: float = PREFLIGHT_GROM_MANIFOLD_EXCELLENT
    bimodal_good: float = PREFLIGHT_BIMODAL_GOOD
    tetno_bimodal: float = PREFLIGHT_TETNO_BIMODAL
    tetno_linear_overkill: float = PREFLIGHT_TETNO_LINEAR_OVERKILL
    compat_score_caa_excellent: float = PREFLIGHT_COMPAT_SCORE_CAA_EXCELLENT
    compat_score_caa_good: float = PREFLIGHT_COMPAT_SCORE_CAA_GOOD
    compat_score_caa_poor: float = PREFLIGHT_COMPAT_SCORE_CAA_POOR
    compat_score_caa_default: float = PREFLIGHT_COMPAT_SCORE_CAA_DEFAULT
    compat_score_tecza_excellent: float = PREFLIGHT_COMPAT_SCORE_TECZA_EXCELLENT
    compat_score_tecza_overkill: float = PREFLIGHT_COMPAT_SCORE_TECZA_OVERKILL
    compat_score_tecza_default: float = PREFLIGHT_COMPAT_SCORE_TECZA_DEFAULT
    compat_score_tetno_default: float = PREFLIGHT_COMPAT_SCORE_TETNO_DEFAULT
    compat_min_compatible: float = PREFLIGHT_COMPAT_MIN_COMPATIBLE
    compat_unknown_default: float = PREFLIGHT_COMPAT_UNKNOWN_DEFAULT
    sparse_high: float = PREFLIGHT_SPARSE_HIGH


def complete_method_compatibility_check(
    method: str,
    best_structure: StructureType,
    structure_scores: Dict[str, float],
    warnings: list,
    STRUCTURE_TO_METHODS: Dict,
    *,
    preflight_compat_unknown_default: float,
    preflight_compat_min_compatible: float,
    preflight_sparse_high: float,
) -> Tuple[bool, float, list]:
    """Complete the method compatibility check for unknown methods.

    Handles the final branch of check_method_compatibility for methods
    that are not explicitly handled (unknown methods), and adds
    structure-specific warnings for manifold and sparse data.

    Args:
        method: Steering method name
        best_structure: Best detected structure type
        structure_scores: Dict of structure type scores
        warnings: List of PreflightWarning objects accumulated so far
        STRUCTURE_TO_METHODS: Mapping of structure types to recommended methods
        preflight_compat_unknown_default: Default compat score for unknown methods
        preflight_compat_min_compatible: Minimum compat score threshold
        preflight_sparse_high: Threshold for sparse structure warnings

    Returns:
        Tuple of (is_compatible, compat_score, warnings)
    """
    from wisent.core.control.steering_methods.preflight import PreflightWarning

    method_lower = method.lower()

    # Unknown method - give generic advice
    compat_score = preflight_compat_unknown_default
    warnings.append(PreflightWarning(
        severity="info",
        message=f"Unknown method '{method}' - cannot provide specific "
                f"compatibility check",
        suggestion=f"Recommended methods for {best_structure.value}: "
                   f"{', '.join(STRUCTURE_TO_METHODS.get(best_structure, ['caa']))}",
    ))

    # Add structure-specific warnings
    if best_structure == StructureType.MANIFOLD and method_lower not in ["grom"]:
        warnings.append(PreflightWarning(
            severity="warning",
            message="Data has non-linear manifold structure",
            details="Intrinsic dimensionality much lower than ambient dimension",
            suggestion="GROM with learned gating may capture this structure "
                       "better",
        ))

    if (structure_scores.get("sparse", SCORE_RANGE_MIN) > preflight_sparse_high
            and method_lower not in ["sae", "sparse_steering"]):
        warnings.append(PreflightWarning(
            severity="info",
            message="Data shows sparse structure - few neurons are active",
            details=f"Sparse score: {structure_scores.get('sparse', SCORE_RANGE_MIN):.2f}",
            suggestion="Consider SAE-based steering for more targeted "
                       "intervention",
        ))

    is_compatible = compat_score >= preflight_compat_min_compatible
    return is_compatible, compat_score, warnings


def print_preflight_report(result) -> None:
    """Print a human-readable pre-flight check report."""
    print("\n" + "=" * SEPARATOR_WIDTH_STANDARD)
    print("STEERING METHOD PRE-FLIGHT CHECK")
    print("=" * SEPARATOR_WIDTH_STANDARD)
    print(f"\nChosen Method: {result.chosen_method}")
    print(f"Detected Structure: {result.geometry_result.best_structure.value}")
    print(f"Structure Score: {result.geometry_result.best_score:.3f}")
    compat_text = "OK" if result.is_compatible else "WARNING"
    print(f"Compatibility: {compat_text} ({result.compatibility_score:.0%})")
    if result.recommended_methods:
        print(f"\nRecommended Methods: {', '.join(result.recommended_methods)}")
    if result.warnings:
        print(f"\n{'='*SEPARATOR_WIDTH_STANDARD}")
        print("WARNINGS")
        print("=" * SEPARATOR_WIDTH_STANDARD)
        for w in result.warnings:
            icon = "!" if w.severity == "error" else "~" if w.severity == "warning" else "i"
            print(f"\n[{icon}] [{w.severity.upper()}] {w.message}")
            if w.details:
                print(f"   Details: {w.details}")
            if w.suggestion:
                print(f"   Suggestion: {w.suggestion}")
    print(f"\n{'='*SEPARATOR_WIDTH_STANDARD}")
    print(f"Recommendation: {result.geometry_result.recommendation}")
    print("=" * SEPARATOR_WIDTH_STANDARD + "\n")

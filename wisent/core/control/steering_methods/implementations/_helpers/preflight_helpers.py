"""Preflight thresholds dataclass and compatibility check helpers."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from wisent.core.reading.diagnostics.control_vectors.geometry.geometry_types import StructureType
from wisent.core.utils.config_tools.constants import (
    SCORE_RANGE_MIN, SEPARATOR_WIDTH_STANDARD,
)


@dataclass(frozen=True)
class PreflightThresholds:
    """All preflight compatibility threshold values.

    All fields are required - callers must provide explicit values.
    """

    linear_excellent: float
    linear_good: float
    cone_good: float
    manifold_high: float
    linear_overkill: float
    linear_very_high: float
    grom_default: float
    grom_manifold_excellent: float
    bimodal_good: float
    tetno_bimodal: float
    tetno_linear_overkill: float
    compat_score_caa_excellent: float
    compat_score_caa_good: float
    compat_score_caa_poor: float
    compat_score_caa_default: float
    compat_score_tecza_excellent: float
    compat_score_tecza_overkill: float
    compat_score_tecza_default: float
    compat_score_tetno_default: float
    compat_min_compatible: float
    compat_unknown_default: float
    sparse_high: float


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

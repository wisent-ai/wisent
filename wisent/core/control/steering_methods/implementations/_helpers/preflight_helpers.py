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

    Only keeps the one threshold that gates a real decision: is_compatible.
    All method-specific scoring thresholds have been removed — raw geometry
    scores are returned directly instead of being transformed through
    unvalidated heuristic decision trees.
    """

    compat_min_compatible: float


def complete_method_compatibility_check(
    method: str,
    best_structure: StructureType,
    structure_scores: Dict[str, float],
    warnings: list,
    *,
    preflight_compat_min_compatible: float,
) -> Tuple[bool, float, list]:
    """Complete the method compatibility check for unknown methods.

    Reports raw geometry scores with an info warning. No structure-specific
    recommendations or method suggestions.

    Args:
        method: Steering method name
        best_structure: Best detected structure type
        structure_scores: Dict of structure type scores
        warnings: List of PreflightWarning objects accumulated so far
        preflight_compat_min_compatible: Minimum compat score threshold

    Returns:
        Tuple of (is_compatible, compat_score, warnings)
    """
    from wisent.core.control.steering_methods.preflight import PreflightWarning

    # Use best structure score as compatibility score
    best_score = max(structure_scores.values()) if structure_scores else SCORE_RANGE_MIN
    compat_score = best_score

    warnings.append(PreflightWarning(
        severity="info",
        message=f"Unknown method '{method}' - cannot provide specific "
                f"compatibility check",
        details=f"Best structure: {best_structure.value}, scores: "
                f"{', '.join(f'{k}={v:.2f}' for k, v in structure_scores.items())}",
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

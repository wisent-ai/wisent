"""Extracted from preflight.py - check_method_compatibility tail."""

from typing import Dict, List, Tuple
from wisent.core.reading.diagnostics.control_vectors.geometry.geometry_types import StructureType
from wisent.core.utils.config_tools.constants import (
    PREFLIGHT_COMPAT_DEFAULT, PREFLIGHT_COMPAT_THRESHOLD, PREFLIGHT_MANIFOLD_HIGH,
)


def complete_method_compatibility_check(
    method: str,
    best_structure: StructureType,
    structure_scores: Dict[str, float],
    warnings: list,
    STRUCTURE_TO_METHODS: Dict,
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

    Returns:
        Tuple of (is_compatible, compat_score, warnings)
    """
    from wisent.core.control.steering_methods.preflight import PreflightWarning

    method_lower = method.lower()

    # Unknown method - give generic advice
    compat_score = PREFLIGHT_COMPAT_DEFAULT
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

    if (structure_scores.get("sparse", 0) > PREFLIGHT_MANIFOLD_HIGH
            and method_lower not in ["sae", "sparse_steering"]):
        warnings.append(PreflightWarning(
            severity="info",
            message="Data shows sparse structure - few neurons are active",
            details=f"Sparse score: {structure_scores.get('sparse', 0):.2f}",
            suggestion="Consider SAE-based steering for more targeted "
                       "intervention",
        ))

    is_compatible = compat_score >= PREFLIGHT_COMPAT_THRESHOLD
    return is_compatible, compat_score, warnings

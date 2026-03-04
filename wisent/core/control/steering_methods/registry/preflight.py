"""
Pre-flight checks for steering method selection.

Analyzes activation geometry and warns if the chosen steering method
may not be optimal for the data structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import torch

from wisent.core.primitives.contrastive_pairs.diagnostics.control_vectors import (
    detect_geometry_structure,
    GeometryAnalysisConfig,
    GeometryAnalysisResult,
    StructureType,
)
from wisent.core.utils.config_tools.constants import (
    SEPARATOR_WIDTH_STANDARD,
    SCORE_RANGE_MIN,
)
from wisent.core.control.steering_methods._helpers.preflight_helpers import (
    PreflightThresholds,
    complete_method_compatibility_check,
)

__all__ = [
    "PreflightCheckResult",
    "PreflightWarning",
    "run_preflight_check",
    "check_method_compatibility",
]

# Mapping of structure types to recommended methods
STRUCTURE_TO_METHODS: Dict[StructureType, List[str]] = {
    StructureType.LINEAR: ["caa", "mean_diff"],
    StructureType.CONE: ["tecza", "grom"],
    StructureType.CLUSTER: ["cluster_steering", "grom"],
    StructureType.MANIFOLD: ["grom"],
    StructureType.SPARSE: ["sae", "sparse_steering"],
    StructureType.BIMODAL: ["tetno", "grom"],
    StructureType.ORTHOGONAL: ["ica_steering", "multi_caa"],
    StructureType.UNKNOWN: ["caa", "grom"],
}

# Methods that work reasonably well for any structure
UNIVERSAL_METHODS = ["grom", "caa"]

# Method descriptions for warnings
METHOD_DESCRIPTIONS: Dict[str, str] = {
    "caa": "Contrastive Activation Addition (single direction)",
    "mean_diff": "Mean Difference (simple single direction)",
    "tecza": "TECZA (multi-directional manifold)",
    "grom": "GROM (adaptive multi-component steering)",
    "tetno": "TETNO (conditional gating)",
    "sae": "Sparse Autoencoder based steering",
    "cluster_steering": "Cluster-based steering",
    "ica_steering": "ICA-based independent component steering",
    "multi_caa": "Multiple independent CAA vectors",
    "sparse_steering": "Sparse neuron targeting",
}


@dataclass
class PreflightWarning:
    """A single warning from pre-flight check."""
    severity: str  # "info", "warning", "error"
    message: str
    details: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class PreflightCheckResult:
    """Results from pre-flight steering method check."""
    chosen_method: str
    geometry_result: GeometryAnalysisResult
    is_compatible: bool
    compatibility_score: float
    warnings: List[PreflightWarning] = field(default_factory=list)
    recommended_methods: List[str] = field(default_factory=list)

    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return bool(self.warnings)

    def has_errors(self) -> bool:
        """Check if there are any error-level warnings."""
        return any(w.severity == "error" for w in self.warnings)

    def print_report(self) -> None:
        """Print a human-readable report."""
        from wisent.core.control.steering_methods._helpers.preflight_helpers import print_preflight_report
        print_preflight_report(self)


def run_preflight_check(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    chosen_method: str,
    config: Optional[GeometryAnalysisConfig] = None,
    *,
    thresholds: Optional[PreflightThresholds] = None,
) -> PreflightCheckResult:
    """Run pre-flight check for a steering method."""
    if thresholds is None:
        thresholds = PreflightThresholds()
    geo_result = detect_geometry_structure(pos_activations, neg_activations, config)
    is_compatible, compat_score, warnings = check_method_compatibility(
        chosen_method, geo_result, thresholds=thresholds,
    )
    recommended = STRUCTURE_TO_METHODS.get(geo_result.best_structure, UNIVERSAL_METHODS)
    return PreflightCheckResult(
        chosen_method=chosen_method,
        geometry_result=geo_result,
        is_compatible=is_compatible,
        compatibility_score=compat_score,
        warnings=warnings,
        recommended_methods=recommended,
    )


def check_method_compatibility(
    method: str,
    geo_result: GeometryAnalysisResult,
    *,
    thresholds: Optional[PreflightThresholds] = None,
) -> Tuple[bool, float, List[PreflightWarning]]:
    """Check if a steering method is compatible with detected geometry."""
    if thresholds is None:
        thresholds = PreflightThresholds()
    warnings: List[PreflightWarning] = []
    method_lower = method.lower()
    best_structure = geo_result.best_structure
    structure_scores = {name: s.score for name, s in geo_result.all_scores.items()}
    linear_score = structure_scores.get("linear", SCORE_RANGE_MIN)
    cone_score = structure_scores.get("cone", SCORE_RANGE_MIN)
    manifold_score = structure_scores.get("manifold", SCORE_RANGE_MIN)
    compat_score = thresholds.compat_min_compatible
    if method_lower in ["caa", "mean_diff"]:
        compat_score = _check_caa(linear_score, cone_score, manifold_score, best_structure, thresholds, warnings)
    elif method_lower == "tecza":
        compat_score = _check_tecza(linear_score, cone_score, thresholds, warnings)
    elif method_lower == "grom":
        compat_score = _check_grom(linear_score, manifold_score, thresholds, warnings)
    elif method_lower == "tetno":
        bimodal_score = structure_scores.get("bimodal", SCORE_RANGE_MIN)
        compat_score = _check_tetno(linear_score, bimodal_score, thresholds, warnings)
    else:
        return complete_method_compatibility_check(
            method, best_structure, structure_scores, warnings, STRUCTURE_TO_METHODS,
            preflight_compat_unknown_default=thresholds.compat_unknown_default,
            preflight_compat_min_compatible=thresholds.compat_min_compatible,
            preflight_sparse_high=thresholds.sparse_high,
        )
    is_compatible = compat_score >= thresholds.compat_min_compatible
    return is_compatible, compat_score, warnings


def _check_caa(linear_score, cone_score, manifold_score, best_structure, t, warnings):
    """CAA/mean_diff compatibility check."""
    if linear_score > t.linear_excellent:
        warnings.append(PreflightWarning(
            severity="info", message="Excellent choice - data has strong linear structure",
            details=f"Linear score: {linear_score:.2f}",
        ))
        return t.compat_score_caa_excellent
    if linear_score > t.linear_good:
        warnings.append(PreflightWarning(
            severity="info", message="Good choice - data has moderate linear structure",
            details=f"Linear score: {linear_score:.2f}",
        ))
        return t.compat_score_caa_good
    if cone_score > t.cone_good or manifold_score > t.manifold_high:
        warnings.append(PreflightWarning(
            severity="warning",
            message=f"CAA may miss important structure - data appears to be {best_structure.value}",
            details=f"Linear: {linear_score:.2f}, Cone: {cone_score:.2f}, Manifold: {manifold_score:.2f}",
            suggestion="Consider using TECZA or GROM for better coverage",
        ))
        return t.compat_score_caa_poor
    warnings.append(PreflightWarning(severity="info", message="CAA is a reasonable baseline choice"))
    return t.compat_score_caa_default


def _check_tecza(linear_score, cone_score, t, warnings):
    """TECZA compatibility check."""
    if cone_score > t.cone_good:
        warnings.append(PreflightWarning(
            severity="info", message="Excellent choice - data has cone structure",
            details=f"Cone score: {cone_score:.2f}",
        ))
        return t.compat_score_tecza_excellent
    if linear_score > t.linear_overkill:
        warnings.append(PreflightWarning(
            severity="warning", message="TECZA may be overkill - data is mostly linear",
            details=f"Linear score: {linear_score:.2f}",
            suggestion="CAA would be simpler and equally effective",
        ))
        return t.compat_score_tecza_overkill
    warnings.append(PreflightWarning(severity="info", message="TECZA is a good choice for multi-directional steering"))
    return t.compat_score_tecza_default


def _check_grom(linear_score, manifold_score, t, warnings):
    """GROM compatibility check."""
    if linear_score > t.linear_very_high:
        warnings.append(PreflightWarning(
            severity="info", message="GROM will adapt to linear structure (may simplify to CAA-like behavior)",
            details=f"Linear score: {linear_score:.2f}",
        ))
        return t.grom_default
    if manifold_score > t.manifold_high:
        warnings.append(PreflightWarning(
            severity="info", message="Excellent choice - data has manifold structure that GROM handles well",
            details=f"Manifold score: {manifold_score:.2f}",
        ))
        return t.grom_manifold_excellent
    warnings.append(PreflightWarning(severity="info", message="GROM is adaptive and should work well with detected structure"))
    return t.grom_default


def _check_tetno(linear_score, bimodal_score, t, warnings):
    """TETNO compatibility check."""
    if bimodal_score > t.bimodal_good:
        warnings.append(PreflightWarning(
            severity="info", message="Good choice - data shows bimodal characteristics",
            details=f"Bimodal score: {bimodal_score:.2f}",
        ))
        return t.tetno_bimodal
    if linear_score > t.linear_overkill:
        warnings.append(PreflightWarning(
            severity="warning", message="TETNO gating may not be necessary for linear structure",
            suggestion="Consider simpler CAA if gating is not needed",
        ))
        return t.tetno_linear_overkill
    return t.compat_score_tetno_default

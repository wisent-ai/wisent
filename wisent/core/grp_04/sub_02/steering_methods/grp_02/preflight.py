"""
Pre-flight checks for steering method selection.

Analyzes activation geometry and warns if the chosen steering method
may not be optimal for the data structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import torch

from wisent.core.contrastive_pairs.diagnostics.control_vectors import (
    detect_geometry_structure,
    GeometryAnalysisConfig,
    GeometryAnalysisResult,
    StructureType,
)
from wisent.core.constants import (
    PREFLIGHT_COMPAT_DEFAULT, PREFLIGHT_COMPAT_EXCELLENT, PREFLIGHT_COMPAT_GOOD,
    PREFLIGHT_COMPAT_ADAPTIVE, PREFLIGHT_COMPAT_POOR, PREFLIGHT_COMPAT_NEUTRAL,
    PREFLIGHT_COMPAT_THRESHOLD, PREFLIGHT_LINEAR_HIGH, PREFLIGHT_LINEAR_VERY_HIGH,
    PREFLIGHT_LINEAR_MODERATE, PREFLIGHT_CONE_HIGH, PREFLIGHT_MANIFOLD_HIGH,
    PREFLIGHT_BIMODAL_MODERATE,
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
    """The method user chose to use."""
    geometry_result: GeometryAnalysisResult
    """Full geometry analysis results."""
    is_compatible: bool
    """Whether chosen method is compatible with detected structure."""
    compatibility_score: float
    """0-1 score of how well method matches structure (1 = perfect match)."""
    warnings: List[PreflightWarning] = field(default_factory=list)
    """List of warnings/suggestions."""
    recommended_methods: List[str] = field(default_factory=list)
    """Methods recommended for this structure."""

    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    def has_errors(self) -> bool:
        """Check if there are any error-level warnings."""
        return any(w.severity == "error" for w in self.warnings)

    def print_report(self) -> None:
        """Print a human-readable report."""
        print("\n" + "=" * 60)
        print("STEERING METHOD PRE-FLIGHT CHECK")
        print("=" * 60)
        print(f"\nChosen Method: {self.chosen_method}")
        print(f"Detected Structure: {self.geometry_result.best_structure.value}")
        print(f"Structure Score: {self.geometry_result.best_score:.3f}")
        print(f"Compatibility: {'OK' if self.is_compatible else 'WARNING'} ({self.compatibility_score:.0%})")
        if self.recommended_methods:
            print(f"\nRecommended Methods: {', '.join(self.recommended_methods)}")
        if self.warnings:
            print(f"\n{'='*60}")
            print("WARNINGS")
            print("=" * 60)
            for w in self.warnings:
                icon = "ℹ️" if w.severity == "info" else "⚠️" if w.severity == "warning" else "❌"
                print(f"\n{icon} [{w.severity.upper()}] {w.message}")
                if w.details:
                    print(f"   Details: {w.details}")
                if w.suggestion:
                    print(f"   Suggestion: {w.suggestion}")
        print(f"\n{'='*60}")
        print(f"Recommendation: {self.geometry_result.recommendation}")
        print("=" * 60 + "\n")


def run_preflight_check(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    chosen_method: str,
    config: Optional[GeometryAnalysisConfig] = None,
) -> PreflightCheckResult:
    """Run pre-flight check for a steering method."""
    geo_result = detect_geometry_structure(pos_activations, neg_activations, config)
    is_compatible, compat_score, warnings = check_method_compatibility(
        chosen_method, geo_result
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
) -> Tuple[bool, float, List[PreflightWarning]]:
    """Check if a steering method is compatible with detected geometry."""
    warnings: List[PreflightWarning] = []
    method_lower = method.lower()
    best_structure = geo_result.best_structure
    structure_scores = {name: s.score for name, s in geo_result.all_scores.items()}
    linear_score = structure_scores.get("linear", 0)
    cone_score = structure_scores.get("cone", 0)
    manifold_score = structure_scores.get("manifold", 0)
    is_compatible = True
    compat_score = PREFLIGHT_COMPAT_DEFAULT
    if method_lower in ["caa", "mean_diff"]:
        if linear_score > PREFLIGHT_LINEAR_HIGH:
            compat_score = PREFLIGHT_COMPAT_EXCELLENT
            warnings.append(PreflightWarning(
                severity="info",
                message="Excellent choice - data has strong linear structure",
                details=f"Linear score: {linear_score:.2f}",
            ))
        elif linear_score > PREFLIGHT_LINEAR_MODERATE:
            compat_score = PREFLIGHT_COMPAT_GOOD
            warnings.append(PreflightWarning(
                severity="info",
                message="Good choice - data has moderate linear structure",
                details=f"Linear score: {linear_score:.2f}",
            ))
        elif cone_score > PREFLIGHT_CONE_HIGH or manifold_score > PREFLIGHT_MANIFOLD_HIGH:
            compat_score = PREFLIGHT_COMPAT_POOR
            is_compatible = False
            warnings.append(PreflightWarning(
                severity="warning",
                message=f"CAA may miss important structure - data appears to be {best_structure.value}",
                details=f"Linear: {linear_score:.2f}, Cone: {cone_score:.2f}, Manifold: {manifold_score:.2f}",
                suggestion="Consider using TECZA or GROM for better coverage",
            ))
        else:
            compat_score = PREFLIGHT_COMPAT_NEUTRAL
            warnings.append(PreflightWarning(severity="info", message="CAA is a reasonable baseline choice"))
    elif method_lower == "tecza":
        if cone_score > PREFLIGHT_CONE_HIGH:
            compat_score = PREFLIGHT_COMPAT_EXCELLENT
            warnings.append(PreflightWarning(
                severity="info", message="Excellent choice - data has cone structure",
                details=f"Cone score: {cone_score:.2f}",
            ))
        elif linear_score > PREFLIGHT_LINEAR_HIGH:
            compat_score = PREFLIGHT_COMPAT_NEUTRAL
            warnings.append(PreflightWarning(
                severity="warning", message="TECZA may be overkill - data is mostly linear",
                details=f"Linear score: {linear_score:.2f}",
                suggestion="CAA would be simpler and equally effective",
            ))
        else:
            compat_score = PREFLIGHT_COMPAT_GOOD
            warnings.append(PreflightWarning(severity="info", message="TECZA is a good choice for multi-directional steering"))
    elif method_lower == "grom":
        compat_score = PREFLIGHT_COMPAT_ADAPTIVE
        if linear_score > PREFLIGHT_LINEAR_VERY_HIGH:
            warnings.append(PreflightWarning(
                severity="info", message="GROM will adapt to linear structure (may simplify to CAA-like behavior)",
                details=f"Linear score: {linear_score:.2f}",
            ))
        elif manifold_score > PREFLIGHT_MANIFOLD_HIGH:
            compat_score = PREFLIGHT_COMPAT_EXCELLENT
            warnings.append(PreflightWarning(
                severity="info", message="Excellent choice - data has manifold structure that GROM handles well",
                details=f"Manifold score: {manifold_score:.2f}",
            ))
        else:
            warnings.append(PreflightWarning(severity="info", message="GROM is adaptive and should work well with detected structure"))
    elif method_lower == "tetno":
        bimodal_score = structure_scores.get("bimodal", 0)
        if bimodal_score > PREFLIGHT_BIMODAL_MODERATE:
            compat_score = PREFLIGHT_COMPAT_ADAPTIVE
            warnings.append(PreflightWarning(
                severity="info", message="Good choice - data shows bimodal characteristics",
                details=f"Bimodal score: {bimodal_score:.2f}",
            ))
        elif linear_score > PREFLIGHT_LINEAR_HIGH:
            compat_score = PREFLIGHT_COMPAT_NEUTRAL
            warnings.append(PreflightWarning(
                severity="warning", message="TETNO gating may not be necessary for linear structure",
                suggestion="Consider simpler CAA if gating is not needed",
            ))
        else:
            compat_score = PREFLIGHT_COMPAT_GOOD
    else:
        from wisent.core.steering_methods._helpers.preflight_helpers import complete_method_compatibility_check
        return complete_method_compatibility_check(method, best_structure, structure_scores, warnings, STRUCTURE_TO_METHODS)
    is_compatible = compat_score >= PREFLIGHT_COMPAT_THRESHOLD
    return is_compatible, compat_score, warnings

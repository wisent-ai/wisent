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


__all__ = [
    "PreflightCheckResult",
    "PreflightWarning",
    "run_preflight_check",
    "check_method_compatibility",
]


# Mapping of structure types to recommended methods
STRUCTURE_TO_METHODS: Dict[StructureType, List[str]] = {
    StructureType.LINEAR: ["caa", "mean_diff"],
    StructureType.CONE: ["prism", "titan"],
    StructureType.CLUSTER: ["cluster_steering", "titan"],
    StructureType.MANIFOLD: ["titan"],
    StructureType.SPARSE: ["sae", "sparse_steering"],
    StructureType.BIMODAL: ["pulse", "titan"],
    StructureType.ORTHOGONAL: ["ica_steering", "multi_caa"],
    StructureType.UNKNOWN: ["caa", "titan"],
}

# Methods that work reasonably well for any structure
UNIVERSAL_METHODS = ["titan", "caa"]

# Method descriptions for warnings
METHOD_DESCRIPTIONS: Dict[str, str] = {
    "caa": "Contrastive Activation Addition (single direction)",
    "mean_diff": "Mean Difference (simple single direction)",
    "prism": "PRISM (multi-directional manifold)",
    "titan": "TITAN (adaptive multi-component steering)",
    "pulse": "PULSE (conditional gating)",
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
    """
    Run pre-flight check for a steering method.
    
    Analyzes the geometry of activation data and checks if the chosen
    steering method is appropriate.
    
    Arguments:
        pos_activations: Positive example activations [N_pos, hidden_dim]
        neg_activations: Negative example activations [N_neg, hidden_dim]
        chosen_method: Name of the steering method to use
        config: Optional geometry analysis config
        
    Returns:
        PreflightCheckResult with compatibility info and warnings
    """
    # Run geometry analysis
    geo_result = detect_geometry_structure(pos_activations, neg_activations, config)
    
    # Check compatibility
    is_compatible, compat_score, warnings = check_method_compatibility(
        chosen_method, geo_result
    )
    
    # Get recommended methods
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
    """
    Check if a steering method is compatible with detected geometry.
    
    Returns:
        Tuple of (is_compatible, compatibility_score, warnings)
    """
    warnings: List[PreflightWarning] = []
    method_lower = method.lower()
    
    best_structure = geo_result.best_structure
    structure_scores = {name: s.score for name, s in geo_result.all_scores.items()}
    
    linear_score = structure_scores.get("linear", 0)
    cone_score = structure_scores.get("cone", 0)
    manifold_score = structure_scores.get("manifold", 0)
    
    # Default compatibility
    is_compatible = True
    compat_score = 0.5
    
    # Check specific methods
    if method_lower in ["caa", "mean_diff"]:
        # CAA/mean_diff is optimal for linear, suboptimal for complex structures
        if linear_score > 0.8:
            compat_score = 1.0
            warnings.append(PreflightWarning(
                severity="info",
                message="Excellent choice - data has strong linear structure",
                details=f"Linear score: {linear_score:.2f}",
            ))
        elif linear_score > 0.5:
            compat_score = 0.7
            warnings.append(PreflightWarning(
                severity="info",
                message="Good choice - data has moderate linear structure",
                details=f"Linear score: {linear_score:.2f}",
            ))
        elif cone_score > 0.7 or manifold_score > 0.8:
            compat_score = 0.3
            is_compatible = False
            warnings.append(PreflightWarning(
                severity="warning",
                message=f"CAA may miss important structure - data appears to be {best_structure.value}",
                details=f"Linear: {linear_score:.2f}, Cone: {cone_score:.2f}, Manifold: {manifold_score:.2f}",
                suggestion="Consider using PRISM or TITAN for better coverage",
            ))
        else:
            compat_score = 0.5
            warnings.append(PreflightWarning(
                severity="info",
                message="CAA is a reasonable baseline choice",
            ))
    
    elif method_lower == "prism":
        # PRISM is optimal for cone structure
        if cone_score > 0.7:
            compat_score = 1.0
            warnings.append(PreflightWarning(
                severity="info",
                message="Excellent choice - data has cone structure",
                details=f"Cone score: {cone_score:.2f}",
            ))
        elif linear_score > 0.8:
            compat_score = 0.5
            warnings.append(PreflightWarning(
                severity="warning",
                message="PRISM may be overkill - data is mostly linear",
                details=f"Linear score: {linear_score:.2f}",
                suggestion="CAA would be simpler and equally effective",
            ))
        else:
            compat_score = 0.7
            warnings.append(PreflightWarning(
                severity="info",
                message="PRISM is a good choice for multi-directional steering",
            ))
    
    elif method_lower == "titan":
        # TITAN adapts to structure, always reasonable
        compat_score = 0.9  # High baseline
        
        if linear_score > 0.9:
            warnings.append(PreflightWarning(
                severity="info",
                message="TITAN will adapt to linear structure (may simplify to CAA-like behavior)",
                details=f"Linear score: {linear_score:.2f}",
            ))
        elif manifold_score > 0.8:
            compat_score = 1.0
            warnings.append(PreflightWarning(
                severity="info",
                message="Excellent choice - data has manifold structure that TITAN handles well",
                details=f"Manifold score: {manifold_score:.2f}",
            ))
        else:
            warnings.append(PreflightWarning(
                severity="info",
                message="TITAN is adaptive and should work well with detected structure",
            ))
    
    elif method_lower == "pulse":
        # PULSE is good for bimodal/conditional steering
        bimodal_score = structure_scores.get("bimodal", 0)
        if bimodal_score > 0.6:
            compat_score = 0.9
            warnings.append(PreflightWarning(
                severity="info",
                message="Good choice - data shows bimodal characteristics",
                details=f"Bimodal score: {bimodal_score:.2f}",
            ))
        elif linear_score > 0.8:
            compat_score = 0.5
            warnings.append(PreflightWarning(
                severity="warning",
                message="PULSE gating may not be necessary for linear structure",
                suggestion="Consider simpler CAA if gating is not needed",
            ))
        else:
            compat_score = 0.7
    
    else:
        # Unknown method - give generic advice
        warnings.append(PreflightWarning(
            severity="info",
            message=f"Unknown method '{method}' - cannot provide specific compatibility check",
            suggestion=f"Recommended methods for {best_structure.value}: {', '.join(STRUCTURE_TO_METHODS.get(best_structure, ['caa']))}",
        ))
    
    # Add structure-specific warnings
    if best_structure == StructureType.MANIFOLD and method_lower not in ["titan"]:
        warnings.append(PreflightWarning(
            severity="warning",
            message="Data has non-linear manifold structure",
            details=f"Intrinsic dimensionality much lower than ambient dimension",
            suggestion="TITAN with learned gating may capture this structure better",
        ))
    
    if structure_scores.get("sparse", 0) > 0.8 and method_lower not in ["sae", "sparse_steering"]:
        warnings.append(PreflightWarning(
            severity="info",
            message="Data shows sparse structure - few neurons are active",
            details=f"Sparse score: {structure_scores.get('sparse', 0):.2f}",
            suggestion="Consider SAE-based steering for more targeted intervention",
        ))
    
    return is_compatible, compat_score, warnings

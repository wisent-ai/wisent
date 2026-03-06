"""
Pre-flight checks for steering method selection.

Analyzes activation geometry and reports raw geometry scores for the
chosen steering method. Does not make unvalidated method recommendations.
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

# Method-to-structure mapping: which geometry metric is relevant for each method.
# This is NOT a heuristic — it just selects which raw score to report.
_METHOD_STRUCTURE_KEY: Dict[str, str] = {
    "caa": "linear",
    "mean_diff": "linear",
    "tecza": "cone",
    "grom": "manifold",
    "tetno": "bimodal",
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
    thresholds: PreflightThresholds,
) -> PreflightCheckResult:
    """Run pre-flight check for a steering method."""
    geo_result = detect_geometry_structure(pos_activations, neg_activations, config)
    is_compatible, compat_score, warnings = check_method_compatibility(
        chosen_method, geo_result, thresholds=thresholds,
    )
    return PreflightCheckResult(
        chosen_method=chosen_method,
        geometry_result=geo_result,
        is_compatible=is_compatible,
        compatibility_score=compat_score,
        warnings=warnings,
        recommended_methods=[],
    )


def check_method_compatibility(
    method: str,
    geo_result: GeometryAnalysisResult,
    *,
    thresholds: PreflightThresholds,
) -> Tuple[bool, float, List[PreflightWarning]]:
    """Check if a steering method is compatible with detected geometry.

    Returns raw geometry scores instead of transforming them through
    unvalidated heuristic decision trees. The compatibility score is
    the raw geometry score for the structure relevant to the method.
    """
    warnings: List[PreflightWarning] = []
    method_lower = method.lower()
    structure_scores = {name: s.score for name, s in geo_result.all_scores.items()}

    structure_key = _METHOD_STRUCTURE_KEY.get(method_lower)

    if structure_key is not None:
        compat_score = structure_scores.get(structure_key, SCORE_RANGE_MIN)
        score_details = ", ".join(f"{k}={v:.2f}" for k, v in structure_scores.items())
        warnings.append(PreflightWarning(
            severity="info",
            message=f"Geometry scores for method '{method}'",
            details=f"Relevant score ({structure_key}): {compat_score:.2f}. "
                    f"All scores: {score_details}",
        ))
        is_compatible = compat_score >= thresholds.compat_min_compatible
        return is_compatible, compat_score, warnings

    # Unknown method — delegate to helper
    return complete_method_compatibility_check(
        method, geo_result.best_structure, structure_scores, warnings,
        preflight_compat_min_compatible=thresholds.compat_min_compatible,
    )

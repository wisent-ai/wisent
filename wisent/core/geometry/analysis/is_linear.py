"""Rigorous linearity validation for the Linear Representation Hypothesis.

Implements econometric-style diagnostics to test whether the LRH holds:
1. Statistical test on linear-nonlinear accuracy gap (paired t-test)
2. Residual analysis - check if linear probe errors cluster
3. Ramsey-style polynomial feature test
4. Bootstrap confidence intervals on the gap
5. Cross-context transfer validation (Lampinen-style)
"""
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import torch
from scipy import stats

from .linearity_utils import (
    compute_probe_accuracies,
    analyze_residuals,
    bootstrap_gap_ci,
    ramsey_polynomial_test,
    test_cross_context_linearity,
)


@dataclass
class LinearityTestResult:
    """Complete result of linearity validation."""
    is_linear: bool
    confidence: float
    diagnosis: str

    linear_accuracy: float
    nonlinear_accuracy: float
    gap: float
    gap_ci_lower: float
    gap_ci_upper: float

    t_statistic: float
    p_value: float

    residual_silhouette: float
    residuals_cluster: bool

    ramsey_improvement: float
    ramsey_significant: bool

    n_diagnostics_passed: int
    n_diagnostics_total: int
    diagnostics: Dict[str, Any]


def test_linearity(
    pos: torch.Tensor,
    neg: torch.Tensor,
    gap_threshold: float = 0.05,
    p_threshold: float = 0.05,
    residual_threshold: float = 0.3,
    ramsey_threshold: float = 0.03,
    n_bootstrap: int = 100,
    contexts: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    random_state: int = 42,
) -> LinearityTestResult:
    """Run complete linearity validation suite.

    5 diagnostics are run:
    1. Gap statistical test - is gap significantly > 0? (paired t-test on fold scores)
    2. Residual analysis - do linear probe errors cluster?
    3. Ramsey test - do polynomial features improve fit?
    4. Bootstrap CI - does CI on gap exclude 0?
    5. Cross-context - do directions transfer? (if contexts provided)

    Linearity is diagnosed if MAJORITY of applicable diagnostics pass.

    Args:
        pos: Positive activations [n_samples, dim]
        neg: Negative activations [n_samples, dim]
        gap_threshold: Minimum gap to consider meaningful
        p_threshold: Significance level for statistical tests
        residual_threshold: Silhouette threshold for residual clustering
        ramsey_threshold: Minimum improvement for Ramsey test to fail linearity
        n_bootstrap: Number of bootstrap iterations
        contexts: Optional list of (pos, neg) tuples for cross-context test
        random_state: Random seed

    Returns:
        LinearityTestResult with all diagnostics and final diagnosis.
    """
    diagnostics = {}

    linear_acc, nonlinear_acc, linear_scores, nonlinear_scores = compute_probe_accuracies(
        pos, neg, n_splits=5, random_state=random_state
    )
    gap = nonlinear_acc - linear_acc

    diagnostics["probe_accuracies"] = {
        "linear": linear_acc,
        "nonlinear": nonlinear_acc,
        "gap": gap,
        "linear_fold_scores": linear_scores.tolist(),
        "nonlinear_fold_scores": nonlinear_scores.tolist(),
    }

    t_stat, p_value = stats.ttest_rel(nonlinear_scores, linear_scores, alternative="greater")
    gap_significant = p_value < p_threshold and gap > gap_threshold
    diagnostics["gap_test"] = {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": gap_significant,
    }

    gap_mean, ci_lower, ci_upper = bootstrap_gap_ci(
        pos, neg, n_bootstrap=n_bootstrap, random_state=random_state
    )
    ci_excludes_zero = ci_lower > 0
    diagnostics["bootstrap_ci"] = {
        "gap_mean": gap_mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "excludes_zero": ci_excludes_zero,
    }

    residual_result = analyze_residuals(pos, neg, random_state=random_state)
    residuals_cluster = (
        residual_result["clusters_found"]
        and residual_result["residual_silhouette"] > residual_threshold
    )
    diagnostics["residual_analysis"] = {
        **residual_result,
        "clusters": residuals_cluster,
    }

    ramsey_result = ramsey_polynomial_test(pos, neg, random_state=random_state)
    ramsey_significant = ramsey_result["improvement"] > ramsey_threshold
    diagnostics["ramsey_test"] = {
        **ramsey_result,
        "significant": ramsey_significant,
    }

    diagnostics_passed = 0
    diagnostics_total = 4

    if not gap_significant:
        diagnostics_passed += 1
    if not ci_excludes_zero:
        diagnostics_passed += 1
    if not residuals_cluster:
        diagnostics_passed += 1
    if not ramsey_significant:
        diagnostics_passed += 1

    if contexts is not None and len(contexts) >= 2:
        cross_context_result = test_cross_context_linearity(contexts, random_state=random_state)
        diagnostics["cross_context"] = cross_context_result
        diagnostics_total += 1
        if cross_context_result["transfer_accuracy"] > 0.7:
            diagnostics_passed += 1

    is_linear = diagnostics_passed >= (diagnostics_total / 2)
    confidence = diagnostics_passed / diagnostics_total

    if is_linear:
        if confidence >= 0.8:
            diagnosis = "LINEAR_HIGH_CONFIDENCE"
        else:
            diagnosis = "LINEAR_MODERATE_CONFIDENCE"
    else:
        if confidence <= 0.2:
            diagnosis = "NONLINEAR_HIGH_CONFIDENCE"
        else:
            diagnosis = "NONLINEAR_MODERATE_CONFIDENCE"

    return LinearityTestResult(
        is_linear=is_linear,
        confidence=confidence,
        diagnosis=diagnosis,
        linear_accuracy=linear_acc,
        nonlinear_accuracy=nonlinear_acc,
        gap=gap,
        gap_ci_lower=ci_lower,
        gap_ci_upper=ci_upper,
        t_statistic=float(t_stat),
        p_value=float(p_value),
        residual_silhouette=residual_result["residual_silhouette"],
        residuals_cluster=residuals_cluster,
        ramsey_improvement=ramsey_result["improvement"],
        ramsey_significant=ramsey_significant,
        n_diagnostics_passed=diagnostics_passed,
        n_diagnostics_total=diagnostics_total,
        diagnostics=diagnostics,
    )


def summarize_linearity_result(result: LinearityTestResult) -> str:
    """Generate human-readable summary of linearity test."""
    lines = [
        f"Linearity Diagnosis: {result.diagnosis}",
        f"  Is Linear: {result.is_linear} (confidence: {result.confidence:.0%})",
        f"  Diagnostics Passed: {result.n_diagnostics_passed}/{result.n_diagnostics_total}",
        "",
        "Probe Accuracies:",
        f"  Linear: {result.linear_accuracy:.3f}",
        f"  Nonlinear: {result.nonlinear_accuracy:.3f}",
        f"  Gap: {result.gap:.3f} (95% CI: [{result.gap_ci_lower:.3f}, {result.gap_ci_upper:.3f}])",
        "",
        "Statistical Tests:",
        f"  Gap t-test: t={result.t_statistic:.2f}, p={result.p_value:.4f}",
        f"  Ramsey improvement: {result.ramsey_improvement:.3f} (significant: {result.ramsey_significant})",
        f"  Residual silhouette: {result.residual_silhouette:.3f} (clusters: {result.residuals_cluster})",
    ]
    return "\n".join(lines)

"""Main diagnostic runner combining all curse of dimensionality checks."""

import numpy as np
import torch
from typing import Optional
from dataclasses import dataclass

from .sample_adequacy import compute_sample_dimension_ratio, compute_effective_sample_size
from .power_analysis import compute_statistical_power
from .shrinkage import compute_shrinkage_covariance


@dataclass
class DimensionalityDiagnostics:
    """Complete dimensionality diagnostic results."""
    n_samples: int
    ambient_dim: int
    effective_dim: float
    sample_dim_ratio: float
    effective_sample_dim_ratio: float
    ratio_adequate: bool
    ratio_warning: Optional[str]
    effective_n: float
    eigenvalue_decay_rate: float
    condition_number: float
    detectable_effect_size: float
    power_at_medium_effect: float
    required_n_for_80_power: int
    degrees_of_freedom: int
    shrinkage_intensity: float
    covariance_estimator: str
    curse_severity: str
    recommendations: list


def run_dimensionality_diagnostics(
    pos: torch.Tensor,
    neg: torch.Tensor,
    alpha: float = 0.05,
) -> DimensionalityDiagnostics:
    """Run complete curse of dimensionality diagnostics."""
    pos_np = pos.cpu().numpy() if isinstance(pos, torch.Tensor) else pos
    neg_np = neg.cpu().numpy() if isinstance(neg, torch.Tensor) else neg

    n = min(len(pos_np), len(neg_np))
    X = np.vstack([pos_np[:n], neg_np[:n]])
    diff = pos_np[:n] - neg_np[:n]

    n_samples = len(X)
    ambient_dim = X.shape[1]

    eff_sample = compute_effective_sample_size(diff, method="eigenvalue")
    effective_dim = eff_sample["effective_dim"]
    effective_n = eff_sample["effective_n"]

    ratio_check = compute_sample_dimension_ratio(n_samples, ambient_dim, effective_dim)
    power_analysis = compute_statistical_power(n_samples, effective_dim, alpha=alpha)
    _, shrinkage_info = compute_shrinkage_covariance(diff)

    severity = ratio_check["severity"]
    recommendations = []

    if severity in ["moderate", "severe"]:
        recommendations.append(f"Increase sample size: need {power_analysis['required_n_for_80_power']} for 80% power")
    if shrinkage_info["shrinkage_intensity"] > 0.3:
        recommendations.append("Use regularized methods (shrinkage covariance, L2 regularization)")
    if power_analysis["power_at_medium_effect"] < 0.5:
        recommendations.append("Current sample can only reliably detect large effects (d > 0.8)")
    if eff_sample["condition_number"] > 1000:
        recommendations.append("High condition number: consider dimensionality reduction")
    if not recommendations:
        recommendations.append("Sample size appears adequate for analysis")

    return DimensionalityDiagnostics(
        n_samples=n_samples,
        ambient_dim=ambient_dim,
        effective_dim=effective_dim,
        sample_dim_ratio=ratio_check["sample_dim_ratio"],
        effective_sample_dim_ratio=ratio_check["effective_sample_dim_ratio"],
        ratio_adequate=ratio_check["ratio_adequate"],
        ratio_warning=ratio_check["ratio_warning"],
        effective_n=effective_n,
        eigenvalue_decay_rate=eff_sample["eigenvalue_decay_rate"],
        condition_number=eff_sample["condition_number"],
        detectable_effect_size=power_analysis["minimum_detectable_effect"],
        power_at_medium_effect=power_analysis["power_at_medium_effect"],
        required_n_for_80_power=power_analysis["required_n_for_80_power"],
        degrees_of_freedom=power_analysis["degrees_of_freedom"],
        shrinkage_intensity=shrinkage_info["shrinkage_intensity"],
        covariance_estimator=shrinkage_info["covariance_estimator"],
        curse_severity=severity,
        recommendations=recommendations,
    )


def format_diagnostics_report(diag: DimensionalityDiagnostics) -> str:
    """Format diagnostics as human-readable report."""
    lines = [
        "=" * 60,
        "CURSE OF DIMENSIONALITY DIAGNOSTICS",
        "=" * 60,
        "",
        "SAMPLE SIZE ADEQUACY:",
        f"  Samples (n):          {diag.n_samples}",
        f"  Ambient dimension:    {diag.ambient_dim}",
        f"  Effective dimension:  {diag.effective_dim:.1f}",
        f"  n/d ratio:            {diag.sample_dim_ratio:.2f}",
        f"  n/d_eff ratio:        {diag.effective_sample_dim_ratio:.2f}",
        f"  Adequate:             {'Yes' if diag.ratio_adequate else 'NO'}",
    ]
    if diag.ratio_warning:
        lines.append(f"  WARNING: {diag.ratio_warning}")
    lines.extend([
        "",
        "EFFECTIVE SAMPLE SIZE:",
        f"  Effective n:          {diag.effective_n:.1f}",
        f"  Eigenvalue decay:     {diag.eigenvalue_decay_rate:.3f}",
        f"  Condition number:     {diag.condition_number:.1f}",
        "",
        "STATISTICAL POWER:",
        f"  Degrees of freedom:   {diag.degrees_of_freedom}",
        f"  Min detectable effect:{diag.detectable_effect_size:.2f}",
        f"  Power (medium effect):{diag.power_at_medium_effect:.0%}",
        f"  Required n (80% pwr): {diag.required_n_for_80_power}",
        "",
        "COVARIANCE ESTIMATION:",
        f"  Shrinkage intensity:  {diag.shrinkage_intensity:.2f}",
        f"  Estimator:            {diag.covariance_estimator}",
        "",
        f"OVERALL SEVERITY: {diag.curse_severity.upper()}",
        "",
        "RECOMMENDATIONS:",
    ])
    for rec in diag.recommendations:
        lines.append(f"  - {rec}")
    lines.append("=" * 60)
    return "\n".join(lines)

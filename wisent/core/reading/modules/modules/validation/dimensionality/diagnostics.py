"""Main diagnostic runner combining all curse of dimensionality checks."""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional

from .sample_adequacy import compute_sample_dimension_ratio, compute_effective_sample_size
from .power_analysis import compute_statistical_power
from wisent.core.utils.config_tools.constants import (
    STAT_ALPHA,
    EFFECT_SIZE_LARGE,
    SEPARATOR_WIDTH_STANDARD,
)
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
    alpha: float = STAT_ALPHA,
    *,
    shrinkage_low: float | None = None,
    shrinkage_moderate: float | None = None,
    shrinkage_high: float | None = None,
    power_excellent_threshold: float | None = None,
    power_low_threshold: float | None = None,
    mde_small_threshold: float | None = None,
    mde_medium_threshold: float | None = None,
    mde_large_threshold: float | None = None,
    shrinkage_intensity_threshold: float | None = None,
    statistical_power_threshold: float | None = None,
    sample_ratio_adequate: int | None = None,
    sample_ratio_good: int | None = None,
    sample_ratio_acceptable: int | None = None,
    sample_ratio_marginal: int | None = None,
) -> DimensionalityDiagnostics:
    """Run complete curse of dimensionality diagnostics."""
    if shrinkage_low is None:
        raise ValueError("shrinkage_low is required")
    if shrinkage_moderate is None:
        raise ValueError("shrinkage_moderate is required")
    if shrinkage_high is None:
        raise ValueError("shrinkage_high is required")
    if power_excellent_threshold is None:
        raise ValueError("power_excellent_threshold is required")
    if power_low_threshold is None:
        raise ValueError("power_low_threshold is required")
    if mde_small_threshold is None:
        raise ValueError("mde_small_threshold is required")
    if mde_medium_threshold is None:
        raise ValueError("mde_medium_threshold is required")
    if mde_large_threshold is None:
        raise ValueError("mde_large_threshold is required")
    if shrinkage_intensity_threshold is None:
        raise ValueError("shrinkage_intensity_threshold is required")
    if statistical_power_threshold is None:
        raise ValueError("statistical_power_threshold is required")
    if sample_ratio_adequate is None:
        raise ValueError("sample_ratio_adequate is required")
    if sample_ratio_good is None:
        raise ValueError("sample_ratio_good is required")
    if sample_ratio_acceptable is None:
        raise ValueError("sample_ratio_acceptable is required")
    if sample_ratio_marginal is None:
        raise ValueError("sample_ratio_marginal is required")
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

    ratio_check = compute_sample_dimension_ratio(
        n_samples, ambient_dim, effective_dim,
        sample_ratio_adequate=sample_ratio_adequate, sample_ratio_good=sample_ratio_good,
        sample_ratio_acceptable=sample_ratio_acceptable, sample_ratio_marginal=sample_ratio_marginal,
    )
    power_analysis = compute_statistical_power(
        n_samples, effective_dim, alpha=alpha,
        power_excellent_threshold=power_excellent_threshold,
        power_low_threshold=power_low_threshold,
        mde_small_threshold=mde_small_threshold,
        mde_medium_threshold=mde_medium_threshold,
        mde_large_threshold=mde_large_threshold,
    )
    _, shrinkage_info = compute_shrinkage_covariance(
        diff, shrinkage_low=shrinkage_low, shrinkage_moderate=shrinkage_moderate, shrinkage_high=shrinkage_high,
    )

    severity = ratio_check["severity"]
    recommendations = []

    if severity in ["moderate", "severe"]:
        recommendations.append(f"Increase sample size: need {power_analysis['required_n_for_80_power']} for 80% power")
    if shrinkage_info["shrinkage_intensity"] > shrinkage_intensity_threshold:
        recommendations.append("Use regularized methods (shrinkage covariance, L2 regularization)")
    if power_analysis["power_at_medium_effect"] < statistical_power_threshold:
        recommendations.append(f"Current sample can only reliably detect large effects (d > {EFFECT_SIZE_LARGE})")
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
        "=" * SEPARATOR_WIDTH_STANDARD,
        "CURSE OF DIMENSIONALITY DIAGNOSTICS",
        "=" * SEPARATOR_WIDTH_STANDARD,
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
    lines.append("=" * SEPARATOR_WIDTH_STANDARD)
    return "\n".join(lines)

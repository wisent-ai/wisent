"""Statistical power analysis with degrees of freedom calculations."""

import numpy as np
from typing import Dict, Any
from scipy import stats


def compute_statistical_power(
    n_samples: int,
    effective_dim: float,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> Dict[str, Any]:
    """
    Compute statistical power for detecting effects in high-dimensional setting.

    Uses approximations for:
    - Two-sample t-test equivalent (for classification)
    - Effect size conventions (Cohen's d: 0.2=small, 0.5=medium, 0.8=large)

    Returns:
    - Minimum detectable effect size at 80% power
    - Power to detect medium effect (d=0.5)
    - Required n for 80% power at medium effect
    - Degrees of freedom
    """
    df_simple = max(1, n_samples - 2)
    df_adjusted = max(1, n_samples - effective_dim - 1)
    df = min(df_simple, df_adjusted)

    t_crit = stats.t.ppf(1 - alpha / 2, df)
    n_per_group = n_samples / 2

    # Minimum detectable effect size
    if n_per_group > 0:
        ncp_target = stats.t.ppf(target_power, df)
        mde = (t_crit + ncp_target) * np.sqrt(2 / n_per_group)
    else:
        mde = float('inf')

    # Power at different effect sizes
    d_small, d_medium, d_large = 0.2, 0.5, 0.8
    if n_per_group > 0:
        power_small = 1 - stats.t.cdf(t_crit, df, loc=d_small * np.sqrt(n_per_group / 2))
        power_medium = 1 - stats.t.cdf(t_crit, df, loc=d_medium * np.sqrt(n_per_group / 2))
        power_large = 1 - stats.t.cdf(t_crit, df, loc=d_large * np.sqrt(n_per_group / 2))
    else:
        power_small = power_medium = power_large = 0.0

    # Required n for 80% power at medium effect
    required_n = 8
    for test_n in range(8, 10000, 2):
        test_n_per_group = test_n / 2
        test_df = max(1, test_n - 2)
        test_t_crit = stats.t.ppf(1 - alpha / 2, test_df)
        test_power = 1 - stats.t.cdf(test_t_crit, test_df, loc=d_medium * np.sqrt(test_n_per_group / 2))
        if test_power >= target_power:
            required_n = test_n
            break

    return {
        "degrees_of_freedom": int(df),
        "degrees_of_freedom_simple": int(df_simple),
        "degrees_of_freedom_adjusted": int(df_adjusted),
        "minimum_detectable_effect": float(mde),
        "power_at_small_effect": float(power_small),
        "power_at_medium_effect": float(power_medium),
        "power_at_large_effect": float(power_large),
        "required_n_for_80_power": int(required_n),
        "alpha": alpha,
        "interpretation": _interpret_power(power_medium, mde),
    }


def _interpret_power(power_medium: float, mde: float) -> str:
    """Generate human-readable power interpretation."""
    if power_medium >= 0.9:
        power_str = "excellent"
    elif power_medium >= 0.8:
        power_str = "adequate"
    elif power_medium >= 0.5:
        power_str = "moderate"
    else:
        power_str = "low"

    if mde <= 0.3:
        mde_str = "can detect small effects"
    elif mde <= 0.6:
        mde_str = "can detect medium effects"
    elif mde <= 1.0:
        mde_str = "can only detect large effects"
    else:
        mde_str = "severely underpowered"

    return f"Power is {power_str} ({power_medium:.0%}); {mde_str} (MDE={mde:.2f})"

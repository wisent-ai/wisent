"""Statistical power analysis with degrees of freedom calculations."""

import numpy as np
from typing import Dict, Any
from scipy import stats
from wisent.core.utils.config_tools.constants import (
    STAT_ALPHA, TARGET_POWER,
    EFFECT_SIZE_SMALL, EFFECT_SIZE_MEDIUM, EFFECT_SIZE_LARGE,
    POWER_ADEQUATE_THRESHOLD,
)


def compute_statistical_power(
    n_samples: int,
    effective_dim: float,
    power_analysis_min_n: int,
    power_analysis_max_n: int,
    power_analysis_step: int,
    alpha: float = STAT_ALPHA,
    target_power: float = TARGET_POWER,
    *,
    power_excellent_threshold: float | None = None,
    power_low_threshold: float | None = None,
    mde_small_threshold: float | None = None,
    mde_medium_threshold: float | None = None,
    mde_large_threshold: float | None = None,
) -> Dict[str, Any]:
    """
    Compute statistical power for detecting effects in high-dimensional setting.

    Uses approximations for:
    - Two-sample t-test equivalent (for classification)
    - Effect size conventions (Cohen's d: EFFECT_SIZE_SMALL=small, EFFECT_SIZE_MEDIUM=medium, EFFECT_SIZE_LARGE=large)

    Returns:
    - Minimum detectable effect size at TARGET_POWER power
    - Power to detect medium effect (d=EFFECT_SIZE_MEDIUM)
    - Required n for eighty-percent power at medium effect
    - Degrees of freedom
    """
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
    d_small, d_medium, d_large = EFFECT_SIZE_SMALL, EFFECT_SIZE_MEDIUM, EFFECT_SIZE_LARGE
    if n_per_group > 0:
        power_small = 1 - stats.t.cdf(t_crit, df, loc=d_small * np.sqrt(n_per_group / 2))
        power_medium = 1 - stats.t.cdf(t_crit, df, loc=d_medium * np.sqrt(n_per_group / 2))
        power_large = 1 - stats.t.cdf(t_crit, df, loc=d_large * np.sqrt(n_per_group / 2))
    else:
        power_small = power_medium = power_large = 0.0

    # Required n for 80% power at medium effect
    required_n = power_analysis_min_n
    for test_n in range(power_analysis_min_n, power_analysis_max_n, power_analysis_step):
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
        "interpretation": _interpret_power(
            power_medium, mde,
            power_excellent_threshold=power_excellent_threshold,
            power_low_threshold=power_low_threshold,
            mde_small_threshold=mde_small_threshold,
            mde_medium_threshold=mde_medium_threshold,
            mde_large_threshold=mde_large_threshold,
        ),
    }


def _interpret_power(
    power_medium: float, mde: float, *,
    power_excellent_threshold: float,
    power_low_threshold: float,
    mde_small_threshold: float,
    mde_medium_threshold: float,
    mde_large_threshold: float,
) -> str:
    """Generate human-readable power interpretation."""
    if power_medium >= power_excellent_threshold:
        power_str = "excellent"
    elif power_medium >= POWER_ADEQUATE_THRESHOLD:
        power_str = "adequate"
    elif power_medium >= power_low_threshold:
        power_str = "moderate"
    else:
        power_str = "low"

    if mde <= mde_small_threshold:
        mde_str = "can detect small effects"
    elif mde <= mde_medium_threshold:
        mde_str = "can detect medium effects"
    elif mde <= mde_large_threshold:
        mde_str = "can only detect large effects"
    else:
        mde_str = "severely underpowered"

    return f"Power is {power_str} ({power_medium:.0%}); {mde_str} (MDE={mde:.2f})"

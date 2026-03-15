"""Core scoring orchestrator: rank all methods by predicted effectiveness."""
from typing import Any, Dict, List, Optional, Tuple

from wisent.core.utils.config_tools.constants import (
    COMBO_OFFSET, SCORE_RANGE_MIN, SCORE_RANGE_MAX,
    SCORER_SIGNAL_Z_NORMALIZER, SCORER_TOP_RECOMMENDED,
)


def flatten_zwiad_results(
    signal: Any, geometry: Any, decomposition: Any,
    editability: Optional[Any] = None,
) -> Dict[str, float]:
    """Flatten ZWIAD test result dataclasses into a single metric dict.

    Accepts SignalTestResult, GeometryTestResult, DecompositionTestResult,
    and EditabilityTestResult via duck typing (no import needed).
    """
    flat: Dict[str, float] = {}
    flat["signal_strength"] = min(
        signal.max_z_score / SCORER_SIGNAL_Z_NORMALIZER, SCORE_RANGE_MAX,
    )
    flat["min_p_value"] = signal.min_p_value
    _extract_permutation_metrics(signal.permutation_metrics, flat)
    _extract_geometry_metrics(geometry, flat)
    _extract_decomposition_metrics(decomposition, flat)
    if editability is not None:
        _extract_editability_metrics(editability, flat)
    return flat


def _extract_permutation_metrics(
    perm: Dict[str, Any], flat: Dict[str, float],
) -> None:
    """Pull real metric values from signal permutation results."""
    for key, entry in perm.items():
        if isinstance(entry, dict):
            real_val = entry.get("real")
            if real_val is not None:
                flat[key] = float(real_val)


def _extract_geometry_metrics(geo: Any, flat: Dict[str, float]) -> None:
    """Pull geometry test fields into flat dict."""
    flat["linear_probe_accuracy"] = geo.linear_accuracy
    flat["nonlinear_accuracy"] = geo.nonlinear_accuracy
    flat["gap"] = geo.gap
    _optional_attr = [
        "confidence", "t_statistic", "residual_silhouette", "ramsey_improvement",
    ]
    for attr in _optional_attr:
        val = getattr(geo, attr, None)
        if val is not None:
            flat[attr] = float(val)


def _extract_decomposition_metrics(dec: Any, flat: Dict[str, float]) -> None:
    """Pull decomposition test fields into flat dict."""
    flat["n_concepts"] = float(dec.n_concepts)
    flat["silhouette_score"] = dec.silhouette_score
    flat["is_fragmented"] = SCORE_RANGE_MAX if dec.is_fragmented else SCORE_RANGE_MIN


def _extract_editability_metrics(edit: Any, flat: Dict[str, float]) -> None:
    """Pull editability test fields into flat dict."""
    for attr in [
        "composite_editability", "steering_survival",
        "spectral_concentration", "spectral_sharpness",
        "attention_entropy", "jacobian_sensitivity",
    ]:
        flat[attr] = float(getattr(edit, attr))


def rank_methods(metrics: Dict[str, float]) -> List[Tuple[str, float]]:
    """Rank all steering methods by predicted effectiveness.

    Args:
        metrics: Flat dict of ZWIAD metrics (from flatten_zwiad_results
                 or from GeometryProfile.metrics).

    Returns:
        List of (method_name, score) sorted descending by score.
    """
    from wisent.core.control.steering_method_score import ALL_SCORERS
    results = []
    for method_name, scorer_fn in ALL_SCORERS.items():
        score = scorer_fn(metrics)
        results.append((method_name, score))
    results.sort(key=lambda pair: pair[COMBO_OFFSET], reverse=True)
    return results


def top_method_names(
    metrics: Dict[str, float], n: int = SCORER_TOP_RECOMMENDED,
) -> List[str]:
    """Return top N method names from scoring."""
    ranked = rank_methods(metrics)
    return [name for name, _ in ranked[:n]]

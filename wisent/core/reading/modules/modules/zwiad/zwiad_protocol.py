"""
Zwiad protocol: 5-step representation discovery and characterization.

1. Signal Test   2. Geometry Test   3. Decomposition Test
4. Intervention Selection   5. Editability Test (EOT sensitivity)
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
import numpy as np
from wisent.core.utils.config_tools.constants import (
    DEFAULT_SCORE, DEFAULT_SCALE, ROUNDING_PRECISION, STAT_ALPHA, BLEND_DEFAULT,
    ZWIAD_MIN_SILHOUETTE, ZWIAD_EDITABILITY_THRESHOLD, ZWIAD_PRZELOM_BONUS_MAX,
    ZWIAD_SCORE_PRIMARY, ZWIAD_SCORE_SECONDARY, ZWIAD_SCORE_TERTIARY,
)
from .zwiad_config import adaptive_gap_threshold, adaptive_min_silhouette, ZwiadProtocolConfig


@dataclass
class SignalTestResult:
    """Result of signal existence test."""
    max_z_score: float
    min_p_value: float
    passed: bool
    permutation_metrics: Dict[str, Dict[str, float]]
    nonsense_metrics: Optional[Dict[str, Dict[str, float]]] = None


@dataclass
class GeometryTestResult:
    """Result of geometry (linear vs nonlinear) test."""
    linear_accuracy: float
    nonlinear_accuracy: float
    gap: float
    diagnosis: str
    confidence: float = DEFAULT_SCORE
    p_value: float = DEFAULT_SCALE
    gap_ci_lower: float = DEFAULT_SCORE
    gap_ci_upper: float = DEFAULT_SCORE
    n_diagnostics_passed: int = 0
    n_diagnostics_total: int = 0
    t_statistic: float = DEFAULT_SCORE
    residual_silhouette: float = DEFAULT_SCORE
    residuals_cluster: bool = False
    ramsey_improvement: float = DEFAULT_SCORE
    ramsey_significant: bool = False
    diagnostics: Optional[Dict[str, Any]] = None


@dataclass
class DecompositionTestResult:
    """Result of concept decomposition test."""
    n_concepts: int
    cluster_labels: List[int]
    silhouette_score: float
    is_fragmented: bool
    per_concept_sizes: Dict[int, int]


@dataclass
class InterventionResult:
    """Result of intervention method selection."""
    recommended_method: str
    confidence: float
    reasoning: List[str]
    method_scores: Dict[str, float]


@dataclass
class EditabilityTestResult:
    """Result of EOT editability test (Step 5)."""
    composite_editability: float
    steering_survival: float
    spectral_concentration: float
    spectral_sharpness: float
    attention_entropy: float
    jacobian_sensitivity: float


def test_signal(
    pos: torch.Tensor, neg: torch.Tensor, metric_keys: List[str],
    device: str,
    model=None, tokenizer=None, layer: int = None,
    p_threshold: float = STAT_ALPHA,
) -> SignalTestResult:
    """Step 1: Test if a learnable signal exists relative to null."""
    from ..validation.null_tests.signal_null_tests import compute_signal_vs_null, compute_signal_vs_nonsense, compute_aggregate_signal
    perm_metrics = compute_signal_vs_null(pos, neg, metric_keys)
    max_z, min_p, any_sig = compute_aggregate_signal(perm_metrics, correction="bonferroni")
    nonsense_metrics = None
    if model is not None and tokenizer is not None:
        try:
            nonsense_metrics = compute_signal_vs_nonsense(
                pos, neg, model, tokenizer, metric_keys, device=device, layer=layer)
            nonsense_z, _, nonsense_sig = compute_aggregate_signal(nonsense_metrics, correction="bonferroni")
            max_z = max(max_z, nonsense_z)
            passed = (min_p < p_threshold) and nonsense_sig
        except Exception:
            passed = min_p < p_threshold
    else:
        passed = min_p < p_threshold
    return SignalTestResult(max_z, min_p, passed, perm_metrics, nonsense_metrics)


def test_geometry(pos: torch.Tensor, neg: torch.Tensor) -> GeometryTestResult:
    """Step 2: Test if geometry is linear or nonlinear."""
    from ..analysis.is_linear import test_linearity
    r = test_linearity(pos, neg)
    return GeometryTestResult(
        linear_accuracy=r.linear_accuracy, nonlinear_accuracy=r.nonlinear_accuracy,
        gap=r.gap, diagnosis=r.diagnosis, confidence=r.confidence, p_value=r.p_value,
        gap_ci_lower=r.gap_ci_lower, gap_ci_upper=r.gap_ci_upper,
        n_diagnostics_passed=r.n_diagnostics_passed, n_diagnostics_total=r.n_diagnostics_total,
        t_statistic=r.t_statistic, residual_silhouette=r.residual_silhouette,
        residuals_cluster=r.residuals_cluster, ramsey_improvement=r.ramsey_improvement,
        ramsey_significant=r.ramsey_significant, diagnostics=r.diagnostics,
    )


def test_decomposition(
    pos: torch.Tensor, neg: torch.Tensor, min_silhouette: float = ZWIAD_MIN_SILHOUETTE,
) -> DecompositionTestResult:
    """Step 3: Test if concept is fragmented into sub-concepts."""
    from ..metrics.distribution.decomposition_metrics import find_optimal_clustering
    diff = pos - neg
    n_concepts, labels, sil = find_optimal_clustering(diff, min_silhouette=min_silhouette)
    labels_np = np.array(labels)
    sizes = {int(k): int((labels_np == k).sum()) for k in range(n_concepts)}
    fragmented = n_concepts > 1 and sil >= min_silhouette
    return DecompositionTestResult(n_concepts, labels, sil, fragmented, sizes)


def test_editability(pos: torch.Tensor, neg: torch.Tensor) -> EditabilityTestResult:
    """Step 5: Test EOT editability of the activation space."""
    from .editability import compute_eot_editability
    r = compute_eot_editability(pos, neg)
    return EditabilityTestResult(
        composite_editability=r.composite_editability, steering_survival=r.steering_survival,
        spectral_concentration=r.spectral_concentration, spectral_sharpness=r.spectral_sharpness,
        attention_entropy=r.attention_entropy, jacobian_sensitivity=r.jacobian_sensitivity,
    )


def select_intervention(
    signal: SignalTestResult, geometry: GeometryTestResult,
    decomposition: DecompositionTestResult, metrics: Optional[Dict[str, Any]] = None,
    editability: Optional[EditabilityTestResult] = None,
) -> InterventionResult:
    """Step 4: Select optimal intervention method."""
    _all = {"CAA": DEFAULT_SCORE, "Ostrze": DEFAULT_SCORE, "MLP": DEFAULT_SCORE, "TECZA": DEFAULT_SCORE, "TETNO": DEFAULT_SCORE, "GROM": DEFAULT_SCORE, "Concept Flow": DEFAULT_SCORE, "PRZELOM": DEFAULT_SCORE}
    if not signal.passed:
        return InterventionResult("NONE", DEFAULT_SCORE, ["No signal (p > 0.05)"], _all)
    if metrics:
        from ..steering.analysis.steering_recommendation import compute_steering_recommendation
        rec = compute_steering_recommendation(metrics)
        reasoning = rec["reasoning"]
        reasoning.append(f"Z: {signal.max_z_score:.2f}, p: {signal.min_p_value:.4f}, gap: {geometry.gap:.3f}")
        scores = rec["method_scores"]
        scores.setdefault("PRZELOM", DEFAULT_SCORE)
        if editability and editability.composite_editability > ZWIAD_EDITABILITY_THRESHOLD:
            scores["PRZELOM"] = max(scores.get("PRZELOM", DEFAULT_SCORE), ZWIAD_PRZELOM_BONUS_MAX)
            reasoning.append(f"High EOT editability ({editability.composite_editability:.2f}) -> PRZELOM viable")
        return InterventionResult(rec["recommended_method"], round(rec["confidence"], ROUNDING_PRECISION), reasoning, scores)
    scores = dict(_all)
    reasoning = []
    is_linear = geometry.diagnosis.startswith("LINEAR")
    is_frag = decomposition.is_fragmented
    n = decomposition.n_concepts
    if is_linear and not is_frag:
        scores["CAA"], scores["Ostrze"], scores["Concept Flow"] = ZWIAD_SCORE_PRIMARY, ZWIAD_SCORE_TERTIARY, ZWIAD_SCORE_TERTIARY
        recommended, msg = "CAA", "Linear + single concept -> CAA"
    elif is_linear and is_frag:
        scores["TECZA"], scores["GROM"], scores["Concept Flow"] = ZWIAD_SCORE_PRIMARY, ZWIAD_SCORE_TERTIARY, ZWIAD_SCORE_SECONDARY
        recommended, msg = "TECZA", f"Linear + {n} concepts -> TECZA"
    elif not is_linear and not is_frag:
        scores["MLP"], scores["Ostrze"], scores["TETNO"], scores["Concept Flow"] = ZWIAD_SCORE_PRIMARY, ZWIAD_SCORE_TERTIARY, ZWIAD_SCORE_TERTIARY, ZWIAD_SCORE_TERTIARY
        recommended, msg = "MLP", "Nonlinear + single -> MLP"
    else:
        scores["GROM"], scores["TETNO"], scores["Concept Flow"] = ZWIAD_SCORE_PRIMARY, ZWIAD_SCORE_TERTIARY, ZWIAD_SCORE_SECONDARY
        recommended, msg = "GROM", f"Nonlinear + {n} concepts -> GROM"
    reasoning.append(msg)
    if editability and editability.composite_editability > ZWIAD_EDITABILITY_THRESHOLD:
        scores["PRZELOM"] = max(scores.get("PRZELOM", DEFAULT_SCORE), ZWIAD_PRZELOM_BONUS_MAX)
        reasoning.append(f"High EOT editability ({editability.composite_editability:.2f}) -> PRZELOM viable")
    reasoning.append(f"Z: {signal.max_z_score:.2f}, p: {signal.min_p_value:.4f}, gap: {geometry.gap:.3f}")
    return InterventionResult(recommended, BLEND_DEFAULT, reasoning, scores)


_GEO_FIELDS = ["linear_accuracy", "nonlinear_accuracy", "gap", "diagnosis",
    "confidence", "p_value", "gap_ci_lower", "gap_ci_upper",
    "n_diagnostics_passed", "n_diagnostics_total", "t_statistic",
    "residual_silhouette", "residuals_cluster", "ramsey_improvement",
    "ramsey_significant", "diagnostics"]


def run_full_protocol(
    pos: torch.Tensor, neg: torch.Tensor, device: str,
    model=None, tokenizer=None,
    layer: int = None, signal_keys: Optional[List[str]] = None,
    p_threshold: float = STAT_ALPHA, gap_threshold: Optional[float] = None,
    min_silhouette: Optional[float] = None, include_dimensionality_diagnostics: bool = True,
) -> Dict[str, Any]:
    """Run complete 5-step Zwiad protocol."""
    if signal_keys is None:
        signal_keys = ["knn_accuracy", "knn_pca_accuracy", "mlp_probe_accuracy"]
    n_samples = len(pos)
    if gap_threshold is None:
        gap_threshold = adaptive_gap_threshold(n_samples)
    if min_silhouette is None:
        min_silhouette = adaptive_min_silhouette(n_samples)
    dim_diag = None
    if include_dimensionality_diagnostics:
        from ..validation.dimensionality import run_dimensionality_diagnostics
        dim_diag = run_dimensionality_diagnostics(pos, neg)
    sig = test_signal(pos, neg, signal_keys, device, model, tokenizer, layer, p_threshold)
    geo = test_geometry(pos, neg)
    dec = test_decomposition(pos, neg, min_silhouette)
    edit = test_editability(pos, neg)
    inter = select_intervention(sig, geo, dec, editability=edit)
    result = {
        "protocol_config": {"n_samples": n_samples, "p_threshold": p_threshold,
                            "gap_threshold": gap_threshold, "min_silhouette": min_silhouette},
        "signal_test": {"max_z_score": sig.max_z_score, "min_p_value": sig.min_p_value,
                        "passed": sig.passed, "permutation_metrics": sig.permutation_metrics,
                        "nonsense_metrics": sig.nonsense_metrics},
        "geometry_test": {f: getattr(geo, f) for f in _GEO_FIELDS},
        "decomposition_test": {"n_concepts": dec.n_concepts, "silhouette_score": dec.silhouette_score,
                               "min_silhouette": min_silhouette, "is_fragmented": dec.is_fragmented,
                               "per_concept_sizes": dec.per_concept_sizes, "cluster_labels": dec.cluster_labels},
        "intervention": {"recommended_method": inter.recommended_method, "confidence": inter.confidence,
                         "reasoning": inter.reasoning, "method_scores": inter.method_scores},
        "editability_test": {"composite_editability": edit.composite_editability,
            "steering_survival": edit.steering_survival, "spectral_concentration": edit.spectral_concentration,
            "spectral_sharpness": edit.spectral_sharpness, "attention_entropy": edit.attention_entropy,
            "jacobian_sensitivity": edit.jacobian_sensitivity},
    }
    try:
        from .geometry_types import profile_benchmark
        geo_dict = {f: getattr(geo, f) for f in _GEO_FIELDS}
        geo_dict["n_concepts"] = dec.n_concepts
        result["geometry_profile"] = profile_benchmark(geo_dict).to_dict()
    except Exception:
        pass
    if dim_diag is not None:
        from dataclasses import asdict
        result["dimensionality_diagnostics"] = asdict(dim_diag)
    return result


run_zwiad_protocol = run_full_protocol

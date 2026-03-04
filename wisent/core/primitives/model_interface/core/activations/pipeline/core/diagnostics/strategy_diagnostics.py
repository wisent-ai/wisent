"""
Strategy diagnostics dataclass and single-strategy analysis.

Provides StrategyDiagnostics dataclass that holds all metrics for one extraction
strategy, with an overall_score property for ranking strategies.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import torch

from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy
from wisent.core.utils.config_tools.constants import PERCENT_MULTIPLIER
"""Formerly imported STRATEGY_DIAG_* constants are now required parameters."""
from .metrics import (
    compute_pairwise_consistency,
    compute_linear_nonlinear_accuracy,
    analyze_mc_confound,
    compute_steering_quality,
)


@dataclass
class StrategyDiagnostics:
    """Diagnostic results for a single extraction strategy."""

    strategy: str
    linear_accuracy: float
    nonlinear_accuracy: float
    accuracy_gap: float
    geometry: str  # "LINEAR" or "NONLINEAR"
    consistency: float
    consistency_std: float

    # For MC strategies: A/B confound analysis
    is_mc_strategy: bool = False
    ab_variance_fraction: Optional[float] = None
    semantic_variance_fraction: Optional[float] = None
    semantic_consistency: Optional[float] = None

    # Steering quality
    steering_accuracy: Optional[float] = None
    mean_effect_size: Optional[float] = None

    # Recommendation
    recommended_method: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    # Strategy diagnostics weights (set by run_strategy_diagnostics)
    _w_linear: float = field(default=None, repr=False)
    _w_consistency: float = field(default=None, repr=False)
    _w_steering: float = field(default=None, repr=False)
    _ab_threshold: float = field(default=None, repr=False)
    _ab_penalty: float = field(default=None, repr=False)

    @property
    def overall_score(self) -> float:
        """Compute overall quality score."""
        if self._w_linear is None:
            raise ValueError("Strategy weights not set")
        from wisent.core.utils.config_tools.constants import SCORE_RANGE_MAX
        score = self._w_linear * self.linear_accuracy
        norm_consistency = (self.consistency + SCORE_RANGE_MAX) / (SCORE_RANGE_MAX + SCORE_RANGE_MAX)
        score += self._w_consistency * norm_consistency
        if self.steering_accuracy is not None:
            score += self._w_steering * self.steering_accuracy
        else:
            score += self._w_steering * self.linear_accuracy
        if self.ab_variance_fraction is not None and self.ab_variance_fraction > self._ab_threshold:
            score *= (SCORE_RANGE_MAX - self.ab_variance_fraction * self._ab_penalty)
        return score


def run_strategy_diagnostics(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    strategy: ExtractionStrategy,
    letter_assignments: Optional[List[str]] = None,
    *,
    w_linear: float,
    w_consistency: float,
    w_steering: float,
    ab_threshold: float,
    ab_penalty: float,
    nonlinear_gap: float,
    low_consistency: float,
    low_linear_acc: float,
    low_semantic: float,
    cv_folds: int,
    diagnostic_mlp_hidden_sizes: tuple,
) -> StrategyDiagnostics:
    """
    Run comprehensive diagnostics on activations from a single strategy.

    Args:
        pos_activations: Tensor [n_samples, hidden_dim] for positive examples
        neg_activations: Tensor [n_samples, hidden_dim] for negative examples
        strategy: The extraction strategy used
        letter_assignments: For MC strategies, list of "A" or "B" per sample

    Returns:
        StrategyDiagnostics with all metrics
    """
    pos = pos_activations.cpu().float()
    neg = neg_activations.cpu().float()
    directions = pos - neg

    linear_acc, nonlinear_acc = compute_linear_nonlinear_accuracy(pos, neg, cv_folds=cv_folds, diagnostic_mlp_hidden_sizes=diagnostic_mlp_hidden_sizes)
    gap = nonlinear_acc - linear_acc
    geometry = "NONLINEAR" if gap > nonlinear_gap else "LINEAR"
    consistency, consistency_std = compute_pairwise_consistency(directions)
    steering_acc, effect_size = compute_steering_quality(directions)
    recommended = "CAA" if geometry == "LINEAR" else "Ostrze"

    warnings = []
    if consistency < low_consistency:
        warnings.append(f"Low consistency ({consistency:.3f})")
    if linear_acc < low_linear_acc:
        warnings.append(f"Low linear accuracy ({linear_acc:.3f})")

    is_mc = strategy in (ExtractionStrategy.MC_BALANCED, ExtractionStrategy.MC_COMPLETION)
    ab_var, sem_var, sem_consist = None, None, None

    if is_mc and letter_assignments is not None:
        confound = analyze_mc_confound(directions, letter_assignments)
        ab_var = confound["ab_variance_fraction"]
        sem_var = confound["semantic_variance_fraction"]
        sem_consist = confound["semantic_consistency"]
        if ab_var > ab_threshold:
            warnings.append(f"A/B confound: {ab_var*PERCENT_MULTIPLIER:.1f}% variance")
        if sem_consist is not None and sem_consist < low_semantic:
            warnings.append(f"No semantic signal (consistency={sem_consist:.4f})")

    return StrategyDiagnostics(
        strategy=strategy.value,
        linear_accuracy=linear_acc,
        nonlinear_accuracy=nonlinear_acc,
        accuracy_gap=gap,
        geometry=geometry,
        consistency=consistency,
        consistency_std=consistency_std,
        is_mc_strategy=is_mc,
        ab_variance_fraction=ab_var,
        semantic_variance_fraction=sem_var,
        semantic_consistency=sem_consist,
        steering_accuracy=steering_acc,
        mean_effect_size=effect_size,
        recommended_method=recommended,
        warnings=warnings,
        _w_linear=w_linear,
        _w_consistency=w_consistency,
        _w_steering=w_steering,
        _ab_threshold=ab_threshold,
        _ab_penalty=ab_penalty,
    )

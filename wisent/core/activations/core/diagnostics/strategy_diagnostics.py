"""
Strategy diagnostics dataclass and single-strategy analysis.

Provides StrategyDiagnostics dataclass that holds all metrics for one extraction
strategy, with an overall_score property for ranking strategies.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import torch

from wisent.core.activations import ExtractionStrategy
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
    recommended_method: str = "CAA"
    warnings: List[str] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """Compute overall quality score (0-1)."""
        score = 0.0
        score += 0.3 * self.linear_accuracy
        norm_consistency = (self.consistency + 1) / 2
        score += 0.4 * norm_consistency
        if self.steering_accuracy is not None:
            score += 0.3 * self.steering_accuracy
        else:
            score += 0.3 * self.linear_accuracy
        if self.ab_variance_fraction is not None and self.ab_variance_fraction > 0.5:
            score *= (1 - self.ab_variance_fraction * 0.5)
        return score


def run_strategy_diagnostics(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    strategy: ExtractionStrategy,
    letter_assignments: Optional[List[str]] = None,
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

    linear_acc, nonlinear_acc = compute_linear_nonlinear_accuracy(pos, neg)
    gap = nonlinear_acc - linear_acc
    geometry = "NONLINEAR" if gap > 0.05 else "LINEAR"
    consistency, consistency_std = compute_pairwise_consistency(directions)
    steering_acc, effect_size = compute_steering_quality(directions)
    recommended = "CAA" if geometry == "LINEAR" else "Hyperplane"

    warnings = []
    if consistency < 0.1:
        warnings.append(f"Low consistency ({consistency:.3f})")
    if linear_acc < 0.7:
        warnings.append(f"Low linear accuracy ({linear_acc:.3f})")

    is_mc = strategy in (ExtractionStrategy.MC_BALANCED, ExtractionStrategy.MC_COMPLETION)
    ab_var, sem_var, sem_consist = None, None, None

    if is_mc and letter_assignments is not None:
        confound = analyze_mc_confound(directions, letter_assignments)
        ab_var = confound["ab_variance_fraction"]
        sem_var = confound["semantic_variance_fraction"]
        sem_consist = confound["semantic_consistency"]
        if ab_var > 0.5:
            warnings.append(f"A/B confound: {ab_var*100:.1f}% variance")
        if sem_consist is not None and sem_consist < 0.05:
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
    )

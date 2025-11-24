"""
Multi-objective optimization for steering parameters using Optuna.

Adapts Heretic's Optuna-based optimization to Wisent's steering paradigm.
Optimizes for:
1. Task performance (accuracy, exact match, etc.)
2. KL divergence (preservation of original model capabilities)

This creates a Pareto frontier of solutions trading off task improvement
vs. model preservation.

Requires: pip install optuna
"""

from __future__ import annotations

import optuna
from optuna.samplers import TPESampler
from optuna.study import StudyDirection
from typing import TYPE_CHECKING, Callable
from dataclasses import dataclass

if TYPE_CHECKING:
    from torch import Tensor
    from optuna import Trial

__all__ = [
    "MultiObjectiveOptimizer",
    "SteeringParameters",
    "OptimizationResult",
]


@dataclass
class SteeringParameters:
    """
    Parameters for steering optimization.

    Attributes:
        alpha: Global steering strength
        direction_index: Float layer index for global direction (None = per-layer)
        layer_weights: Per-layer scaling factors (None = uniform)
        direction_scope: "global" or "per_layer"
    """

    alpha: float
    direction_index: float | None = None
    layer_weights: dict[int, float] | None = None
    direction_scope: str = "per_layer"

    def __repr__(self) -> str:
        return (
            f"SteeringParameters(alpha={self.alpha:.2f}, "
            f"scope={self.direction_scope}, "
            f"dir_idx={self.direction_index})"
        )


@dataclass
class OptimizationResult:
    """
    Result from multi-objective optimization.

    Attributes:
        study: Optuna study object
        best_trials: List of Pareto-optimal trials
        best_params: SteeringParameters from user-selected trial
        metrics: Final evaluation metrics
    """

    study: optuna.Study
    best_trials: list[optuna.trial.FrozenTrial]
    best_params: SteeringParameters | None = None
    metrics: dict[str, float] | None = None


class MultiObjectiveOptimizer:
    """
    Multi-objective optimizer for steering parameters.

    Optimizes two objectives simultaneously:
    1. Maximize task performance (accuracy, etc.)
    2. Minimize KL divergence from baseline

    Uses Optuna's TPE sampler with multivariate optimization.

    Adapted from Heretic's optimization loop in main.py lines 256-388.

    Usage:
        optimizer = MultiObjectiveOptimizer(
            model=model,
            steering_vectors=steering_vectors,
            evaluate_fn=lambda params: (accuracy, kl_div)
        )
        result = optimizer.optimize(n_trials=100)
        best_params = result.select_best_trial()
    """

    def __init__(
        self,
        model,
        steering_vectors: dict[int, Tensor],
        evaluate_fn: Callable[[SteeringParameters], tuple[float, float]],
        kl_divergence_scale: float = 1.0,
        direction: str = "maximize",  # for task performance
    ):
        """
        Initialize multi-objective optimizer.

        Args:
            model: Model to optimize steering for
            steering_vectors: Base steering vectors (per-layer)
            evaluate_fn: Function that takes SteeringParameters and returns
                        (task_metric, kl_divergence). task_metric should be
                        higher-is-better (accuracy, F1, etc.)
            kl_divergence_scale: Scale factor for KL divergence objective
            direction: "maximize" or "minimize" for task metric
        """
        self.model = model
        self.steering_vectors = steering_vectors
        self.evaluate_fn = evaluate_fn
        self.kl_divergence_scale = kl_divergence_scale
        self.direction = direction

        self.num_layers = len(steering_vectors)

    def suggest_parameters(self, trial: Trial) -> SteeringParameters:
        """
        Suggest steering parameters for this trial.

        Adapted from Heretic's parameter suggestion (lines 264-326).

        Args:
            trial: Optuna trial

        Returns:
            SteeringParameters for this trial
        """
        # Direction scope: global or per-layer
        direction_scope = trial.suggest_categorical(
            "direction_scope",
            ["global", "per_layer"],
        )

        # Alpha (steering strength)
        alpha = trial.suggest_float("alpha", 0.1, 10.0, log=True)

        # Direction index (for global scope)
        direction_index = None
        if direction_scope == "global":
            # Suggest float index in range [0.4*(L-1), 0.9*(L-1)]
            min_idx = 0.4 * (self.num_layers - 1)
            max_idx = 0.9 * (self.num_layers - 1)
            direction_index = trial.suggest_float(
                "direction_index",
                min_idx,
                max_idx,
            )

        # Per-layer weights (optional refinement)
        use_layer_weights = trial.suggest_categorical(
            "use_layer_weights",
            [False, True],
        )

        layer_weights = None
        if use_layer_weights:
            # Suggest kernel parameters for layer weight distribution
            max_weight = trial.suggest_float("max_weight", 0.5, 2.0)
            max_weight_position = trial.suggest_float(
                "max_weight_position",
                0.4 * (self.num_layers - 1),
                self.num_layers - 1,
            )
            min_weight = trial.suggest_float("min_weight", 0.0, max_weight)
            min_weight_distance = trial.suggest_float(
                "min_weight_distance",
                1.0,
                0.6 * (self.num_layers - 1),
            )

            # Compute per-layer weights using kernel
            layer_weights = self._compute_layer_weights(
                max_weight,
                max_weight_position,
                min_weight,
                min_weight_distance,
            )

        return SteeringParameters(
            alpha=alpha,
            direction_index=direction_index,
            layer_weights=layer_weights,
            direction_scope=direction_scope,
        )

    def _compute_layer_weights(
        self,
        max_weight: float,
        max_weight_position: float,
        min_weight: float,
        min_weight_distance: float,
    ) -> dict[int, float]:
        """
        Compute per-layer weights using Heretic-style kernel.

        Creates a kernel shape with maximum at max_weight_position,
        tapering to min_weight at edges.

        Args:
            max_weight: Peak weight
            max_weight_position: Layer index of peak
            min_weight: Minimum weight
            min_weight_distance: Distance over which weight decays

        Returns:
            Dictionary mapping layer index to weight
        """
        weights = {}

        for layer_idx in range(self.num_layers):
            distance = abs(layer_idx - max_weight_position)

            if distance > min_weight_distance:
                # Too far from center - skip or use minimum
                weight = 0.0
            else:
                # Linear interpolation from max_weight to min_weight
                weight = max_weight + (distance / min_weight_distance) * (
                    min_weight - max_weight
                )

            weights[layer_idx] = weight

        return weights

    def objective(self, trial: Trial) -> tuple[float, float]:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial

        Returns:
            Tuple of (task_metric_neg, kl_divergence) for minimization
            Note: task_metric is negated if direction="maximize"
        """
        # Suggest parameters
        params = self.suggest_parameters(trial)

        # Evaluate
        task_metric, kl_divergence = self.evaluate_fn(params)

        # Negate task_metric if maximizing
        if self.direction == "maximize":
            task_metric_neg = -task_metric
        else:
            task_metric_neg = task_metric

        # Scale KL divergence
        kl_scaled = kl_divergence / self.kl_divergence_scale

        return (task_metric_neg, kl_scaled)

    def optimize(
        self,
        n_trials: int = 100,
        n_startup_trials: int = 30,
        n_ei_candidates: int = 128,
        show_progress: bool = True,
    ) -> OptimizationResult:
        """
        Run multi-objective optimization.

        Args:
            n_trials: Total number of trials
            n_startup_trials: Number of random startup trials
            n_ei_candidates: Number of Expected Improvement candidates
            show_progress: Whether to show progress bar

        Returns:
            OptimizationResult with Pareto-optimal trials
        """
        # Create study with TPE sampler
        study = optuna.create_study(
            sampler=TPESampler(
                n_startup_trials=n_startup_trials,
                n_ei_candidates=n_ei_candidates,
                multivariate=True,  # Model joint parameter distributions
            ),
            directions=[
                StudyDirection.MINIMIZE,  # Minimize task_metric_neg (or maximize task_metric)
                StudyDirection.MINIMIZE,  # Minimize KL divergence
            ],
        )

        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            show_progress_bar=show_progress,
        )

        # Get Pareto-optimal trials
        best_trials = study.best_trials

        return OptimizationResult(
            study=study,
            best_trials=best_trials,
        )

    def print_pareto_frontier(self, result: OptimizationResult) -> None:
        """
        Print Pareto-optimal trials in a formatted table.

        Args:
            result: OptimizationResult from optimize()
        """
        print("\n" + "=" * 80)
        print("PARETO-OPTIMAL TRIALS")
        print("=" * 80)
        print(f"{'Trial':>6} | {'Task Metric':>12} | {'KL Div':>10} | {'Alpha':>8} | {'Scope':>10}")
        print("-" * 80)

        for i, trial in enumerate(result.best_trials):
            task_metric = -trial.values[0] if self.direction == "maximize" else trial.values[0]
            kl_div = trial.values[1] * self.kl_divergence_scale
            alpha = trial.params.get("alpha", 0.0)
            scope = trial.params.get("direction_scope", "per_layer")

            print(
                f"{trial.number:>6} | {task_metric:>12.4f} | {kl_div:>10.4f} | {alpha:>8.2f} | {scope:>10}"
            )

        print("=" * 80 + "\n")

    def select_best_trial(
        self,
        result: OptimizationResult,
        task_weight: float = 0.7,
    ) -> SteeringParameters:
        """
        Select best trial from Pareto frontier using weighted score.

        Args:
            result: OptimizationResult from optimize()
            task_weight: Weight for task metric (1-task_weight for KL div)

        Returns:
            SteeringParameters from selected trial
        """
        if not result.best_trials:
            raise ValueError("No Pareto-optimal trials found")

        # Compute weighted scores for each trial
        scores = []
        for trial in result.best_trials:
            task_metric = -trial.values[0] if self.direction == "maximize" else trial.values[0]
            kl_div = trial.values[1]

            # Normalize to [0, 1] range
            task_normalized = task_metric  # Assume already in [0, 1]
            kl_normalized = kl_div  # Lower is better

            # Weighted score (higher is better)
            score = task_weight * task_normalized - (1 - task_weight) * kl_normalized
            scores.append((score, trial))

        # Select trial with highest score
        best_score, best_trial = max(scores, key=lambda x: x[0])

        # Reconstruct parameters
        return SteeringParameters(
            alpha=best_trial.params["alpha"],
            direction_index=best_trial.params.get("direction_index"),
            layer_weights=None,  # Would need to reconstruct from kernel params
            direction_scope=best_trial.params["direction_scope"],
        )


def quick_optimize_steering(
    model,
    steering_vectors: dict[int, Tensor],
    evaluate_fn: Callable[[float], tuple[float, float]],
    n_trials: int = 50,
    direction: str = "maximize",
) -> tuple[float, float, float]:
    """
    Quick optimization function for simple alpha-only tuning.

    Args:
        model: Model to optimize for
        steering_vectors: Steering vectors
        evaluate_fn: Function taking alpha and returning (task_metric, kl_div)
        n_trials: Number of trials
        direction: "maximize" or "minimize" task metric

    Returns:
        Tuple of (best_alpha, best_task_metric, best_kl_div)

    Example:
        >>> def evaluate(alpha):
        ...     # Apply steering with alpha
        ...     # Return accuracy and KL divergence
        ...     return accuracy, kl_div
        >>> best_alpha, acc, kl = quick_optimize_steering(model, vectors, evaluate)
    """
    def wrapper(params: SteeringParameters) -> tuple[float, float]:
        return evaluate_fn(params.alpha)

    optimizer = MultiObjectiveOptimizer(
        model=model,
        steering_vectors=steering_vectors,
        evaluate_fn=wrapper,
        direction=direction,
    )

    result = optimizer.optimize(n_trials=n_trials, n_startup_trials=min(20, n_trials // 3))
    optimizer.print_pareto_frontier(result)

    best_params = optimizer.select_best_trial(result)
    task_metric, kl_div = evaluate_fn(best_params.alpha)

    return best_params.alpha, task_metric, kl_div

"""
Weight modification optimizer using BaseOptimizer.

Optimizes weight modification parameters (strength, max_weight, min_weight, position)
to achieve target metrics like compliance rate or task accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import optuna
import torch

from wisent.core.opti.core.atoms import BaseOptimizer, Direction

__all__ = [
    "WeightsOptimizer",
    "WeightsOptimizerConfig",
]


@dataclass(slots=True)
class WeightsOptimizerConfig:
    """
    Configuration for weight modification optimization.

    attributes:
        strength_range:
            Range for strength parameter (min, max).
        max_weight_range:
            Range for max_weight parameter (min, max).
        min_weight_range:
            Range for min_weight parameter (min, max).
        position_range:
            Range for max_weight_position as ratio of layers (min, max).
        method:
            Weight modification method: "directional" or "bake".
        components:
            Number of components for weight modification.
        norm_preserve:
            Whether to preserve norms during directional projection.
        optimize_direction_index:
            Whether to optimize direction index as well.
        target_metric:
            Metric to optimize (e.g., "compliance_rate", "accuracy").
        target_value:
            Target value to achieve (for early stopping).
    """
    strength_range: tuple[float, float] = (0.5, 2.0)
    max_weight_range: tuple[float, float] = (0.1, 1.0)
    min_weight_range: tuple[float, float] = (0.0, 0.3)
    position_range: tuple[float, float] = (0.3, 0.7)
    method: Literal["directional", "bake"] = "directional"
    components: int = 1
    norm_preserve: bool = True
    optimize_direction_index: bool = False
    target_metric: str = "compliance_rate"
    target_value: float | None = None


class WeightsOptimizer(BaseOptimizer):
    """
    Optuna optimizer for weight modification parameters.

    Finds optimal parameters (strength, max_weight, min_weight, position) for
    applying steering vectors as permanent weight modifications.

    arguments:
        model:
            The model to optimize (will be modified in-place during trials).
        base_state_dict:
            Original model state dict for restoration between trials.
        steering_vectors:
            Dict mapping layer indices to steering vector tensors.
        evaluator:
            Callable that takes (model, tokenizer) and returns dict with metrics.
        tokenizer:
            Tokenizer for the model.
        config:
            WeightsOptimizerConfig with search space and method settings.
        num_layers:
            Number of layers in the model.

    example:
        >>> from wisent.opti.methods.opti_weights import WeightsOptimizer, WeightsOptimizerConfig
        >>> from wisent.opti.core.atoms import HPOConfig
        >>>
        >>> config = WeightsOptimizerConfig(
        ...     strength_range=(0.5, 2.0),
        ...     max_weight_range=(0.1, 1.0),
        ...     target_metric="compliance_rate",
        ... )
        >>>
        >>> optimizer = WeightsOptimizer(
        ...     model=model,
        ...     base_state_dict=base_state_dict,
        ...     steering_vectors=steering_vectors,
        ...     evaluator=my_evaluator,
        ...     tokenizer=tokenizer,
        ...     config=config,
        ...     num_layers=32,
        ... )
        >>>
        >>> result = optimizer.optimize(HPOConfig(n_trials=50, direction="maximize"))
        >>> print("Best params:", result.best_params)
        >>> print("Best score:", result.best_value)
    """

    name = "weights-optimizer"
    direction: Direction = "maximize"

    def __init__(
        self,
        model: Any,
        base_state_dict: dict[str, torch.Tensor],
        steering_vectors: dict[int, torch.Tensor],
        evaluator: Callable[[Any, Any], dict[str, float]],
        tokenizer: Any,
        config: WeightsOptimizerConfig,
        num_layers: int,
    ) -> None:
        self.model = model
        self.base_state_dict = base_state_dict
        self.steering_vectors = steering_vectors
        self.evaluator = evaluator
        self.tokenizer = tokenizer
        self.config = config
        self.num_layers = num_layers

        # Set direction based on target metric
        if config.target_metric in ["refusal_rate", "kl_divergence"]:
            self.direction = "minimize"
        else:
            self.direction = "maximize"

    def _objective(self, trial: optuna.Trial) -> float:
        """
        One trial: apply weight modification with suggested params and evaluate.

        arguments:
            trial: Optuna trial object.

        returns:
            float, value of the target metric.
        """
        # Suggest parameters
        params = {
            "strength": trial.suggest_float(
                "strength",
                self.config.strength_range[0],
                self.config.strength_range[1],
            ),
            "max_weight": trial.suggest_float(
                "max_weight",
                self.config.max_weight_range[0],
                self.config.max_weight_range[1],
            ),
            "min_weight": trial.suggest_float(
                "min_weight",
                self.config.min_weight_range[0],
                self.config.min_weight_range[1],
            ),
            "max_weight_position": trial.suggest_float(
                "max_weight_position",
                self.config.position_range[0],
                self.config.position_range[1],
            ),
        }

        # Optional direction index optimization
        if self.config.optimize_direction_index:
            params["direction_index"] = trial.suggest_float(
                "direction_index",
                0.0,
                self.num_layers - 1,
            )

        # Restore base model weights
        self.model.load_state_dict(self.base_state_dict)

        # Apply weight modification
        self._apply_weight_modification(params)

        # Evaluate
        eval_result = self.evaluator(self.model, self.tokenizer)

        # Get target metric score
        score = eval_result.get(
            self.config.target_metric,
            eval_result.get("score", 0.0),
        )

        # Check for early stopping if target is set
        if self.config.target_value is not None:
            if self.direction == "maximize" and score >= self.config.target_value:
                trial.study.stop()
            elif self.direction == "minimize" and score <= self.config.target_value:
                trial.study.stop()

        return float(score)

    def _apply_weight_modification(self, params: dict[str, float]) -> None:
        """
        Apply weight modification with given parameters.

        arguments:
            params: Dict with strength, max_weight, min_weight, max_weight_position.
        """
        from wisent.core.weight_modification import (
            project_with_kernel,
            bake_steering_with_kernel,
        )

        # Convert position ratio to layer index
        max_weight_position = params["max_weight_position"] * (self.num_layers - 1)

        # Compute min_weight_distance from position
        min_weight_distance = 0.6 * (self.num_layers - 1)

        if self.config.method == "directional":
            project_with_kernel(
                self.model,
                self.steering_vectors,
                max_weight=params["max_weight"] * params["strength"],
                max_weight_position=max_weight_position,
                min_weight=params["min_weight"],
                min_weight_distance=min_weight_distance,
                components=self.config.components,
                norm_preserve=self.config.norm_preserve,
                verbose=False,
            )
        else:
            bake_steering_with_kernel(
                self.model,
                self.steering_vectors,
                max_alpha=params["max_weight"] * params["strength"],
                max_alpha_position=max_weight_position,
                min_alpha=params["min_weight"],
                components=self.config.components,
                verbose=False,
            )

    def apply_best_params(self, best_params: dict[str, float]) -> None:
        """
        Apply the best parameters found during optimization.

        Restores base weights first, then applies weight modification
        with the optimal parameters.

        arguments:
            best_params: Best parameters from optimization result.
        """
        self.model.load_state_dict(self.base_state_dict)
        self._apply_weight_modification(best_params)

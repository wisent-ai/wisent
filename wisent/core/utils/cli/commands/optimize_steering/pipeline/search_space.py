"""Search space generation for steering optimization methods."""
from typing import Iterator, Sequence, Tuple

from wisent.core.utils.cli.optimize_steering.method_configs import (
    MethodConfig, CAAConfig, OstrzeConfig, MLPConfig,
    TECZAConfig, TETNOConfig, GROMConfig,
    STEERING_STRATEGIES,
)


def get_search_space(
    method: str,
    num_layers: int,
    search_mlp_hidden_dims: tuple = (),
    search_mlp_num_layers: tuple = (),
    mlp_input_divisor: int = None,
    mlp_early_stopping_patience: int = None,
    mlp_gating_hidden_dim_divisor: int = None,
    *,
    tetno_search_condition_thresholds: Tuple[float, ...],
    tetno_search_gate_temperatures: Tuple[float, ...],
    tetno_search_max_alphas: Tuple[float, ...],
    tecza_search_num_directions: Tuple[int, ...],
    tecza_search_retain_weights: Tuple[float, ...],
    grom_search_num_directions: Tuple[int, ...],
    grom_search_gate_hidden_dims: Tuple[int, ...],
    grom_search_intensity_hidden_dims: Tuple[int, ...],
    grom_search_behavior_weights: Tuple[float, ...],
    sensor_layer_sampling_divisor: int,
    search_steering_ranges: Sequence[int],
) -> Iterator[MethodConfig]:
    """
    Generate search space for a method.

    Includes:
    - extraction_strategy: how to collect activations (chat_last, chat_mean)
    - steering_strategy: how to apply steering during generation (constant, diminishing, etc.)
    - method-specific parameters

    Args:
        method: Steering method name.
        num_layers: Number of model layers.
        search_mlp_hidden_dims: Hidden dims to search for MLP.
        search_mlp_num_layers: Layer counts to search for MLP.
        mlp_input_divisor: MLP hidden-dim input divisor.
        mlp_early_stopping_patience: MLP early-stopping patience.
    """
    extraction_strategies = ["chat_last", "chat_mean"]
    steering_strategies = STEERING_STRATEGIES  # constant, initial_only, diminishing, increasing, gaussian

    if method.upper() == "CAA":
        for layer in range(num_layers):
            for ext_strategy in extraction_strategies:
                for steer_strategy in steering_strategies:
                    yield CAAConfig(
                        method="CAA",
                        layer=layer,
                        extraction_strategy=ext_strategy,
                        steering_strategy=steer_strategy,
                    )
    
    elif method.upper() == "OSTRZE":
        for layer in range(num_layers):
            for ext_strategy in extraction_strategies:
                for steer_strategy in steering_strategies:
                    yield OstrzeConfig(
                        method="Ostrze",
                        layer=layer,
                        extraction_strategy=ext_strategy,
                        steering_strategy=steer_strategy,
                    )
    
    elif method.upper() == "MLP":
        for layer in range(num_layers):
            for hidden_dim in search_mlp_hidden_dims:
                for num_layers_mlp in search_mlp_num_layers:
                    for ext_strategy in extraction_strategies:
                        for steer_strategy in steering_strategies:
                            yield MLPConfig(
                                method="MLP",
                                layer=layer,
                                hidden_dim=hidden_dim,
                                num_layers=num_layers_mlp,
                                mlp_input_divisor=mlp_input_divisor,
                                mlp_early_stopping_patience=mlp_early_stopping_patience,
                                mlp_gating_hidden_dim_divisor=mlp_gating_hidden_dim_divisor,
                                extraction_strategy=ext_strategy,
                                steering_strategy=steer_strategy,
                            )
    
    elif method.upper() == "TECZA":
        for layer in range(num_layers):
            for num_directions in tecza_search_num_directions:
                for direction_weighting in ["primary_only", "equal"]:
                    for retain_weight in tecza_search_retain_weights:
                        for ext_strategy in extraction_strategies:
                            for steer_strategy in steering_strategies:
                                yield TECZAConfig(
                                    method="TECZA",
                                    layer=layer,
                                    num_directions=num_directions,
                                    direction_weighting=direction_weighting,
                                    retain_weight=retain_weight,
                                    extraction_strategy=ext_strategy,
                                    steering_strategy=steer_strategy,
                                )
    
    elif method.upper() == "TETNO":
        sensor_layers = list(range(0, num_layers, max(1, num_layers // sensor_layer_sampling_divisor)))
        for sensor_layer in sensor_layers:
            for steering_range in search_steering_ranges:
                start = max(0, sensor_layer - steering_range // 2)
                end = min(num_layers, start + steering_range)
                steering_layers = list(range(start, end))
                for threshold in tetno_search_condition_thresholds:
                    for gate_temp in tetno_search_gate_temperatures:
                        for max_alpha in tetno_search_max_alphas:
                            for ext_strategy in extraction_strategies:
                                for steer_strategy in steering_strategies:
                                    yield TETNOConfig(
                                        method="TETNO",
                                        sensor_layer=sensor_layer,
                                        steering_layers=steering_layers,
                                        condition_threshold=threshold,
                                        gate_temperature=gate_temp,
                                        max_alpha=max_alpha,
                                        extraction_strategy=ext_strategy,
                                        steering_strategy=steer_strategy,
                                    )
    
    elif method.upper() == "GROM":
        sensor_layers = list(range(0, num_layers, max(1, num_layers // sensor_layer_sampling_divisor)))
        for sensor_layer in sensor_layers:
            for steering_range in search_steering_ranges:
                start = max(0, sensor_layer - steering_range // 2)
                end = min(num_layers, start + steering_range)
                steering_layers = list(range(start, end))
                for num_directions in grom_search_num_directions:
                    for gate_hidden in grom_search_gate_hidden_dims:
                        for intensity_hidden in grom_search_intensity_hidden_dims:
                            for behavior_weight in grom_search_behavior_weights:
                                for ext_strategy in extraction_strategies:
                                    for steer_strategy in steering_strategies:
                                        yield GROMConfig(
                                            method="GROM",
                                            sensor_layer=sensor_layer,
                                            steering_layers=steering_layers,
                                            num_directions=num_directions,
                                            gate_hidden_dim=gate_hidden,
                                            intensity_hidden_dim=intensity_hidden,
                                            behavior_weight=behavior_weight,
                                            extraction_strategy=ext_strategy,
                                            steering_strategy=steer_strategy,
                                        )


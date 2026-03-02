"""Search space generation for steering optimization methods."""
from typing import Iterator

from wisent.core.utils.cli.optimize_steering.method_configs import (
    MethodConfig, CAAConfig, OstrzeConfig, MLPConfig,
    TECZAConfig, TETNOConfig, GROMConfig,
    STEERING_STRATEGIES,
)
from wisent.core.utils.config_tools.constants import (
    SEARCH_MLP_HIDDEN_DIMS, SEARCH_MLP_NUM_LAYERS, TECZA_SEARCH_NUM_DIRECTIONS,
    TECZA_SEARCH_RETAIN_WEIGHTS, SEARCH_STEERING_RANGES,
    TETNO_SEARCH_CONDITION_THRESHOLDS, TETNO_SEARCH_GATE_TEMPERATURES,
    TETNO_SEARCH_MAX_ALPHAS, GROM_SEARCH_NUM_DIRECTIONS,
    SEARCH_GROM_GATE_HIDDEN_DIMS, SEARCH_GROM_INTENSITY_HIDDEN_DIMS,
    GROM_SEARCH_BEHAVIOR_WEIGHTS, SENSOR_LAYER_SAMPLING_DIVISOR,
)


def get_search_space(method: str, num_layers: int) -> Iterator[MethodConfig]:
    """
    Generate search space for a method.
    
    Includes:
    - extraction_strategy: how to collect activations (chat_last, chat_mean)
    - steering_strategy: how to apply steering during generation (constant, diminishing, etc.)
    - method-specific parameters
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
            for hidden_dim in SEARCH_MLP_HIDDEN_DIMS:
                for num_layers_mlp in SEARCH_MLP_NUM_LAYERS:
                    for ext_strategy in extraction_strategies:
                        for steer_strategy in steering_strategies:
                            yield MLPConfig(
                                method="MLP",
                                layer=layer,
                                hidden_dim=hidden_dim,
                                num_layers=num_layers_mlp,
                                extraction_strategy=ext_strategy,
                                steering_strategy=steer_strategy,
                            )
    
    elif method.upper() == "TECZA":
        for layer in range(num_layers):
            for num_directions in TECZA_SEARCH_NUM_DIRECTIONS:
                for direction_weighting in ["primary_only", "equal"]:
                    for retain_weight in TECZA_SEARCH_RETAIN_WEIGHTS:
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
        sensor_layers = list(range(0, num_layers, max(1, num_layers // SENSOR_LAYER_SAMPLING_DIVISOR)))
        for sensor_layer in sensor_layers:
            for steering_range in SEARCH_STEERING_RANGES:
                start = max(0, sensor_layer - steering_range // 2)
                end = min(num_layers, start + steering_range)
                steering_layers = list(range(start, end))
                for threshold in TETNO_SEARCH_CONDITION_THRESHOLDS:
                    for gate_temp in TETNO_SEARCH_GATE_TEMPERATURES:
                        for max_alpha in TETNO_SEARCH_MAX_ALPHAS:
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
        sensor_layers = list(range(0, num_layers, max(1, num_layers // SENSOR_LAYER_SAMPLING_DIVISOR)))
        for sensor_layer in sensor_layers:
            for steering_range in SEARCH_STEERING_RANGES:
                start = max(0, sensor_layer - steering_range // 2)
                end = min(num_layers, start + steering_range)
                steering_layers = list(range(start, end))
                for num_directions in GROM_SEARCH_NUM_DIRECTIONS:
                    for gate_hidden in SEARCH_GROM_GATE_HIDDEN_DIMS:
                        for intensity_hidden in SEARCH_GROM_INTENSITY_HIDDEN_DIMS:
                            for behavior_weight in GROM_SEARCH_BEHAVIOR_WEIGHTS:
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


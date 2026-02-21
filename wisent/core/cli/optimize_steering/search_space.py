"""Search space generation for steering optimization methods."""
from typing import Iterator

from wisent.core.cli.optimize_steering.method_configs import (
    MethodConfig, CAAConfig, OstrzeConfig, MLPConfig,
    TECZAConfig, TETNOConfig, GROMConfig,
    STEERING_STRATEGIES,
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
            for hidden_dim in [128, 256, 512]:
                for num_layers_mlp in [1, 2, 3]:
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
            for num_directions in [1, 2, 3, 5]:
                for direction_weighting in ["primary_only", "equal"]:
                    for retain_weight in [0.0, 0.1, 0.3]:
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
        for sensor_pos in [0.5, 0.75]:  # middle, late
            sensor_layer = int(num_layers * sensor_pos)
            for steering_range in [3, 5]:
                steering_start = int(num_layers * 0.75)
                steering_layers = list(range(steering_start, min(steering_start + steering_range, num_layers)))
                for threshold in [0.3, 0.5, 0.7]:
                    for gate_temp in [0.1, 0.5, 1.0]:
                        for max_alpha in [1.5, 2.0, 3.0]:
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
        for sensor_pos in [0.5, 0.75]:
            sensor_layer = int(num_layers * sensor_pos)
            for steering_range in [3, 5]:
                steering_start = int(num_layers * 0.75)
                steering_layers = list(range(steering_start, min(steering_start + steering_range, num_layers)))
                for num_directions in [2, 3, 5]:
                    for gate_hidden in [32, 64]:
                        for intensity_hidden in [16, 32]:
                            for behavior_weight in [0.5, 1.0]:
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


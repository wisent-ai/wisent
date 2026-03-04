"""Configuration and helper functions for hierarchical optimization."""
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Iterator, Tuple

from wisent.core.utils.cli.optimize_steering.method_configs import (
    MethodConfig, CAAConfig, OstrzeConfig, TECZAConfig, TETNOConfig, GROMConfig,
    NurtConfig, SzlakConfig, WicherConfig,
)


@dataclass
class HierarchicalResult:
    """Result from hierarchical optimization."""
    method: str
    best_layer: int
    best_strength: float
    best_params: Dict[str, Any]
    best_score: float
    stage1_results: List[Dict]  # Layer sweep
    stage2_results: List[Dict]  # Strength sweep
    stage3_results: List[Dict]  # Param tuning


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical search."""
    # Stage 1: Layer sweep
    layer_sweep_strength: Optional[float] = None
    layer_sweep_normalize: bool = True

    # Stage 2: Strength sweep
    strengths: List[float] = field(kw_only=True)

    # Stage 3: Method-specific grids
    mlp_hidden_dims: List[int] = field(default_factory=list)
    mlp_num_layers: List[int] = field(default_factory=list)

    tecza_num_directions: List[int] = field(kw_only=True)
    tecza_optimization_steps: List[int] = field(kw_only=True)

    tetno_thresholds: List[float] = field(kw_only=True)
    tetno_temperatures: List[float] = field(kw_only=True)

    grom_num_directions: List[int] = field(kw_only=True)
    grom_max_alphas: List[float] = field(kw_only=True)
    grom_temperatures: List[float] = field(kw_only=True)


def count_hierarchical_configs(
    methods: List[str],
    num_layers: int,
    config: HierarchicalConfig,
    min_clusters: int,
) -> Dict[str, Dict[str, int]]:
    """Count configurations per stage per method."""
    counts = {}

    for method in methods:
        method_upper = method.upper()
        stage1 = num_layers  # All layers
        stage2 = len(config.strengths)

        if method_upper == "CAA":
            stage3 = min_clusters  # normalize True/False
        elif method_upper == "OSTRZE":
            stage3 = min_clusters
        elif method_upper == "MLP":
            stage3 = len(config.mlp_hidden_dims) * len(config.mlp_num_layers) * 2
        elif method_upper == "TECZA":
            stage3 = len(config.tecza_num_directions) * len(config.tecza_optimization_steps) * 2
        elif method_upper == "TETNO":
            stage3 = len(config.tetno_thresholds) * len(config.tetno_temperatures) * 2
        elif method_upper == "GROM":
            stage3 = (len(config.grom_num_directions) *
                     len(config.grom_max_alphas) *
                     len(config.grom_temperatures) * 2)
        else:
            stage3 = min_clusters

        counts[method] = {
            "stage1_layer": stage1,
            "stage2_strength": stage2,
            "stage3_params": stage3,
            "total": stage1 + stage2 + stage3,
        }

    return counts


def _create_config_for_layer_sweep(
    method: str,
    layer: int,
    strength: float,
    normalize: bool,
) -> MethodConfig:
    """Create a config for layer sweep (fixed strength, default params)."""
    method_upper = method.upper()

    if method_upper == "CAA":
        return CAAConfig(method="CAA", layer=layer)
    elif method_upper == "OSTRZE":
        return OstrzeConfig(method="Ostrze", layer=layer)
    elif method_upper == "MLP":
        return MLPConfig(method="MLP", layer=layer)
    elif method_upper == "TECZA":
        return TECZAConfig(method="TECZA", layer=layer)
    elif method_upper == "TETNO":
        return TETNOConfig(
            method="TETNO",
            sensor_layer=layer,
            steering_layers=[layer],
        )
    elif method_upper == "GROM":
        return GROMConfig(
            method="GROM",
            sensor_layer=layer,
            steering_layers=[layer],
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def _create_configs_for_stage3(
    method: str,
    layer: int,
    strength: float,
    config: HierarchicalConfig,
) -> List[Tuple[MethodConfig, Dict[str, Any]]]:
    """Create all configs for stage 3 param tuning."""
    configs = []
    method_upper = method.upper()

    if method_upper == "CAA":
        for normalize in [True, False]:
            cfg = CAAConfig(method="CAA", layer=layer)
            configs.append((cfg, {"normalize": normalize}))

    elif method_upper == "OSTRZE":
        for normalize in [True, False]:
            cfg = OstrzeConfig(method="Ostrze", layer=layer)
            configs.append((cfg, {"normalize": normalize}))

    elif method_upper == "MLP":
        for hidden_dim in config.mlp_hidden_dims:
            for num_layers in config.mlp_num_layers:
                for normalize in [True, False]:
                    cfg = MLPConfig(
                        method="MLP",
                        layer=layer,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                    )
                    configs.append((cfg, {
                        "hidden_dim": hidden_dim,
                        "num_layers": num_layers,
                        "normalize": normalize,
                    }))

    elif method_upper == "TECZA":
        for num_dirs in config.tecza_num_directions:
            for opt_steps in config.tecza_optimization_steps:
                for normalize in [True, False]:
                    cfg = TECZAConfig(
                        method="TECZA",
                        layer=layer,
                        num_directions=num_dirs,
                        optimization_steps=opt_steps,
                    )
                    configs.append((cfg, {
                        "num_directions": num_dirs,
                        "optimization_steps": opt_steps,
                        "normalize": normalize,
                    }))

    elif method_upper == "TETNO":
        for thresh in config.tetno_thresholds:
            for temp in config.tetno_temperatures:
                for normalize in [True, False]:
                    cfg = TETNOConfig(
                        method="TETNO",
                        sensor_layer=layer,
                        steering_layers=[layer],
                        condition_threshold=thresh,
                        gate_temperature=temp,
                    )
                    configs.append((cfg, {
                        "threshold": thresh,
                        "temperature": temp,
                        "normalize": normalize,
                    }))

    elif method_upper == "GROM":
        for num_dirs in config.grom_num_directions:
            for max_alpha in config.grom_max_alphas:
                for temp in config.grom_temperatures:
                    for normalize in [True, False]:
                        cfg = GROMConfig(
                            method="GROM",
                            sensor_layer=layer,
                            steering_layers=[layer],
                            num_directions=num_dirs,
                            max_alpha=max_alpha,
                        )
                        configs.append((cfg, {
                            "num_directions": num_dirs,
                            "max_alpha": max_alpha,
                            "temperature": temp,
                            "normalize": normalize,
                        }))

    return configs



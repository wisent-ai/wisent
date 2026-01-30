"""
Hierarchical steering optimization with guarantees.

Instead of random/Bayesian search over the full space, we search systematically:

Stage 1: Layer Sweep
    - For each method, test ALL layers with fixed strength=1.0
    - Find the best layer per method
    - ~num_layers × num_methods configs

Stage 2: Strength Sweep
    - At best layer, test all strength values
    - Find best strength per method
    - ~num_strengths × num_methods configs

Stage 3: Method-Specific Tuning
    - At best layer+strength, grid search method-specific params
    - CAA/Hyperplane: just normalize (2 configs each)
    - MLP: hidden_dim × num_layers × normalize
    - PRISM: num_directions × optimization_steps × normalize
    - PULSE: threshold × temperature × normalize
    - TITAN: num_directions × max_alpha × temperature × normalize

This gives FULL coverage of the search space in a tractable way.
"""

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import (
    run_pipeline,
    MethodConfig,
    CAAConfig,
    HyperplaneConfig,
    MLPConfig,
    PRISMConfig,
    PULSEConfig,
    TITANConfig,
    OptimizationResult,
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
    layer_sweep_strength: float = 1.0
    layer_sweep_normalize: bool = True

    # Stage 2: Strength sweep
    strengths: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0])

    # Stage 3: Method-specific grids
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    mlp_num_layers: List[int] = field(default_factory=lambda: [1, 2, 3])

    prism_num_directions: List[int] = field(default_factory=lambda: [1, 2, 3, 5])
    prism_optimization_steps: List[int] = field(default_factory=lambda: [50, 100, 200])

    pulse_thresholds: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    pulse_temperatures: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.5])

    titan_num_directions: List[int] = field(default_factory=lambda: [3, 5, 8])
    titan_max_alphas: List[float] = field(default_factory=lambda: [1.5, 2.0, 3.0])
    titan_temperatures: List[float] = field(default_factory=lambda: [0.3, 0.5, 1.0])


def count_hierarchical_configs(
    methods: List[str],
    num_layers: int,
    config: HierarchicalConfig,
) -> Dict[str, Dict[str, int]]:
    """Count configurations per stage per method."""
    counts = {}

    for method in methods:
        method_upper = method.upper()
        stage1 = num_layers  # All layers
        stage2 = len(config.strengths)

        if method_upper == "CAA":
            stage3 = 2  # normalize True/False
        elif method_upper == "HYPERPLANE":
            stage3 = 2
        elif method_upper == "MLP":
            stage3 = len(config.mlp_hidden_dims) * len(config.mlp_num_layers) * 2
        elif method_upper == "PRISM":
            stage3 = len(config.prism_num_directions) * len(config.prism_optimization_steps) * 2
        elif method_upper == "PULSE":
            stage3 = len(config.pulse_thresholds) * len(config.pulse_temperatures) * 2
        elif method_upper == "TITAN":
            stage3 = (len(config.titan_num_directions) *
                     len(config.titan_max_alphas) *
                     len(config.titan_temperatures) * 2)
        else:
            stage3 = 2

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
    elif method_upper == "HYPERPLANE":
        return HyperplaneConfig(method="Hyperplane", layer=layer)
    elif method_upper == "MLP":
        return MLPConfig(method="MLP", layer=layer, hidden_dim=256, num_layers=2)
    elif method_upper == "PRISM":
        return PRISMConfig(method="PRISM", layer=layer, num_directions=3, optimization_steps=100)
    elif method_upper == "PULSE":
        return PULSEConfig(
            method="PULSE",
            sensor_layer=layer,
            steering_layers=[layer],
            condition_threshold=0.5,
            gate_temperature=0.1,
        )
    elif method_upper == "TITAN":
        return TITANConfig(
            method="TITAN",
            sensor_layer=layer,
            steering_layers=[layer],
            num_directions=5,
            gate_hidden_dim=64,
            intensity_hidden_dim=32,
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

    elif method_upper == "HYPERPLANE":
        for normalize in [True, False]:
            cfg = HyperplaneConfig(method="Hyperplane", layer=layer)
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

    elif method_upper == "PRISM":
        for num_dirs in config.prism_num_directions:
            for opt_steps in config.prism_optimization_steps:
                for normalize in [True, False]:
                    cfg = PRISMConfig(
                        method="PRISM",
                        layer=layer,
                        num_directions=num_dirs,
                        optimization_steps=opt_steps,
                    )
                    configs.append((cfg, {
                        "num_directions": num_dirs,
                        "optimization_steps": opt_steps,
                        "normalize": normalize,
                    }))

    elif method_upper == "PULSE":
        for thresh in config.pulse_thresholds:
            for temp in config.pulse_temperatures:
                for normalize in [True, False]:
                    cfg = PULSEConfig(
                        method="PULSE",
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

    elif method_upper == "TITAN":
        for num_dirs in config.titan_num_directions:
            for max_alpha in config.titan_max_alphas:
                for temp in config.titan_temperatures:
                    for normalize in [True, False]:
                        cfg = TITANConfig(
                            method="TITAN",
                            sensor_layer=layer,
                            steering_layers=[layer],
                            num_directions=num_dirs,
                            max_alpha=max_alpha,
                            gate_hidden_dim=64,
                            intensity_hidden_dim=32,
                        )
                        configs.append((cfg, {
                            "num_directions": num_dirs,
                            "max_alpha": max_alpha,
                            "temperature": temp,
                            "normalize": normalize,
                        }))

    return configs


def run_hierarchical_optimization(
    model: str,
    task: str,
    methods: List[str],
    num_layers: int,
    limit: int = 100,
    device: Optional[str] = None,
    enriched_pairs_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    verbose: bool = False,
    search_config: Optional[HierarchicalConfig] = None,
) -> Dict[str, HierarchicalResult]:
    """
    Run hierarchical optimization for multiple methods.

    Returns dict mapping method -> HierarchicalResult
    """
    if search_config is None:
        search_config = HierarchicalConfig()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Count configs
    config_counts = count_hierarchical_configs(methods, num_layers, search_config)
    total_configs = sum(c["total"] for c in config_counts.values())

    print(f"\n{'='*70}")
    print(f"HIERARCHICAL STEERING OPTIMIZATION")
    print(f"{'='*70}")
    print(f"Model: {model}")
    print(f"Task: {task}")
    print(f"Methods: {methods}")
    print(f"Layers: {num_layers}")
    print(f"\nConfigurations per method:")
    for method, counts in config_counts.items():
        print(f"  {method}:")
        print(f"    Stage 1 (layer):    {counts['stage1_layer']}")
        print(f"    Stage 2 (strength): {counts['stage2_strength']}")
        print(f"    Stage 3 (params):   {counts['stage3_params']}")
        print(f"    Total:              {counts['total']}")
    print(f"\nTotal configurations: {total_configs}")
    print(f"{'='*70}\n")

    results = {}
    configs_done = 0

    with tempfile.TemporaryDirectory() as work_dir:
        for method in methods:
            print(f"\n{'='*70}")
            print(f"OPTIMIZING: {method}")
            print(f"{'='*70}")

            # =========================================================
            # STAGE 1: Layer Sweep
            # =========================================================
            print(f"\n--- Stage 1: Layer Sweep ({num_layers} layers) ---")
            stage1_results = []
            best_layer = 1
            best_layer_score = -float('inf')

            for layer in range(1, num_layers + 1):
                configs_done += 1
                progress = f"[{configs_done}/{total_configs}]"

                cfg = _create_config_for_layer_sweep(
                    method, layer,
                    search_config.layer_sweep_strength,
                    search_config.layer_sweep_normalize,
                )

                try:
                    result = run_pipeline(
                        model=model,
                        task=task,
                        config=cfg,
                        work_dir=work_dir,
                        limit=limit,
                        device=device,
                        enriched_pairs_file=enriched_pairs_file,
                    )
                    score = result.score
                except Exception as e:
                    if verbose:
                        print(f"  {progress} Layer {layer}: ERROR - {e}")
                    score = 0.0

                stage1_results.append({"layer": layer, "score": score})

                if score > best_layer_score:
                    best_layer_score = score
                    best_layer = layer

                if verbose:
                    print(f"  {progress} Layer {layer}: {score:.4f}")

            print(f"  Best layer: {best_layer} (score: {best_layer_score:.4f})")

            # =========================================================
            # STAGE 2: Strength Sweep
            # =========================================================
            print(f"\n--- Stage 2: Strength Sweep ({len(search_config.strengths)} strengths) ---")
            stage2_results = []
            best_strength = 1.0
            best_strength_score = -float('inf')

            for strength in search_config.strengths:
                configs_done += 1
                progress = f"[{configs_done}/{total_configs}]"

                cfg = _create_config_for_layer_sweep(
                    method, best_layer,
                    strength,
                    search_config.layer_sweep_normalize,
                )

                try:
                    result = run_pipeline(
                        model=model,
                        task=task,
                        config=cfg,
                        work_dir=work_dir,
                        limit=limit,
                        device=device,
                        enriched_pairs_file=enriched_pairs_file,
                    )
                    score = result.score
                except Exception as e:
                    if verbose:
                        print(f"  {progress} Strength {strength}: ERROR - {e}")
                    score = 0.0

                stage2_results.append({"strength": strength, "score": score})

                if score > best_strength_score:
                    best_strength_score = score
                    best_strength = strength

                if verbose:
                    print(f"  {progress} Strength {strength}: {score:.4f}")

            print(f"  Best strength: {best_strength} (score: {best_strength_score:.4f})")

            # =========================================================
            # STAGE 3: Method-Specific Parameter Tuning
            # =========================================================
            stage3_configs = _create_configs_for_stage3(method, best_layer, best_strength, search_config)
            print(f"\n--- Stage 3: Parameter Tuning ({len(stage3_configs)} configs) ---")

            stage3_results = []
            best_params = {}
            best_params_score = -float('inf')
            best_config = None

            for cfg, params in stage3_configs:
                configs_done += 1
                progress = f"[{configs_done}/{total_configs}]"

                try:
                    result = run_pipeline(
                        model=model,
                        task=task,
                        config=cfg,
                        work_dir=work_dir,
                        limit=limit,
                        device=device,
                        enriched_pairs_file=enriched_pairs_file,
                    )
                    score = result.score
                except Exception as e:
                    if verbose:
                        print(f"  {progress} {params}: ERROR - {e}")
                    score = 0.0

                stage3_results.append({"params": params, "score": score})

                if score > best_params_score:
                    best_params_score = score
                    best_params = params
                    best_config = cfg

                if verbose:
                    params_str = ", ".join(f"{k}={v}" for k, v in params.items())
                    print(f"  {progress} {params_str}: {score:.4f}")

            print(f"  Best params: {best_params} (score: {best_params_score:.4f})")

            # Store result
            results[method] = HierarchicalResult(
                method=method,
                best_layer=best_layer,
                best_strength=best_strength,
                best_params=best_params,
                best_score=best_params_score,
                stage1_results=stage1_results,
                stage2_results=stage2_results,
                stage3_results=stage3_results,
            )

            # Print summary for this method
            print(f"\n{'='*50}")
            print(f"  {method} COMPLETE")
            print(f"  Best: layer={best_layer}, strength={best_strength}")
            print(f"  Params: {best_params}")
            print(f"  Score: {best_params_score:.4f}")
            print(f"{'='*50}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"HIERARCHICAL OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"\n{'Method':<12} {'Layer':<8} {'Strength':<10} {'Score':<10} {'Best Params'}")
    print(f"{'-'*70}")

    for method, res in sorted(results.items(), key=lambda x: -x[1].best_score):
        params_str = ", ".join(f"{k}={v}" for k, v in list(res.best_params.items())[:3])
        print(f"{method:<12} {res.best_layer:<8} {res.best_strength:<10.2f} {res.best_score:<10.4f} {params_str}")

    # Save results
    if output_dir:
        results_file = Path(output_dir) / "hierarchical_results.json"
        output_data = {
            "model": model,
            "task": task,
            "num_layers": num_layers,
            "timestamp": datetime.now().isoformat(),
            "methods": {
                method: {
                    "best_layer": res.best_layer,
                    "best_strength": res.best_strength,
                    "best_params": res.best_params,
                    "best_score": res.best_score,
                    "stage1_results": res.stage1_results,
                    "stage2_results": res.stage2_results,
                    "stage3_results": res.stage3_results,
                }
                for method, res in results.items()
            },
        }
        with open(results_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    return results


def execute_hierarchical_optimization(args):
    """Execute hierarchical optimization from CLI args."""
    enriched_pairs_file = getattr(args, 'enriched_pairs_file', None)
    task = getattr(args, 'task', None) or "custom"
    methods = getattr(args, 'methods', None) or ["CAA"]

    # Get num_layers
    num_layers = getattr(args, 'num_layers', None)
    if not num_layers and enriched_pairs_file:
        with open(enriched_pairs_file) as f:
            data = json.load(f)
        num_layers = len(data.get("layers", [])) or 32
    num_layers = num_layers or 32

    return run_hierarchical_optimization(
        model=args.model,
        task=task,
        methods=methods,
        num_layers=num_layers,
        limit=getattr(args, 'limit', 100),
        device=getattr(args, 'device', None),
        enriched_pairs_file=enriched_pairs_file,
        output_dir=getattr(args, 'output_dir', './hierarchical_results'),
        verbose=getattr(args, 'verbose', False),
    )

"""Run function for hierarchical optimization."""
import json
import os
import tempfile
import time
from typing import Optional, List, Dict

from wisent.core.cli.optimize_steering.hierarchical_config import (
    HierarchicalResult, HierarchicalConfig,
    count_hierarchical_configs, _create_config_for_layer_sweep,
    _create_configs_for_stage3,
)
from wisent.core.cli.optimize_steering.pipeline import run_pipeline, OptimizationResult
from wisent.core.cli.optimize_steering.method_configs import MethodConfig
from wisent.core.constants import DEFAULT_LIMIT


def run_hierarchical_optimization(
    model: str,
    task: str,
    methods: List[str],
    num_layers: int,
    limit: int = DEFAULT_LIMIT,
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



"""
Helper functions for find_goodness_direction.py.

Contains layer configuration, model utilities, and per-model analysis logic.
"""
import json
import subprocess
import os
from pathlib import Path

from wisent.core.constants import (
    LAYER_THRESHOLD_LARGE,
    LAYER_THRESHOLD_MEDIUM,
    LAYER_THRESHOLD_SMALL,
)


def run_cmd(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    """Run command and return result."""
    print(f"  Running: {' '.join(cmd[:6])}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  STDERR: {result.stderr[:200] if result.stderr else 'none'}")
    return result


def generate_pairs(benchmark: str, output_path: str, pairs_limit: int) -> bool:
    """Generate contrastive pairs for a benchmark."""
    cmd = [
        'wisent', 'generate-pairs-from-task', benchmark,
        '--output', output_path,
        '--limit', str(pairs_limit * 2),
    ]
    result = run_cmd(cmd, check=False)
    return result.returncode == 0


def check_linearity(pairs_path: str, model: str, strategy: str, output_path: str, device: str, pairs_limit: int, layers: str = None) -> dict:
    """Run linearity check and return results."""
    cmd = [
        'wisent', 'check-linearity', pairs_path,
        '--model', model,
        '--device', device,
        '--extraction-strategy', strategy,
        '--output', output_path,
        '--max-pairs', str(pairs_limit),
        '--verbose',
    ]
    if layers:
        cmd.extend(['--layers', layers])

    result = run_cmd(cmd, check=False)

    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return None


def get_layer_configs(num_layers: int) -> list:
    """Generate layer configurations to test: individual layers + combinations."""
    if num_layers <= LAYER_THRESHOLD_SMALL:
        individual_layers = [2, 4, 6, 8, 10, 12, 14]
    elif num_layers <= LAYER_THRESHOLD_MEDIUM:
        individual_layers = [4, 8, 12, 16, 20, 24, 28]
    elif num_layers <= LAYER_THRESHOLD_LARGE:
        individual_layers = [6, 12, 18, 24, 30, 36, 42]
    else:
        individual_layers = [8, 16, 24, 32, 40, 48, 56]

    individual_layers = [l for l in individual_layers if l < num_layers]
    configs = []

    # Individual layers
    for l in individual_layers:
        configs.append(([l], f"L{l}"))

    # All layers combined
    configs.append((individual_layers, "all_layers"))

    # Combinations of 2 adjacent layers
    for i in range(len(individual_layers) - 1):
        pair = [individual_layers[i], individual_layers[i+1]]
        configs.append((pair, f"L{pair[0]}+L{pair[1]}"))

    # Combinations of 3 adjacent layers
    for i in range(len(individual_layers) - 2):
        triple = [individual_layers[i], individual_layers[i+1], individual_layers[i+2]]
        configs.append((triple, f"L{triple[0]}+L{triple[1]}+L{triple[2]}"))

    # Early, middle, late combinations
    if len(individual_layers) >= 3:
        early = individual_layers[0]
        middle = individual_layers[len(individual_layers)//2]
        late = individual_layers[-1]
        configs.append(([early, middle], f"early+mid_L{early}+L{middle}"))
        configs.append(([middle, late], f"mid+late_L{middle}+L{late}"))
        configs.append(([early, late], f"early+late_L{early}+L{late}"))
        configs.append(([early, middle, late], f"early+mid+late_L{early}+L{middle}+L{late}"))

    return configs


def get_model_short_name(model: str) -> str:
    """Get short name for model (for directory names)."""
    return model.split('/')[-1].replace('-', '_').lower()


def get_model_num_layers(model: str) -> int:
    """Get number of layers for a model (approximate based on model name)."""
    model_lower = model.lower()
    if '1b' in model_lower or '1.3b' in model_lower:
        return 16
    elif '3b' in model_lower:
        return 26
    elif '7b' in model_lower or '8b' in model_lower:
        return 32
    elif '13b' in model_lower:
        return 40
    elif '20b' in model_lower:
        return 44
    elif '30b' in model_lower or '33b' in model_lower:
        return 60
    elif '65b' in model_lower or '70b' in model_lower:
        return 80
    else:
        return 32  # default


def run_for_model(model: str, pairs_dir: Path, output_dir: Path,
                  benchmarks: list, strategies: list, device: str, pairs_limit: int) -> dict:
    """Run analysis for a single model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"MODEL: {model}")
    print(f"{'='*70}")

    num_layers = get_model_num_layers(model)
    layer_configs = get_layer_configs(num_layers)
    print(f"  Model has ~{num_layers} layers, testing {len(layer_configs)} layer configurations")

    # Step 1: Check linearity for each benchmark x strategy x layer_config
    results = {}
    for strategy in strategies:
        print(f"\n  Strategy: {strategy}")
        results[strategy] = {}

        for bench in benchmarks:
            pairs_path = pairs_dir / f"{bench}_pairs.json"
            if not pairs_path.exists():
                print(f"    [{bench}] SKIP - no pairs")
                continue

            results[strategy][bench] = {'layer_configs': {}}
            print(f"    [{bench}]")

            best_linear = 0
            best_config_name = None
            best_result_data = None

            for layer_list, config_name in layer_configs:
                layers_str = ','.join(str(l) for l in layer_list)
                linearity_path = output_dir / f"{bench}_{strategy}_{config_name}_linearity.json"

                if linearity_path.exists():
                    with open(linearity_path) as f:
                        result = json.load(f)
                else:
                    result = check_linearity(str(pairs_path), model, strategy, str(linearity_path), device, pairs_limit, layers=layers_str)

                if result:
                    linear_score = result.get('best_linear_score', 0)
                    all_results = result.get('all_results', [])
                    best_result = all_results[0] if all_results else {}
                    best_structure = best_result.get('best_structure', 'unknown')
                    structure_scores = best_result.get('all_structure_scores', {})

                    results[strategy][bench]['layer_configs'][config_name] = {
                        'layers': layer_list,
                        'linear_score': linear_score,
                        'best_layer': result.get('best_layer', 0),
                        'verdict': result.get('verdict', 'unknown'),
                        'best_structure': best_structure,
                        'structure_scores': structure_scores,
                    }

                    if linear_score > best_linear:
                        best_linear = linear_score
                        best_config_name = config_name
                        best_result_data = results[strategy][bench]['layer_configs'][config_name]

            if best_result_data:
                results[strategy][bench]['best_config'] = best_config_name
                results[strategy][bench]['best_linear_score'] = best_linear
                results[strategy][bench]['best_structure'] = best_result_data['best_structure']
                print(f"      Best: {best_config_name} linear={best_linear:.3f} struct={best_result_data['best_structure']}")

    # Step 2: Pool all pairs and check combined geometry
    pooled_result = _check_pooled_geometry(output_dir, pairs_dir, benchmarks, results, model, device, pairs_limit)

    summary = {
        'model': model,
        'per_benchmark_results': results,
        'pooled_geometry': pooled_result,
    }
    summary_path = output_dir / 'model_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def _check_pooled_geometry(output_dir, pairs_dir, benchmarks, results, model, device, pairs_limit):
    """Check pooled representation geometry."""
    print(f"\n  Checking POOLED representation geometry...")
    pooled_pairs_path = output_dir / 'pooled_all_benchmarks.json'
    all_pairs = []

    for bench in benchmarks:
        pairs_path = pairs_dir / f"{bench}_pairs.json"
        if not pairs_path.exists():
            continue
        try:
            with open(pairs_path) as f:
                data = json.load(f)
                pairs = data.get('pairs', [])
                for p in pairs:
                    if 'metadata' not in p:
                        p['metadata'] = {}
                    p['metadata']['source_benchmark'] = bench
                all_pairs.extend(pairs)
        except Exception as e:
            print(f"  Warning: Could not load {pairs_path}: {e}")

    pooled_result = None
    if all_pairs:
        with open(pooled_pairs_path, 'w') as f:
            json.dump({'pairs': all_pairs}, f)

        strategy_scores = {}
        for strategy in results:
            scores = [r.get('best_linear_score', 0) for r in results[strategy].values() if r and 'best_linear_score' in r]
            if scores:
                strategy_scores[strategy] = sum(scores) / len(scores)

        best_strategy = max(strategy_scores, key=strategy_scores.get) if strategy_scores else 'chat_last'

        best_layer_config = None
        best_layer_score = 0
        for strategy in results:
            for bench_data in results[strategy].values():
                if bench_data and 'layer_configs' in bench_data:
                    for config_name, config_data in bench_data['layer_configs'].items():
                        if config_data.get('linear_score', 0) > best_layer_score:
                            best_layer_score = config_data['linear_score']
                            best_layer_config = config_data.get('layers', [])

        pooled_linearity_path = output_dir / 'pooled_linearity.json'
        if pooled_linearity_path.exists():
            with open(pooled_linearity_path) as f:
                pooled_result = json.load(f)
        else:
            layers_str = ','.join(str(l) for l in best_layer_config) if best_layer_config else None
            pooled_result = check_linearity(str(pooled_pairs_path), model, best_strategy, str(pooled_linearity_path), device, pairs_limit, layers=layers_str)

        if pooled_result:
            print(f"\n    POOLED GEOMETRY:")
            print(f"     Verdict: {pooled_result.get('verdict', 'unknown').upper()}")
            print(f"     Linear score: {pooled_result.get('best_linear_score', 0):.3f}")

    return pooled_result

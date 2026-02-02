"""Steering vector loading and generation for weight modification."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Any

import torch

from wisent.core.utils import resolve_default_device


def load_vectors_from_file(path: str, verbose: bool = False) -> Dict[int, torch.Tensor]:
    """Load steering vectors from a file (PT or JSON format)."""
    vector_path = Path(path)

    if vector_path.suffix == '.pt':
        checkpoint = torch.load(path, map_location=resolve_default_device(), weights_only=False)

        if 'steering_vectors' in checkpoint:
            raw_vectors = checkpoint['steering_vectors']
            first_key = next(iter(raw_vectors.keys()))
            if isinstance(first_key, str):
                steering_vectors = {
                    int(layer): vec if isinstance(vec, torch.Tensor) else torch.tensor(vec)
                    for layer, vec in raw_vectors.items()
                }
            else:
                steering_vectors = {
                    layer: vec if isinstance(vec, torch.Tensor) else torch.tensor(vec)
                    for layer, vec in raw_vectors.items()
                }
        elif 'vector' in checkpoint:
            layer = checkpoint.get('layer', checkpoint.get('best_layer', 14))
            steering_vectors = {layer: checkpoint['vector']}
        else:
            steering_vectors = {
                int(k): v if isinstance(v, torch.Tensor) else torch.tensor(v)
                for k, v in checkpoint.items()
                if isinstance(k, (int, str)) and str(k).isdigit()
            }

        if verbose:
            print(f"Loaded {len(steering_vectors)} steering vectors from .pt file")
            if 'metadata' in checkpoint:
                meta = checkpoint['metadata']
                if 'benchmarks_used' in meta:
                    print(f"  Trained on {len(meta['benchmarks_used'])} benchmarks")
                if 'optimal_scale' in meta:
                    print(f"  Optimal steering scale: {meta['optimal_scale']}")
    else:
        with open(path, 'r') as f:
            vector_data = json.load(f)

        if "steering_vectors" in vector_data:
            vectors_dict = vector_data["steering_vectors"]
        elif "vectors" in vector_data:
            vectors_dict = vector_data["vectors"]
        else:
            raise ValueError("No steering vectors found in file")

        steering_vectors = {int(layer): torch.tensor(vector) for layer, vector in vectors_dict.items()}

        if verbose:
            print(f"Loaded {len(steering_vectors)} steering vectors (layers: {list(steering_vectors.keys())})")

    return steering_vectors


def generate_personalization_vectors(args, verbose: bool = False) -> Dict[int, torch.Tensor]:
    """Generate steering vectors from trait-based synthetic pairs."""
    from wisent.core.cli.generate_vector_from_synthetic import execute_generate_vector_from_synthetic

    class VectorArgs:
        pass

    vector_args = VectorArgs()
    vector_args.trait = args.trait
    vector_args.model = args.model
    vector_args.num_pairs = args.num_pairs
    vector_args.similarity_threshold = getattr(args, 'similarity_threshold', 0.8)
    vector_args.layers = str(args.layers) if args.layers is not None else "all"
    vector_args.method = getattr(args, 'steering_method', 'caa')
    vector_args.normalize = args.normalize_vectors
    vector_args.verbose = verbose
    vector_args.timing = getattr(args, 'timing', False)
    vector_args.intermediate_dir = None
    vector_args.keep_intermediate = False
    vector_args.device = None
    vector_args.accept_low_quality_vector = getattr(args, 'accept_low_quality_vector', False)
    vector_args.pairs_cache_dir = getattr(args, 'pairs_cache_dir', None)
    vector_args.force_regenerate = getattr(args, 'force_regenerate', False)

    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    vector_args.output = temp_file.name
    temp_file.close()

    execute_generate_vector_from_synthetic(vector_args)

    with open(vector_args.output, 'r') as f:
        vector_data = json.load(f)

    vectors_dict = vector_data.get("steering_vectors") or vector_data.get("vectors", {})
    steering_vectors = {int(layer) - 1: torch.tensor(vector) for layer, vector in vectors_dict.items()}

    if getattr(args, 'save_steering_vectors', None):
        import shutil
        shutil.copy(vector_args.output, args.save_steering_vectors)

    os.unlink(vector_args.output)
    return steering_vectors


def generate_multi_benchmark_vectors(args, verbose: bool = False) -> Dict[int, torch.Tensor]:
    """Generate steering vectors from multiple benchmarks using unified goodness."""
    from wisent.core.cli.train_unified_goodness import execute_train_unified_goodness

    benchmarks = [b.strip() for b in args.task.split(",")]
    if verbose:
        print(f"Generating vectors from {len(benchmarks)} benchmarks: {', '.join(benchmarks)}...")

    class UnifiedArgs:
        pass

    unified_args = UnifiedArgs()
    unified_args.task = args.task
    unified_args.exclude_benchmarks = None
    unified_args.max_benchmarks = getattr(args, 'max_benchmarks', None)
    unified_args.cap_pairs_per_benchmark = getattr(args, 'cap_pairs_per_benchmark', None)
    unified_args.train_ratio = getattr(args, 'train_ratio', 0.8)
    unified_args.seed = getattr(args, 'seed', 42)
    unified_args.model = args.model
    unified_args.device = getattr(args, 'device', None)
    unified_args.layer = None
    unified_args.layers = args.layers
    unified_args.method = "caa"
    unified_args.normalize = args.normalize_vectors if hasattr(args, 'normalize_vectors') else False
    unified_args.no_normalize = not unified_args.normalize
    unified_args.skip_evaluation = True
    unified_args.evaluate_steering_scales = "0.0,1.0"
    unified_args.save_pairs = None
    unified_args.save_report = None
    unified_args.verbose = verbose
    unified_args.timing = args.timing if hasattr(args, 'timing') else False

    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.pt', delete=False)
    unified_args.output = temp_file.name
    temp_file.close()

    execute_train_unified_goodness(unified_args)

    checkpoint = torch.load(unified_args.output, map_location=resolve_default_device(), weights_only=False)

    if 'steering_vectors' in checkpoint:
        raw_vectors = checkpoint['steering_vectors']
    elif 'all_layer_vectors' in checkpoint:
        raw_vectors = checkpoint['all_layer_vectors']
    elif 'steering_vector' in checkpoint and 'layer_index' in checkpoint:
        raw_vectors = {checkpoint['layer_index']: checkpoint['steering_vector']}
    else:
        raw_vectors = {k: v for k, v in checkpoint.items() if isinstance(k, (int, str)) and str(k).isdigit()}

    steering_vectors = {}
    for layer, vec in raw_vectors.items():
        layer_idx = int(layer) if isinstance(layer, str) else layer
        steering_vectors[layer_idx] = vec if isinstance(vec, torch.Tensor) else torch.tensor(vec)

    if hasattr(args, 'save_steering_vectors') and args.save_steering_vectors:
        import shutil
        shutil.copy(unified_args.output, args.save_steering_vectors)

    os.unlink(unified_args.output)
    return steering_vectors


def generate_task_vectors(args, verbose: bool = False) -> Dict[int, torch.Tensor]:
    """Generate steering vectors from a single benchmark task."""
    from wisent.core.cli.generate_vector_from_task import execute_generate_vector_from_task
    from wisent.core.activations import ExtractionStrategy

    optimal_config = _get_optimal_config(args)

    if verbose:
        print(f"Generating steering vectors from task '{args.task}'...")

    class VectorArgs:
        pass

    vector_args = VectorArgs()
    vector_args.task = args.task
    vector_args.trait_label = getattr(args, 'trait_label', 'correctness')
    vector_args.model = args.model
    vector_args.num_pairs = args.num_pairs

    if optimal_config and args.layers is None:
        vector_args.layers = str(optimal_config['layer'])
    else:
        vector_args.layers = str(args.layers) if args.layers is not None else "all"

    if optimal_config:
        vector_args.extraction_strategy = _map_token_aggregation(optimal_config)
        vector_args.method = optimal_config['method'].lower()
    else:
        vector_args.extraction_strategy = ExtractionStrategy.default().value
        steering_method = getattr(args, 'steering_method', 'caa')
        if steering_method == 'auto':
            steering_method = 'caa'
        vector_args.method = steering_method

    vector_args.normalize = args.normalize_vectors
    vector_args.verbose = verbose
    vector_args.timing = getattr(args, 'timing', False)
    vector_args.intermediate_dir = None
    vector_args.keep_intermediate = False
    vector_args.device = None
    vector_args.accept_low_quality_vector = getattr(args, 'accept_low_quality_vector', False)
    vector_args.use_optimal = getattr(args, 'use_optimal', True)

    if optimal_config:
        vector_args._optimal_config = optimal_config

    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    vector_args.output = temp_file.name
    temp_file.close()

    execute_generate_vector_from_task(vector_args)

    with open(vector_args.output, 'r') as f:
        vector_data = json.load(f)

    method_name = vector_data.get("method", "caa")

    if method_name == "titan":
        directions = vector_data.get("directions", {})
        direction_weights = vector_data.get("direction_weights", {})

        steering_vectors = {}
        for layer_str, dirs in directions.items():
            layer = int(layer_str)
            dirs_tensor = torch.tensor(dirs)
            weights = direction_weights.get(layer_str)
            if weights is not None:
                weights_tensor = torch.tensor(weights)
                effective_dir = (dirs_tensor * weights_tensor.unsqueeze(-1)).sum(0)
            else:
                effective_dir = dirs_tensor.mean(0)
            steering_vectors[layer] = effective_dir
    else:
        vectors_dict = vector_data.get("steering_vectors") or vector_data.get("vectors", {})
        steering_vectors = {int(layer) - 1: torch.tensor(vector) for layer, vector in vectors_dict.items()}

    if getattr(args, 'save_steering_vectors', None):
        import shutil
        shutil.copy(vector_args.output, args.save_steering_vectors)

    os.unlink(vector_args.output)
    return steering_vectors


def _get_optimal_config(args) -> Optional[Dict[str, Any]]:
    """Get optimal steering config from cache if available."""
    use_optimal = getattr(args, 'use_optimal', True)
    if not use_optimal:
        return None

    try:
        from wisent.core.config_manager import get_cached_optimization
        optimal_result = get_cached_optimization(args.model, args.task, method="*")
        if optimal_result:
            return {
                "method": optimal_result.method,
                "layer": optimal_result.layer,
                "strength": optimal_result.strength,
                "strategy": optimal_result.strategy,
                "token_aggregation": optimal_result.token_aggregation,
                "prompt_strategy": optimal_result.prompt_strategy,
                "score": optimal_result.score,
                "num_directions": optimal_result.num_directions,
                "direction_weighting": optimal_result.direction_weighting,
            }
    except Exception:
        pass
    return None


def _map_token_aggregation(optimal_config: Dict[str, Any]) -> str:
    """Map token aggregation to extraction strategy."""
    from wisent.core.activations import ExtractionStrategy
    if optimal_config.get('extraction_strategy'):
        return optimal_config['extraction_strategy']
    if optimal_config.get('token_aggregation'):
        mapping = {"last_token": "last_token", "mean_pooling": "mean_pooling", "first_token": "first_token",
                   "max_pooling": "max_pooling", "final": "last_token", "average": "mean_pooling"}
        return mapping.get(optimal_config['token_aggregation'], ExtractionStrategy.default().value)
    return ExtractionStrategy.default().value

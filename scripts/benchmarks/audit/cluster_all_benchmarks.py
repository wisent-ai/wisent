#!/usr/bin/env python3
"""
Cluster ALL benchmarks by direction similarity AND analyze geometry structure.
Tests 7 extraction strategies and multiple layers per model.

Run on AWS with: ./run_on_aws.sh --model meta-llama/Llama-3.2-1B-Instruct "python scripts/cluster_all_benchmarks.py"
"""
import torch
import numpy as np
import random
import json
import sys
import os
import logging
import gc
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('datasets').setLevel(logging.WARNING)

import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModelForCausalLM

_repo = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_repo))
from wisent.core.utils.services.benchmarks.registry.benchmark_registry import get_all_benchmarks
from wisent.core.utils.infra_tools.data.loaders.lm_eval.lm_loader import LMEvalDataLoader
from wisent.core.primitives.contrastive_pairs.diagnostics import (
    detect_geometry_structure,
    GeometryAnalysisConfig,
)

from wisent.core.utils.config_tools.constants import DEFAULT_RANDOM_SEED, GEOMETRY_DEFAULT_NUM_COMPONENTS, GEOMETRY_OPTIMIZATION_STEPS_DEFAULT, PROGRESS_LOG_INTERVAL

from cluster_all_benchmarks_strategies import (
    compute_directions_for_strategy,
    load_benchmark_pairs,
)
from cluster_all_benchmarks_clustering import (
    find_optimal_clusters,
    evaluate_directions,
    save_and_print_final_results,
)

STRATEGIES = [
    "chat_mean", "chat_first", "chat_last", "chat_max_norm",
    "chat_weighted", "role_play", "mc_balanced",
]


@dataclass
class ConfigResult:
    layer: int
    strategy: str
    n_benchmarks: int
    global_accuracy: float
    cluster_accuracy: float
    optimal_clusters: int
    combined_geometry: str
    geometry_distribution: Dict[str, int]


def load_model(model_name: str, device: str):
    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dtype = torch.bfloat16 if device == 'cuda' else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device, trust_remote_code=True,
    )
    return model, tokenizer


def get_layers_to_test(model) -> List[int]:
    """Get layers based on model depth."""
    num_layers = model.config.num_hidden_layers
    if num_layers <= 16:
        test_layers = [4, 6, 8, 10, 12, 14]
    elif num_layers <= 32:
        test_layers = [8, 12, 16, 20, 24, 28]
    else:
        test_layers = [10, 20, 30, 40, 50, 60]
    return [l for l in test_layers if l < num_layers]


def main():
    MODEL_NAME = os.environ.get('MODEL_NAME', 'meta-llama/Llama-3.2-1B-Instruct')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR', '/home/ubuntu/output'))
    PAIRS_PER_BENCHMARK = int(os.environ.get('PAIRS_PER_BENCHMARK', '50'))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(DEFAULT_RANDOM_SEED)
    np.random.seed(DEFAULT_RANDOM_SEED)
    torch.manual_seed(DEFAULT_RANDOM_SEED)

    model, tokenizer = load_model(MODEL_NAME, DEVICE)
    LAYERS = get_layers_to_test(model)

    logger.info(f"Model: {MODEL_NAME}, Device: {DEVICE}, Layers: {LAYERS}")
    logger.info(f"Total configs: {len(LAYERS)} layers x {len(STRATEGIES)} strategies = {len(LAYERS) * len(STRATEGIES)}")

    # Phase 1: Load all benchmark pairs
    all_benchmarks = get_all_benchmarks()
    logger.info(f"\nPhase 1: Loading {len(all_benchmarks)} benchmarks...")
    loader = LMEvalDataLoader()
    all_pairs = {}
    for i, bench in enumerate(all_benchmarks):
        if (i + 1) % 20 == 0:
            logger.info(f"  [{i+1}/{len(all_benchmarks)}] Loaded {len(all_pairs)} benchmarks so far...")
        try:
            pairs = load_benchmark_pairs(bench, loader, limit=PAIRS_PER_BENCHMARK)
            if pairs and len(pairs) >= 10:
                all_pairs[bench] = pairs
        except:
            pass
    logger.info(f"Loaded {len(all_pairs)} benchmarks with sufficient pairs")

    # Phase 2: Test all layer x strategy combinations
    logger.info(f"\nPhase 2: Testing {len(LAYERS) * len(STRATEGIES)} combinations...")
    geo_config = GeometryAnalysisConfig(num_components=GEOMETRY_DEFAULT_NUM_COMPONENTS, optimization_steps=GEOMETRY_OPTIMIZATION_STEPS_DEFAULT)
    all_config_results = []
    best_config = None
    best_cluster_acc = 0

    for layer in LAYERS:
        for strategy in STRATEGIES:
            logger.info(f"\nConfig: Layer {layer}, Strategy {strategy}")
            directions, activations, geometry_dist = {}, {}, {}

            for bench_idx, (bench, pairs) in enumerate(all_pairs.items()):
                if (bench_idx + 1) % PROGRESS_LOG_INTERVAL == 0:
                    logger.info(f"  [{bench_idx+1}/{len(all_pairs)}] Processing...")
                try:
                    direction, pos_t, neg_t = compute_directions_for_strategy(
                        model, tokenizer, pairs, layer, DEVICE, strategy
                    )
                    if direction is not None:
                        directions[bench] = direction
                        activations[bench] = {'pos': pos_t, 'neg': neg_t}
                        try:
                            geo = detect_geometry_structure(pos_t, neg_t, geo_config)
                            gtype = geo.best_structure.value
                            geometry_dist[gtype] = geometry_dist.get(gtype, 0) + 1
                        except:
                            pass
                except:
                    pass

            if len(directions) < 10:
                logger.warning(f"  Only {len(directions)} benchmarks - skipping")
                continue

            bench_names = list(directions.keys())
            n = len(bench_names)
            sim_matrix = np.zeros((n, n))
            for i, n1 in enumerate(bench_names):
                for j, n2 in enumerate(bench_names):
                    sim_matrix[i, j] = torch.dot(directions[n1], directions[n2]).item()

            optimal_n, clusters = find_optimal_clusters(sim_matrix, bench_names)
            global_acc, cluster_acc = evaluate_directions(directions, activations, clusters)

            all_pos = torch.cat([activations[b]['pos'] for b in activations])
            all_neg = torch.cat([activations[b]['neg'] for b in activations])
            try:
                combined_geo = detect_geometry_structure(all_pos, all_neg, geo_config)
                combined_type = combined_geo.best_structure.value
            except:
                combined_type = "error"

            result = ConfigResult(
                layer=layer, strategy=strategy, n_benchmarks=n,
                global_accuracy=float(global_acc), cluster_accuracy=float(cluster_acc),
                optimal_clusters=optimal_n, combined_geometry=combined_type,
                geometry_distribution=geometry_dist
            )
            all_config_results.append(result)
            logger.info(f"  Benchmarks: {n}, Global: {global_acc:.3f}, Cluster: {cluster_acc:.3f}")

            if cluster_acc > best_cluster_acc:
                best_cluster_acc = cluster_acc
                best_config = {
                    'layer': layer, 'strategy': strategy,
                    'directions': {k: v.tolist() for k, v in directions.items()},
                    'sim_matrix': sim_matrix.tolist(), 'bench_names': bench_names,
                    'clusters': clusters, 'global_acc': float(global_acc),
                    'cluster_acc': float(cluster_acc), 'geometry_dist': geometry_dist,
                    'combined_geometry': combined_type, 'optimal_n': optimal_n,
                }

            intermediate = {
                'model': MODEL_NAME,
                'configs_completed': len(all_config_results),
                'best_so_far': {
                    'layer': best_config['layer'] if best_config else None,
                    'strategy': best_config['strategy'] if best_config else None,
                    'cluster_acc': best_cluster_acc,
                },
                'all_results': [asdict(r) for r in all_config_results],
            }
            with open(OUTPUT_DIR / 'intermediate_results.json', 'w') as f:
                json.dump(intermediate, f, indent=2)

    save_and_print_final_results(best_config, all_config_results, MODEL_NAME, model, OUTPUT_DIR, LAYERS, STRATEGIES)

    del model
    del tokenizer
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    elif DEVICE == 'mps':
        torch.mps.empty_cache()


if __name__ == '__main__':
    main()

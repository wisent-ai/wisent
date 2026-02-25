#!/usr/bin/env python3
"""
CLI command for clustering benchmarks by direction similarity with geometry analysis.
Tests 8 extraction strategies and multiple layers per model.

Usage: wisent cluster-benchmarks --model meta-llama/Llama-3.2-1B-Instruct --output ./results
"""
import torch
import numpy as np
import random
import json
import gc
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.cluster import AgglomerativeClustering

from wisent.core.benchmarks import get_all_benchmarks
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import lm_build_contrastive_pairs
from wisent.core.data_loaders.loaders.lm_eval.lm_loader import LMEvalDataLoader
from wisent.core.contrastive_pairs.huggingface_pairs.hf_extractor_manifest import HF_EXTRACTORS
from wisent.core.contrastive_pairs.diagnostics import (
    detect_geometry_structure,
    GeometryAnalysisConfig,
)

logger = logging.getLogger(__name__)

STRATEGIES = [
    "chat_mean",
    "chat_first",
    "chat_last",
    "chat_max_norm",
    "chat_weighted",
    "role_play",
    "mc_balanced",
]

RANDOM_TOKENS = ["I", "Well", "The", "Sure", "Let", "That", "It", "This", "My", "To"]


from wisent.core.constants import (
    NORM_EPS, CLUSTER_PROGRESS_INTERVAL, CLUSTER_MIN_PAIRS,
    GEOMETRY_DEFAULT_NUM_COMPONENTS, GEOMETRY_OPTIMIZATION_STEPS_DEFAULT,
    DEFAULT_RANDOM_SEED, JSON_INDENT, CHANCE_LEVEL_ACCURACY,
)
from wisent.core.cli.analysis.diagnosis.cluster_benchmarks_activations import (
    ConfigResult, get_layers_to_test, get_activation, get_mc_balanced_activations,
    load_benchmark_pairs, compute_directions_for_strategy,
)


def evaluate_directions(directions, activations, clusters):
    all_pos = torch.cat([activations[b]['pos'] for b in activations])
    all_neg = torch.cat([activations[b]['neg'] for b in activations])
    global_dir = all_pos.mean(dim=0) - all_neg.mean(dim=0)
    norm = torch.norm(global_dir)
    if norm > NORM_EPS:
        global_dir = global_dir / norm
    
    cluster_dirs = {}
    bench_to_cluster = {}
    for cid, members in clusters.items():
        valid = [m for m in members if m in activations]
        if valid:
            p = torch.cat([activations[m]['pos'] for m in valid])
            n = torch.cat([activations[m]['neg'] for m in valid])
            d = p.mean(dim=0) - n.mean(dim=0)
            norm = torch.norm(d)
            if norm > NORM_EPS:
                cluster_dirs[cid] = d / norm
            for m in members:
                bench_to_cluster[m] = cid
    
    global_accs, cluster_accs = [], []
    for bench, acts in activations.items():
        pos, neg = acts['pos'], acts['neg']
        n = min(len(pos), len(neg))
        
        g_correct = sum(1 for i in range(n) if torch.dot(pos[i], global_dir) > torch.dot(neg[i], global_dir))
        global_accs.append(g_correct / n if n > 0 else CHANCE_LEVEL_ACCURACY)
        
        cid = bench_to_cluster.get(bench)
        if cid in cluster_dirs:
            c_correct = sum(1 for i in range(n) if torch.dot(pos[i], cluster_dirs[cid]) > torch.dot(neg[i], cluster_dirs[cid]))
            cluster_accs.append(c_correct / n if n > 0 else CHANCE_LEVEL_ACCURACY)
        else:
            cluster_accs.append(global_accs[-1])
    
    return np.mean(global_accs), np.mean(cluster_accs)


def execute_cluster_benchmarks(args):
    """Execute cluster-benchmarks command."""
    model = args.model
    output = args.output
    pairs_per_benchmark = args.pairs_per_benchmark
    device = args.device
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    random.seed(DEFAULT_RANDOM_SEED)
    np.random.seed(DEFAULT_RANDOM_SEED)
    torch.manual_seed(DEFAULT_RANDOM_SEED)
    
    logger.info(f"Loading {model}...")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    from wisent.core.utils import device_optimized_dtype
    dtype = device_optimized_dtype(device)
    llm = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype, device_map=device, trust_remote_code=True)
    
    layers = get_layers_to_test(llm)
    
    logger.info(f"Model: {model}")
    logger.info(f"Device: {device}")
    logger.info(f"Layers: {layers} (of {llm.config.num_hidden_layers})")
    logger.info(f"Strategies: {STRATEGIES}")
    
    # Load benchmarks
    all_benchmarks = get_all_benchmarks()
    logger.info(f"\nLoading {len(all_benchmarks)} benchmarks...")
    
    loader = LMEvalDataLoader()
    all_pairs = {}
    
    for i, bench in enumerate(all_benchmarks):
        if (i + 1) % CLUSTER_PROGRESS_INTERVAL == 0:
            logger.info(f"  [{i+1}/{len(all_benchmarks)}] Loaded {len(all_pairs)} benchmarks...")
        try:
            pairs = load_benchmark_pairs(bench, loader, limit=pairs_per_benchmark)
            if pairs and len(pairs) >= CLUSTER_MIN_PAIRS:
                all_pairs[bench] = pairs
        except:
            pass
    
    logger.info(f"Loaded {len(all_pairs)} benchmarks")
    
    # Test configurations
    geo_config = GeometryAnalysisConfig(num_components=GEOMETRY_DEFAULT_NUM_COMPONENTS, optimization_steps=GEOMETRY_OPTIMIZATION_STEPS_DEFAULT)
    all_results = []
    best_config = None
    best_acc = 0
    
    for layer in layers:
        for strategy in STRATEGIES:
            logger.info(f"\nTesting: Layer {layer}, Strategy {strategy}")
            
            directions, activations, geo_dist = {}, {}, {}
            
            for bench, pairs in all_pairs.items():
                try:
                    direction, pos_t, neg_t = compute_directions_for_strategy(llm, tokenizer, pairs, layer, device, strategy)
                    if direction is not None:
                        directions[bench] = direction
                        activations[bench] = {'pos': pos_t, 'neg': neg_t}
                        try:
                            geo = detect_geometry_structure(pos_t, neg_t, geo_config)
                            gtype = geo.best_structure.value
                            geo_dist[gtype] = geo_dist.get(gtype, 0) + 1
                        except:
                            pass
                except:
                    pass
            
            if len(directions) < CLUSTER_MIN_PAIRS:
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
            
            result = ConfigResult(layer, strategy, n, float(global_acc), float(cluster_acc), optimal_n, combined_type, geo_dist)
            all_results.append(result)
            
            logger.info(f"  Global: {global_acc:.3f}, Cluster: {cluster_acc:.3f}, Geo: {combined_type}")
            
            if cluster_acc > best_acc:
                best_acc = cluster_acc
                best_config = {
                    'layer': layer, 'strategy': strategy, 'bench_names': bench_names,
                    'clusters': clusters, 'global_acc': float(global_acc),
                    'cluster_acc': float(cluster_acc), 'geo_dist': geo_dist,
                    'combined_geometry': combined_type, 'optimal_n': optimal_n,
                    'sim_matrix': sim_matrix.tolist(),
                }
            
            # Save intermediate
            with open(output_dir / 'intermediate.json', 'w') as f:
                json.dump({'results': [asdict(r) for r in all_results], 'best_acc': best_acc}, f, indent=JSON_INDENT)
    
    # Save final
    if best_config:
        summary = {
            'model': model,
            'layers_tested': layers,
            'strategies_tested': STRATEGIES,
            'best_layer': best_config['layer'],
            'best_strategy': best_config['strategy'],
            'n_benchmarks': len(best_config['bench_names']),
            'optimal_clusters': best_config['optimal_n'],
            'global_accuracy': best_config['global_acc'],
            'cluster_accuracy': best_config['cluster_acc'],
            'clusters': best_config['clusters'],
            'combined_geometry': best_config['combined_geometry'],
            'geometry_distribution': best_config['geo_dist'],
            'all_configs': [asdict(r) for r in all_results],
        }
        
        with open(output_dir / 'cluster_summary.json', 'w') as f:
            json.dump(summary, f, indent=JSON_INDENT)
        
        print(f"\nBest: Layer {best_config['layer']}, Strategy {best_config['strategy']}")
        print(f"Global: {best_config['global_acc']:.3f}, Cluster: {best_config['cluster_acc']:.3f}")
    
    del llm
    gc.collect()

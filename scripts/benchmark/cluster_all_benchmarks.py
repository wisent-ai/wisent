#!/usr/bin/env python3
"""
Cluster ALL benchmarks by direction similarity AND analyze geometry structure.
Tests 8 extraction strategies and multiple layers per model.

Strategies:
- chat_mean: Direct Q+A chat format, mean of answer tokens
- chat_first: Direct Q+A chat format, first answer token
- chat_last: Direct Q+A chat format, last token
- chat_max_norm: Direct Q+A chat format, token with max norm in answer
- chat_weighted: Direct Q+A chat format, position-weighted mean
- role_play: "Behave like person who answers Q with A" format, last token
- mc_balanced: Multiple choice format, last token

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
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('datasets').setLevel(logging.WARNING)

import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.cluster import AgglomerativeClustering

sys.path.insert(0, str(Path(__file__).parent.parent))
from wisent.core.benchmark_registry import get_all_benchmarks
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import lm_build_contrastive_pairs
from wisent.core.data_loaders.loaders.lm_eval.lm_loader import LMEvalDataLoader
from wisent.core.contrastive_pairs.huggingface_pairs.hf_extractor_manifest import HF_EXTRACTORS
from wisent.core.contrastive_pairs.diagnostics import (
    detect_geometry_structure,
    GeometryAnalysisConfig,
)

# 7 extraction strategies
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
    """Get layers based on model depth (same as test_all_models_tasks.py)."""
    num_layers = model.config.num_hidden_layers
    if num_layers <= 16:
        test_layers = [4, 6, 8, 10, 12, 14]
    elif num_layers <= 32:
        test_layers = [8, 12, 16, 20, 24, 28]
    else:
        test_layers = [10, 20, 30, 40, 50, 60]
    return [l for l in test_layers if l < num_layers]


def get_last_token_act(model, tokenizer, text: str, layer: int, device: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    return outputs.hidden_states[layer][0, -1, :].cpu().float()


def get_mean_answer_tokens_act(model, tokenizer, text: str, answer: str, layer: int, device: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    answer_tokens = tokenizer(answer, add_special_tokens=False)["input_ids"]
    num_answer_tokens = len(answer_tokens)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0]
    if num_answer_tokens > 0 and num_answer_tokens < hidden.shape[0]:
        answer_hidden = hidden[-num_answer_tokens-1:-1, :]
        return answer_hidden.mean(dim=0).cpu().float()
    return hidden[-1].cpu().float()


def get_first_answer_token_act(model, tokenizer, text: str, answer: str, layer: int, device: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    answer_tokens = tokenizer(answer, add_special_tokens=False)["input_ids"]
    num_answer_tokens = len(answer_tokens)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0]
    if num_answer_tokens > 0 and num_answer_tokens < hidden.shape[0]:
        first_answer_idx = hidden.shape[0] - num_answer_tokens - 1
        return hidden[first_answer_idx, :].cpu().float()
    return hidden[-1].cpu().float()


def get_generation_point_act(model, tokenizer, text: str, answer: str, layer: int, device: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    answer_tokens = tokenizer(answer, add_special_tokens=False)["input_ids"]
    num_answer_tokens = len(answer_tokens)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0]
    gen_point_idx = max(0, hidden.shape[0] - num_answer_tokens - 2)
    return hidden[gen_point_idx, :].cpu().float()


def get_max_norm_answer_act(model, tokenizer, text: str, answer: str, layer: int, device: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    answer_tokens = tokenizer(answer, add_special_tokens=False)["input_ids"]
    num_answer_tokens = len(answer_tokens)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0]
    if num_answer_tokens > 0 and num_answer_tokens < hidden.shape[0]:
        answer_hidden = hidden[-num_answer_tokens-1:-1, :]
        norms = torch.norm(answer_hidden, dim=1)
        max_idx = torch.argmax(norms)
        return answer_hidden[max_idx, :].cpu().float()
    return hidden[-1].cpu().float()


def get_weighted_mean_answer_act(model, tokenizer, text: str, answer: str, layer: int, device: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    answer_tokens = tokenizer(answer, add_special_tokens=False)["input_ids"]
    num_answer_tokens = len(answer_tokens)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0]
    if num_answer_tokens > 0 and num_answer_tokens < hidden.shape[0]:
        answer_hidden = hidden[-num_answer_tokens-1:-1, :]
        weights = torch.exp(-torch.arange(answer_hidden.shape[0], dtype=torch.float32) * 0.5)
        weights = weights / weights.sum()
        weighted_mean = (answer_hidden * weights.unsqueeze(1).to(answer_hidden.device)).sum(dim=0)
        return weighted_mean.cpu().float()
    return hidden[-1].cpu().float()


def get_activation(model, tokenizer, prompt: str, response: str, layer: int, device: str, strategy: str) -> torch.Tensor:
    """Get activation using specified strategy."""
    random_token = RANDOM_TOKENS[hash(prompt) % len(RANDOM_TOKENS)]
    
    if strategy.startswith("chat_"):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt[:500]}, {"role": "assistant", "content": response}],
            tokenize=False, add_generation_prompt=False
        )
        
        if strategy == "chat_mean":
            return get_mean_answer_tokens_act(model, tokenizer, text, response, layer, device)
        elif strategy == "chat_first":
            return get_first_answer_token_act(model, tokenizer, text, response, layer, device)
        elif strategy == "chat_last":
            return get_last_token_act(model, tokenizer, text, layer, device)
        elif strategy == "chat_max_norm":
            return get_max_norm_answer_act(model, tokenizer, text, response, layer, device)
        elif strategy == "chat_weighted":
            return get_weighted_mean_answer_act(model, tokenizer, text, response, layer, device)
    
    elif strategy == "role_play":
        instruction = f"Behave like a person that would answer {prompt[:300]} with {response[:200]}"
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}, {"role": "assistant", "content": random_token}],
            tokenize=False, add_generation_prompt=False
        )
        return get_last_token_act(model, tokenizer, text, layer, device)
    
    elif strategy == "mc_balanced":
        # This needs both positive and negative - handled separately
        raise ValueError("mc_balanced requires special handling")
    
    # Fallback
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt[:500]}, {"role": "assistant", "content": response}],
        tokenize=False, add_generation_prompt=False
    )
    return get_last_token_act(model, tokenizer, text, layer, device)


def get_mc_balanced_activations(model, tokenizer, prompt: str, pos_response: str, neg_response: str, layer: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get activations for mc_balanced strategy (needs both responses)."""
    pos_goes_in_b = hash(prompt) % 2 == 0
    if pos_goes_in_b:
        mc_text = f"Which is correct?\nA. {neg_response[:200]}\nB. {pos_response[:200]}\nAnswer:"
        pos_ans, neg_ans = "B", "A"
    else:
        mc_text = f"Which is correct?\nA. {pos_response[:200]}\nB. {neg_response[:200]}\nAnswer:"
        pos_ans, neg_ans = "A", "B"
    
    pos_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": mc_text}, {"role": "assistant", "content": pos_ans}],
        tokenize=False, add_generation_prompt=False
    )
    neg_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": mc_text}, {"role": "assistant", "content": neg_ans}],
        tokenize=False, add_generation_prompt=False
    )
    
    pos_act = get_last_token_act(model, tokenizer, pos_text, layer, device)
    neg_act = get_last_token_act(model, tokenizer, neg_text, layer, device)
    return pos_act, neg_act


def load_benchmark_pairs(benchmark_name: str, loader: LMEvalDataLoader, limit: int = 60) -> List:
    task_name_lower = benchmark_name.lower()
    is_hf = task_name_lower in {k.lower() for k in HF_EXTRACTORS.keys()}
    
    if is_hf:
        pairs = lm_build_contrastive_pairs(task_name=benchmark_name, lm_eval_task=None, limit=limit)
    else:
        task_obj = loader.load_lm_eval_task(benchmark_name)
        if isinstance(task_obj, dict):
            pairs = []
            for subname, subtask in list(task_obj.items())[:3]:
                try:
                    sub_pairs = lm_build_contrastive_pairs(task_name=subname, lm_eval_task=subtask, limit=limit//3)
                    pairs.extend(sub_pairs)
                except:
                    pass
        else:
            pairs = lm_build_contrastive_pairs(task_name=benchmark_name, lm_eval_task=task_obj, limit=limit)
    return pairs


def compute_directions_for_strategy(
    model, tokenizer, pairs: List, layer: int, device: str, strategy: str, max_pairs: int = 50
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Compute steering direction using specified strategy."""
    pos_acts, neg_acts = [], []
    
    for pair in pairs[:max_pairs]:
        try:
            prompt = pair.prompt
            pos_response = pair.positive_response.model_response
            neg_response = pair.negative_response.model_response
            
            if strategy == "mc_balanced":
                pos_act, neg_act = get_mc_balanced_activations(
                    model, tokenizer, prompt, pos_response, neg_response, layer, device
                )
            else:
                pos_act = get_activation(model, tokenizer, prompt, pos_response, layer, device, strategy)
                neg_act = get_activation(model, tokenizer, prompt, neg_response, layer, device, strategy)
            
            pos_acts.append(pos_act)
            neg_acts.append(neg_act)
        except Exception as e:
            continue
    
    if len(pos_acts) < 10:
        return None, None, None
    
    pos_tensor = torch.stack(pos_acts)
    neg_tensor = torch.stack(neg_acts)
    direction = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
    norm = torch.norm(direction)
    if norm > 1e-8:
        direction = direction / norm
    return direction, pos_tensor, neg_tensor


def find_optimal_clusters(sim_matrix: np.ndarray, names: List[str], max_clusters: int = 10):
    dist_matrix = 1 - sim_matrix
    best_score, best_n, best_clusters = -1, 2, None
    
    for n_clusters in range(2, min(max_clusters + 1, len(names))):
        try:
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
            labels = clustering.fit_predict(dist_matrix)
            
            clusters = {}
            for i, label in enumerate(labels):
                clusters.setdefault(label, []).append(i)
            
            within_sims = []
            for members in clusters.values():
                if len(members) > 1:
                    for i in range(len(members)):
                        for j in range(i+1, len(members)):
                            within_sims.append(sim_matrix[members[i], members[j]])
            
            score = np.mean(within_sims) if within_sims else 0
            if score > best_score:
                best_score, best_n = score, n_clusters
                best_clusters = {k: [names[i] for i in v] for k, v in clusters.items()}
        except:
            pass
    
    return best_n, best_clusters


def evaluate_directions(directions, activations, clusters):
    all_pos = torch.cat([activations[b]['pos'] for b in activations])
    all_neg = torch.cat([activations[b]['neg'] for b in activations])
    global_dir = all_pos.mean(dim=0) - all_neg.mean(dim=0)
    norm = torch.norm(global_dir)
    if norm > 1e-8:
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
            if norm > 1e-8:
                cluster_dirs[cid] = d / norm
            for m in members:
                bench_to_cluster[m] = cid
    
    global_accs, cluster_accs = [], []
    for bench, acts in activations.items():
        pos, neg = acts['pos'], acts['neg']
        n = min(len(pos), len(neg))
        
        g_correct = sum(1 for i in range(n) if torch.dot(pos[i], global_dir) > torch.dot(neg[i], global_dir))
        global_accs.append(g_correct / n if n > 0 else 0.5)
        
        cid = bench_to_cluster.get(bench)
        if cid in cluster_dirs:
            c_correct = sum(1 for i in range(n) if torch.dot(pos[i], cluster_dirs[cid]) > torch.dot(neg[i], cluster_dirs[cid]))
            cluster_accs.append(c_correct / n if n > 0 else 0.5)
        else:
            cluster_accs.append(global_accs[-1])
    
    return np.mean(global_accs), np.mean(cluster_accs)


def main():
    MODEL_NAME = os.environ.get('MODEL_NAME', 'meta-llama/Llama-3.2-1B-Instruct')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR', '/home/ubuntu/output'))
    PAIRS_PER_BENCHMARK = int(os.environ.get('PAIRS_PER_BENCHMARK', '50'))
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    model, tokenizer = load_model(MODEL_NAME, DEVICE)
    
    LAYERS = get_layers_to_test(model)
    
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Layers: {LAYERS} (of {model.config.num_hidden_layers})")
    logger.info(f"Strategies: {STRATEGIES}")
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
    logger.info(f"\nPhase 2: Testing {len(LAYERS) * len(STRATEGIES)} layer x strategy combinations...")
    
    geo_config = GeometryAnalysisConfig(num_components=5, optimization_steps=50)
    all_config_results = []
    best_config = None
    best_cluster_acc = 0
    
    for layer in LAYERS:
        for strategy in STRATEGIES:
            config_key = f"layer{layer}_{strategy}"
            logger.info(f"\n{'='*60}")
            logger.info(f"Config: Layer {layer}, Strategy {strategy}")
            logger.info(f"{'='*60}")
            
            directions, activations, geometry_dist = {}, {}, {}
            
            for bench_idx, (bench, pairs) in enumerate(all_pairs.items()):
                if (bench_idx + 1) % 50 == 0:
                    logger.info(f"  [{bench_idx+1}/{len(all_pairs)}] Processing benchmarks...")
                
                try:
                    direction, pos_t, neg_t = compute_directions_for_strategy(
                        model, tokenizer, pairs, layer, DEVICE, strategy
                    )
                    if direction is not None:
                        directions[bench] = direction
                        activations[bench] = {'pos': pos_t, 'neg': neg_t}
                        
                        # Geometry detection
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
            
            # Build similarity matrix and cluster
            bench_names = list(directions.keys())
            n = len(bench_names)
            sim_matrix = np.zeros((n, n))
            for i, n1 in enumerate(bench_names):
                for j, n2 in enumerate(bench_names):
                    sim_matrix[i, j] = torch.dot(directions[n1], directions[n2]).item()
            
            optimal_n, clusters = find_optimal_clusters(sim_matrix, bench_names)
            global_acc, cluster_acc = evaluate_directions(directions, activations, clusters)
            
            # Combined geometry
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
            
            logger.info(f"  Benchmarks: {n}")
            logger.info(f"  Global acc: {global_acc:.3f}, Cluster acc: {cluster_acc:.3f}")
            logger.info(f"  Clusters: {optimal_n}, Combined geometry: {combined_type}")
            logger.info(f"  Geometry dist: {geometry_dist}")
            
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
            
            # Save intermediate results
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
    
    # Save final results
    logger.info(f"\n{'='*80}")
    logger.info(f"FINAL RESULTS")
    logger.info(f"{'='*80}")
    
    if best_config:
        logger.info(f"Best: Layer {best_config['layer']}, Strategy {best_config['strategy']}")
        logger.info(f"Global acc: {best_config['global_acc']:.3f}")
        logger.info(f"Cluster acc: {best_config['cluster_acc']:.3f}")
        
        summary = {
            'model': MODEL_NAME,
            'num_layers': model.config.num_hidden_layers,
            'layers_tested': LAYERS,
            'strategies_tested': STRATEGIES,
            'best_layer': best_config['layer'],
            'best_strategy': best_config['strategy'],
            'n_benchmarks': len(best_config['bench_names']),
            'optimal_clusters': best_config['optimal_n'],
            'global_accuracy': best_config['global_acc'],
            'cluster_accuracy': best_config['cluster_acc'],
            'improvement': best_config['cluster_acc'] - best_config['global_acc'],
            'clusters': best_config['clusters'],
            'combined_geometry': best_config['combined_geometry'],
            'geometry_distribution': best_config['geometry_dist'],
            'all_configs': [asdict(r) for r in all_config_results],
        }
        
        with open(OUTPUT_DIR / 'cluster_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save similarity matrix for best config
        with open(OUTPUT_DIR / 'best_similarity_matrix.json', 'w') as f:
            json.dump({
                'bench_names': best_config['bench_names'],
                'sim_matrix': best_config['sim_matrix'],
            }, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("BENCHMARK CLUSTERING ANALYSIS - FINAL RESULTS")
        print("="*80)
        print(f"Model: {MODEL_NAME}")
        print(f"Layers tested: {LAYERS}")
        print(f"Strategies tested: {STRATEGIES}")
        print(f"\nBest config: Layer {best_config['layer']}, Strategy {best_config['strategy']}")
        print(f"Benchmarks: {len(best_config['bench_names'])}")
        print(f"Global accuracy: {best_config['global_acc']:.3f}")
        print(f"Cluster accuracy: {best_config['cluster_acc']:.3f}")
        print(f"Improvement: {(best_config['cluster_acc'] - best_config['global_acc'])*100:+.1f}%")
        print(f"\nOptimal clusters: {best_config['optimal_n']}")
        for cid, members in best_config['clusters'].items():
            print(f"  Cluster {cid} ({len(members)} benchmarks): {members[:5]}...")
        print(f"\nCombined geometry: {best_config['combined_geometry']}")
        print(f"Per-benchmark geometry: {best_config['geometry_dist']}")
        print("\nAll configs tested:")
        for r in sorted(all_config_results, key=lambda x: -x.cluster_accuracy):
            print(f"  L{r.layer:2d}_{r.strategy:15s}: global={r.global_accuracy:.3f}, cluster={r.cluster_accuracy:.3f}, geo={r.combined_geometry}")
        print("="*80)
    
    # Cleanup
    del model
    del tokenizer
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    elif DEVICE == 'mps':
        torch.mps.empty_cache()


if __name__ == '__main__':
    main()

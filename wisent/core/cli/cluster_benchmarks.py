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

from wisent.core.benchmark_registry import get_all_benchmarks
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import lm_build_contrastive_pairs
from wisent.core.data_loaders.loaders.lm_loader import LMEvalDataLoader
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


def get_layers_to_test(model) -> List[int]:
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
        weights = torch.exp(-torch.arange(answer_hidden.shape[0], dtype=answer_hidden.dtype, device=answer_hidden.device) * 0.5)
        weights = weights / weights.sum()
        weighted_mean = (answer_hidden * weights.unsqueeze(1)).sum(dim=0)
        return weighted_mean.cpu().float()
    return hidden[-1].cpu().float()


def get_activation(model, tokenizer, prompt: str, response: str, layer: int, device: str, strategy: str) -> torch.Tensor:
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
        raise ValueError("mc_balanced requires special handling")
    
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt[:500]}, {"role": "assistant", "content": response}],
        tokenize=False, add_generation_prompt=False
    )
    return get_last_token_act(model, tokenizer, text, layer, device)


def get_mc_balanced_activations(model, tokenizer, prompt: str, pos_response: str, neg_response: str, layer: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
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


def compute_directions_for_strategy(model, tokenizer, pairs: List, layer: int, device: str, strategy: str, max_pairs: int = 50):
    pos_acts, neg_acts = [], []
    
    for pair in pairs[:max_pairs]:
        try:
            prompt = pair.prompt
            pos_response = pair.positive_response.model_response
            neg_response = pair.negative_response.model_response
            
            if strategy == "mc_balanced":
                pos_act, neg_act = get_mc_balanced_activations(model, tokenizer, prompt, pos_response, neg_response, layer, device)
            else:
                pos_act = get_activation(model, tokenizer, prompt, pos_response, layer, device, strategy)
                neg_act = get_activation(model, tokenizer, prompt, neg_response, layer, device, strategy)
            
            pos_acts.append(pos_act)
            neg_acts.append(neg_act)
        except:
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
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    logger.info(f"Loading {model}...")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    from wisent.core.utils.device import device_optimized_dtype
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
        if (i + 1) % 20 == 0:
            logger.info(f"  [{i+1}/{len(all_benchmarks)}] Loaded {len(all_pairs)} benchmarks...")
        try:
            pairs = load_benchmark_pairs(bench, loader, limit=pairs_per_benchmark)
            if pairs and len(pairs) >= 10:
                all_pairs[bench] = pairs
        except:
            pass
    
    logger.info(f"Loaded {len(all_pairs)} benchmarks")
    
    # Test configurations
    geo_config = GeometryAnalysisConfig(num_components=5, optimization_steps=50)
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
            
            if len(directions) < 10:
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
                json.dump({'results': [asdict(r) for r in all_results], 'best_acc': best_acc}, f, indent=2)
    
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
            json.dump(summary, f, indent=2)
        
        print(f"\nBest: Layer {best_config['layer']}, Strategy {best_config['strategy']}")
        print(f"Global: {best_config['global_acc']:.3f}, Cluster: {best_config['cluster_acc']:.3f}")
    
    del llm
    gc.collect()

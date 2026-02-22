"""
Reward-Shaped Cost Transport (RSCT) — RL for PRZELOM.

Uses the attention EOT cost matrix as a reward-shaped optimization surface.
Iteratively refines the transport plan using per-example evaluation rewards.
C_shaping -= lr * advantage * T_current (REINFORCE on transport plan).
"""
from __future__ import annotations
import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple
import torch
from wisent.core.steering_methods.steering_object import SteeringObjectMetadata
from wisent.core.steering_methods.methods.przelom.przelom_steering_object import (
    PrzelomSteeringObject,
)
from wisent.core.steering_methods.methods.przelom.create import _regularized_pinv
from wisent.core.steering_methods.methods.szlak.transport import (
    compute_attention_affinity_cost,
)
from wisent.core.cli.optimize_steering.data.responses import execute_generate_responses
from wisent.core.cli.optimize_steering.scores import execute_evaluate_responses
from wisent.core.cli.optimize_steering.pipeline import _make_args
from wisent.core.utils import preferred_dtype


def _precompute_layer_data(
    data: dict, epsilon: float,
) -> Tuple[Dict[int, dict], SteeringObjectMetadata]:
    """One-time C_original / q_neg / k_pos / T_current computation per layer."""
    dtype = preferred_dtype()
    pairs_list = data.get('pairs', [])
    num_heads = data.get('num_attention_heads')
    num_kv_heads = data.get('num_key_value_heads')
    layer_act = defaultdict(lambda: {
        "positive": [], "negative": [],
        "q_proj_activations": [], "k_proj_activations": [],
    })
    for pair in pairs_list:
        for ls, al in pair['positive_response'].get('layers_activations', {}).items():
            if al is not None:
                layer_act[ls]["positive"].append(torch.tensor(al, dtype=dtype))
        for ls, al in pair['negative_response'].get('layers_activations', {}).items():
            if al is not None:
                layer_act[ls]["negative"].append(torch.tensor(al, dtype=dtype))
        for ls, qv in pair['negative_response'].get('q_proj_activations', {}).items():
            if qv is not None:
                layer_act[ls]["q_proj_activations"].append(torch.tensor(qv, dtype=dtype))
        for ls, kv in pair['positive_response'].get('k_proj_activations', {}).items():
            if kv is not None:
                layer_act[ls]["k_proj_activations"].append(torch.tensor(kv, dtype=dtype))

    all_layers = sorted(layer_act.keys(), key=lambda x: int(x))
    hidden_dim = layer_act[all_layers[0]]["positive"][0].shape[-1]
    trait_label = data.get('trait_label', 'unknown')
    parts = trait_label.split('/')
    cal_raw = data.get('calibration_norms', {})
    metadata = SteeringObjectMetadata(
        method="przelom", model_name=data.get('model', 'unknown'),
        benchmark=parts[-1], category=parts[0] if len(parts) > 1 else 'unknown',
        extraction_strategy=data.get('token_aggregation', 'unknown'),
        num_pairs=len(pairs_list), layers=[int(l) for l in all_layers],
        hidden_dim=hidden_dim, created_at=datetime.now().isoformat(),
        calibration_norms={int(k): float(v) for k, v in cal_raw.items()},
        extraction_component=data.get('extraction_component', 'residual_stream'),
        extra={'num_attention_heads': num_heads, 'num_key_value_heads': num_kv_heads},
    )
    layer_data: Dict[int, dict] = {}
    for layer_str in all_layers:
        la = layer_act[layer_str]
        if not la["positive"] or not la["negative"]:
            continue
        if not la["q_proj_activations"] or not la["k_proj_activations"]:
            print(f"   Skipping layer {layer_str}: no Q/K projections")
            continue
        pos = torch.stack([t.detach().float().reshape(-1) for t in la["positive"]], dim=0)
        neg = torch.stack([t.detach().float().reshape(-1) for t in la["negative"]], dim=0)
        q_neg = torch.stack([t.detach().float().reshape(-1) for t in la["q_proj_activations"]], dim=0)
        k_pos = torch.stack([t.detach().float().reshape(-1) for t in la["k_proj_activations"]], dim=0)
        C = compute_attention_affinity_cost(q_neg, k_pos, num_heads=num_heads, num_kv_heads=num_kv_heads)
        T_current = torch.softmax(-C / epsilon, dim=1)
        li = int(layer_str)
        layer_data[li] = {
            "C_original": C, "T_current": T_current,
            "q_neg": q_neg, "k_pos": k_pos, "neg": neg, "pos": pos,
        }
        print(f"   Layer {layer_str}: n_neg={neg.shape[0]}, n_pos={pos.shape[0]}, "
              f"C_range=[{C.min().item():.3f}, {C.max().item():.3f}]")
    return layer_data, metadata


def _update_cost_shaping(
    C_shaping: Dict[int, torch.Tensor],
    layer_data: Dict[int, dict],
    rewards: List[float],
    learning_rate: float,
) -> None:
    """REINFORCE update: lower cost where high-reward transport happened."""
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    if rewards_t.std() < 1e-8:
        return
    advantage = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
    for layer_int, data in layer_data.items():
        T_current = data["T_current"]
        adv = advantage[:T_current.shape[0]].unsqueeze(1)
        C_shaping[layer_int] -= learning_rate * (adv * T_current)


def _create_przelom_from_shaping(
    metadata: SteeringObjectMetadata,
    layer_data: Dict[int, dict],
    C_shaping: Dict[int, torch.Tensor],
    epsilon: float,
    regularization: float,
    inference_k: int,
) -> PrzelomSteeringObject:
    """Create steering object from current shaped cost matrices."""
    num_heads = metadata.extra.get('num_attention_heads')
    num_kv_heads = metadata.extra.get('num_key_value_heads')
    source_points, disps = {}, {}
    for layer_int, data in layer_data.items():
        C_eff = data["C_original"] + C_shaping[layer_int]
        T_target = torch.softmax(-C_eff / epsilon, dim=1)
        T_current = data["T_current"]
        delta_C = epsilon * (torch.log(T_target.clamp(min=1e-12))
                             - torch.log(T_current.clamp(min=1e-12)))
        q_dim = data["q_neg"].shape[-1]
        k_dim = data["k_pos"].shape[-1]
        if q_dim == k_dim:
            k_pos_pinv = _regularized_pinv(data["k_pos"], regularization)
            delta_q = -math.sqrt(q_dim) * (delta_C @ k_pos_pinv.T)
        else:
            head_dim = q_dim // num_heads
            groups = num_heads // num_kv_heads
            k_by_head = data["k_pos"].reshape(-1, num_kv_heads, head_dim)
            delta_q_parts = []
            for g in range(num_kv_heads):
                k_g = k_by_head[:, g, :]
                k_g_pinv = _regularized_pinv(k_g, regularization)
                dq_g = -math.sqrt(head_dim) * (delta_C @ k_g_pinv.T)
                for _ in range(groups):
                    delta_q_parts.append(dq_g)
            delta_q = torch.cat(delta_q_parts, dim=-1)
        source_points[layer_int] = data["neg"].detach()
        disps[layer_int] = delta_q.detach()
    return PrzelomSteeringObject(
        metadata=metadata, source_points=source_points,
        displacements=disps, inference_k=inference_k,
    )


def _extract_per_example_rewards(scores_file: str) -> List[float]:
    """Extract per-example binary rewards from evaluation scores file."""
    with open(scores_file) as f:
        scores_data = json.load(f)
    rewards = []
    for ev in scores_data.get("evaluations", []):
        evaluation = ev.get("evaluation", {})
        if isinstance(evaluation, dict) and "correct" in evaluation:
            rewards.append(1.0 if evaluation["correct"] else 0.0)
        elif isinstance(evaluation, dict):
            rewards.append(evaluation.get("score", 0.0))
        else:
            rewards.append(0.0)
    return rewards


def _extract_aggregate_score(scores_file: str) -> float:
    """Extract aggregate accuracy from scores file."""
    with open(scores_file) as f:
        sd = json.load(f)
    return sd.get("aggregated_metrics", {}).get("acc") or sd.get("accuracy") or sd.get("acc") or 0.0


def execute_transport_rl(args):
    """Execute RL-based steering optimization. Dispatches by method.

    Transport methods (przelom, szlak): cost-matrix shaping with REINFORCE.
    All other methods: evolutionary strategy on steering vectors.
    """
    method = getattr(args, 'method', 'przelom').lower()
    if method not in ('przelom', 'szlak'):
        from wisent.core.cli.optimize_steering.transport.vector_rl import run_vector_rl_loop
        return run_vector_rl_loop(args)

    # --- Transport cost-shaping path (PRZELOM / SZLAK) ---
    model = args.model
    task = args.task
    epf = args.enriched_pairs_file
    max_iter = getattr(args, 'max_iterations', 10)
    lr = getattr(args, 'learning_rate', 0.1)
    epsilon = getattr(args, 'epsilon', 1.0)
    reg = getattr(args, 'regularization', 1e-4)
    inf_k = getattr(args, 'inference_k', 5)
    limit = getattr(args, 'limit', 100)
    output_path = getattr(args, 'output', 'best_transport_rl.pt')
    device = getattr(args, 'device', None)

    print(f"\n{'=' * 70}")
    print(f"REWARD-SHAPED COST TRANSPORT (RSCT)")
    print(f"{'=' * 70}")
    print(f"   Model: {model}  |  Task: {task}")
    print(f"   Iterations: {max_iter}  |  LR: {lr}  |  Epsilon: {epsilon}")
    print(f"   Regularization: {reg}  |  Inference K: {inf_k}  |  Limit: {limit}")
    print(f"{'=' * 70}\n")

    # 1. Load and precompute
    print("Loading enriched pairs and precomputing layer data...")
    with open(epf, 'r') as f:
        data = json.load(f)
    layer_data, metadata = _precompute_layer_data(data, epsilon)
    if not layer_data:
        raise ValueError("No layers with Q/K projections found in enriched pairs")

    # 2. Initialize cost shaping
    C_shaping: Dict[int, torch.Tensor] = {
        li: torch.zeros_like(ld["C_original"]) for li, ld in layer_data.items()
    }
    best_score = -float('inf')
    best_obj = None

    # 3. RL loop
    for it in range(1, max_iter + 1):
        print(f"\n--- Iteration {it}/{max_iter} ---")
        with tempfile.TemporaryDirectory() as wd:
            sf = os.path.join(wd, "steering.pt")
            rf = os.path.join(wd, "responses.json")
            ef = os.path.join(wd, "scores.json")
            # Create steering object from shaped cost
            obj = _create_przelom_from_shaping(metadata, layer_data, C_shaping, epsilon, reg, inf_k)
            obj.save(sf)
            # Generate steered responses
            execute_generate_responses(_make_args(
                task=task, input_file=epf, model=model, output=rf,
                num_questions=limit, steering_object=sf, steering_strength=1.0,
                steering_strategy="constant", use_steering=True, device=device,
                max_new_tokens=128, temperature=0.7, top_p=0.95,
                verbose=False, cached_model=None,
            ))
            # Evaluate
            execute_evaluate_responses(_make_args(input=rf, output=ef, task=task, verbose=False))
            rewards = _extract_per_example_rewards(ef)
            agg = _extract_aggregate_score(ef)
            n = max(len(rewards), 1)
            print(f"   Score: {agg:.4f} (mean_reward={sum(rewards)/n:.3f}, n={len(rewards)})")
            # Update cost shaping
            if rewards:
                _update_cost_shaping(C_shaping, layer_data, rewards, lr)
            # Track best
            if agg > best_score:
                best_score = agg
                best_obj = obj
                print(f"   ** New best: {best_score:.4f}")

    # 4. Save best
    if best_obj is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
        best_obj.save(output_path)
        print(f"\nBest steering object saved to: {output_path} (score: {best_score:.4f})")
    else:
        print("\nNo valid steering object produced.")
    return {"best_score": best_score, "output": output_path}

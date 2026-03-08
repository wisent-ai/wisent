"""
Vector-based evolutionary strategy (ES) RL for general steering methods.

Works with ANY steering method: creates initial steering object using the
specified method, extracts per-layer vectors, then optimizes them via
antithetic ES with evaluation rewards.

For transport methods (PRZELOM, SZLAK), use cost-matrix shaping in transport_rl.py instead.
"""
from __future__ import annotations
import json
import os
import tempfile
from typing import Dict
import torch
from wisent.core.control.steering_methods.steering_object import (
    load_steering_object, CAASteeringObject,
)
from wisent.core.utils.cli.optimize_steering.steering_objects import execute_create_steering_object
from wisent.core.utils.cli.optimize_steering.data.responses import execute_generate_responses
from wisent.core.utils.cli.optimize_steering.scores import execute_evaluate_responses
from wisent.core.utils.cli.optimize_steering.pipeline import _make_args
from wisent.core.control.steering_methods.configs.optimal import get_optimal
from wisent.core.utils.config_tools.constants import (COMPARE_TOL,
    SEPARATOR_WIDTH_WIDE)


def _extract_vectors(obj) -> Dict[int, torch.Tensor]:
    """Extract per-layer steering vectors from any steering object type."""
    if hasattr(obj, 'vectors') and isinstance(obj.vectors, dict):
        return {l: v.float().clone() for l, v in obj.vectors.items()}
    if hasattr(obj, 'displacements') and isinstance(obj.displacements, dict):
        return {l: d.float().mean(dim=0) for l, d in obj.displacements.items()}
    vectors = {}
    for layer in obj.metadata.layers:
        try:
            vectors[layer] = obj.get_steering_vector(layer).float().clone()
        except (KeyError, Exception):
            pass
    return vectors


def _evaluate_with_vectors(vectors, metadata, args, work_dir, layer_sweep_strength) -> float:
    """Build CAASteeringObject from vectors, generate, evaluate, return score."""
    if layer_sweep_strength is None:
        raise ValueError("layer_sweep_strength is required")
    sf = os.path.join(work_dir, "steering.pt")
    rf = os.path.join(work_dir, "responses.json")
    ef = os.path.join(work_dir, "scores.json")
    obj = CAASteeringObject(metadata=metadata, vectors=vectors)
    obj.save(sf)
    execute_generate_responses(_make_args(
        task=args.task, input_file=args.enriched_pairs_file, model=args.model,
        output=rf, num_questions=args.limit, min_load_limit_questions=args.limit,
        steering_object=sf, steering_strength=layer_sweep_strength, steering_strategy=get_optimal("steering_strategy"),
        use_steering=True, device=getattr(args, 'device', None),
        verbose=False, cached_model=None,
    ))
    execute_evaluate_responses(_make_args(
        input=rf, output=ef, task=args.task, verbose=False,
    ))
    with open(ef) as f:
        sd = json.load(f)
    return (
        sd.get("aggregated_metrics", {}).get("acc")
        or sd.get("accuracy") or sd.get("acc") or 0.0
    )


def run_vector_rl_loop(args) -> dict:
    """Run ES-based vector RL optimization for any steering method.

    Algorithm (antithetic ES):
      1. Create initial steering object with specified method
      2. Extract per-layer steering vectors
      3. Each iteration:
         a. Sample noise eps ~ N(0, sigma^2 * I)
         b. Evaluate v+eps -> score_plus, v-eps -> score_minus
         c. Update: v += lr * (score_plus - score_minus) / (2 * sigma) * eps
      4. Save best-scoring vectors as CAASteeringObject
    """
    method = getattr(args, 'method', 'CAA')
    layer_sweep_strength = getattr(args, 'layer_sweep_strength', None)
    if layer_sweep_strength is None:
        raise ValueError("layer_sweep_strength is required")
    max_iter = getattr(args, 'max_iterations', None)
    if max_iter is None:
        raise ValueError("max_iterations is required")
    lr = getattr(args, 'learning_rate', None)
    if lr is None:
        raise ValueError("learning_rate is required")
    noise_scale = getattr(args, 'noise_scale', None)
    if noise_scale is None:
        raise ValueError("noise_scale is required")
    limit = args.limit
    output_path = getattr(args, 'output', 'best_transport_rl.pt')

    print(f"\n{'=' * SEPARATOR_WIDTH_WIDE}")
    print(f"VECTOR RL OPTIMIZATION (ES)")
    print(f"{'=' * SEPARATOR_WIDTH_WIDE}")
    print(f"   Model: {args.model}  |  Task: {args.task}")
    print(f"   Base method: {method}  |  Iterations: {max_iter}  |  LR: {lr}")
    print(f"   Noise scale: {noise_scale}  |  Limit: {limit}")
    print(f"{'=' * SEPARATOR_WIDTH_WIDE}\n")

    # 1. Create initial steering object using specified method
    print(f"Creating initial {method} steering object...")
    with tempfile.TemporaryDirectory() as init_dir:
        init_path = os.path.join(init_dir, "initial.pt")
        execute_create_steering_object(_make_args(
            enriched_pairs_file=args.enriched_pairs_file,
            output=init_path, method=method,
            verbose=False, timing=False, layer=None,
        ))
        base_obj = load_steering_object(init_path)

    # 2. Extract per-layer vectors
    vectors = _extract_vectors(base_obj)
    if not vectors:
        raise ValueError(f"Could not extract steering vectors from {method} object")
    metadata = base_obj.metadata
    print(f"   Extracted vectors for {len(vectors)} layers")
    for l, v in sorted(vectors.items()):
        print(f"   Layer {l}: dim={v.shape[-1]}, norm={v.norm().item():.4f}")

    # 3. ES loop
    best_score = -float('inf')
    best_vectors = None

    for it in range(1, max_iter + 1):
        print(f"\n--- Iteration {it}/{max_iter} ---")

        with tempfile.TemporaryDirectory() as wd:
            # Sample noise scaled relative to vector norms
            noise = {}
            for l, v in vectors.items():
                scale = noise_scale * max(v.norm().item(), COMPARE_TOL)
                noise[l] = torch.randn_like(v) * scale

            # Forward perturbation: v + noise
            v_plus = {l: vectors[l] + noise[l] for l in vectors}
            score_plus = _evaluate_with_vectors(v_plus, metadata, args, wd, layer_sweep_strength)

            # Antithetic perturbation: v - noise
            v_minus = {l: vectors[l] - noise[l] for l in vectors}
            score_minus = _evaluate_with_vectors(v_minus, metadata, args, wd, layer_sweep_strength)

            # ES gradient estimate and update
            grad_est = (score_plus - score_minus) / 2.0
            for l in vectors:
                v_norm = max(vectors[l].norm().item(), COMPARE_TOL)
                vectors[l] += lr * grad_est * noise[l] / (noise_scale * v_norm)

            print(f"   +noise: {score_plus:.4f}  -noise: {score_minus:.4f}  "
                  f"grad_est: {grad_est:+.4f}")

            # Track best
            if score_plus >= score_minus and score_plus > best_score:
                best_score = score_plus
                best_vectors = {l: v_plus[l].clone() for l in vectors}
                print(f"   ** New best: {best_score:.4f}")
            elif score_minus > best_score:
                best_score = score_minus
                best_vectors = {l: v_minus[l].clone() for l in vectors}
                print(f"   ** New best: {best_score:.4f}")

    # 4. Save best
    if best_vectors is not None:
        final_obj = CAASteeringObject(metadata=metadata, vectors=best_vectors)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
        final_obj.save(output_path)
        print(f"\nBest steering object saved to: {output_path} (score: {best_score:.4f})")
    else:
        print("\nNo valid steering object produced.")

    return {"best_score": best_score, "output": output_path}

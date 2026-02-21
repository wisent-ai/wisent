"""Helper functions for the autonomous continual learning loop."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, Optional

import torch

from wisent.core.cli.optimize_steering.pipeline import _make_args
from wisent.core.cli.optimize_steering.data.contrastive_pairs import (
    execute_generate_pairs_from_task,
)
from wisent.core.cli.optimize_steering.data.activations import execute_get_activations
from wisent.core.cli.optimize_steering.data.responses import execute_generate_responses
from wisent.core.cli.optimize_steering.scores import execute_evaluate_responses
from wisent.core.steering_methods.steering_object import (
    load_steering_object,
    CAASteeringObject,
)


def ensure_enriched_pairs(
    task: str,
    model: str,
    pairs_dir: str,
    device: Optional[str],
    limit: int,
) -> str:
    """Generate enriched pairs for a task if they don't already exist.

    Returns the path to the enriched pairs JSON file.
    """
    enriched_path = os.path.join(pairs_dir, f"{task}.json")
    if os.path.exists(enriched_path):
        print(f"   Enriched pairs exist: {enriched_path}")
        return enriched_path

    os.makedirs(pairs_dir, exist_ok=True)
    pairs_file = os.path.join(pairs_dir, f"{task}_raw.json")

    print(f"   Generating contrastive pairs for {task}...")
    execute_generate_pairs_from_task(_make_args(
        task_name=task, output=pairs_file, limit=limit, verbose=False,
    ))

    print(f"   Collecting activations for {task}...")
    execute_get_activations(_make_args(
        pairs_file=pairs_file, model=model, output=enriched_path,
        layers="all", extraction_strategy="chat_last",
        device=device, verbose=False, timing=False, raw=False,
        cached_model=None, capture_qk=True,
    ))

    return enriched_path


def extract_vectors(obj) -> Dict[int, torch.Tensor]:
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


def evaluate_vectors(
    vectors: Dict[int, torch.Tensor],
    metadata,
    model: str,
    task: str,
    enriched_path: str,
    limit: int,
    device: Optional[str],
    work_dir: str,
) -> float:
    """Evaluate steering vectors by generating responses and scoring them."""
    sf = os.path.join(work_dir, "steering.pt")
    rf = os.path.join(work_dir, "responses.json")
    ef = os.path.join(work_dir, "scores.json")

    obj = CAASteeringObject(metadata=metadata, vectors=vectors)
    obj.save(sf)

    execute_generate_responses(_make_args(
        task=task, input_file=enriched_path, model=model, output=rf,
        num_questions=limit, steering_object=sf, steering_strength=1.0,
        steering_strategy="constant", use_steering=True, device=device,
        max_new_tokens=128, temperature=0.7, top_p=0.95,
        verbose=False, cached_model=None,
    ))

    execute_evaluate_responses(_make_args(
        input=rf, output=ef, task=task, verbose=False,
    ))

    with open(ef) as f:
        sd = json.load(f)
    return (
        sd.get("aggregated_metrics", {}).get("acc")
        or sd.get("accuracy") or sd.get("acc") or 0.0
    )


def run_rl_iteration(
    task: str,
    method: str,
    model: str,
    enriched_path: str,
    initial_vectors: Optional[Dict[int, torch.Tensor]],
    args,
) -> Dict:
    """Run a single RL optimization iteration for a task.

    Dispatches to transport RL (przelom/szlak) or vector RL (all others).
    Returns dict with 'best_score', 'output', 'steering_object', 'vectors'.
    """
    method_lower = method.lower()
    with tempfile.TemporaryDirectory() as wd:
        output_path = os.path.join(wd, "optimized.pt")

        if method_lower in ("przelom", "szlak"):
            from wisent.core.cli.optimize_steering.transport.transport_rl import (
                execute_transport_rl,
            )
            rl_args = _make_args(
                model=model, task=task, enriched_pairs_file=enriched_path,
                method=method_lower,
                max_iterations=getattr(args, 'max_iterations', 10),
                learning_rate=getattr(args, 'learning_rate', 0.1),
                epsilon=getattr(args, 'epsilon', 1.0),
                regularization=getattr(args, 'regularization', 1e-4),
                inference_k=getattr(args, 'inference_k', 5),
                noise_scale=getattr(args, 'noise_scale', 0.1),
                limit=getattr(args, 'limit', 100),
                output=output_path,
                device=getattr(args, 'device', None),
            )
            result = execute_transport_rl(rl_args)
        else:
            from wisent.core.cli.optimize_steering.transport.vector_rl import (
                run_vector_rl_loop,
            )
            rl_args = _make_args(
                model=model, task=task, enriched_pairs_file=enriched_path,
                method=method,
                max_iterations=getattr(args, 'max_iterations', 10),
                learning_rate=getattr(args, 'learning_rate', 0.1),
                noise_scale=getattr(args, 'noise_scale', 0.1),
                limit=getattr(args, 'limit', 100),
                output=output_path,
                device=getattr(args, 'device', None),
            )
            result = run_vector_rl_loop(rl_args)

        if os.path.exists(output_path):
            obj = load_steering_object(output_path)
            result["steering_object"] = obj
            result["vectors"] = extract_vectors(obj)
        else:
            result["steering_object"] = None
            result["vectors"] = {}

    return result


def select_method_for_task(task: str, model: str) -> str:
    """Select steering method for a task using zwiad recommendation."""
    try:
        from wisent.core.steering_optimizer import run_auto_steering_optimization
        result = run_auto_steering_optimization(
            model_name=model, task_name=task, limit=20,
            device=None, verbose=False,
        )
        if "error" not in result:
            return result.get("best_method", "CAA")
    except Exception:
        pass
    return "CAA"


def check_convergence(metrics: Dict[str, list], window: int) -> bool:
    """Check if all tasks show no improvement in the last `window` cycles."""
    for task_name, scores in metrics.items():
        if len(scores) < window:
            return False
        recent = scores[-window:]
        if max(recent) > min(recent):
            return False
    return True


def get_task_priority(
    task: str,
    metrics: Dict[str, list],
    task_priorities: Dict[str, float],
    cycle: int,
) -> float:
    """Compute task priority based on staleness and performance.

    Higher priority = longer since last optimization + lower current score.
    """
    history = metrics.get(task, [])
    last_score = history[-1] if history else 0.0
    staleness = cycle - task_priorities.get(task, 0)
    return staleness + (1.0 - last_score)

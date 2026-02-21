"""Autonomous continual learning loop for multi-task steering optimization.

Iterates over tasks by priority, optimizes each with RL, applies EWC constraints
to prevent forgetting, decomposes into shared/task vectors, and periodically
replays past tasks. Runs fully autonomously with no human in the loop.
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict

import torch

from wisent.core.benchmarks.benchmark_registry import get_all_benchmarks
from wisent.core.steering_methods.steering_object import CAASteeringObject
from wisent.core.cli.optimize_steering.continual.state import (
    ContinualState,
    decompose_into_shared_and_task,
    compose_vectors,
    save_state,
    load_state,
    upload_to_s3,
    ewc_constrained_update,
)
from wisent.core.cli.optimize_steering.continual.replay_buffer import detect_forgetting
from wisent.core.cli.optimize_steering.continual._loop_helpers import (
    ensure_enriched_pairs,
    evaluate_vectors,
    run_rl_iteration,
    select_method_for_task,
    check_convergence,
    get_task_priority,
)


def _run_replay_check(state, model, pairs_dir, limit, device,
                      replay_size, forgetting_threshold, result):
    """Sample past tasks, evaluate, detect forgetting."""
    print(f"\n   --- Replay check ---")
    entries = state.replay_buffer.sample(
        min(replay_size, len(state.replay_buffer))
    )
    current_scores: Dict[str, float] = {}
    for rt in set(e.task for e in entries):
        rt_vectors = compose_vectors(
            state.shared_vectors, state.task_vectors.get(rt, {}),
        )
        if not rt_vectors:
            continue
        rt_enriched = os.path.join(pairs_dir, f"{rt}.json")
        if not os.path.exists(rt_enriched):
            continue
        rt_obj = result.get("steering_object")
        if rt_obj is None:
            continue
        with tempfile.TemporaryDirectory() as wd:
            current_scores[rt] = evaluate_vectors(
                rt_vectors, rt_obj.metadata, model, rt,
                rt_enriched, limit, device, wd,
            )
    degraded = detect_forgetting(
        state.replay_buffer.entries, current_scores, forgetting_threshold,
    )
    if degraded:
        print(f"   Forgetting detected in: {degraded}")
        for dt in degraded:
            state.task_priorities[dt] = 0


def execute_continual_learning(args):
    """Execute autonomous continual learning over multiple tasks."""
    model = args.model
    tasks_str = getattr(args, 'tasks', None)
    method = getattr(args, 'method', None)
    max_cycles = getattr(args, 'max_cycles', 100)
    pairs_dir = getattr(args, 'enriched_pairs_dir', './pairs/')
    checkpoint_dir = getattr(args, 'checkpoint_dir', './checkpoints/')
    ewc_lambda = getattr(args, 'ewc_lambda', 1000.0)
    replay_size = getattr(args, 'replay_size', 50)
    replay_interval = getattr(args, 'replay_interval', 5)
    forgetting_threshold = getattr(args, 'forgetting_threshold', 0.9)
    convergence_window = getattr(args, 'convergence_window', 10)
    limit = getattr(args, 'limit', 100)
    device = getattr(args, 'device', None)
    s3_bucket = getattr(args, 's3_bucket', None)

    tasks = ([t.strip() for t in tasks_str.split(",")]
             if tasks_str else get_all_benchmarks())

    print(f"\n{'=' * 70}")
    print(f"AUTONOMOUS CONTINUAL LEARNING")
    print(f"{'=' * 70}")
    print(f"   Model: {model}")
    print(f"   Tasks: {len(tasks)} ({', '.join(tasks[:5])}"
          f"{'...' if len(tasks) > 5 else ''})")
    print(f"   Method: {method or 'auto (zwiad per task)'}")
    print(f"   Max cycles: {max_cycles}  |  EWC lambda: {ewc_lambda}")
    print(f"   Checkpoint: {checkpoint_dir}")
    print(f"{'=' * 70}\n")

    state = load_state(checkpoint_dir) or ContinualState()
    start_cycle = state.current_cycle + 1
    for task in tasks:
        if task not in state.task_priorities:
            state.task_priorities[task] = 0

    for cycle in range(start_cycle, start_cycle + max_cycles):
        print(f"\n{'=' * 50}")
        print(f"CYCLE {cycle}/{start_cycle + max_cycles - 1}")
        print(f"{'=' * 50}")

        # 1. Select highest-priority task
        task_scores = {
            t: get_task_priority(t, state.metrics, state.task_priorities, cycle)
            for t in tasks
        }
        task = max(task_scores, key=task_scores.get)
        print(f"\n   Selected: {task} (priority={task_scores[task]:.2f})")

        # 2-3. Ensure pairs + select method
        enriched_path = ensure_enriched_pairs(task, model, pairs_dir, device, limit)
        task_method = method or select_method_for_task(task, model)
        print(f"   Method: {task_method}")

        # 4. Compose initial vectors
        initial_vectors = compose_vectors(
            state.shared_vectors, state.task_vectors.get(task, {}),
        )

        # 5. Run RL optimization
        result = run_rl_iteration(
            task, task_method, model, enriched_path, initial_vectors, args,
        )
        best_score = result.get("best_score", 0.0)
        new_vectors = result.get("vectors", {})
        print(f"   RL result: score={best_score:.4f}")

        if not new_vectors:
            print(f"   No vectors produced, skipping")
            state.task_priorities[task] = cycle
            continue

        # 6. EWC-constrained update
        if state.fisher_info:
            new_vectors = ewc_constrained_update(
                current_vectors=new_vectors,
                update_vectors={l: torch.zeros_like(v) for l, v in new_vectors.items()},
                fisher_all_tasks=state.fisher_info,
                old_vectors_all_tasks=state.old_vectors,
                ewc_lambda=ewc_lambda,
            )
            print(f"   EWC applied ({len(state.fisher_info)} past tasks)")

        state.fisher_info[task] = {
            l: torch.ones_like(v) * (best_score ** 2) for l, v in new_vectors.items()
        }
        state.old_vectors[task] = {l: v.clone() for l, v in new_vectors.items()}

        # 7. Decompose into shared + task-specific
        shared_update, task_update = decompose_into_shared_and_task(
            new_vectors, state.task_vectors,
        )
        for layer in shared_update:
            if layer in state.shared_vectors:
                state.shared_vectors[layer] += 0.1 * shared_update[layer]
            else:
                state.shared_vectors[layer] = shared_update[layer].clone()
        state.task_vectors[task] = task_update

        # 8. Replay check
        if cycle % replay_interval == 0 and len(state.replay_buffer) > 0:
            _run_replay_check(state, model, pairs_dir, limit, device,
                              replay_size, forgetting_threshold, result)

        # 9. Replay buffer + save steering
        steering_path = os.path.join(checkpoint_dir, f"{task}_latest.pt")
        os.makedirs(checkpoint_dir, exist_ok=True)
        combined = compose_vectors(
            state.shared_vectors, state.task_vectors.get(task, {}),
        )
        if combined and result.get("steering_object") is not None:
            CAASteeringObject(
                metadata=result["steering_object"].metadata, vectors=combined,
            ).save(steering_path)
        state.replay_buffer.add(task, enriched_path, steering_path, best_score, cycle)

        # 10. Update state
        state.metrics.setdefault(task, []).append(best_score)
        state.task_priorities[task] = cycle
        state.current_cycle = cycle

        # 11. Checkpoint
        save_state(state, checkpoint_dir)
        print(f"   Checkpoint saved to {checkpoint_dir}")
        if s3_bucket:
            upload_to_s3(checkpoint_dir, s3_bucket, f"continual/{model}/{cycle}")

        # 12. Convergence check
        if check_convergence(state.metrics, convergence_window):
            print(f"\n   Converged after {convergence_window} cycles w/o improvement")
            break

    print(f"\n{'=' * 70}")
    print(f"CONTINUAL LEARNING COMPLETE  |  Cycles: {state.current_cycle}")
    print(f"{'=' * 70}")
    for t, scores in state.metrics.items():
        print(f"   {t}: best={max(scores):.4f}, latest={scores[-1]:.4f}")
    return state

"""State management + EWC for continual learning: decomposition, persistence, forgetting."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

from wisent.core.cli.optimize_steering.continual.replay_buffer import ReplayBuffer


@dataclass
class ContinualState:
    """Full state of the continual learning system.

    Tracks shared steering vectors (common across tasks), per-task adjustments,
    Fisher information for EWC, replay buffer for forgetting detection, per-task
    score history, and scheduling metadata.
    """

    shared_vectors: Dict[int, torch.Tensor] = field(default_factory=dict)
    task_vectors: Dict[str, Dict[int, torch.Tensor]] = field(default_factory=dict)
    fisher_info: Dict[str, Dict[int, torch.Tensor]] = field(default_factory=dict)
    old_vectors: Dict[str, Dict[int, torch.Tensor]] = field(default_factory=dict)
    replay_buffer: ReplayBuffer = field(default_factory=lambda: ReplayBuffer(max_size=200))
    metrics: Dict[str, list] = field(default_factory=dict)
    current_cycle: int = 0
    task_priorities: Dict[str, float] = field(default_factory=dict)


def decompose_into_shared_and_task(
    new_vectors: Dict[int, torch.Tensor],
    all_task_vectors: Dict[str, Dict[int, torch.Tensor]],
    variance_threshold: float = 0.8,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """Decompose vectors into shared subspace component + task-specific residual.

    Stacks all existing task vectors per layer, runs SVD to find the shared
    subspace (top-k singular vectors explaining variance_threshold of total
    variance), then projects new_vectors onto that subspace.

    Args:
        new_vectors: Newly optimized per-layer vectors for the current task.
        all_task_vectors: All existing per-task vectors {task: {layer: Tensor}}.
        variance_threshold: Fraction of variance to capture in shared subspace.

    Returns:
        (shared_update, task_specific): Shared subspace projection and residual.
    """
    shared_update: Dict[int, torch.Tensor] = {}
    task_specific: Dict[int, torch.Tensor] = {}

    for layer, vec in new_vectors.items():
        # Collect all task vectors for this layer
        layer_vecs = []
        for task_name, tvecs in all_task_vectors.items():
            if layer in tvecs:
                layer_vecs.append(tvecs[layer].float())

        if len(layer_vecs) < 2:
            # Not enough tasks for meaningful decomposition — all goes to task-specific
            shared_update[layer] = torch.zeros_like(vec)
            task_specific[layer] = vec.clone()
            continue

        # Stack: (num_tasks, hidden_dim)
        stacked = torch.stack(layer_vecs, dim=0)
        # SVD to find shared subspace
        U, S, Vh = torch.linalg.svd(stacked, full_matrices=False)

        # Determine k: smallest k such that sum(S[:k]^2) >= threshold * sum(S^2)
        total_var = (S ** 2).sum()
        cumvar = torch.cumsum(S ** 2, dim=0) / total_var
        k = int((cumvar >= variance_threshold).float().argmax().item()) + 1
        k = max(1, min(k, len(S)))

        # Shared subspace: top-k right singular vectors
        V_shared = Vh[:k]  # (k, hidden_dim)

        # Project new vector onto shared subspace
        vec_f = vec.float()
        coeffs = vec_f @ V_shared.T  # (k,)
        projection = coeffs @ V_shared  # (hidden_dim,)

        shared_update[layer] = projection.to(vec.dtype)
        task_specific[layer] = (vec_f - projection).to(vec.dtype)

    return shared_update, task_specific


def compose_vectors(
    shared: Dict[int, torch.Tensor],
    task_specific: Dict[int, torch.Tensor],
) -> Dict[int, torch.Tensor]:
    """Recombine shared and task-specific components for inference.

    Args:
        shared: Shared subspace vectors per layer.
        task_specific: Task-specific residual vectors per layer.

    Returns:
        Combined vectors per layer.
    """
    combined: Dict[int, torch.Tensor] = {}
    all_layers = set(shared.keys()) | set(task_specific.keys())
    for layer in all_layers:
        s = shared.get(layer)
        t = task_specific.get(layer)
        if s is not None and t is not None:
            combined[layer] = s + t
        elif s is not None:
            combined[layer] = s.clone()
        elif t is not None:
            combined[layer] = t.clone()
    return combined


def save_state(state: ContinualState, path: str) -> None:
    """Save continual learning state to disk.

    Saves tensors via torch.save and metadata/replay buffer as JSON sidecar.

    Args:
        state: The ContinualState to persist.
        path: Directory path for the checkpoint.
    """
    os.makedirs(path, exist_ok=True)

    # Save tensor data
    tensor_data = {
        "shared_vectors": state.shared_vectors,
        "task_vectors": state.task_vectors,
        "fisher_info": state.fisher_info,
        "old_vectors": state.old_vectors,
    }
    torch.save(tensor_data, os.path.join(path, "tensors.pt"))

    # Save metadata as JSON
    metadata = {
        "current_cycle": state.current_cycle,
        "task_priorities": state.task_priorities,
        "metrics": state.metrics,
    }
    with open(os.path.join(path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Save replay buffer
    state.replay_buffer.save(os.path.join(path, "replay_buffer.json"))


def load_state(path: str) -> Optional[ContinualState]:
    """Load continual learning state from checkpoint directory.

    Args:
        path: Directory containing checkpoint files.

    Returns:
        ContinualState if checkpoint exists, None otherwise.
    """
    tensors_path = os.path.join(path, "tensors.pt")
    metadata_path = os.path.join(path, "metadata.json")
    replay_path = os.path.join(path, "replay_buffer.json")

    if not os.path.exists(tensors_path):
        return None

    tensor_data = torch.load(tensors_path, map_location="cpu", weights_only=False)

    state = ContinualState(
        shared_vectors=tensor_data.get("shared_vectors", {}),
        task_vectors=tensor_data.get("task_vectors", {}),
        fisher_info=tensor_data.get("fisher_info", {}),
        old_vectors=tensor_data.get("old_vectors", {}),
    )

    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        state.current_cycle = metadata.get("current_cycle", 0)
        state.task_priorities = metadata.get("task_priorities", {})
        state.metrics = metadata.get("metrics", {})

    if os.path.exists(replay_path):
        state.replay_buffer = ReplayBuffer.load(replay_path)

    return state


def upload_to_s3(local_path: str, bucket: str, prefix: str) -> None:
    """Upload checkpoint directory to S3."""
    s3_uri = f"s3://{bucket}/{prefix}/"
    subprocess.run(
        ["aws", "s3", "sync", local_path, s3_uri, "--quiet"],
        check=True,
    )
    print(f"   Checkpoint uploaded to {s3_uri}")


# --- EWC (Elastic Weight Consolidation) ---


def compute_fisher_information(
    vectors: Dict[int, torch.Tensor],
    perturbed_scores: Dict[int, Tuple[float, float]],
    perturbation_scale: float = 0.01,
) -> Dict[int, torch.Tensor]:
    """Diagonal Fisher approximation via score sensitivity to perturbations."""
    fisher: Dict[int, torch.Tensor] = {}
    for layer, vec in vectors.items():
        if layer not in perturbed_scores:
            fisher[layer] = torch.zeros_like(vec)
            continue
        score_plus, score_minus = perturbed_scores[layer]
        grad_est = (score_plus - score_minus) / (2.0 * perturbation_scale)
        fisher[layer] = torch.full_like(vec, grad_est ** 2)
    return fisher


def apply_ewc_penalty(
    new_vectors: Dict[int, torch.Tensor],
    old_vectors: Dict[int, torch.Tensor],
    fisher: Dict[int, torch.Tensor],
    ewc_lambda: float,
) -> float:
    """Compute EWC penalty: lambda/2 * sum_i F_i * (theta_new - theta_old)^2."""
    penalty = 0.0
    for layer in new_vectors:
        if layer not in old_vectors or layer not in fisher:
            continue
        diff = new_vectors[layer] - old_vectors[layer]
        penalty += (fisher[layer] * diff ** 2).sum().item()
    return 0.5 * ewc_lambda * penalty


def ewc_constrained_update(
    current_vectors: Dict[int, torch.Tensor],
    update_vectors: Dict[int, torch.Tensor],
    fisher_all_tasks: Dict[str, Dict[int, torch.Tensor]],
    old_vectors_all_tasks: Dict[str, Dict[int, torch.Tensor]],
    ewc_lambda: float,
    learning_rate: float = 1.0,
) -> Dict[int, torch.Tensor]:
    """Apply update with EWC constraint: shrink updates along Fisher-important directions."""
    result: Dict[int, torch.Tensor] = {}
    for layer in current_vectors:
        base_update = learning_rate * update_vectors.get(
            layer, torch.zeros_like(current_vectors[layer])
        )
        penalty_grad = torch.zeros_like(current_vectors[layer])
        for task_name, fisher in fisher_all_tasks.items():
            old_vecs = old_vectors_all_tasks.get(task_name, {})
            if layer in fisher and layer in old_vecs:
                diff = current_vectors[layer] + base_update - old_vecs[layer]
                penalty_grad += ewc_lambda * fisher[layer] * diff
        result[layer] = current_vectors[layer] + base_update - learning_rate * penalty_grad
    return result

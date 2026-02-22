"""
Optimal extraction strategy for maximum signal.

This module provides functions to extract activations at the token position
with maximum signal, rather than at fixed positions (first/last).

Token-by-token analysis showed:
- chat_first has ~0 signal at token 0
- chat_last captures only 20-40% of available signal
- Optimal position is in the middle (tokens 5-24) with 2-10x more signal

Usage:
    # Two-pass extraction:
    # 1. Get preliminary steering direction from chat_last
    # 2. Re-extract at optimal positions using this module

    from wisent.core.activations.core.optimal_extraction import (
        extract_at_optimal_position,
        find_optimal_positions,
        compute_signal_trajectory,
    )

    # Get optimal extraction for a batch of pairs
    optimal_activations = extract_at_optimal_position(
        pos_raw_activations,  # [N, seq_len, hidden_dim]
        neg_raw_activations,  # [N, seq_len, hidden_dim]
        steering_direction,   # [hidden_dim]
        prompt_lengths,       # [N] - where answer starts for each pair
    )
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional
import torch


@dataclass
class OptimalExtractionResult:
    """Result of optimal extraction for a single pair."""
    pos_activation: torch.Tensor  # [hidden_dim]
    neg_activation: torch.Tensor  # [hidden_dim]
    optimal_position: int  # Token index (relative to answer start)
    signal_strength: float  # Projection difference at optimal position
    trajectory: Optional[torch.Tensor] = None  # [seq_len] signal at each position


def compute_signal_trajectory(
    pos_hidden_states: torch.Tensor,
    neg_hidden_states: torch.Tensor,
    steering_direction: torch.Tensor,
    prompt_len: int,
) -> Tuple[torch.Tensor, int, float]:
    """
    Compute signal strength at each token position.

    Returns:
        Tuple of (trajectory, optimal_pos, max_signal)
    """
    # Cast steering direction to same dtype as hidden states
    steering_direction = steering_direction.to(pos_hidden_states.dtype)
    steering_direction = steering_direction / (torch.norm(steering_direction) + 1e-8)
    pos_answer = pos_hidden_states[prompt_len:]
    neg_answer = neg_hidden_states[prompt_len:]
    min_len = min(len(pos_answer), len(neg_answer))
    if min_len == 0:
        return torch.zeros(1), 0, 0.0
    pos_answer, neg_answer = pos_answer[:min_len], neg_answer[:min_len]
    pos_proj = pos_answer @ steering_direction
    neg_proj = neg_answer @ steering_direction
    trajectory = pos_proj - neg_proj
    optimal_pos = int(torch.argmax(trajectory).item())
    max_signal = float(trajectory[optimal_pos].item())
    return trajectory, optimal_pos, max_signal


def extract_at_optimal_position(
    pos_hidden_states: torch.Tensor,
    neg_hidden_states: torch.Tensor,
    steering_direction: torch.Tensor,
    prompt_len: int,
    return_trajectory: bool = False,
) -> OptimalExtractionResult:
    """Extract activations at the token with maximum signal."""
    trajectory, optimal_pos, max_signal = compute_signal_trajectory(
        pos_hidden_states, neg_hidden_states, steering_direction, prompt_len
    )
    abs_pos = prompt_len + optimal_pos
    pos_activation = pos_hidden_states[min(abs_pos, len(pos_hidden_states) - 1)]
    neg_activation = neg_hidden_states[min(abs_pos, len(neg_hidden_states) - 1)]
    return OptimalExtractionResult(
        pos_activation=pos_activation,
        neg_activation=neg_activation,
        optimal_position=optimal_pos,
        signal_strength=max_signal,
        trajectory=trajectory if return_trajectory else None,
    )


def extract_batch_optimal(
    pos_batch: List[torch.Tensor],
    neg_batch: List[torch.Tensor],
    steering_direction: torch.Tensor,
    prompt_lengths: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[float]]:
    """Extract activations at optimal positions for a batch."""
    pos_acts, neg_acts, positions, strengths = [], [], [], []
    for pos_hs, neg_hs, prompt_len in zip(pos_batch, neg_batch, prompt_lengths):
        result = extract_at_optimal_position(pos_hs, neg_hs, steering_direction, prompt_len)
        pos_acts.append(result.pos_activation)
        neg_acts.append(result.neg_activation)
        positions.append(result.optimal_position)
        strengths.append(result.signal_strength)
    return torch.stack(pos_acts), torch.stack(neg_acts), positions, strengths


@dataclass
class PCADirectionResult:
    """Result of PCA-based direction finding."""
    direction: torch.Tensor  # [hidden_dim] - first principal component
    explained_variance_ratio: List[float]  # Variance explained by each component
    singular_values: torch.Tensor  # Raw singular values
    n_tokens: int  # Total tokens used
    n_pairs: int  # Number of pairs


def find_direction_from_all_tokens(
    pos_batch: List[torch.Tensor],
    neg_batch: List[torch.Tensor],
    prompt_lengths: List[int],
    return_details: bool = False,
) -> torch.Tensor | PCADirectionResult:
    """
    Find steering direction using PCA on ALL token differences. No fixed position needed.

    This avoids the chicken-and-egg problem by finding the dominant direction
    of variation across all tokens and all pairs simultaneously.

    Args:
        return_details: If True, return PCADirectionResult with variance info
    """
    all_diffs = []
    for pos_hs, neg_hs, prompt_len in zip(pos_batch, neg_batch, prompt_lengths):
        pos_answer = pos_hs[prompt_len:]
        neg_answer = neg_hs[prompt_len:]
        min_len = min(len(pos_answer), len(neg_answer))
        if min_len > 0:
            diffs = pos_answer[:min_len] - neg_answer[:min_len]
            all_diffs.append(diffs)
    if not all_diffs:
        zero_dir = torch.zeros(pos_batch[0].shape[-1])
        if return_details:
            return PCADirectionResult(zero_dir, [], torch.zeros(1), 0, 0)
        return zero_dir
    all_diffs = torch.cat(all_diffs, dim=0)  # [total_tokens, hidden_dim]
    n_tokens = all_diffs.shape[0]
    # Convert to float32 for SVD (required by torch.linalg.svd)
    all_diffs = all_diffs.float()
    # Center the data
    mean_diff = all_diffs.mean(dim=0)
    centered = all_diffs - mean_diff
    # SVD to find principal components
    _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    direction = Vh[0]  # First principal component
    # Ensure direction points from neg to pos (positive mean projection)
    if (all_diffs @ direction).mean() < 0:
        direction = -direction
    if not return_details:
        return direction
    # Compute explained variance ratios
    variance = S ** 2
    total_var = variance.sum()
    explained_ratio = (variance / total_var).tolist()[:10]  # Top 10 components
    return PCADirectionResult(
        direction=direction,
        explained_variance_ratio=explained_ratio,
        singular_values=S[:10],
        n_tokens=n_tokens,
        n_pairs=len(pos_batch),
    )


def extract_at_max_diff_norm(
    pos_hidden_states: torch.Tensor,
    neg_hidden_states: torch.Tensor,
    prompt_len: int,
) -> OptimalExtractionResult:
    """
    Extract at position with maximum ||pos - neg|| norm. NO steering direction needed.

    This is a direction-free alternative that finds where the pos/neg difference
    is largest in magnitude, without needing a preliminary steering direction.
    """
    pos_answer = pos_hidden_states[prompt_len:]
    neg_answer = neg_hidden_states[prompt_len:]
    min_len = min(len(pos_answer), len(neg_answer))
    if min_len == 0:
        return OptimalExtractionResult(
            pos_activation=pos_hidden_states[-1],
            neg_activation=neg_hidden_states[-1],
            optimal_position=0,
            signal_strength=0.0,
        )
    pos_answer, neg_answer = pos_answer[:min_len], neg_answer[:min_len]
    diff_norms = torch.norm(pos_answer - neg_answer, dim=1)
    optimal_pos = int(torch.argmax(diff_norms).item())
    max_norm = float(diff_norms[optimal_pos].item())
    abs_pos = prompt_len + optimal_pos
    return OptimalExtractionResult(
        pos_activation=pos_hidden_states[abs_pos],
        neg_activation=neg_hidden_states[abs_pos],
        optimal_position=optimal_pos,
        signal_strength=max_norm,
        trajectory=diff_norms,
    )


def compare_extraction_strategies(
    pos_hidden_states: torch.Tensor,
    neg_hidden_states: torch.Tensor,
    steering_direction: torch.Tensor,
    prompt_len: int,
) -> dict:
    """Compare signal strength across extraction strategies for a single pair."""
    steering_direction = steering_direction / (torch.norm(steering_direction) + 1e-8)
    trajectory, optimal_pos, max_signal = compute_signal_trajectory(
        pos_hidden_states, neg_hidden_states, steering_direction, prompt_len
    )
    pos_answer = pos_hidden_states[prompt_len:]
    neg_answer = neg_hidden_states[prompt_len:]
    min_len = min(len(pos_answer), len(neg_answer))
    if min_len == 0:
        return {"error": "No answer tokens"}

    def get_signal(pos_idx, neg_idx):
        pos_proj = float((pos_hidden_states[pos_idx] @ steering_direction).item())
        neg_proj = float((neg_hidden_states[neg_idx] @ steering_direction).item())
        return pos_proj - neg_proj

    last_signal = get_signal(-1, -1)
    pos_mean = pos_answer[:min_len].mean(dim=0)
    neg_mean = neg_answer[:min_len].mean(dim=0)
    return {
        "chat_first": get_signal(prompt_len, prompt_len),
        "chat_last": last_signal,
        "chat_mean": float(((pos_mean - neg_mean) @ steering_direction).item()),
        "chat_optimal": max_signal,
        "optimal_position": optimal_pos,
        "improvement_over_last": max_signal / (abs(last_signal) + 1e-8),
    }

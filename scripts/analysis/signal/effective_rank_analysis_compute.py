"""
Computation functions for effective rank analysis.

Contains dataclasses, Fisher ratio computation, effective rank metrics,
and linear subset detection.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from wisent.core.utils.config_tools.constants import NORM_EPS, ZERO_THRESHOLD


@dataclass
class EffectiveRankResult:
    """Result of effective rank analysis for one layer."""
    layer: int
    rank_50: int  # dims for 50% variance
    rank_80: int  # dims for 80% variance
    rank_90: int  # dims for 90% variance
    participation_ratio: float  # effective dimensionality
    mean_cosine: float  # mean pairwise cosine similarity
    pc1_variance: float  # variance explained by PC1
    pc1_3_variance: float  # variance explained by PC1-3
    # Fisher ratio metrics - how much each PC carries label signal
    fisher_ratio_pc1: float  # Fisher ratio for PC1
    fisher_ratio_pc2: float  # Fisher ratio for PC2
    fisher_ratio_pc3: float  # Fisher ratio for PC3
    top_fisher_pc: int  # which PC has highest Fisher ratio
    max_fisher_ratio: float  # maximum Fisher ratio across PCs
    cumulative_fisher_top3: float  # sum of Fisher ratios for top 3 PCs


@dataclass
class LinearSubset:
    """A subset of pairs with potentially linear signal."""
    pair_indices: List[int]
    rank_80: int
    mean_cosine: float
    pc1_variance: float


@dataclass
class BenchmarkAnalysis:
    """Full analysis for a benchmark."""
    benchmark: str
    model: str
    strategy: str
    num_pairs: int
    per_layer_results: List[EffectiveRankResult]
    best_layer: int
    best_fisher_ratio: float  # max Fisher ratio across layers
    linear_subsets: Optional[List[LinearSubset]] = None


def compute_fisher_ratio(projections: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Fisher ratio for a 1D projection.

    Fisher ratio = (between-class variance) / (within-class variance)
    Higher = better separation by this component.
    """
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_proj = projections[pos_mask]
    neg_proj = projections[neg_mask]

    if len(pos_proj) < 2 or len(neg_proj) < 2:
        return 0.0

    # Between-class variance
    mean_pos = pos_proj.mean()
    mean_neg = neg_proj.mean()
    between_var = (mean_pos - mean_neg) ** 2

    # Within-class variance (pooled)
    var_pos = pos_proj.var()
    var_neg = neg_proj.var()
    within_var = (var_pos + var_neg) / 2

    if within_var < ZERO_THRESHOLD:
        return 0.0

    return float(between_var / within_var)


def compute_effective_rank(
    pos_np: np.ndarray,
    neg_np: np.ndarray,
) -> EffectiveRankResult:
    """
    Compute effective rank metrics for activation data.

    Args:
        pos_np: positive class activations [N_pos, hidden_dim]
        neg_np: negative class activations [N_neg, hidden_dim]
    """
    # Combine data
    all_data = np.concatenate([pos_np, neg_np], axis=0)
    labels = np.array([1] * len(pos_np) + [0] * len(neg_np))

    # Difference vectors (for backward compatibility with mean cosine)
    n_pairs = min(len(pos_np), len(neg_np))
    diff_np = pos_np[:n_pairs] - neg_np[:n_pairs]

    # Center all data
    all_centered = all_data - all_data.mean(axis=0)

    # SVD on all data
    U, S, Vh = np.linalg.svd(all_centered, full_matrices=False)

    # Variance explained
    var_explained = (S ** 2) / (S ** 2).sum()
    cumvar = np.cumsum(var_explained)

    # Effective rank metrics
    rank_50 = int(np.searchsorted(cumvar, 0.5) + 1)
    rank_80 = int(np.searchsorted(cumvar, 0.8) + 1)
    rank_90 = int(np.searchsorted(cumvar, 0.9) + 1)

    # Participation ratio
    participation_ratio = float((S ** 2).sum() ** 2 / (S ** 4).sum())

    # Mean cosine similarity on difference vectors
    norms = np.linalg.norm(diff_np, axis=1, keepdims=True)
    diff_norm = diff_np / (norms + NORM_EPS)
    cos_sim_matrix = diff_norm @ diff_norm.T
    n = cos_sim_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    mean_cosine = float(cos_sim_matrix[mask].mean())

    # Project all data onto top PCs
    projections = all_centered @ Vh.T  # [N, num_components]

    # Compute Fisher ratio for each PC
    num_pcs = min(10, projections.shape[1])
    fisher_ratios = []
    for pc_idx in range(num_pcs):
        fr = compute_fisher_ratio(projections[:, pc_idx], labels)
        fisher_ratios.append(fr)

    # Find top Fisher PC
    top_fisher_pc = int(np.argmax(fisher_ratios))
    max_fisher_ratio = float(max(fisher_ratios))

    # Cumulative Fisher for top 3
    sorted_fisher = sorted(fisher_ratios, reverse=True)
    cumulative_fisher_top3 = float(sum(sorted_fisher[:3]))

    return EffectiveRankResult(
        layer=-1,  # Will be set by caller
        rank_50=rank_50,
        rank_80=rank_80,
        rank_90=rank_90,
        participation_ratio=participation_ratio,
        mean_cosine=mean_cosine,
        pc1_variance=float(var_explained[0]),
        pc1_3_variance=float(cumvar[min(2, len(cumvar)-1)]),
        fisher_ratio_pc1=float(fisher_ratios[0]) if len(fisher_ratios) > 0 else 0.0,
        fisher_ratio_pc2=float(fisher_ratios[1]) if len(fisher_ratios) > 1 else 0.0,
        fisher_ratio_pc3=float(fisher_ratios[2]) if len(fisher_ratios) > 2 else 0.0,
        top_fisher_pc=top_fisher_pc,
        max_fisher_ratio=max_fisher_ratio,
        cumulative_fisher_top3=cumulative_fisher_top3,
    )


def find_linear_subsets(
    diff_np: np.ndarray,
    threshold: float = 0.15,
    min_subset_size: int = 5,
) -> List[LinearSubset]:
    """Find subsets of pairs with higher similarity (potential linear signal)."""
    norms = np.linalg.norm(diff_np, axis=1, keepdims=True)
    diff_norm = diff_np / (norms + NORM_EPS)

    cos_sim_full = diff_norm @ diff_norm.T
    np.fill_diagonal(cos_sim_full, 0)

    # Greedy clustering by similarity
    mean_sim_per_pair = cos_sim_full.mean(axis=1)

    linear_subsets = []
    remaining = set(range(len(diff_np)))

    while len(remaining) >= min_subset_size:
        # Start new subset with most "central" remaining pair
        subset = []
        current_remaining = set(remaining)

        # Start with highest avg similarity pair
        best_start = max(current_remaining, key=lambda x: mean_sim_per_pair[x])
        subset.append(best_start)
        current_remaining.remove(best_start)

        # Greedily add pairs with high similarity to current subset
        while current_remaining:
            best = None
            best_sim = -1
            for candidate in current_remaining:
                avg_sim = np.mean([cos_sim_full[candidate, p] for p in subset])
                if avg_sim > best_sim:
                    best_sim = avg_sim
                    best = candidate

            if best_sim < threshold:
                break

            subset.append(best)
            current_remaining.remove(best)

        if len(subset) >= min_subset_size:
            # Compute metrics for this subset
            subset_diffs = diff_np[subset]
            half = len(subset_diffs) // 2
            if half >= 2:
                result = compute_effective_rank(subset_diffs[:half], subset_diffs[half:])
            else:
                result = compute_effective_rank(subset_diffs, subset_diffs)

            linear_subsets.append(LinearSubset(
                pair_indices=subset,
                rank_80=result.rank_80,
                mean_cosine=result.mean_cosine,
                pc1_variance=result.pc1_variance,
            ))

            # Remove these pairs from remaining
            remaining -= set(subset)
        else:
            # No more valid subsets, stop
            break

    return linear_subsets

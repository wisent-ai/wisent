"""Helper functions for cone structure analysis."""

from __future__ import annotations

from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F


def compute_pca_directions(
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    n_components: int,
) -> Tuple[torch.Tensor, float]:
    """Compute PCA directions and explained variance ratio."""
    all_activations = torch.cat([pos_tensor, neg_tensor], dim=0)
    mean = all_activations.mean(dim=0, keepdim=True)
    centered = all_activations - mean

    try:
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        k = min(n_components, Vh.shape[0])
        pca_directions = Vh[:k]
        total_var = (S ** 2).sum()
        explained_var = (S[:k] ** 2).sum() / total_var if total_var > 0 else 0.0
        return pca_directions, float(explained_var)
    except Exception:
        return torch.zeros(n_components, pos_tensor.shape[1]), 0.0


def discover_cone_directions(
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    num_directions: int,
    optimization_steps: int,
    learning_rate: float,
    min_cos_sim: float,
    max_cos_sim: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Discover cone directions via gradient optimization."""
    hidden_dim = pos_tensor.shape[1]

    caa_dir = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
    caa_dir = F.normalize(caa_dir, p=2, dim=0)

    directions = torch.randn(num_directions, hidden_dim)
    directions[0] = caa_dir

    for i in range(1, num_directions):
        noise = torch.randn(hidden_dim) * 0.3
        directions[i] = F.normalize(caa_dir + noise, p=2, dim=0)

    directions = F.normalize(directions, p=2, dim=1)
    directions.requires_grad_(True)

    optimizer = torch.optim.Adam([directions], lr=learning_rate)
    training_losses = []
    total_loss = torch.tensor(0.0)

    for step in range(optimization_steps):
        optimizer.zero_grad()
        dirs_norm = F.normalize(directions, p=2, dim=1)

        pos_proj = pos_tensor @ dirs_norm.T
        neg_proj = neg_tensor @ dirs_norm.T
        separation_loss = -((pos_proj.mean(dim=0) - neg_proj.mean(dim=0)).abs().mean())

        cos_sim = dirs_norm @ dirs_norm.T
        off_diag_mask = 1 - torch.eye(num_directions)
        off_diag = cos_sim * off_diag_mask

        negative_penalty = F.relu(-off_diag).sum()
        too_similar = F.relu(off_diag - max_cos_sim).sum()
        too_dissimilar = F.relu(min_cos_sim - off_diag).sum()

        cone_loss = negative_penalty + too_similar + 0.5 * too_dissimilar
        diversity_loss = -off_diag.var()
        total_loss = separation_loss + 0.5 * cone_loss + 0.1 * diversity_loss

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            directions.data = F.normalize(directions.data, p=2, dim=1)
            if directions.shape[0] > 1:
                primary = directions[0:1]
                for i in range(1, directions.shape[0]):
                    if (directions[i:i+1] @ primary.T).item() < 0:
                        directions.data[i] = -directions.data[i]

        if step % 20 == 0:
            training_losses.append(float(total_loss.item()))

    return directions.detach(), {"training_losses": training_losses,
                                  "final_loss": float(total_loss.item())}


def compute_cone_explained_variance(
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    cone_directions: torch.Tensor,
) -> float:
    """Compute how much variance the cone directions explain."""
    all_activations = torch.cat([pos_tensor, neg_tensor], dim=0)
    mean = all_activations.mean(dim=0, keepdim=True)
    centered = all_activations - mean

    total_var = (centered ** 2).sum()
    if total_var == 0:
        return 0.0

    dirs_norm = F.normalize(cone_directions, p=2, dim=1)
    projections = centered @ dirs_norm.T
    reconstructed = projections @ dirs_norm

    explained_var = (reconstructed ** 2).sum() / total_var
    return float(min(explained_var.item(), 1.0))


def check_half_space_consistency(directions: torch.Tensor) -> float:
    """Check what fraction of directions are in the same half-space."""
    if directions.shape[0] <= 1:
        return 1.0

    dirs_norm = F.normalize(directions, p=2, dim=1)
    primary = dirs_norm[0:1]
    cos_with_primary = (dirs_norm @ primary.T).squeeze()
    positive_count = (cos_with_primary > 0).sum().item()
    return positive_count / directions.shape[0]


def test_positive_combinations(
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    directions: torch.Tensor,
) -> float:
    """Test if difference vectors can be represented as positive combinations."""
    diff = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
    diff_norm = F.normalize(diff.unsqueeze(0), p=2, dim=1)
    dirs_norm = F.normalize(directions, p=2, dim=1)

    projections = (diff_norm @ dirs_norm.T).squeeze()
    positive_projections = (projections >= 0).sum().item()
    significant_projections = (projections > 0.1).sum().item()

    pos_ratio = positive_projections / directions.shape[0]
    sig_ratio = significant_projections / directions.shape[0]

    return 0.7 * pos_ratio + 0.3 * sig_ratio


def compute_cosine_similarity_matrix(directions: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity matrix."""
    dirs_norm = F.normalize(directions, p=2, dim=1)
    return dirs_norm @ dirs_norm.T


def compute_avg_off_diagonal(matrix: torch.Tensor) -> float:
    """Compute average of off-diagonal elements."""
    n = matrix.shape[0]
    if n <= 1:
        return 1.0
    mask = 1 - torch.eye(n)
    return float((matrix * mask).sum() / (n * (n - 1)))


def compute_separation_scores(
    pos_tensor: torch.Tensor,
    neg_tensor: torch.Tensor,
    directions: torch.Tensor,
) -> List[float]:
    """Compute separation score for each direction."""
    dirs_norm = F.normalize(directions, p=2, dim=1)
    pos_proj = pos_tensor @ dirs_norm.T
    neg_proj = neg_tensor @ dirs_norm.T
    separation = pos_proj.mean(dim=0) - neg_proj.mean(dim=0)
    return separation.tolist()


def compute_cone_score(
    pca_explained: float,
    cone_explained: float,
    half_space_score: float,
    avg_cos_sim: float,
    pos_combo_score: float,
    separation_scores: List[float],
) -> float:
    """Compute overall cone score combining all metrics."""
    var_ratio = cone_explained / max(pca_explained, 1e-6)
    var_score = min(var_ratio, 1.0)

    half_space_component = half_space_score

    if avg_cos_sim < 0:
        cos_score = 0.0
    elif avg_cos_sim < 0.3:
        cos_score = avg_cos_sim / 0.3 * 0.5
    elif avg_cos_sim <= 0.7:
        cos_score = 1.0
    else:
        cos_score = max(0.5, 1.0 - (avg_cos_sim - 0.7) / 0.3)

    combo_component = pos_combo_score

    significant_directions = sum(1 for s in separation_scores if abs(s) > 0.1)
    multi_dir_score = min(significant_directions / max(len(separation_scores), 1), 1.0)

    cone_score = (
        0.20 * var_score +
        0.25 * half_space_component +
        0.20 * cos_score +
        0.20 * combo_component +
        0.15 * multi_dir_score
    )

    return float(cone_score)

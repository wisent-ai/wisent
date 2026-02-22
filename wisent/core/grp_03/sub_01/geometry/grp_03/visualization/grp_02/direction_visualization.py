"""Visualization of steering directions and their relationships.

Shows multiple steering directions as vectors, their angles, and how they
relate to each other geometrically.
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
from sklearn.decomposition import PCA


def compute_direction_angles(
    directions: List[torch.Tensor],
    names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute pairwise angles and cosine similarities between directions.

    Args:
        directions: List of steering direction tensors
        names: Optional names for each direction

    Returns:
        Dict with cosine_matrix, angle_matrix (degrees), and summary stats.
    """
    n = len(directions)
    if names is None:
        names = [f"dir_{i}" for i in range(n)]

    # Normalize directions
    dirs_np = []
    for d in directions:
        d_np = d.cpu().numpy() if isinstance(d, torch.Tensor) else d
        norm = np.linalg.norm(d_np)
        dirs_np.append(d_np / norm if norm > 1e-10 else d_np)

    dirs_matrix = np.stack(dirs_np)

    # Cosine similarity matrix
    cosine_matrix = dirs_matrix @ dirs_matrix.T

    # Angle matrix (in degrees)
    cosine_clipped = np.clip(cosine_matrix, -1, 1)
    angle_matrix = np.degrees(np.arccos(cosine_clipped))

    # Extract off-diagonal for stats
    mask = ~np.eye(n, dtype=bool)
    off_diag_cos = cosine_matrix[mask]
    off_diag_ang = angle_matrix[mask]

    return {
        "cosine_matrix": cosine_matrix.tolist(),
        "angle_matrix_degrees": angle_matrix.tolist(),
        "names": names,
        "n_directions": n,
        "mean_cosine": float(off_diag_cos.mean()) if n > 1 else 1.0,
        "min_cosine": float(off_diag_cos.min()) if n > 1 else 1.0,
        "max_cosine": float(off_diag_cos.max()) if n > 1 else 1.0,
        "mean_angle_degrees": float(off_diag_ang.mean()) if n > 1 else 0.0,
        "min_angle_degrees": float(off_diag_ang.min()) if n > 1 else 0.0,
        "max_angle_degrees": float(off_diag_ang.max()) if n > 1 else 0.0,
        "orthogonal_pairs": int((np.abs(off_diag_cos) < 0.1).sum() // 2) if n > 1 else 0,
        "aligned_pairs": int((off_diag_cos > 0.9).sum() // 2) if n > 1 else 0,
    }


def plot_direction_similarity_matrix(
    directions: List[torch.Tensor],
    names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Create data for a heatmap of direction similarities.

    Returns data suitable for plotting with matplotlib/seaborn.
    """
    angles = compute_direction_angles(directions, names)

    return {
        "matrix": angles["cosine_matrix"],
        "labels": angles["names"],
        "title": "Steering Direction Cosine Similarities",
        "colorbar_label": "Cosine Similarity",
        "vmin": -1,
        "vmax": 1,
        "summary": {
            "mean_similarity": angles["mean_cosine"],
            "orthogonal_pairs": angles["orthogonal_pairs"],
            "aligned_pairs": angles["aligned_pairs"],
        },
    }


def plot_directions_in_pca_space(
    directions: List[torch.Tensor],
    pos_activations: Optional[torch.Tensor] = None,
    neg_activations: Optional[torch.Tensor] = None,
    names: Optional[List[str]] = None,
    n_components: int = 2,
) -> Dict[str, Any]:
    """
    Project directions to PCA space and visualize as arrows.

    If activations provided, fits PCA on activations and projects directions.
    Otherwise, fits PCA on the directions themselves.

    Args:
        directions: List of steering direction tensors
        pos_activations: Optional positive activations for PCA fitting
        neg_activations: Optional negative activations for PCA fitting
        names: Optional names for directions
        n_components: 2 or 3 for visualization

    Returns:
        Dict with projected directions and activation points for plotting.
    """
    n = len(directions)
    if names is None:
        names = [f"dir_{i}" for i in range(n)]

    # Stack directions
    dirs_np = np.stack([
        d.cpu().numpy() if isinstance(d, torch.Tensor) else d
        for d in directions
    ])

    # Fit PCA
    if pos_activations is not None and neg_activations is not None:
        pos_np = pos_activations.cpu().numpy() if isinstance(pos_activations, torch.Tensor) else pos_activations
        neg_np = neg_activations.cpu().numpy() if isinstance(neg_activations, torch.Tensor) else neg_activations
        X = np.vstack([pos_np, neg_np])
        pca = PCA(n_components=n_components)
        pca.fit(X)
        pos_proj = pca.transform(pos_np)
        neg_proj = pca.transform(neg_np)
    else:
        pca = PCA(n_components=n_components)
        pca.fit(dirs_np)
        pos_proj = None
        neg_proj = None

    # Project directions (as vectors from origin)
    dirs_proj = pca.transform(dirs_np)

    # Normalize for visualization (scale to unit length in projected space)
    norms = np.linalg.norm(dirs_proj, axis=1, keepdims=True)
    dirs_proj_normalized = dirs_proj / (norms + 1e-10)

    return {
        "directions_projected": dirs_proj_normalized.tolist(),
        "directions_raw_projected": dirs_proj.tolist(),
        "names": names,
        "n_components": n_components,
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "pos_projected": pos_proj.tolist() if pos_proj is not None else None,
        "neg_projected": neg_proj.tolist() if neg_proj is not None else None,
        "title": f"Steering Directions in PC1-PC{n_components} Space",
    }


def compute_per_concept_directions(
    pos: torch.Tensor,
    neg: torch.Tensor,
    cluster_labels: List[int],
    n_concepts: int,
) -> Tuple[List[torch.Tensor], List[str]]:
    """
    Compute steering direction for each concept cluster.

    Args:
        pos: Positive activations
        neg: Negative activations
        cluster_labels: Cluster assignment for each sample
        n_concepts: Number of concepts

    Returns:
        Tuple of (directions list, names list).
    """
    labels = np.array(cluster_labels)
    directions = []
    names = []

    for concept_id in range(n_concepts):
        mask = labels == concept_id
        n_samples = mask.sum()

        if n_samples < 2:
            continue

        pos_concept = pos[mask]
        neg_concept = neg[mask]
        direction = pos_concept.mean(dim=0) - neg_concept.mean(dim=0)

        directions.append(direction)
        names.append(f"concept_{concept_id} (n={n_samples})")

    return directions, names


def visualize_concept_directions(
    pos: torch.Tensor,
    neg: torch.Tensor,
    cluster_labels: List[int],
    n_concepts: int,
) -> Dict[str, Any]:
    """
    Full visualization of per-concept steering directions.

    Returns:
        Dict with angles, similarity matrix, and PCA projection data.
    """
    directions, names = compute_per_concept_directions(
        pos, neg, cluster_labels, n_concepts
    )

    if len(directions) < 2:
        return {
            "n_directions": len(directions),
            "error": "Need at least 2 concept directions to visualize relationships",
        }

    angles = compute_direction_angles(directions, names)
    similarity_matrix = plot_direction_similarity_matrix(directions, names)
    pca_viz = plot_directions_in_pca_space(directions, pos, neg, names)

    return {
        "n_directions": len(directions),
        "names": names,
        "angles": angles,
        "similarity_matrix": similarity_matrix,
        "pca_visualization": pca_viz,
        "interpretation": _interpret_direction_relationships(angles),
    }


def _interpret_direction_relationships(angles: Dict[str, Any]) -> List[str]:
    """Generate interpretation of direction relationships."""
    interpretations = []

    mean_cos = angles["mean_cosine"]
    orthogonal = angles["orthogonal_pairs"]
    aligned = angles["aligned_pairs"]
    n = angles["n_directions"]

    if mean_cos > 0.8:
        interpretations.append("Directions are highly aligned - may be same concept")
    elif mean_cos > 0.5:
        interpretations.append("Directions are moderately correlated")
    elif mean_cos > 0.1:
        interpretations.append("Directions are weakly correlated")
    elif mean_cos > -0.1:
        interpretations.append("Directions are approximately orthogonal")
    else:
        interpretations.append("Directions are anti-correlated (opposing)")

    if orthogonal > 0:
        interpretations.append(f"{orthogonal} direction pairs are nearly orthogonal")

    if aligned > 0 and n > 2:
        interpretations.append(f"{aligned} direction pairs are nearly identical")

    if angles["max_angle_degrees"] - angles["min_angle_degrees"] > 60:
        interpretations.append("Direction angles vary widely - concepts are distinct")

    return interpretations

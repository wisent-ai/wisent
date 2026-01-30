"""Null baseline comparisons for geometry detection.

Compares detected geometry (cone, sphere, clusters, translation) against
random data to determine if structure is statistically significant.
"""
import numpy as np
import torch
from typing import Dict

from ..analysis.activation_structure import (
    compute_cone_fit,
    compute_sphere_fit,
    compute_cluster_structure,
    compute_relative_position,
)


def _generate_null_activations(
    n_samples: int,
    n_dims: int,
    rng: np.random.RandomState,
) -> torch.Tensor:
    """Generate random activations for null baseline."""
    return torch.tensor(rng.randn(n_samples, n_dims), dtype=torch.float32)


def compute_cone_null(
    n_samples: int,
    n_dims: int,
    n_bootstrap: int = 10,
    random_state: int = 42,
) -> Dict[str, float]:
    """Compute cone metrics for random (null) data."""
    rng = np.random.RandomState(random_state)

    concentrations = []
    half_angles = []

    for _ in range(n_bootstrap):
        null_acts = _generate_null_activations(n_samples, n_dims, rng)
        result = compute_cone_fit(null_acts)

        if "error" not in result:
            concentrations.append(result["cone_concentration"])
            half_angles.append(result["cone_half_angle_90pct"])

    if not concentrations:
        return {"error": "could not compute null cone"}

    return {
        "cone_concentration_null": float(np.mean(concentrations)),
        "cone_concentration_null_std": float(np.std(concentrations)),
        "cone_half_angle_null": float(np.mean(half_angles)),
        "cone_half_angle_null_std": float(np.std(half_angles)),
    }


def compute_sphere_null(
    n_samples: int,
    n_dims: int,
    n_bootstrap: int = 10,
    random_state: int = 42,
) -> Dict[str, float]:
    """Compute sphere metrics for random (null) data."""
    rng = np.random.RandomState(random_state)

    radius_cvs = []

    for _ in range(n_bootstrap):
        null_acts = _generate_null_activations(n_samples, n_dims, rng)
        result = compute_sphere_fit(null_acts)

        if "error" not in result:
            radius_cvs.append(result["centroid_radius_cv"])

    if not radius_cvs:
        return {"error": "could not compute null sphere"}

    return {
        "radius_cv_null": float(np.mean(radius_cvs)),
        "radius_cv_null_std": float(np.std(radius_cvs)),
    }


def compute_cluster_null(
    n_samples: int,
    n_dims: int,
    n_bootstrap: int = 10,
    random_state: int = 42,
) -> Dict[str, float]:
    """Compute cluster metrics for random (null) data."""
    rng = np.random.RandomState(random_state)

    silhouettes = []
    best_ks = []

    for _ in range(n_bootstrap):
        null_acts = _generate_null_activations(n_samples, n_dims, rng)
        result = compute_cluster_structure(null_acts)

        if "error" not in result:
            silhouettes.append(result["best_silhouette"])
            best_ks.append(result["best_k"])

    if not silhouettes:
        return {"error": "could not compute null clusters"}

    return {
        "silhouette_null": float(np.mean(silhouettes)),
        "silhouette_null_std": float(np.std(silhouettes)),
        "best_k_null": float(np.mean(best_ks)),
    }


def compute_translation_null(
    n_samples: int,
    n_dims: int,
    n_bootstrap: int = 10,
    random_state: int = 42,
) -> Dict[str, float]:
    """Compute translation metrics for random (null) pairs."""
    rng = np.random.RandomState(random_state)

    consistencies = []
    explains_fracs = []

    for _ in range(n_bootstrap):
        pos_null = _generate_null_activations(n_samples, n_dims, rng)
        neg_null = _generate_null_activations(n_samples, n_dims, rng)
        result = compute_relative_position(pos_null, neg_null)

        if "error" not in result:
            consistencies.append(result["translation_consistency"])
            explains_fracs.append(result["shift_explains_fraction"])

    if not consistencies:
        return {"error": "could not compute null translation"}

    return {
        "translation_consistency_null": float(np.mean(consistencies)),
        "translation_consistency_null_std": float(np.std(consistencies)),
        "shift_explains_null": float(np.mean(explains_fracs)),
        "shift_explains_null_std": float(np.std(explains_fracs)),
    }


def _z_score(real_val: float, null_mean: float, null_std: float) -> float:
    """Compute z-score relative to null distribution."""
    if null_std < 1e-10:
        return 0.0
    return (real_val - null_mean) / null_std


def compute_geometry_vs_null(
    pos: torch.Tensor,
    neg: torch.Tensor,
    n_bootstrap: int = 10,
    random_state: int = 42,
) -> Dict[str, any]:
    """
    Compare detected geometry against null baselines.

    Returns z-scores for each geometry type:
    - cone_z > 2: significant cone structure
    - sphere_z < -2: significantly better sphere fit than null
    - cluster_z > 2: significant cluster structure
    - translation_z > 2: significant translation relationship
    """
    n_samples = len(pos)
    n_dims = pos.shape[1]

    # Get real metrics
    cone_pos = compute_cone_fit(pos)
    sphere_pos = compute_sphere_fit(pos)
    clusters_pos = compute_cluster_structure(pos)
    translation = compute_relative_position(pos, neg)

    # Get null baselines
    cone_null = compute_cone_null(n_samples, n_dims, n_bootstrap, random_state)
    sphere_null = compute_sphere_null(n_samples, n_dims, n_bootstrap, random_state)
    cluster_null = compute_cluster_null(n_samples, n_dims, n_bootstrap, random_state)
    trans_null = compute_translation_null(n_samples, n_dims, n_bootstrap, random_state)

    z_scores = {}

    # Cone: higher concentration = more cone-like
    if "error" not in cone_pos and "error" not in cone_null:
        z_scores["cone_concentration_z"] = _z_score(
            cone_pos["cone_concentration"],
            cone_null["cone_concentration_null"],
            cone_null["cone_concentration_null_std"],
        )

    # Sphere: lower CV = better sphere fit
    if "error" not in sphere_pos and "error" not in sphere_null:
        z_scores["sphere_fit_z"] = _z_score(
            sphere_pos["centroid_radius_cv"],
            sphere_null["radius_cv_null"],
            sphere_null["radius_cv_null_std"],
        )

    # Clusters: higher silhouette = better clusters
    if "error" not in clusters_pos and "error" not in cluster_null:
        z_scores["cluster_z"] = _z_score(
            clusters_pos["best_silhouette"],
            cluster_null["silhouette_null"],
            cluster_null["silhouette_null_std"],
        )

    # Translation: higher consistency = more translation-like
    if "error" not in translation and "error" not in trans_null:
        z_scores["translation_z"] = _z_score(
            translation["translation_consistency"],
            trans_null["translation_consistency_null"],
            trans_null["translation_consistency_null_std"],
        )

    # Determine significant structures
    significant = []
    if z_scores.get("cone_concentration_z", 0) > 2:
        significant.append("cone")
    if z_scores.get("sphere_fit_z", 0) < -2:  # Lower CV is better
        significant.append("sphere")
    if z_scores.get("cluster_z", 0) > 2:
        significant.append("clusters")
    if z_scores.get("translation_z", 0) > 2:
        significant.append("translation")

    return {
        "real": {
            "cone": cone_pos,
            "sphere": sphere_pos,
            "clusters": clusters_pos,
            "translation": translation,
        },
        "null": {
            "cone": cone_null,
            "sphere": sphere_null,
            "clusters": cluster_null,
            "translation": trans_null,
        },
        "z_scores": z_scores,
        "significant_structures": significant,
        "has_significant_structure": len(significant) > 0,
    }

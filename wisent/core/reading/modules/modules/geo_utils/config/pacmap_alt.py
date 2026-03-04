"""
Alternative PaCMAP implementation using pynndescent instead of Annoy.

This provides the same functionality as PaCMAP but works on systems
where Annoy hangs (e.g., some Apple Silicon configurations).
"""

import numpy as np
from typing import Tuple
from pynndescent import NNDescent
from sklearn.decomposition import PCA
from wisent.core.utils.config_tools.constants import (NORM_EPS, DEFAULT_RANDOM_SEED,
    N_COMPONENTS_2D, N_JOBS_SINGLE)


def find_neighbors(
    X: np.ndarray,
    n_neighbors: int = None,
) -> np.ndarray:
    """Find k-nearest neighbors using pynndescent."""
    if n_neighbors is None:
        raise ValueError("n_neighbors is required")
    n_samples = X.shape[0]
    n_neighbors = min(n_neighbors, n_samples - 1)

    index = NNDescent(
        X,
        n_neighbors=n_neighbors + 1,
        metric='euclidean',
        n_jobs=N_JOBS_SINGLE,
        random_state=DEFAULT_RANDOM_SEED,
    )
    neighbors, _ = index.neighbor_graph
    return neighbors[:, 1:n_neighbors + 1]


def create_pairs(
    X: np.ndarray,
    neighbors: np.ndarray,
    n_mid: int = None,
    n_far: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create near, mid, and far pairs for PaCMAP optimization."""
    if n_mid is None:
        raise ValueError("n_mid is required")
    if n_far is None:
        raise ValueError("n_far is required")
    n_samples = X.shape[0]
    rng = np.random.RandomState(DEFAULT_RANDOM_SEED)

    # Near pairs from neighbor graph
    near_pairs = []
    for i in range(n_samples):
        for j in neighbors[i]:
            near_pairs.append([i, j])
    near_pairs = np.array(near_pairs, dtype=np.int32)

    # Mid pairs - sample from second-degree neighbors
    mid_pairs = []
    for i in range(n_samples):
        second_neighbors = set()
        for j in neighbors[i]:
            for k in neighbors[j]:
                if k != i and k not in neighbors[i]:
                    second_neighbors.add(k)
        if second_neighbors:
            selected = rng.choice(
                list(second_neighbors),
                size=min(n_mid, len(second_neighbors)),
                replace=False
            )
            for j in selected:
                mid_pairs.append([i, j])
    mid_pairs = np.array(mid_pairs, dtype=np.int32) if mid_pairs else np.zeros((0, 2), dtype=np.int32)

    # Far pairs - random sampling
    far_pairs = []
    for i in range(n_samples):
        candidates = list(set(range(n_samples)) - set(neighbors[i]) - {i})
        if candidates:
            selected = rng.choice(candidates, size=min(n_far, len(candidates)), replace=False)
            for j in selected:
                far_pairs.append([i, j])
    far_pairs = np.array(far_pairs, dtype=np.int32) if far_pairs else np.zeros((0, 2), dtype=np.int32)

    return near_pairs, mid_pairs, far_pairs


def pacmap_embedding(
    X: np.ndarray,
    pacmap_pca_dim_threshold: int,
    n_components: int = N_COMPONENTS_2D,
    n_neighbors: int = None,
    num_iters: int = None,
    learning_rate: float = None,
    random_state: int = DEFAULT_RANDOM_SEED,
    pacmap_config: dict = None,
) -> np.ndarray:
    """
    Compute PaCMAP embedding using pynndescent for neighbor search.

    Args:
        X: Input data (n_samples, n_features)
        n_components: Output dimensions (defaults to N_COMPONENTS_2D)
        n_neighbors: Number of neighbors to use
        num_iters: Number of optimization iterations
        learning_rate: Learning rate for optimization
        random_state: Random seed

    Returns:
        Embedding array (n_samples, n_components)
    """
    if n_neighbors is None:
        raise ValueError("n_neighbors is required")
    if num_iters is None:
        raise ValueError("num_iters is required")
    if learning_rate is None:
        raise ValueError("learning_rate is required")
    if pacmap_config is None:
        raise ValueError("pacmap_config is required (dict with phase params)")
    pc = pacmap_config
    n_samples, n_features = X.shape
    rng = np.random.RandomState(random_state)

    if n_features > pacmap_pca_dim_threshold:
        pca = PCA(n_components=min(pacmap_pca_dim_threshold, n_samples - 1), random_state=random_state)
        X_reduced = pca.fit_transform(X)
    else:
        X_reduced = X

    neighbors = find_neighbors(X_reduced, n_neighbors)
    near_pairs, mid_pairs, far_pairs = create_pairs(X_reduced, neighbors, n_mid=pc["n_mid_pairs"], n_far=pc["n_far_pairs"])

    pca_init = PCA(n_components=n_components, random_state=random_state)
    Y = pca_init.fit_transform(X_reduced).astype(np.float32)
    Y = Y / (np.std(Y) + NORM_EPS) * pc["initial_scale"]

    for iteration in range(num_iters):
        grad = np.zeros_like(Y)
        phase = iteration / num_iters
        if phase < pc["phase_1_end"]:
            w_near_curr, w_mid_curr, w_far_curr = pc["p1_near"], pc["p1_mid"], pc["p1_far"]
        elif phase < pc["phase_2_end"]:
            w_near_curr, w_mid_curr, w_far_curr = pc["p2_near"], pc["p2_mid"], pc["p2_far"]
        else:
            w_near_curr, w_mid_curr, w_far_curr = pc["p3_near"], pc["p3_mid"], pc["p3_far"]

        for i, j in near_pairs:
            diff = Y[i] - Y[j]
            dist_sq = np.sum(diff ** 2) + NORM_EPS
            force = w_near_curr * diff / (1 + dist_sq)
            grad[i] -= force
            grad[j] += force
        for i, j in mid_pairs:
            diff = Y[i] - Y[j]
            dist_sq = np.sum(diff ** 2) + NORM_EPS
            force = w_mid_curr * diff / (1 + dist_sq)
            grad[i] -= force
            grad[j] += force
        for i, j in far_pairs:
            diff = Y[i] - Y[j]
            dist_sq = np.sum(diff ** 2) + NORM_EPS
            force = w_far_curr * diff / (dist_sq + pc["far_repulse_eps"])
            grad[i] += force
            grad[j] -= force

        lr = learning_rate * (1 - iteration / num_iters)
        Y -= lr * grad / (n_samples + NORM_EPS)

    return Y


def plot_pacmap_alt(
    pos_activations: np.ndarray,
    neg_activations: np.ndarray,
    pacmap_pca_dim_threshold: int,
    n_neighbors: int = None,
    num_iters: int = None,
    title: str = "PaCMAP Projection",
) -> dict:
    """
    PaCMAP projection using alternative implementation.

    Drop-in replacement for plot_pacmap_projection.
    """
    if hasattr(pos_activations, 'numpy'):
        pos = pos_activations.float().cpu().numpy()
    else:
        pos = np.asarray(pos_activations, dtype=np.float32)

    if hasattr(neg_activations, 'numpy'):
        neg = neg_activations.float().cpu().numpy()
    else:
        neg = np.asarray(neg_activations, dtype=np.float32)

    X = np.vstack([pos, neg])
    n_samples = len(X)

    if n_samples < 5:
        return {"error": "not enough samples for PaCMAP"}

    n_neighbors = min(n_neighbors, n_samples - 1)

    embedding = pacmap_embedding(
        X,
        pacmap_pca_dim_threshold=pacmap_pca_dim_threshold,
        n_components=N_COMPONENTS_2D,
        n_neighbors=n_neighbors,
        num_iters=num_iters,
    )

    pos_embedded = embedding[:len(pos)]
    neg_embedded = embedding[len(pos):]

    return {
        "pos_projected": pos_embedded,
        "neg_projected": neg_embedded,
        "title": title,
    }

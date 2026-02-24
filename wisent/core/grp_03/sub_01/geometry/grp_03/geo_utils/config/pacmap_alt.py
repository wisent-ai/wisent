"""
Alternative PaCMAP implementation using pynndescent instead of Annoy.

This provides the same functionality as PaCMAP but works on systems
where Annoy hangs (e.g., some Apple Silicon configurations).
"""

import numpy as np
from typing import Optional, Tuple
from pynndescent import NNDescent
from sklearn.decomposition import PCA
from wisent.core.constants import (NORM_EPS, DEFAULT_RANDOM_SEED, PACMAP_INITIAL_SCALE, SPECTRAL_N_NEIGHBORS_DEFAULT,
    PACMAP_W_NEAR, PACMAP_W_MID, PACMAP_W_FAR, PACMAP_PHASE_1_END, PACMAP_PHASE_2_END,
    PACMAP_P1_NEAR, PACMAP_P1_MID, PACMAP_P1_FAR, PACMAP_P2_NEAR, PACMAP_P2_MID,
    PACMAP_P2_FAR, PACMAP_P3_NEAR, PACMAP_P3_MID, PACMAP_P3_FAR, PACMAP_FAR_REPULSE_EPS,
    PACMAP_ALT_NUM_ITERS, PACMAP_LEARNING_RATE_DEFAULT, PACMAP_N_MID_PAIRS,
    PACMAP_N_FAR_PAIRS, N_COMPONENTS_2D, PACMAP_PCA_DIM_THRESHOLD)


def find_neighbors(
    X: np.ndarray,
    n_neighbors: int = SPECTRAL_N_NEIGHBORS_DEFAULT,
) -> np.ndarray:
    """Find k-nearest neighbors using pynndescent."""
    n_samples = X.shape[0]
    n_neighbors = min(n_neighbors, n_samples - 1)

    index = NNDescent(
        X,
        n_neighbors=n_neighbors + 1,
        metric='euclidean',
        n_jobs=1,
        random_state=DEFAULT_RANDOM_SEED,
    )
    neighbors, _ = index.neighbor_graph
    return neighbors[:, 1:n_neighbors + 1]


def create_pairs(
    X: np.ndarray,
    neighbors: np.ndarray,
    n_mid: int = PACMAP_N_MID_PAIRS,
    n_far: int = PACMAP_N_FAR_PAIRS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create near, mid, and far pairs for PaCMAP optimization."""
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
    n_components: int = N_COMPONENTS_2D,
    n_neighbors: int = SPECTRAL_N_NEIGHBORS_DEFAULT,
    num_iters: int = PACMAP_ALT_NUM_ITERS,
    learning_rate: float = PACMAP_LEARNING_RATE_DEFAULT,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> np.ndarray:
    """
    Compute PaCMAP embedding using pynndescent for neighbor search.

    Args:
        X: Input data (n_samples, n_features)
        n_components: Output dimensions (default 2)
        n_neighbors: Number of neighbors to use
        num_iters: Number of optimization iterations
        learning_rate: Learning rate for optimization
        random_state: Random seed

    Returns:
        Embedding array (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    rng = np.random.RandomState(random_state)

    # Reduce dimensions with PCA if needed for speed
    if n_features > PACMAP_PCA_DIM_THRESHOLD:
        pca = PCA(n_components=min(PACMAP_PCA_DIM_THRESHOLD, n_samples - 1), random_state=random_state)
        X_reduced = pca.fit_transform(X)
    else:
        X_reduced = X

    # Find neighbors
    neighbors = find_neighbors(X_reduced, n_neighbors)

    # Create pairs
    near_pairs, mid_pairs, far_pairs = create_pairs(X_reduced, neighbors)

    # Initialize embedding with PCA
    pca_init = PCA(n_components=n_components, random_state=random_state)
    Y = pca_init.fit_transform(X_reduced).astype(np.float32)
    Y = Y / (np.std(Y) + NORM_EPS) * PACMAP_INITIAL_SCALE

    # Optimization weights
    w_near = PACMAP_W_NEAR
    w_mid = PACMAP_W_MID
    w_far = PACMAP_W_FAR

    # Gradient descent optimization
    for iteration in range(num_iters):
        grad = np.zeros_like(Y)

        # Phase-dependent weights (like original PaCMAP)
        phase = iteration / num_iters
        if phase < PACMAP_PHASE_1_END:
            w_near_curr, w_mid_curr, w_far_curr = PACMAP_P1_NEAR, PACMAP_P1_MID, PACMAP_P1_FAR
        elif phase < PACMAP_PHASE_2_END:
            w_near_curr, w_mid_curr, w_far_curr = PACMAP_P2_NEAR, PACMAP_P2_MID, PACMAP_P2_FAR
        else:
            w_near_curr, w_mid_curr, w_far_curr = PACMAP_P3_NEAR, PACMAP_P3_MID, PACMAP_P3_FAR

        # Near pair attractive forces
        for i, j in near_pairs:
            diff = Y[i] - Y[j]
            dist_sq = np.sum(diff ** 2) + NORM_EPS
            force = w_near_curr * diff / (1 + dist_sq)
            grad[i] -= force
            grad[j] += force

        # Mid pair forces
        for i, j in mid_pairs:
            diff = Y[i] - Y[j]
            dist_sq = np.sum(diff ** 2) + NORM_EPS
            force = w_mid_curr * diff / (1 + dist_sq)
            grad[i] -= force
            grad[j] += force

        # Far pair repulsive forces
        for i, j in far_pairs:
            diff = Y[i] - Y[j]
            dist_sq = np.sum(diff ** 2) + NORM_EPS
            force = w_far_curr * diff / (dist_sq + PACMAP_FAR_REPULSE_EPS)
            grad[i] += force
            grad[j] -= force

        # Update with adaptive learning rate
        lr = learning_rate * (1 - iteration / num_iters)
        Y -= lr * grad / (n_samples + NORM_EPS)

    return Y


def plot_pacmap_alt(
    pos_activations: np.ndarray,
    neg_activations: np.ndarray,
    n_neighbors: int = SPECTRAL_N_NEIGHBORS_DEFAULT,
    num_iters: int = PACMAP_ALT_NUM_ITERS,
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
        n_components=2,
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

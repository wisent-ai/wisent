"""
Alternative PaCMAP implementation using pynndescent instead of Annoy.

This provides the same functionality as PaCMAP but works on systems
where Annoy hangs (e.g., some Apple Silicon configurations).
"""

import numpy as np
from typing import Optional, Tuple
from pynndescent import NNDescent
from sklearn.decomposition import PCA


def find_neighbors(
    X: np.ndarray,
    n_neighbors: int = 10,
) -> np.ndarray:
    """Find k-nearest neighbors using pynndescent."""
    n_samples = X.shape[0]
    n_neighbors = min(n_neighbors, n_samples - 1)

    index = NNDescent(
        X,
        n_neighbors=n_neighbors + 1,
        metric='euclidean',
        n_jobs=1,
        random_state=42,
    )
    neighbors, _ = index.neighbor_graph
    return neighbors[:, 1:n_neighbors + 1]


def create_pairs(
    X: np.ndarray,
    neighbors: np.ndarray,
    n_mid: int = 5,
    n_far: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create near, mid, and far pairs for PaCMAP optimization."""
    n_samples = X.shape[0]
    rng = np.random.RandomState(42)

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
    n_components: int = 2,
    n_neighbors: int = 10,
    num_iters: int = 100,
    learning_rate: float = 1.0,
    random_state: int = 42,
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
    if n_features > 50:
        pca = PCA(n_components=min(50, n_samples - 1), random_state=random_state)
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
    Y = Y / (np.std(Y) + 1e-8) * 0.01  # Small initial scale

    # Optimization weights
    w_near = 1.0
    w_mid = 0.5
    w_far = 0.1

    # Gradient descent optimization
    for iteration in range(num_iters):
        grad = np.zeros_like(Y)

        # Phase-dependent weights (like original PaCMAP)
        phase = iteration / num_iters
        if phase < 0.1:
            w_near_curr, w_mid_curr, w_far_curr = 2.0, 0.5, 0.01
        elif phase < 0.35:
            w_near_curr, w_mid_curr, w_far_curr = 3.0, 1.0, 0.1
        else:
            w_near_curr, w_mid_curr, w_far_curr = 1.0, 0.5, 1.0

        # Near pair attractive forces
        for i, j in near_pairs:
            diff = Y[i] - Y[j]
            dist_sq = np.sum(diff ** 2) + 1e-8
            force = w_near_curr * diff / (1 + dist_sq)
            grad[i] -= force
            grad[j] += force

        # Mid pair forces
        for i, j in mid_pairs:
            diff = Y[i] - Y[j]
            dist_sq = np.sum(diff ** 2) + 1e-8
            force = w_mid_curr * diff / (1 + dist_sq)
            grad[i] -= force
            grad[j] += force

        # Far pair repulsive forces
        for i, j in far_pairs:
            diff = Y[i] - Y[j]
            dist_sq = np.sum(diff ** 2) + 1e-8
            force = w_far_curr * diff / (dist_sq + 0.01)
            grad[i] += force
            grad[j] -= force

        # Update with adaptive learning rate
        lr = learning_rate * (1 - iteration / num_iters)
        Y -= lr * grad / (n_samples + 1e-8)

    return Y


def plot_pacmap_alt(
    pos_activations: np.ndarray,
    neg_activations: np.ndarray,
    n_neighbors: int = 10,
    num_iters: int = 100,
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

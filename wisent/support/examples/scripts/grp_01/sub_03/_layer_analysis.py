"""Layer-wise analysis for mixed concept detection."""

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import pacmap
    HAS_PACMAP = True
except ImportError:
    HAS_PACMAP = False

from wisent.core.models.wisent_model import WisentModel
from wisent.core.constants import (
    ZERO_THRESHOLD, DEFAULT_RANDOM_SEED, LINEARITY_N_INIT,
    VIZ_N_NEIGHBORS, VIZ_MIN_DIST, STABILITY_N_CLUSTERS,
    TOKENIZER_MAX_LENGTH_GEOMETRY, N_COMPONENTS_2D,
    PROGRESS_LOG_INTERVAL_20,
)


def get_activations_all_layers(model: WisentModel, text: str) -> Dict[int, torch.Tensor]:
    """Extract last token activation from ALL layers."""
    inputs = model.tokenizer(text, return_tensors="pt", truncation=True, max_length=TOKENIZER_MAX_LENGTH_GEOMETRY)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    activations = {}
    handles = []
    
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                activations[layer_idx] = output[0][:, -1, :].detach().cpu()
            else:
                activations[layer_idx] = output[:, -1, :].detach().cpu()
        return hook_fn
    
    layers = model._layers
    for idx, layer in enumerate(layers):
        handle = layer.register_forward_hook(make_hook(idx))
        handles.append(handle)
    
    with torch.no_grad():
        model.hf_model(**inputs)
    
    for handle in handles:
        handle.remove()
    
    return {k: v.squeeze(0) for k, v in activations.items()}


def extract_difference_vectors_all_layers(
    model: WisentModel,
    pairs: List[Dict],
    show_progress: bool = True
) -> Dict[int, np.ndarray]:
    """
    Extract difference vectors for all layers.
    
    Returns:
        Dict mapping layer_idx -> [N, hidden_dim] array
    """
    all_diffs = {i: [] for i in range(model.num_layers)}
    
    total = len(pairs)
    for i, pair in enumerate(pairs):
        if show_progress and (i + 1) % PROGRESS_LOG_INTERVAL_20 == 0:
            print(f"  Extracting activations: {i+1}/{total}")
        
        prompt = pair["question"]
        
        pos_text = model.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": pair["positive"]}],
            tokenize=False, add_generation_prompt=False
        )
        neg_text = model.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": pair["negative"]}],
            tokenize=False, add_generation_prompt=False
        )
        
        pos_acts = get_activations_all_layers(model, pos_text)
        neg_acts = get_activations_all_layers(model, neg_text)
        
        for layer_idx in range(model.num_layers):
            diff = (pos_acts[layer_idx] - neg_acts[layer_idx]).numpy()
            all_diffs[layer_idx].append(diff)
    
    return {k: np.array(v) for k, v in all_diffs.items()}


def compute_projection(
    diff_vectors: np.ndarray,
    method: str = "pca",
    n_components: int = N_COMPONENTS_2D,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Tuple[np.ndarray, str]:
    """
    Project difference vectors to 2D using various methods.
    
    Args:
        diff_vectors: [N, hidden_dim] array
        method: "pca", "umap", or "pacmap"
        n_components: number of output dimensions
        seed: random seed
        
    Returns:
        projected: [N, n_components] array
        method_used: actual method used (may differ if requested not available)
    """
    if method == "umap":
        if not HAS_UMAP:
            print("  UMAP not installed, falling back to PCA")
            method = "pca"
        else:
            reducer = umap.UMAP(n_components=n_components, random_state=seed, n_neighbors=VIZ_N_NEIGHBORS, min_dist=VIZ_MIN_DIST)
            projected = reducer.fit_transform(diff_vectors)
            return projected, "umap"
    
    if method == "pacmap":
        if not HAS_PACMAP:
            print("  PaCMAP not installed, falling back to PCA")
            method = "pca"
        else:
            reducer = pacmap.PaCMAP(n_components=n_components, random_state=seed)
            projected = reducer.fit_transform(diff_vectors)
            return projected, "pacmap"
    
    # Default: PCA
    pca = PCA(n_components=n_components, random_state=seed)
    projected = pca.fit_transform(diff_vectors)
    return projected, "pca"


def analyze_layer_separability(
    diff_vectors_by_layer: Dict[int, np.ndarray],
    sources: List[str],
) -> Dict[int, Dict]:
    """
    Analyze how well concepts are separated at each layer.
    
    For each layer, compute:
    - Silhouette score for k=2 clustering
    - Cluster direction similarity
    - Cluster purity (how well clusters align with true sources)
    """
    results = {}
    
    for layer_idx, diffs in diff_vectors_by_layer.items():
        # Cluster
        km = KMeans(n_clusters=STABILITY_N_CLUSTERS, random_state=DEFAULT_RANDOM_SEED, n_init=LINEARITY_N_INIT)
        labels = km.fit_predict(diffs)
        
        # Silhouette
        sil = silhouette_score(diffs, labels)
        
        # Direction similarity
        mask0 = labels == 0
        mask1 = labels == 1
        if mask0.sum() >= 3 and mask1.sum() >= 3:
            dir0 = diffs[mask0].mean(axis=0)
            dir1 = diffs[mask1].mean(axis=0)
            dir0 = dir0 / (np.linalg.norm(dir0) + ZERO_THRESHOLD)
            dir1 = dir1 / (np.linalg.norm(dir1) + ZERO_THRESHOLD)
            dir_sim = np.abs(np.dot(dir0, dir1))
        else:
            dir_sim = 1.0
        
        # Cluster purity
        from collections import Counter
        c0_sources = [sources[i] for i in range(len(sources)) if labels[i] == 0]
        c1_sources = [sources[i] for i in range(len(sources)) if labels[i] == 1]
        
        if c0_sources and c1_sources:
            c0_purity = max(Counter(c0_sources).values()) / len(c0_sources)
            c1_purity = max(Counter(c1_sources).values()) / len(c1_sources)
            avg_purity = (c0_purity + c1_purity) / 2
        else:
            avg_purity = 0.5
        
        results[layer_idx] = {
            'silhouette': sil,
            'direction_similarity': dir_sim,
            'cluster_purity': avg_purity,
            'separability_score': (1 - dir_sim) * avg_purity,  # Higher = better separation aligned with sources
        }
    
    return results

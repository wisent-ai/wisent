"""Visualization functions (part 1) for mixed concept detection."""

from collections import Counter
from typing import Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

from ._layer_analysis import compute_projection
from wisent.core.constants import (
    ZERO_THRESHOLD, VIZ_DPI, DEFAULT_RANDOM_SEED,
    DISPLAY_TRUNCATION_ERROR, DISPLAY_TRUNCATION_SHORT,
    LINEARITY_N_INIT, VIZ_MARKER_SIZE, VIZ_ALPHA_LIGHT,
    VIZ_ALPHA_MEDIUM, VIZ_LINEWIDTH_NORMAL,
    VIZ_MARKERSIZE_LINE_SMALL,
    VIZ_FONTSIZE_SUPTITLE, SEPARATOR_WIDTH_WIDE,
)


def visualize_multi_method(
    diff_vectors: np.ndarray,
    sources: List[str],
    title: str = "Multi-Method Projection",
    output_path: Optional[str] = None,
    show_plot: bool = True,
):
    """
    Visualize projections using PCA, UMAP, and PaCMAP side by side.
    """
    methods = ["pca"]
    if HAS_UMAP:
        methods.append("umap")
    if HAS_PACMAP:
        methods.append("pacmap")
    
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=VIZ_FONTSIZE_SUPTITLE, fontweight='bold')
    
    source_colors = {'truthfulqa': '#e74c3c', 'hellaswag': '#3498db', 'unknown': '#95a5a6'}
    unique_sources = list(set(sources))
    
    for ax, method in zip(axes, methods):
        proj, method_used = compute_projection(diff_vectors, method=method)
        
        for source in unique_sources:
            mask = np.array([s == source for s in sources])
            color = source_colors.get(source, '#95a5a6')
            ax.scatter(proj[mask, 0], proj[mask, 1], c=color, label=source, alpha=VIZ_ALPHA_MEDIUM, s=VIZ_MARKER_SIZE)
        
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.set_title(f'{method_used.upper()}')
        ax.legend()
        ax.grid(True, alpha=VIZ_ALPHA_LIGHT)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=VIZ_DPI, bbox_inches='tight')
        print(f"  Saved to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_layer_analysis(
    layer_results: Dict[int, Dict],
    title: str = "Layer-wise Separability Analysis",
    output_path: Optional[str] = None,
    show_plot: bool = True,
):
    """
    Visualize how concept separability varies across layers.
    """
    layers = sorted(layer_results.keys())
    silhouettes = [layer_results[l]['silhouette'] for l in layers]
    dir_sims = [layer_results[l]['direction_similarity'] for l in layers]
    purities = [layer_results[l]['cluster_purity'] for l in layers]
    sep_scores = [layer_results[l]['separability_score'] for l in layers]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=VIZ_FONTSIZE_SUPTITLE, fontweight='bold')
    
    # Silhouette
    ax1 = axes[0, 0]
    ax1.plot(layers, silhouettes, 'b-o', linewidth=VIZ_LINEWIDTH_NORMAL, markersize=VIZ_MARKERSIZE_LINE_SMALL)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Cluster Quality (higher = better clusters)')
    ax1.grid(True, alpha=VIZ_ALPHA_LIGHT)
    
    # Direction similarity
    ax2 = axes[0, 1]
    ax2.plot(layers, dir_sims, 'r-o', linewidth=VIZ_LINEWIDTH_NORMAL, markersize=VIZ_MARKERSIZE_LINE_SMALL)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Direction Similarity')
    ax2.set_title('Cluster Direction Similarity (lower = more distinct)')
    ax2.axhline(y=0.2, color='g', linestyle='--', label='Threshold (0.2)')
    ax2.legend()
    ax2.grid(True, alpha=VIZ_ALPHA_LIGHT)
    
    # Cluster purity
    ax3 = axes[1, 0]
    ax3.plot(layers, purities, 'g-o', linewidth=VIZ_LINEWIDTH_NORMAL, markersize=VIZ_MARKERSIZE_LINE_SMALL)
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Cluster Purity')
    ax3.set_title('Alignment with True Sources (higher = better)')
    ax3.grid(True, alpha=VIZ_ALPHA_LIGHT)
    
    # Combined separability
    ax4 = axes[1, 1]
    ax4.plot(layers, sep_scores, 'm-o', linewidth=VIZ_LINEWIDTH_NORMAL, markersize=VIZ_MARKERSIZE_LINE_SMALL)
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Separability Score')
    ax4.set_title('Combined Score: (1 - dir_sim) * purity')
    best_layer = layers[np.argmax(sep_scores)]
    ax4.axvline(x=best_layer, color='orange', linestyle='--', label=f'Best: Layer {best_layer}')
    ax4.legend()
    ax4.grid(True, alpha=VIZ_ALPHA_LIGHT)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=VIZ_DPI, bbox_inches='tight')
        print(f"  Saved to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return best_layer


def attribute_pairs_to_concepts(
    diff_vectors: np.ndarray,
    pairs: List[Dict],
    detection_result: Dict,
) -> Dict:
    """
    Attribute each pair to its detected concept.
    
    Given the k-concept detection result, assign each original pair
    to one of the detected concepts. This allows you to:
    1. See which pairs belong to which concept
    2. Analyze the semantic content of each detected concept
    3. Validate that the detection aligns with true sources (if known)
    
    Args:
        diff_vectors: [N, hidden_dim] difference vectors
        pairs: Original list of contrastive pairs (with 'question', 'positive', 'negative')
        detection_result: Output from detect_k_concepts()
        
    Returns:
        Dictionary with:
        - concept_assignments: List[int] - concept ID for each pair
        - concepts: Dict mapping concept_id -> list of pair indices
        - concept_details: Dict with statistics and sample pairs for each concept
    """
    optimal_k = detection_result['optimal_k']
    k_result = detection_result['results_by_k'][optimal_k]
    concept_groups = k_result['concept_groups']  # List of sets of cluster IDs
    
    # Cluster all pairs
    km = KMeans(n_clusters=optimal_k, random_state=DEFAULT_RANDOM_SEED, n_init=LINEARITY_N_INIT)
    cluster_labels = km.fit_predict(diff_vectors)
    
    # Map cluster -> concept
    cluster_to_concept = {}
    for concept_id, cluster_set in enumerate(concept_groups):
        for cluster_id in cluster_set:
            cluster_to_concept[cluster_id] = concept_id
    
    # Assign each pair to a concept
    concept_assignments = [cluster_to_concept[c] for c in cluster_labels]
    
    # Group pairs by concept
    concepts = {i: [] for i in range(len(concept_groups))}
    for pair_idx, concept_id in enumerate(concept_assignments):
        concepts[concept_id].append(pair_idx)
    
    # Build detailed info for each concept
    concept_details = {}
    for concept_id, pair_indices in concepts.items():
        # Get sample pairs
        sample_indices = pair_indices[:5]  # First 5 as samples
        sample_pairs = [pairs[i] for i in sample_indices]
        
        # Compute concept direction
        concept_diffs = diff_vectors[pair_indices]
        concept_direction = concept_diffs.mean(axis=0)
        concept_direction = concept_direction / (np.linalg.norm(concept_direction) + ZERO_THRESHOLD)
        
        # Source distribution (if sources are available)
        sources_in_concept = [pairs[i].get('source', 'unknown') for i in pair_indices]
        from collections import Counter
        source_distribution = dict(Counter(sources_in_concept))
        
        # Clusters in this concept
        clusters_in_concept = list(concept_groups[concept_id])
        
        concept_details[concept_id] = {
            'num_pairs': len(pair_indices),
            'pair_indices': pair_indices,
            'sample_pairs': sample_pairs,
            'source_distribution': source_distribution,
            'clusters': clusters_in_concept,
            'direction_norm': float(np.linalg.norm(concept_diffs.mean(axis=0))),
        }
    
    return {
        'concept_assignments': concept_assignments,
        'concepts': concepts,
        'concept_details': concept_details,
        'num_concepts': len(concept_groups),
    }


def print_concept_attribution(attribution: Dict, show_samples: bool = True):
    """Print a summary of concept attributions."""
    print(f"\n{'=' * SEPARATOR_WIDTH_WIDE}")
    print(f"CONCEPT ATTRIBUTION RESULTS")
    print(f"{'=' * SEPARATOR_WIDTH_WIDE}")
    print(f"\nDetected {attribution['num_concepts']} distinct concepts\n")
    
    for concept_id, details in attribution['concept_details'].items():
        print(f"\n--- Concept {concept_id} ---")
        print(f"  Pairs: {details['num_pairs']}")
        print(f"  Clusters: {details['clusters']}")
        print(f"  Source distribution: {details['source_distribution']}")
        
        if show_samples and details['sample_pairs']:
            print(f"\n  Sample pairs:")
            for i, pair in enumerate(details['sample_pairs'][:3]):
                q = pair['question'][:DISPLAY_TRUNCATION_ERROR] + '...' if len(pair['question']) > DISPLAY_TRUNCATION_ERROR else pair['question']
                p = pair['positive'][:DISPLAY_TRUNCATION_SHORT] + '...' if len(pair['positive']) > DISPLAY_TRUNCATION_SHORT else pair['positive']
                n = pair['negative'][:DISPLAY_TRUNCATION_SHORT] + '...' if len(pair['negative']) > DISPLAY_TRUNCATION_SHORT else pair['negative']
                source = pair.get('source', 'unknown')
                print(f"    [{i+1}] Source: {source}")
                print(f"        Q: {q}")
                print(f"        +: {p}")
                print(f"        -: {n}")


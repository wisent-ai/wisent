"""Advanced visualization functions for mixed concept detection."""

from collections import Counter
from typing import Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from wisent.core import constants as _C


def visualize_k_concepts(
    diff_vectors: np.ndarray,
    sources: List[str],
    detection_result: Dict,
    title: str = "Multi-Concept Detection",
    output_path: Optional[str] = None,
    show_plot: bool = True,
):
    """
    Visualize the k-concept detection results.
    """
    optimal_k = detection_result['optimal_k']
    k_result = detection_result['results_by_k'][optimal_k]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"{title}\nDetected {detection_result['detected_concepts']} distinct concepts", 
                 fontsize=_C.VIZ_FONTSIZE_SUPTITLE, fontweight='bold')
    
    # PCA projection
    pca = PCA(n_components=_C.VIZ_N_COMPONENTS_2D)
    proj_2d = pca.fit_transform(diff_vectors)

    # Cluster with optimal k
    km = KMeans(n_clusters=optimal_k, random_state=_C.DEFAULT_RANDOM_SEED, n_init=_C.LINEARITY_N_INIT)
    cluster_labels = km.fit_predict(diff_vectors)
    
    # Color palettes
    source_colors = {'truthfulqa': '#e74c3c', 'hellaswag': '#3498db', 'mmlu': '#2ecc71', 
                     'gsm8k': '#f39c12', 'unknown': '#95a5a6'}
    cluster_cmap = plt.cm.Set2
    
    # === Plot 1: By True Source ===
    ax1 = axes[0, 0]
    unique_sources = list(set(sources))
    for source in unique_sources:
        mask = np.array([s == source for s in sources])
        color = source_colors.get(source, '#95a5a6')
        ax1.scatter(proj_2d[mask, 0], proj_2d[mask, 1], c=color, label=source, alpha=_C.VIZ_ALPHA_MEDIUM, s=_C.VIZ_MARKER_SIZE)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax1.set_title('Ground Truth Sources')
    ax1.legend()
    ax1.grid(True, alpha=_C.VIZ_ALPHA_LIGHT)
    
    # === Plot 2: By Cluster ===
    ax2 = axes[0, 1]
    for i in range(optimal_k):
        mask = cluster_labels == i
        color = cluster_cmap(i / optimal_k)
        ax2.scatter(proj_2d[mask, 0], proj_2d[mask, 1], c=[color], label=f'Cluster {i}', alpha=_C.VIZ_ALPHA_MEDIUM, s=_C.VIZ_MARKER_SIZE)
    ax2.set_xlabel(f'PC1')
    ax2.set_ylabel(f'PC2')
    ax2.set_title(f'K-Means Clustering (k={optimal_k})')
    ax2.legend()
    ax2.grid(True, alpha=_C.VIZ_ALPHA_LIGHT)
    
    # === Plot 3: By Detected Concept ===
    ax3 = axes[0, 2]
    concept_groups = k_result['concept_groups']
    concept_cmap = plt.cm.Set1
    
    for concept_id, cluster_group in enumerate(concept_groups):
        mask = np.isin(cluster_labels, list(cluster_group))
        color = concept_cmap(concept_id / len(concept_groups))
        ax3.scatter(proj_2d[mask, 0], proj_2d[mask, 1], c=[color],
                   label=f'Concept {concept_id} (clusters {cluster_group})', alpha=_C.VIZ_ALPHA_MEDIUM, s=_C.VIZ_MARKER_SIZE)
    ax3.set_xlabel(f'PC1')
    ax3.set_ylabel(f'PC2')
    ax3.set_title(f'Detected Concepts ({len(concept_groups)} found)')
    ax3.legend()
    ax3.grid(True, alpha=_C.VIZ_ALPHA_LIGHT)
    
    # === Plot 4: Pairwise Similarity Matrix ===
    ax4 = axes[1, 0]
    sim_matrix = np.eye(optimal_k)
    for pair_info in k_result['pairwise_similarities']:
        i, j = pair_info['clusters']
        sim_matrix[i, j] = pair_info['similarity']
        sim_matrix[j, i] = pair_info['similarity']
    
    im = ax4.imshow(sim_matrix, cmap='RdYlGn', vmin=_C.VIZ_HEATMAP_VMIN_ZERO, vmax=_C.VIZ_HEATMAP_VMAX_ONE)
    ax4.set_xticks(range(optimal_k))
    ax4.set_yticks(range(optimal_k))
    ax4.set_xticklabels([f'C{i}' for i in range(optimal_k)])
    ax4.set_yticklabels([f'C{i}' for i in range(optimal_k)])
    ax4.set_title('Cluster Direction Similarity\n(Red=Different, Green=Same)')
    plt.colorbar(im, ax=ax4)
    
    # Add text annotations
    for i in range(optimal_k):
        for j in range(optimal_k):
            ax4.text(j, i, f'{sim_matrix[i,j]:.2f}', ha='center', va='center', fontsize=_C.VIZ_FONTSIZE_ANNOTATION)
    
    # === Plot 5: Concepts vs K ===
    ax5 = axes[1, 1]
    ks = list(detection_result['results_by_k'].keys())
    concepts = [detection_result['results_by_k'][k]['num_distinct_concepts'] for k in ks]
    silhouettes = [detection_result['results_by_k'][k]['silhouette'] for k in ks]
    
    ax5.plot(ks, concepts, 'bo-', label='Distinct Concepts', linewidth=_C.VIZ_LINEWIDTH_NORMAL, markersize=_C.VIZ_MARKERSIZE_LINE)
    ax5.axhline(y=detection_result['detected_concepts'], color='r', linestyle='--', 
                label=f'Detected: {detection_result["detected_concepts"]}')
    ax5.axvline(x=optimal_k, color='g', linestyle='--', label=f'Optimal k={optimal_k}')
    ax5.set_xlabel('Number of Clusters (k)')
    ax5.set_ylabel('Distinct Concepts Found')
    ax5.set_title('Concepts Detected vs. k')
    ax5.legend()
    ax5.grid(True, alpha=_C.VIZ_ALPHA_LIGHT)
    ax5.set_xticks(ks)
    
    # === Plot 6: Cluster Composition ===
    ax6 = axes[1, 2]
    
    cluster_compositions = []
    for i in range(optimal_k):
        mask = cluster_labels == i
        cluster_sources = [sources[j] for j in range(len(sources)) if mask[j]]
        cluster_compositions.append(Counter(cluster_sources))
    
    x = np.arange(optimal_k)
    width = 0.8 / len(unique_sources)
    
    for idx, source in enumerate(unique_sources):
        counts = [cluster_compositions[i].get(source, 0) for i in range(optimal_k)]
        color = source_colors.get(source, '#95a5a6')
        ax6.bar(x + idx * width, counts, width, label=source, color=color)
    
    ax6.set_xticks(x + width * (len(unique_sources) - 1) / 2)
    ax6.set_xticklabels([f'Cluster {i}' for i in range(optimal_k)])
    ax6.set_ylabel('Count')
    ax6.set_title('Cluster Composition by Source')
    ax6.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=_C.VIZ_DPI, bbox_inches='tight')
        print(f"  Saved visualization to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_concept_detection(
    diff_vectors: np.ndarray,
    sources: List[str],
    title: str = "Concept Detection Visualization",
    output_path: Optional[str] = None,
    show_plot: bool = True,
):
    """
    Create a visualization showing:
    1. PCA projection colored by true source (if known)
    2. PCA projection colored by k=2 clustering
    3. The cluster directions as arrows
    4. Key metrics
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=_C.VIZ_FONTSIZE_SUPTITLE, fontweight='bold')
    
    # PCA projection
    pca = PCA(n_components=_C.VIZ_N_COMPONENTS_2D)
    proj_2d = pca.fit_transform(diff_vectors)

    # K-means clustering
    km = KMeans(n_clusters=_C.STABILITY_N_CLUSTERS, random_state=_C.DEFAULT_RANDOM_SEED, n_init=_C.LINEARITY_N_INIT)
    cluster_labels = km.fit_predict(diff_vectors)
    
    # Compute cluster directions
    cluster_0_mask = cluster_labels == 0
    cluster_1_mask = cluster_labels == 1
    
    dir_0 = diff_vectors[cluster_0_mask].mean(axis=0)
    dir_1 = diff_vectors[cluster_1_mask].mean(axis=0)
    dir_0_norm = dir_0 / (np.linalg.norm(dir_0) + _C.ZERO_THRESHOLD)
    dir_1_norm = dir_1 / (np.linalg.norm(dir_1) + _C.ZERO_THRESHOLD)
    cluster_sim = np.abs(np.dot(dir_0_norm, dir_1_norm))
    
    # Project directions to 2D for visualization
    dir_0_2d = pca.transform(dir_0.reshape(1, -1))[0]
    dir_1_2d = pca.transform(dir_1.reshape(1, -1))[0]
    
    # Get unique sources for coloring
    unique_sources = list(set(sources))
    source_colors = {'truthfulqa': '#e74c3c', 'hellaswag': '#3498db', 'unknown': '#95a5a6'}
    cluster_colors = {0: '#2ecc71', 1: '#9b59b6'}
    
    # === Plot 1: Colored by TRUE SOURCE ===
    ax1 = axes[0]
    for source in unique_sources:
        mask = np.array([s == source for s in sources])
        color = source_colors.get(source, '#95a5a6')
        ax1.scatter(proj_2d[mask, 0], proj_2d[mask, 1], 
                   c=color, label=source, alpha=_C.VIZ_ALPHA_MEDIUM, s=_C.VIZ_MARKER_SIZE, edgecolors='white', linewidth=_C.VIZ_LINEWIDTH_FINE)
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax1.set_title('Colored by True Source\n(Ground Truth)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=_C.VIZ_ALPHA_LIGHT)
    
    # === Plot 2: Colored by CLUSTER ===
    ax2 = axes[1]
    for cluster_id in [0, 1]:
        mask = cluster_labels == cluster_id
        color = cluster_colors[cluster_id]
        ax2.scatter(proj_2d[mask, 0], proj_2d[mask, 1],
                   c=color, label=f'Cluster {cluster_id}', alpha=_C.VIZ_ALPHA_MEDIUM, s=_C.VIZ_MARKER_SIZE, edgecolors='white', linewidth=_C.VIZ_LINEWIDTH_FINE)
    
    # Draw cluster direction arrows
    center = proj_2d.mean(axis=0)
    arrow_scale = 2.0
    ax2.annotate('', xy=center + dir_0_2d * arrow_scale, xytext=center,
                arrowprops=dict(arrowstyle='->', color=cluster_colors[0], lw=3))
    ax2.annotate('', xy=center + dir_1_2d * arrow_scale, xytext=center,
                arrowprops=dict(arrowstyle='->', color=cluster_colors[1], lw=3))
    
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax2.set_title(f'Colored by K-Means Cluster\nDirection Similarity: {cluster_sim:.3f}')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=_C.VIZ_ALPHA_LIGHT)
    
    # === Plot 3: Cluster Composition ===
    ax3 = axes[2]
    
    # Count sources per cluster
    cluster_0_sources = [sources[i] for i in range(len(sources)) if cluster_labels[i] == 0]
    cluster_1_sources = [sources[i] for i in range(len(sources)) if cluster_labels[i] == 1]
    
    c0_counts = Counter(cluster_0_sources)
    c1_counts = Counter(cluster_1_sources)
    
    # Create stacked bar chart
    x = np.arange(2)
    width = 0.6
    
    bottom_0 = 0
    bottom_1 = 0
    
    for source in unique_sources:
        color = source_colors.get(source, '#95a5a6')
        heights = [c0_counts.get(source, 0), c1_counts.get(source, 0)]
        ax3.bar(x, heights, width, bottom=[bottom_0, bottom_1], label=source, color=color, edgecolor='white')
        bottom_0 += heights[0]
        bottom_1 += heights[1]
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Cluster 0', 'Cluster 1'])
    ax3.set_ylabel('Count')
    ax3.set_title('Cluster Composition\n(How well clusters separate sources)')
    ax3.legend(loc='upper right')
    
    # Add purity annotation
    total_0 = len(cluster_0_sources)
    total_1 = len(cluster_1_sources)
    if total_0 > 0 and total_1 > 0:
        max_0 = max(c0_counts.values()) if c0_counts else 0
        max_1 = max(c1_counts.values()) if c1_counts else 0
        purity_0 = max_0 / total_0
        purity_1 = max_1 / total_1
        avg_purity = (purity_0 + purity_1) / 2
        ax3.text(0.5, -0.15, f'Cluster Purity: {avg_purity:.1%}', 
                transform=ax3.transAxes, ha='center', fontsize=_C.VIZ_FONTSIZE_TITLE, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=_C.VIZ_DPI, bbox_inches='tight')
        print(f"  Saved visualization to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return {
        "cluster_similarity": cluster_sim,
        "cluster_0_composition": dict(c0_counts),
        "cluster_1_composition": dict(c1_counts),
    }


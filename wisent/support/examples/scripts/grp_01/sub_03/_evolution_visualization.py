"""Concept evolution visualization - dimensionality reduction and direction plots."""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple

from wisent.core.constants import (
    DEFAULT_RANDOM_SEED, VIZ_N_NEIGHBORS, VIZ_MIN_DIST,
    VIZ_PERPLEXITY, TSNE_N_ITER, VIZ_DPI, VIZ_N_COMPONENTS_2D,
    VIZ_FONTSIZE_BODY, VIZ_FONTSIZE_TITLE, VIZ_FONTSIZE_SUBTITLE,
    VIZ_ALPHA_LIGHT, VIZ_ALPHA_MEDIUM, VIZ_ALPHA_HALF,
    VIZ_MARKER_SIZE_MOVEMENT, VIZ_LINEWIDTH_NORMAL, VIZ_FONTSIZE_TINY,
)
from wisent.examples.scripts._pair_generators_neutral import ConceptMetrics
from wisent.examples.scripts._evolution_visualization_plots import (
    visualize_concept_evolution_plots,
)


def visualize_concept_evolution(
    activations_by_concept: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    concept_metrics: Dict[str, Dict[int, ConceptMetrics]],
    direction_comparisons: Dict[int, Dict[str, float]],
    layers: List[int],
    output_dir: str,
    model_name: str,
):
    """Create visualizations of concept evolution."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Pick a representative layer (middle layer with good signal)
    mid_layer = layers[len(layers) // 2]
    
    colors_pos = {'hitler': 'red', 'fascism': 'orange', 'harmful_ideology': 'purple', 'neutral_baseline': 'green'}
    colors_neg = {'hitler': 'lightcoral', 'fascism': 'moccasin', 'harmful_ideology': 'plum', 'neutral_baseline': 'lightgreen'}
    
    # Collect all activations for dimensionality reduction
    all_pos = []
    all_neg = []
    concept_order = ['hitler', 'fascism', 'harmful_ideology', 'neutral_baseline']
    
    for concept in concept_order:
        if concept not in activations_by_concept:
            continue
        if mid_layer not in activations_by_concept[concept]:
            continue
        pos, neg = activations_by_concept[concept][mid_layer]
        all_pos.append(pos.cpu().numpy())
        all_neg.append(neg.cpu().numpy())
    
    if not all_pos:
        print("No activations to visualize")
        return
    
    all_pos_np = np.vstack(all_pos)
    all_neg_np = np.vstack(all_neg)
    all_data = np.vstack([all_pos_np, all_neg_np])
    n_pos_total = len(all_pos_np)
    
    # Create labels for coloring
    pos_labels = []
    neg_labels = []
    for concept in concept_order:
        if concept not in activations_by_concept or mid_layer not in activations_by_concept[concept]:
            continue
        n = len(activations_by_concept[concept][mid_layer][0])
        pos_labels.extend([concept] * n)
        neg_labels.extend([concept] * n)
    
    # =========================================================================
    # COMPARISON: PCA vs UMAP vs PaCMAP
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # --- PCA ---
    pca = PCA(n_components=VIZ_N_COMPONENTS_2D)
    pca_2d = pca.fit_transform(all_data)
    pca_pos = pca_2d[:n_pos_total]
    pca_neg = pca_2d[n_pos_total:]
    
    ax = axes[0]
    idx = 0
    for concept in concept_order:
        if concept not in activations_by_concept or mid_layer not in activations_by_concept[concept]:
            continue
        n = len(activations_by_concept[concept][mid_layer][0])
        ax.scatter(pca_pos[idx:idx+n, 0], pca_pos[idx:idx+n, 1], c=colors_pos[concept], label=f'{concept} +', alpha=VIZ_ALPHA_MEDIUM, s=VIZ_MARKER_SIZE_MOVEMENT, marker='o')
        ax.scatter(pca_neg[idx:idx+n, 0], pca_neg[idx:idx+n, 1], c=colors_neg[concept], label=f'{concept} -', alpha=VIZ_ALPHA_MEDIUM, s=VIZ_MARKER_SIZE_MOVEMENT, marker='x')
        idx += n
    ax.set_title(f'PCA (Linear)\nVar explained: {pca.explained_variance_ratio_.sum()*100:.1f}%')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=VIZ_FONTSIZE_TINY, loc='upper right')
    ax.grid(True, alpha=VIZ_ALPHA_LIGHT)
    
    # --- UMAP ---
    try:
        import umap
        reducer_umap = umap.UMAP(n_components=VIZ_N_COMPONENTS_2D, n_neighbors=VIZ_N_NEIGHBORS, min_dist=VIZ_MIN_DIST, random_state=DEFAULT_RANDOM_SEED)
        umap_2d = reducer_umap.fit_transform(all_data)
        umap_pos = umap_2d[:n_pos_total]
        umap_neg = umap_2d[n_pos_total:]
        
        ax = axes[1]
        idx = 0
        for concept in concept_order:
            if concept not in activations_by_concept or mid_layer not in activations_by_concept[concept]:
                continue
            n = len(activations_by_concept[concept][mid_layer][0])
            ax.scatter(umap_pos[idx:idx+n, 0], umap_pos[idx:idx+n, 1], c=colors_pos[concept], label=f'{concept} +', alpha=VIZ_ALPHA_MEDIUM, s=VIZ_MARKER_SIZE_MOVEMENT, marker='o')
            ax.scatter(umap_neg[idx:idx+n, 0], umap_neg[idx:idx+n, 1], c=colors_neg[concept], label=f'{concept} -', alpha=VIZ_ALPHA_MEDIUM, s=VIZ_MARKER_SIZE_MOVEMENT, marker='x')
            idx += n
        ax.set_title('UMAP (Nonlinear)\nPreserves local structure')
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.legend(fontsize=VIZ_FONTSIZE_TINY, loc='upper right')
        ax.grid(True, alpha=VIZ_ALPHA_LIGHT)
    except (ImportError, Exception) as e:
        axes[1].text(0.5, 0.5, f'UMAP unavailable\n{type(e).__name__}', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('UMAP (error)')
    
    # --- t-SNE as alternative to PaCMAP (which segfaults) ---
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=VIZ_N_COMPONENTS_2D, perplexity=VIZ_PERPLEXITY, random_state=DEFAULT_RANDOM_SEED, n_iter=TSNE_N_ITER)
        tsne_2d = tsne.fit_transform(all_data)
        tsne_pos = tsne_2d[:n_pos_total]
        tsne_neg = tsne_2d[n_pos_total:]
        
        ax = axes[2]
        idx = 0
        for concept in concept_order:
            if concept not in activations_by_concept or mid_layer not in activations_by_concept[concept]:
                continue
            n = len(activations_by_concept[concept][mid_layer][0])
            ax.scatter(tsne_pos[idx:idx+n, 0], tsne_pos[idx:idx+n, 1], c=colors_pos[concept], label=f'{concept} +', alpha=VIZ_ALPHA_MEDIUM, s=VIZ_MARKER_SIZE_MOVEMENT, marker='o')
            ax.scatter(tsne_neg[idx:idx+n, 0], tsne_neg[idx:idx+n, 1], c=colors_neg[concept], label=f'{concept} -', alpha=VIZ_ALPHA_MEDIUM, s=VIZ_MARKER_SIZE_MOVEMENT, marker='x')
            idx += n
        ax.set_title('t-SNE (Nonlinear)\nPreserves local structure')
        ax.set_xlabel('t-SNE1')
        ax.set_ylabel('t-SNE2')
        ax.legend(fontsize=VIZ_FONTSIZE_TINY, loc='upper right')
        ax.grid(True, alpha=VIZ_ALPHA_LIGHT)
    except (ImportError, Exception) as e:
        axes[2].text(0.5, 0.5, f't-SNE unavailable\n{type(e).__name__}', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('t-SNE (error)')
    
    plt.suptitle(f'Dimensionality Reduction Comparison - Layer {mid_layer} - {model_name}\n(o) = positive/harmful, (x) = negative/safe', fontsize=VIZ_FONTSIZE_SUBTITLE)
    plt.tight_layout()
    plt.savefig(output_path / 'concept_dimred_comparison.png', dpi=VIZ_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'concept_dimred_comparison.png'}")
    
    # =========================================================================
    # Direction vectors in each space
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    directions = {}
    for concept in concept_order:
        if concept not in activations_by_concept or mid_layer not in activations_by_concept[concept]:
            continue
        pos, neg = activations_by_concept[concept][mid_layer]
        direction = (pos.mean(dim=0) - neg.mean(dim=0)).cpu().numpy()
        directions[concept] = direction
    
    if len(directions) >= 2:
        dir_matrix = np.stack(list(directions.values()))
        
        # PCA on directions
        pca_dir = PCA(n_components=VIZ_N_COMPONENTS_2D)
        dir_pca = pca_dir.fit_transform(dir_matrix)
        
        ax = axes[0]
        for i, concept in enumerate(directions.keys()):
            color = colors_pos.get(concept, 'gray')
            ax.arrow(0, 0, dir_pca[i, 0], dir_pca[i, 1], head_width=0.5, head_length=0.3, fc=color, ec=color, linewidth=VIZ_LINEWIDTH_NORMAL)
            ax.annotate(concept.replace('_', '\n'), (dir_pca[i, 0]*1.15, dir_pca[i, 1]*1.15), fontsize=VIZ_FONTSIZE_BODY, ha='center', color=color, fontweight='bold')
        max_val = np.abs(dir_pca).max() * 1.4
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=VIZ_ALPHA_HALF)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=VIZ_ALPHA_HALF)
        ax.set_aspect('equal')
        ax.set_title(f'PCA Direction Vectors\nVar: {pca_dir.explained_variance_ratio_.sum()*100:.1f}%')
        ax.grid(True, alpha=VIZ_ALPHA_LIGHT)
        
        # UMAP on directions
        try:
            import umap
            if len(dir_matrix) >= 4:
                reducer = umap.UMAP(n_components=VIZ_N_COMPONENTS_2D, n_neighbors=min(3, len(dir_matrix)-1), min_dist=VIZ_MIN_DIST, random_state=DEFAULT_RANDOM_SEED)
                dir_umap = reducer.fit_transform(dir_matrix)
                
                ax = axes[1]
                for i, concept in enumerate(directions.keys()):
                    color = colors_pos.get(concept, 'gray')
                    ax.arrow(0, 0, dir_umap[i, 0], dir_umap[i, 1], head_width=0.3, head_length=0.2, fc=color, ec=color, linewidth=VIZ_LINEWIDTH_NORMAL)
                    ax.annotate(concept.replace('_', '\n'), (dir_umap[i, 0]*1.15, dir_umap[i, 1]*1.15), fontsize=VIZ_FONTSIZE_BODY, ha='center', color=color, fontweight='bold')
                max_val = np.abs(dir_umap).max() * 1.4
                ax.set_xlim(-max_val, max_val)
                ax.set_ylim(-max_val, max_val)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=VIZ_ALPHA_HALF)
                ax.axvline(x=0, color='gray', linestyle='--', alpha=VIZ_ALPHA_HALF)
                ax.set_aspect('equal')
                ax.set_title('UMAP Direction Vectors')
                ax.grid(True, alpha=VIZ_ALPHA_LIGHT)
            else:
                axes[1].text(0.5, 0.5, 'Need >= 4 concepts for UMAP', ha='center', va='center', transform=axes[1].transAxes)
        except ImportError:
            axes[1].text(0.5, 0.5, 'UMAP not installed', ha='center', va='center', transform=axes[1].transAxes)
        
        # t-SNE on directions (replacing PaCMAP which segfaults)
        # Note: t-SNE doesn't preserve global structure well for 4 points, so just show message
        axes[2].text(0.5, 0.5, 'Only 4 direction vectors\n(t-SNE not meaningful)', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('t-SNE Direction Vectors\n(N/A for 4 points)')
    
    plt.suptitle(f'Direction Vectors Comparison - Layer {mid_layer} - {model_name}', fontsize=VIZ_FONTSIZE_SUBTITLE)
    plt.tight_layout()
    plt.savefig(output_path / 'concept_directions_comparison.png', dpi=VIZ_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'concept_directions_comparison.png'}")

    # Continue with additional plots
    visualize_concept_evolution_plots(
        activations_by_concept=activations_by_concept,
        concept_metrics=concept_metrics,
        direction_comparisons=direction_comparisons,
        layers=layers,
        output_dir=output_dir,
        model_name=model_name,
        mid_layer=mid_layer,
    )

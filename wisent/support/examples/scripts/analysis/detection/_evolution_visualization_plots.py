"""Concept evolution visualization - PCA, heatmaps, similarity, and combined plots."""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from wisent.core.utils.config_tools.constants import (
    VIZ_DPI, VIZ_MARKER_SIZE, VIZ_FONTSIZE_SMALL, VIZ_FONTSIZE_SUBTITLE,
    VIZ_FONTSIZE_SUPTITLE, VIZ_ALPHA_LIGHT, VIZ_ALPHA_MEDIUM,
    VIZ_ALPHA_HIGH, VIZ_ALPHA_HALF,
    VIZ_LINEWIDTH_NORMAL, VIZ_LINEWIDTH_FINE, VIZ_MARKERSIZE_LINE,
    VIZ_MARKER_SIZE_MOVEMENT, VIZ_CORRELATION_VMIN, VIZ_CORRELATION_VMAX,
)
from wisent.examples.scripts._pair_generators_neutral import ConceptMetrics


def visualize_concept_evolution_plots(
    activations_by_concept: Dict[str, Dict[int, Tuple]],
    concept_metrics: Dict[str, Dict[int, ConceptMetrics]],
    direction_comparisons: Dict[int, Dict[str, float]],
    layers: List[int],
    output_dir: str,
    model_name: str,
    mid_layer: int,
):
    """Create PCA, heatmap, similarity evolution, and combined visualizations."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. PCA visualization of all concepts at one layer
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Collect all activations for PCA
    all_pos = []
    all_neg = []
    labels = []
    colors_pos = {'hitler': 'red', 'fascism': 'orange', 'harmful_ideology': 'purple', 'neutral_baseline': 'green'}
    colors_neg = {'hitler': 'lightcoral', 'fascism': 'moccasin', 'harmful_ideology': 'plum', 'neutral_baseline': 'lightgreen'}
    
    for concept in ['hitler', 'fascism', 'harmful_ideology', 'neutral_baseline']:
        if concept not in activations_by_concept:
            continue
        if mid_layer not in activations_by_concept[concept]:
            continue
        pos, neg = activations_by_concept[concept][mid_layer]
        all_pos.append(pos.cpu().numpy())
        all_neg.append(neg.cpu().numpy())
        labels.extend([concept] * len(pos))
    
    if all_pos:
        all_pos_np = np.vstack(all_pos)
        all_neg_np = np.vstack(all_neg)
        all_data = np.vstack([all_pos_np, all_neg_np])
        
        pca = PCA(n_components=2)
        all_2d = pca.fit_transform(all_data)
        
        n_pos = len(all_pos_np)
        pos_2d = all_2d[:n_pos]
        neg_2d = all_2d[n_pos:]
        
        # Plot positive (harmful) responses
        ax = axes[0]
        idx = 0
        for concept in ['hitler', 'fascism', 'harmful_ideology', 'neutral_baseline']:
            if concept not in activations_by_concept or mid_layer not in activations_by_concept[concept]:
                continue
            n = len(activations_by_concept[concept][mid_layer][0])
            ax.scatter(pos_2d[idx:idx+n, 0], pos_2d[idx:idx+n, 1],
                      c=colors_pos[concept], label=f'{concept} (pos)', alpha=VIZ_ALPHA_HIGH, s=VIZ_MARKER_SIZE)
            idx += n
        ax.set_title(f'Positive Responses (Layer {mid_layer})')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.legend()
        ax.grid(True, alpha=VIZ_ALPHA_LIGHT)

        # Plot negative (safe) responses
        ax = axes[1]
        idx = 0
        for concept in ['hitler', 'fascism', 'harmful_ideology', 'neutral_baseline']:
            if concept not in activations_by_concept or mid_layer not in activations_by_concept[concept]:
                continue
            n = len(activations_by_concept[concept][mid_layer][1])
            ax.scatter(neg_2d[idx:idx+n, 0], neg_2d[idx:idx+n, 1],
                      c=colors_neg[concept], label=f'{concept} (neg)', alpha=VIZ_ALPHA_HIGH, s=VIZ_MARKER_SIZE)
            idx += n
        ax.set_title(f'Negative Responses (Layer {mid_layer})')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.legend()
        ax.grid(True, alpha=VIZ_ALPHA_LIGHT)
    
    plt.suptitle(f'Concept Activations - {model_name}', fontsize=VIZ_FONTSIZE_SUPTITLE)
    plt.tight_layout()
    plt.savefig(output_path / 'concept_pca_activations.png', dpi=VIZ_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'concept_pca_activations.png'}")
    
    # 2. Direction vectors visualization - project onto shared PCA space
    fig, ax = plt.subplots(figsize=(10, 8))
    
    directions = {}
    for concept in ['hitler', 'fascism', 'harmful_ideology', 'neutral_baseline']:
        if concept not in activations_by_concept or mid_layer not in activations_by_concept[concept]:
            continue
        pos, neg = activations_by_concept[concept][mid_layer]
        direction = (pos.mean(dim=0) - neg.mean(dim=0)).cpu().numpy()
        directions[concept] = direction
    
    if len(directions) >= 2:
        dir_matrix = np.stack(list(directions.values()))
        pca_dir = PCA(n_components=2)
        dir_2d = pca_dir.fit_transform(dir_matrix)
        
        # Plot directions as arrows from origin
        for i, (concept, _) in enumerate(directions.items()):
            color = colors_pos.get(concept, 'gray')
            ax.arrow(0, 0, dir_2d[i, 0], dir_2d[i, 1],
                    head_width=0.05, head_length=0.03, fc=color, ec=color, linewidth=VIZ_LINEWIDTH_NORMAL)
            ax.annotate(concept.replace('_', '\n'), (dir_2d[i, 0]*1.1, dir_2d[i, 1]*1.1),
                       fontsize=VIZ_FONTSIZE_TITLE, ha='center', color=color, fontweight='bold')
        
        # Set equal aspect and limits
        max_val = np.abs(dir_2d).max() * 1.3
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=VIZ_ALPHA_HALF)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=VIZ_ALPHA_HALF)
        ax.set_aspect('equal')
        ax.set_title(f'Concept Direction Vectors (Layer {mid_layer})\nArrows show pos-neg direction for each concept')
        ax.set_xlabel(f'PC1 ({pca_dir.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca_dir.explained_variance_ratio_[1]*100:.1f}%)')
        ax.grid(True, alpha=VIZ_ALPHA_LIGHT)

    plt.tight_layout()
    plt.savefig(output_path / 'concept_direction_vectors.png', dpi=VIZ_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'concept_direction_vectors.png'}")
    
    # 3. Cosine similarity heatmap across layers
    concepts = ['hitler', 'fascism', 'harmful_ideology', 'neutral_baseline']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, layer in enumerate(layers):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        # Build similarity matrix
        sim_matrix = np.zeros((len(concepts), len(concepts)))
        
        for i, c1 in enumerate(concepts):
            for j, c2 in enumerate(concepts):
                if i == j:
                    sim_matrix[i, j] = 1.0
                elif i < j:
                    key = f"{c1}_vs_{c2}"
                    if layer in direction_comparisons and key in direction_comparisons[layer]:
                        sim_matrix[i, j] = direction_comparisons[layer][key]
                        sim_matrix[j, i] = direction_comparisons[layer][key]
        
        im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=VIZ_CORRELATION_VMIN, vmax=VIZ_CORRELATION_VMAX)
        ax.set_xticks(range(len(concepts)))
        ax.set_yticks(range(len(concepts)))
        ax.set_xticklabels([c.replace('_', '\n') for c in concepts], fontsize=VIZ_FONTSIZE_SMALL)
        ax.set_yticklabels([c.replace('_', '\n') for c in concepts], fontsize=VIZ_FONTSIZE_SMALL)
        ax.set_title(f'Layer {layer}')
        
        # Add text annotations
        for i in range(len(concepts)):
            for j in range(len(concepts)):
                text = f'{sim_matrix[i, j]:.2f}'
                color = 'white' if abs(sim_matrix[i, j]) > 0.5 else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=VIZ_FONTSIZE_SMALL)
    
    plt.suptitle(f'Direction Cosine Similarities Across Layers - {model_name}', fontsize=VIZ_FONTSIZE_SUPTITLE)
    fig.colorbar(im, ax=axes, shrink=0.6, label='Cosine Similarity')
    plt.tight_layout()
    plt.savefig(output_path / 'concept_similarity_heatmaps.png', dpi=VIZ_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'concept_similarity_heatmaps.png'}")
    
    # 4. Layer-wise similarity evolution plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pairs_to_plot = [
        ('hitler_vs_fascism', 'Hitler <-> Fascism', 'red'),
        ('fascism_vs_harmful_ideology', 'Fascism <-> Harmful', 'orange'),
        ('hitler_vs_harmful_ideology', 'Hitler <-> Harmful', 'purple'),
        ('hitler_vs_neutral_baseline', 'Hitler <-> Neutral', 'gray'),
    ]
    
    for key, label, color in pairs_to_plot:
        sims = []
        valid_layers = []
        for layer in layers:
            if layer in direction_comparisons and key in direction_comparisons[layer]:
                sims.append(direction_comparisons[layer][key])
                valid_layers.append(layer)
        if sims:
            ax.plot(valid_layers, sims, 'o-', label=label, color=color, linewidth=VIZ_LINEWIDTH_NORMAL, markersize=VIZ_MARKERSIZE_LINE)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=VIZ_ALPHA_LIGHT)
    ax.set_xlabel('Layer', fontsize=VIZ_FONTSIZE_SUBTITLE)
    ax.set_ylabel('Cosine Similarity', fontsize=VIZ_FONTSIZE_SUBTITLE)
    ax.set_title(f'Concept Direction Similarity Across Layers - {model_name}', fontsize=VIZ_FONTSIZE_SUPTITLE)
    ax.legend(loc='best')
    ax.grid(True, alpha=VIZ_ALPHA_LIGHT)
    ax.set_ylim(-0.6, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_path / 'concept_similarity_evolution.png', dpi=VIZ_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'concept_similarity_evolution.png'}")
    
    # 5. Combined pos/neg with directions overlay
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if all_pos:
        # Re-use PCA from earlier
        all_pos_np = np.vstack(all_pos)
        all_neg_np = np.vstack(all_neg)
        
        # Plot all points
        idx = 0
        for concept in ['hitler', 'fascism', 'harmful_ideology', 'neutral_baseline']:
            if concept not in activations_by_concept or mid_layer not in activations_by_concept[concept]:
                continue
            n = len(activations_by_concept[concept][mid_layer][0])
            
            # Positive (filled)
            ax.scatter(pos_2d[idx:idx+n, 0], pos_2d[idx:idx+n, 1],
                      c=colors_pos[concept], label=f'{concept} +', alpha=VIZ_ALPHA_MEDIUM, s=VIZ_MARKER_SIZE_MOVEMENT, marker='o')
            # Negative (hollow)
            ax.scatter(neg_2d[idx:idx+n, 0], neg_2d[idx:idx+n, 1],
                      c=colors_neg[concept], label=f'{concept} -', alpha=VIZ_ALPHA_MEDIUM, s=VIZ_MARKER_SIZE_MOVEMENT, marker='x')
            
            # Draw arrow from neg centroid to pos centroid
            pos_centroid = pos_2d[idx:idx+n].mean(axis=0)
            neg_centroid = neg_2d[idx:idx+n].mean(axis=0)
            ax.annotate('', xy=pos_centroid, xytext=neg_centroid,
                       arrowprops=dict(arrowstyle='->', color=colors_pos[concept], lw=3))
            
            idx += n
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title(f'All Concepts: Positive (o) vs Negative (x) with Direction Arrows\nLayer {mid_layer} - {model_name}')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=VIZ_ALPHA_LIGHT)

    plt.tight_layout()
    plt.savefig(output_path / 'concept_combined_visualization.png', dpi=VIZ_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'concept_combined_visualization.png'}")
    
    print(f"\nAll visualizations saved to: {output_path}")


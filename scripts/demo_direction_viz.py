#!/usr/bin/env python3
"""Visualization of steering directions for multiple concepts using REAL data."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from wisent.core.geometry.database_loaders import load_activations_from_database
from wisent.core.geometry.decomposition_metrics import find_optimal_clustering
from wisent.core.geometry.direction_visualization import (
    compute_direction_angles,
    compute_per_concept_directions,
    plot_directions_in_pca_space,
)


def main():
    print("Loading REAL activations from database...")

    # Load real activations
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    task_name = "truthfulqa_custom"
    layer = 8  # Middle layer

    pos, neg = load_activations_from_database(
        model_name=model_name,
        task_name=task_name,
        layer=layer,
    )

    print(f"Loaded {len(pos)} pairs, hidden_dim={pos.shape[1]}")

    # Cluster into concepts
    print("\nClustering into concepts...")
    diff = pos - neg
    n_concepts, labels, silhouette = find_optimal_clustering(diff)
    print(f"Found {n_concepts} concepts (silhouette={silhouette:.3f})")

    # Get per-concept directions
    directions, names = compute_per_concept_directions(pos, neg, labels, n_concepts)
    print(f"Computed {len(directions)} direction vectors")

    if len(directions) < 2:
        print("Need at least 2 concepts to visualize relationships")
        return

    # Compute angles
    angles = compute_direction_angles(directions, names)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # === Plot 1: Cosine Similarity Heatmap ===
    ax1 = axes[0]
    matrix = np.array(angles["cosine_matrix"])
    n = len(names)

    im = ax1.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    short_names = [f"C{i}" for i in range(n)]
    ax1.set_xticklabels(short_names)
    ax1.set_yticklabels(short_names)

    # Add text annotations
    for i in range(n):
        for j in range(n):
            color = 'white' if abs(matrix[i, j]) > 0.5 else 'black'
            ax1.text(j, i, f"{matrix[i, j]:.2f}", ha='center', va='center', color=color, fontsize=10)

    ax1.set_title(f"Cosine Similarity Between Concept Directions\n(mean={angles['mean_cosine']:.2f})")
    plt.colorbar(im, ax=ax1, label='Cosine Similarity')

    # === Plot 2: Directions as Arrows in PCA Space ===
    ax2 = axes[1]
    pca_data = plot_directions_in_pca_space(directions, pos, neg, names)

    # Plot activation point clouds (faded)
    if pca_data["pos_projected"] is not None:
        pos_proj = np.array(pca_data["pos_projected"])
        neg_proj = np.array(pca_data["neg_projected"])
        ax2.scatter(pos_proj[:, 0], pos_proj[:, 1], c='blue', alpha=0.1, s=10, label='pos')
        ax2.scatter(neg_proj[:, 0], neg_proj[:, 1], c='red', alpha=0.1, s=10, label='neg')

    # Plot direction arrows from origin
    colors = plt.cm.tab10(np.linspace(0, 1, len(directions)))
    dirs_proj = np.array(pca_data["directions_projected"])

    for i, (name, proj, color) in enumerate(zip(names, dirs_proj, colors)):
        scale = 2.0
        ax2.annotate('', xy=(proj[0]*scale, proj[1]*scale), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color=color, lw=2))
        ax2.text(proj[0]*scale*1.1, proj[1]*scale*1.1, f"C{i}", fontsize=10, color=color, fontweight='bold')

    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax2.set_xlabel(f"PC1 ({pca_data['explained_variance'][0]:.1%} var)")
    ax2.set_ylabel(f"PC2 ({pca_data['explained_variance'][1]:.1%} var)")
    ax2.set_title("Steering Directions in PCA Space")
    ax2.set_aspect('equal')

    legend_text = "\n".join([f"C{i}: {n.split('(')[1].rstrip(')')}" for i, n in enumerate(names)])
    ax2.text(0.02, 0.98, legend_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save figure to current directory
    output_path = os.path.join(os.path.dirname(__file__), "direction_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    # Print angle summary
    print("\n=== Pairwise Angles ===")
    for i in range(n):
        for j in range(i+1, n):
            angle = angles["angle_matrix_degrees"][i][j]
            cos = angles["cosine_matrix"][i][j]
            print(f"  C{i} <-> C{j}: {angle:.1f} deg (cos={cos:.3f})")

    print(f"\nMean angle: {angles['mean_angle_degrees']:.1f} deg")
    print(f"Orthogonal pairs: {angles['orthogonal_pairs']}")
    print(f"Aligned pairs: {angles['aligned_pairs']}")


if __name__ == "__main__":
    main()

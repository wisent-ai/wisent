"""
3D PCA Analysis for SST2 Activations

Performs 3D PCA on activations from each layer (1-28) using continuation_token aggregation.
Creates a 4x7 grid of 3D PCA plots showing the separation between positive and negative samples.

Uses 100 questions from SST2 benchmark.
For each layer:
  - Extract activations for positive and negative responses (200 samples total)
  - Perform PCA to reduce to 3D
  - Plot positive (red) vs negative (blue) samples in 3D space
"""

from __future__ import annotations

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
import gc

from activation_matrix import load_pairs
from wisent_guard.core.models.wisent_model import WisentModel
from wisent_guard.core.activations.core.activations_collector import ActivationCollector
from wisent_guard.core.activations.core.atoms import ActivationAggregationStrategy


def extract_activations(
    model_name: str,
    num_questions: int,
    aggregation: ActivationAggregationStrategy,
):
    """
    Extract activations for all layers.

    Args:
        model_name: HuggingFace model name
        num_questions: Number of questions to use
        aggregation: Aggregation strategy

    Returns:
        Dict containing activations for all layers
    """
    print(f"Extracting activations from {num_questions} SST2 questions")
    print("=" * 80)

    # Load contrastive pairs
    print(f"\nLoading {num_questions} contrastive pairs...")
    pairs = load_pairs(limit=num_questions, preferred_doc="training")

    if len(pairs) < num_questions:
        print(f"Warning: Only got {len(pairs)} pairs, requested {num_questions}")
        num_questions = len(pairs)

    print(f"✓ Loaded {num_questions} pairs ({num_questions * 2} samples)")

    # Load model
    print(f"\nLoading model: {model_name}")
    model = WisentModel(model_name=model_name, layers={})
    num_layers = model.num_layers
    print(f"Model loaded: {num_layers} layers, hidden_dim={model.hidden_size}")

    # Initialize collector
    collector = ActivationCollector(model=model, store_device="cpu", dtype=torch.float32)
    layer_names = [str(i) for i in range(1, num_layers + 1)]

    # Initialize storage: data[layer] = {positive: [...], negative: [...]}
    data = {str(i): {"positive": [], "negative": []} for i in range(1, num_layers + 1)}

    # Extract activations
    print(f"\nExtracting activations for all {num_layers} layers...")

    for pair_idx, pair in enumerate(pairs):
        if (pair_idx + 1) % 20 == 0 or pair_idx == 0:
            print(f"  Processing question {pair_idx + 1}/{num_questions}...")

        try:
            updated_pair = collector.collect_for_pair(
                pair=pair,
                layers=layer_names,
                aggregation=aggregation,
                return_full_sequence=False,
            )

            pos_acts = updated_pair.positive_response.layers_activations
            neg_acts = updated_pair.negative_response.layers_activations

            for layer_idx in range(1, num_layers + 1):
                layer_name = str(layer_idx)

                if layer_name in pos_acts and pos_acts[layer_name] is not None:
                    data[layer_name]["positive"].append(pos_acts[layer_name].cpu())

                if layer_name in neg_acts and neg_acts[layer_name] is not None:
                    data[layer_name]["negative"].append(neg_acts[layer_name].cpu())

        except Exception as e:
            print(f"  Error processing question {pair_idx}: {e}")
            continue

    print("\n" + "=" * 80)
    print("Activation extraction complete!")

    # Verify data
    print("\nVerifying data completeness...")
    for layer_idx in range(1, num_layers + 1):
        layer_name = str(layer_idx)
        n_pos = len(data[layer_name]["positive"])
        n_neg = len(data[layer_name]["negative"])
        if n_pos != num_questions or n_neg != num_questions:
            print(f"  Warning: Layer {layer_name} has {n_pos} pos, {n_neg} neg (expected {num_questions} each)")

    # Clean up model
    del model
    del collector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(10)

    return data, num_layers

def perform_pca_analysis_3d(data: dict, num_layers: int, num_questions: int):
    """
    Perform 3D PCA on each layer and create visualization.

    Args:
        data: Dict with activations per layer
        num_layers: Number of layers
        num_questions: Number of questions used
    """
    print("\nPerforming 3D PCA analysis...")
    print("=" * 80)

    # Create figure with 4x7 subplots (28 layers) with 3D projection
    fig = plt.figure(figsize=(28, 16))
    fig.suptitle(f'3D PCA Analysis: SST2 Activations (continuation_token)\n'
                 f'{num_questions} questions ({num_questions * 2} samples)',
                 fontsize=20, fontweight='bold')

    for layer_idx in range(1, num_layers + 1):
        layer_name = str(layer_idx)

        # Create 3D subplot
        ax = fig.add_subplot(4, 7, layer_idx, projection='3d')

        # Prepare data
        pos_activations = torch.stack(data[layer_name]["positive"]).numpy()
        neg_activations = torch.stack(data[layer_name]["negative"]).numpy()

        # Combine positive and negative
        X = np.vstack([pos_activations, neg_activations])
        y = np.array([1] * len(pos_activations) + [0] * len(neg_activations))

        print(f"Layer {layer_idx}: Running 3D PCA on {X.shape[0]} samples, dim={X.shape[1]}")

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform PCA (3 components for 3D visualization)
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)

        # Split back into positive/negative
        pos_mask = y == 1
        neg_mask = y == 0

        # Plot in 3D
        ax.scatter(X_pca[pos_mask, 0], X_pca[pos_mask, 1], X_pca[pos_mask, 2],
                   c='red', alpha=0.6, s=10, label='Positive', edgecolors='none')
        ax.scatter(X_pca[neg_mask, 0], X_pca[neg_mask, 1], X_pca[neg_mask, 2],
                   c='blue', alpha=0.6, s=10, label='Negative', edgecolors='none')

        # Calculate explained variance
        explained_var = pca.explained_variance_ratio_
        total_var = explained_var.sum() * 100

        ax.set_title(f'Layer {layer_idx}\nVar: {total_var:.1f}%',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)', fontsize=7)
        ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)', fontsize=7)
        ax.set_zlabel(f'PC3 ({explained_var[2]*100:.1f}%)', fontsize=7)

        # Adjust tick label size
        ax.tick_params(labelsize=6)

        # Set viewing angle for better visualization
        ax.view_init(elev=20, azim=45)

        # Add legend only to first subplot
        if layer_idx == 1:
            ax.legend(loc='upper right', fontsize=7, framealpha=0.9)

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / "plots" / "pca_analysis_all_layers_3d.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 3D PCA plot saved to: {output_path}")

    plt.show()

    print("=" * 80)
    print("3D PCA analysis complete!")


def main():
    """Main function to run 3D PCA analysis."""

    print("\n" + "#" * 80)
    print("# 3D PCA ANALYSIS - SST2")
    print("#" * 80)
    print()

    # Configuration
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    num_questions = 100
    aggregation = ActivationAggregationStrategy.CONTINUATION_TOKEN

    print(f"Model: {model_name}")
    print(f"Aggregation: {aggregation.value}")
    print(f"Questions: {num_questions}")
    print(f"Total samples: {num_questions * 2} (positive + negative)")
    print()

    # Extract activations
    data, num_layers = extract_activations(
        model_name=model_name,
        num_questions=num_questions,
        aggregation=aggregation,
    )

    # Perform 3D PCA and create visualization
    perform_pca_analysis_3d(data, num_layers, num_questions)

    print("\n" + "#" * 80)
    print("# DONE!")
    print("#" * 80)


if __name__ == "__main__":
    main()

"""SAE feature analysis and visualization functions."""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple

from wisent.core.constants import VIZ_DPI, VIZ_HISTOGRAM_BINS, SAE_TOP_K_ANALYSIS, SAE_TOP_FEATURES_DISPLAY, SAE_TOP_FEATURES_MAX, VIZ_FONTSIZE_ANNOTATION, VIZ_FONTSIZE_BODY, VIZ_FONTSIZE_SUPTITLE, VIZ_ALPHA_LIGHT, VIZ_ALPHA_HALF, VIZ_ALPHA_HIGH, VIZ_HEATMAP_VMIN_ZERO, VIZ_HEATMAP_VMAX_ONE
from wisent.examples.scripts._sae import SparseAutoencoder


def analyze_sae_features(
    sae: SparseAutoencoder,
    activations_by_concept: Dict[str, torch.Tensor],
    top_k: int = SAE_TOP_K_ANALYSIS,
    device: str = "cpu",
) -> Dict:
    """
    Analyze which SAE features activate for each concept.
    
    Returns:
        Dictionary with feature analysis results
    """
    results = {
        "feature_activations": {},  # concept -> feature activation means
        "top_features": {},         # concept -> top-k most active features
        "feature_overlap": {},      # pair -> jaccard similarity of top features
        "unique_features": {},      # concept -> features unique to this concept
        "shared_features": [],      # features active across all concepts
    }
    
    # Get feature activations for each concept
    all_features = {}
    for concept, acts in activations_by_concept.items():
        acts = acts.to(device).float()
        acts_norm = (acts - sae.mean.to(device)) / sae.std.to(device)
        
        with torch.no_grad():
            features = sae.encode(acts_norm)
        
        # Mean activation per feature
        mean_activations = features.mean(dim=0).cpu().numpy()
        all_features[concept] = mean_activations
        results["feature_activations"][concept] = mean_activations.tolist()
        
        # Top-k most active features
        top_indices = np.argsort(mean_activations)[-top_k:][::-1]
        results["top_features"][concept] = top_indices.tolist()
    
    # Compute feature overlap between concepts
    concepts = list(activations_by_concept.keys())
    for i, c1 in enumerate(concepts):
        for c2 in concepts[i+1:]:
            set1 = set(results["top_features"][c1])
            set2 = set(results["top_features"][c2])
            
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            jaccard = intersection / union if union > 0 else 0
            
            results["feature_overlap"][f"{c1}_vs_{c2}"] = {
                "jaccard": jaccard,
                "shared_count": intersection,
                "shared_features": list(set1 & set2),
            }
    
    # Find features unique to each concept
    for concept in concepts:
        concept_set = set(results["top_features"][concept])
        other_sets = [set(results["top_features"][c]) for c in concepts if c != concept]
        all_others = set.union(*other_sets) if other_sets else set()
        unique = concept_set - all_others
        results["unique_features"][concept] = list(unique)
    
    # Find features shared across all harmful concepts
    harmful_concepts = [c for c in concepts if c != "neutral_baseline"]
    if len(harmful_concepts) >= 2:
        harmful_sets = [set(results["top_features"][c]) for c in harmful_concepts]
        shared = set.intersection(*harmful_sets)
        results["shared_features"] = list(shared)
    
    return results


def visualize_sae_analysis(
    sae: SparseAutoencoder,
    activations_by_concept: Dict[str, torch.Tensor],
    sae_results: Dict,
    output_path: Path,
    layer: int,
    model_name: str,
    device: str = "cpu",
):
    """Create visualizations of SAE feature analysis."""
    import matplotlib.pyplot as plt
    
    concepts = list(activations_by_concept.keys())
    colors = {'hitler': 'red', 'fascism': 'orange', 'harmful_ideology': 'purple', 'neutral_baseline': 'green'}
    
    # 1. Feature activation heatmap for top features
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Collect top features across all concepts
    all_top_features = set()
    for concept in concepts:
        all_top_features.update(sae_results["top_features"][concept][:SAE_TOP_FEATURES_DISPLAY])
    all_top_features = sorted(list(all_top_features))[:SAE_TOP_FEATURES_MAX]
    
    # Create heatmap data
    heatmap_data = []
    for concept in concepts:
        activations = sae_results["feature_activations"][concept]
        row = [activations[f] for f in all_top_features]
        heatmap_data.append(row)
    
    ax = axes[0, 0]
    im = ax.imshow(heatmap_data, aspect='auto', cmap='hot')
    ax.set_yticks(range(len(concepts)))
    ax.set_yticklabels(concepts)
    ax.set_xlabel('Feature Index')
    ax.set_title(f'SAE Feature Activations (Top Features)\nLayer {layer}')
    plt.colorbar(im, ax=ax, label='Mean Activation')
    
    # 2. Feature overlap matrix
    ax = axes[0, 1]
    overlap_matrix = np.zeros((len(concepts), len(concepts)))
    for i, c1 in enumerate(concepts):
        for j, c2 in enumerate(concepts):
            if i == j:
                overlap_matrix[i, j] = 1.0
            elif i < j:
                key = f"{c1}_vs_{c2}"
                if key in sae_results["feature_overlap"]:
                    overlap_matrix[i, j] = sae_results["feature_overlap"][key]["jaccard"]
                    overlap_matrix[j, i] = overlap_matrix[i, j]
            # else already set by symmetry
    
    im = ax.imshow(overlap_matrix, cmap='Blues', vmin=VIZ_HEATMAP_VMIN_ZERO, vmax=VIZ_HEATMAP_VMAX_ONE)
    ax.set_xticks(range(len(concepts)))
    ax.set_yticks(range(len(concepts)))
    ax.set_xticklabels([c.replace('_', '\n') for c in concepts], fontsize=VIZ_FONTSIZE_ANNOTATION)
    ax.set_yticklabels([c.replace('_', '\n') for c in concepts], fontsize=VIZ_FONTSIZE_ANNOTATION)
    ax.set_title('Feature Overlap (Jaccard Similarity)\nof Top-20 Features')
    
    for i in range(len(concepts)):
        for j in range(len(concepts)):
            ax.text(j, i, f'{overlap_matrix[i,j]:.2f}', ha='center', va='center', fontsize=VIZ_FONTSIZE_BODY)
    
    plt.colorbar(im, ax=ax)
    
    # 3. Unique vs shared features bar chart
    ax = axes[1, 0]
    x = np.arange(len(concepts))
    width = 0.35
    
    unique_counts = [len(sae_results["unique_features"].get(c, [])) for c in concepts]
    shared_count = len(sae_results.get("shared_features", []))
    
    bars1 = ax.bar(x - width/2, unique_counts, width, label='Unique Features', 
                   color=[colors.get(c, 'gray') for c in concepts])
    bars2 = ax.bar(x + width/2, [shared_count]*len(concepts), width, label='Shared (all harmful)', 
                   color='gray', alpha=VIZ_ALPHA_HALF)
    
    ax.set_ylabel('Number of Features')
    ax.set_title('Unique vs Shared Features (Top-20)')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in concepts], fontsize=VIZ_FONTSIZE_ANNOTATION)
    ax.legend()
    ax.grid(True, alpha=VIZ_ALPHA_LIGHT)
    
    # 4. Feature activation distribution
    ax = axes[1, 1]
    
    for concept in concepts:
        acts = activations_by_concept[concept].to(device).float()
        acts_norm = (acts - sae.mean.to(device)) / sae.std.to(device)
        
        with torch.no_grad():
            features = sae.encode(acts_norm)
        
        # Number of active features per sample
        active_per_sample = (features > 0.1).sum(dim=1).cpu().numpy()
        ax.hist(active_per_sample, bins=VIZ_HISTOGRAM_BINS, alpha=VIZ_ALPHA_HALF, label=concept, color=colors.get(concept, 'gray'))
    
    ax.set_xlabel('Number of Active Features per Sample')
    ax.set_ylabel('Count')
    ax.set_title('Feature Sparsity Distribution')
    ax.legend()
    ax.grid(True, alpha=VIZ_ALPHA_LIGHT)
    
    plt.suptitle(f'Sparse Autoencoder Feature Analysis - {model_name}\nHidden dim: {sae.hidden_dim}, L1: {sae.l1_coef}', fontsize=VIZ_FONTSIZE_SUPTITLE)
    plt.tight_layout()
    plt.savefig(output_path / 'sae_feature_analysis.png', dpi=VIZ_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'sae_feature_analysis.png'}")
    
    # 5. Feature activation comparison plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot mean activations for each concept across all features
    n_features_to_show = min(100, sae.hidden_dim)
    feature_indices = np.arange(n_features_to_show)
    
    for concept in concepts:
        activations = np.array(sae_results["feature_activations"][concept][:n_features_to_show])
        ax.plot(feature_indices, activations, label=concept, color=colors.get(concept, 'gray'), alpha=VIZ_ALPHA_HIGH)
    
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Mean Activation')
    ax.set_title(f'SAE Feature Activations by Concept (first {n_features_to_show} features)')
    ax.legend()
    ax.grid(True, alpha=VIZ_ALPHA_LIGHT)
    
    plt.tight_layout()
    plt.savefig(output_path / 'sae_feature_profiles.png', dpi=VIZ_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'sae_feature_profiles.png'}")


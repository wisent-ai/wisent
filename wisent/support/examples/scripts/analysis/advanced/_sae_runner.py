"""SAE analysis runner that orchestrates training, analysis, and visualization."""

import json
import torch
from pathlib import Path
from typing import Dict, Tuple

from wisent.examples.scripts._sae import train_sparse_autoencoder
from wisent.core.utils.config_tools.constants import DEFAULT_CLASSIFIER_LR, SAE_L1_COEF_DEFAULT, SAE_N_EPOCHS_DEFAULT, SAE_BATCH_SIZE_DEFAULT, SAE_HIDDEN_DIM_MULTIPLIER, SAE_TOP_K_ANALYSIS, DISPLAY_TOP_N_SMALL, JSON_INDENT
from wisent.examples.scripts._sae_analysis import (
    analyze_sae_features,
    visualize_sae_analysis,
)


def run_sae_analysis(
    activations_by_concept: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    layer: int,
    output_dir: str,
    model_name: str,
    hidden_dim_multiplier: int = SAE_HIDDEN_DIM_MULTIPLIER,
    l1_coef: float = SAE_L1_COEF_DEFAULT,
    n_epochs: int = SAE_N_EPOCHS_DEFAULT,
    device: str = "cpu",
) -> Dict:
    """
    Run full SAE analysis on collected activations.
    
    Args:
        activations_by_concept: Dict mapping concept -> layer -> (pos_activations, neg_activations)
        layer: Which layer to analyze
        output_dir: Where to save results
        model_name: Name of the model (for titles)
        hidden_dim_multiplier: SAE hidden dim = input_dim * this
        l1_coef: Sparsity coefficient
        n_epochs: Training epochs
        device: Device for training
    
    Returns:
        SAE analysis results dictionary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("SPARSE AUTOENCODER ANALYSIS")
    print(f"{'='*60}")
    print(f"Layer: {layer}")
    print(f"L1 coefficient: {l1_coef}")
    print(f"Training epochs: {n_epochs}")
    
    # Collect all activations for training
    all_activations = []
    concept_activations = {}
    
    for concept, layer_dict in activations_by_concept.items():
        if layer not in layer_dict:
            print(f"Warning: Layer {layer} not found for concept {concept}")
            continue
        
        pos, neg = layer_dict[layer]
        # Use positive activations (harmful content) for analysis
        concept_activations[concept] = pos
        all_activations.append(pos)
    
    if not all_activations:
        print("No activations found for SAE training")
        return {}
    
    # Combine all activations for training
    combined = torch.cat(all_activations, dim=0)
    input_dim = combined.shape[1]
    hidden_dim = input_dim * hidden_dim_multiplier
    
    print(f"\nTraining data: {combined.shape[0]} samples, {input_dim} dimensions")
    print(f"SAE architecture: {input_dim} -> {hidden_dim} -> {input_dim}")
    print(f"\nTraining SAE...")
    
    # Train SAE
    sae = train_sparse_autoencoder(
        combined,
        hidden_dim=hidden_dim,
        l1_coef=l1_coef,
        n_epochs=n_epochs,
        batch_size=SAE_BATCH_SIZE_DEFAULT,
        lr=DEFAULT_CLASSIFIER_LR,
        device=device,
        verbose=True,
    )
    
    # Analyze features
    print(f"\nAnalyzing SAE features...")
    sae_results = analyze_sae_features(sae, concept_activations, top_k=SAE_TOP_K_ANALYSIS, device=device)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SAE ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    print("\nTop-20 Feature Overlap (Jaccard Similarity):")
    for pair, data in sae_results["feature_overlap"].items():
        print(f"  {pair}: {data['jaccard']:.3f} ({data['shared_count']} shared features)")
    
    print("\nUnique Features per Concept:")
    for concept, unique in sae_results["unique_features"].items():
        print(f"  {concept}: {len(unique)} unique features")
    
    print(f"\nFeatures shared across ALL harmful concepts: {len(sae_results['shared_features'])}")
    if sae_results['shared_features']:
        print(f"  Feature indices: {sae_results['shared_features'][:DISPLAY_TOP_N_SMALL]}...")
    
    # Visualize
    print(f"\nGenerating visualizations...")
    visualize_sae_analysis(sae, concept_activations, sae_results, output_path, layer, model_name, device)
    
    # Save results
    results_file = output_path / f"sae_analysis_layer{layer}.json"
    with open(results_file, "w") as f:
        # Convert numpy arrays to lists for JSON
        json_results = {
            "layer": layer,
            "hidden_dim": hidden_dim,
            "l1_coef": l1_coef,
            "feature_overlap": sae_results["feature_overlap"],
            "unique_features": sae_results["unique_features"],
            "shared_features": sae_results["shared_features"],
            "top_features": sae_results["top_features"],
        }
        json.dump(json_results, f, indent=JSON_INDENT)
    print(f"Saved: {results_file}")
    
    return sae_results


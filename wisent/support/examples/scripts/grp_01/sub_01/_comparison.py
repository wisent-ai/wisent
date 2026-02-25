"""Direction comparison and main analysis runner."""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from wisent.core.constants import PAIR_GENERATORS_DEFAULT_N, JSON_INDENT
from wisent.core.models.wisent_model import WisentModel
from wisent.core.activations.activations_collector import ActivationCollector
from wisent.core.activations import ExtractionStrategy

from wisent.examples.scripts._pair_generators_neutral import (
    CONCEPT_GENERATORS,
    ConceptEvolutionResult,
)
from wisent.examples.scripts._concept_metrics import (
    compute_concept_direction,
    compute_cosine_similarity,
    analyze_concept,
)
from wisent.examples.scripts._evolution_visualization import (
    visualize_concept_evolution,
)


def compare_concept_directions(
    activations_by_concept: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    layers: List[int],
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Compare directions between all concept pairs at each layer."""
    
    concepts = list(activations_by_concept.keys())
    comparisons_by_layer = {}
    
    for layer in layers:
        layer_comparisons = {}
        directions = {}
        
        # Compute directions for each concept
        for concept in concepts:
            if layer not in activations_by_concept[concept]:
                continue
            pos, neg = activations_by_concept[concept][layer]
            directions[concept] = compute_concept_direction(pos, neg)
        
        # Compute pairwise similarities
        for i, c1 in enumerate(concepts):
            if c1 not in directions:
                continue
            for c2 in concepts[i+1:]:
                if c2 not in directions:
                    continue
                
                sim = compute_cosine_similarity(directions[c1], directions[c2])
                key = f"{c1}_vs_{c2}"
                layer_comparisons[key] = sim
        
        comparisons_by_layer[layer] = layer_comparisons
    
    return comparisons_by_layer


def run_analysis(
    model_name: str,
    layers_to_analyze: Optional[List[int]] = None,
    n_pairs: int = PAIR_GENERATORS_DEFAULT_N,
    output_dir: str = "/tmp/concept_evolution",
    device: str = "cuda",
):
    """Run the full concept evolution analysis."""
    
    print(f"\n{'#'*70}")
    print(f"CONCEPT EVOLUTION ANALYSIS")
    print(f"Model: {model_name}")
    print(f"Pairs per concept: {n_pairs}")
    print(f"{'#'*70}")
    
    # Load model
    print("\nLoading model...")
    model = WisentModel(model_name, device=device)
    print(f"  Loaded: {model.num_layers} layers, hidden_size={model.hidden_size}")
    
    # Determine layers to analyze
    if layers_to_analyze is None:
        # Sample layers: early, middle, late
        n_layers = model.num_layers
        layers_to_analyze = [
            1,                    # Very early
            n_layers // 4,        # Early-middle
            n_layers // 2,        # Middle
            3 * n_layers // 4,    # Late-middle
            n_layers - 1,         # Late
            n_layers,             # Final
        ]
        layers_to_analyze = sorted(set(layers_to_analyze))
    
    print(f"  Analyzing layers: {layers_to_analyze}")
    
    collector = ActivationCollector(model=model, store_device="cpu")
    
    concept_metrics = {}
    activations_by_concept = {}
    
    for concept_name, concept_data in CONCEPT_GENERATORS.items():
        metrics, activations = analyze_concept(
            model, collector, concept_name, concept_data,
            layers_to_analyze,
            n_pairs=n_pairs,
            strategy=ExtractionStrategy.CHAT_LAST,
        )
        concept_metrics[concept_name] = metrics
        activations_by_concept[concept_name] = activations
    
    # Compare directions between concepts
    print(f"\n{'='*60}")
    print("DIRECTION COMPARISONS (Cosine Similarity)")
    print(f"{'='*60}")
    
    direction_comparisons = compare_concept_directions(
        activations_by_concept, layers_to_analyze
    )
    
    for layer in layers_to_analyze:
        if layer not in direction_comparisons:
            continue
        print(f"\nLayer {layer}:")
        for pair_name, sim in direction_comparisons[layer].items():
            print(f"  {pair_name}: {sim:.4f}")
    
    # Generate summary
    print(f"\n{'='*60}")
    print("EVOLUTION SUMMARY")
    print(f"{'='*60}")
    
    # Find best layer for each concept
    for concept_name in ["hitler", "fascism", "harmful_ideology"]:
        if concept_name not in concept_metrics:
            continue
        
        best_layer = max(
            concept_metrics[concept_name].keys(),
            key=lambda l: concept_metrics[concept_name][l].linear_probe_accuracy
        )
        best_metrics = concept_metrics[concept_name][best_layer]
        
        print(f"\n{concept_name.upper()}:")
        print(f"  Best layer: {best_layer}")
        print(f"  Linear accuracy: {best_metrics.linear_probe_accuracy:.3f}")
        print(f"  k-NN accuracy: {best_metrics.knn_accuracy:.3f}")
        print(f"  Cohen's d: {best_metrics.cohens_d:.2f}")
        print(f"  Structure: {best_metrics.best_structure}")
        print(f"  Is linear: {best_metrics.is_linear}")
    
    # Analyze evolution pattern
    print(f"\n{'='*60}")
    print("CONCEPT HIERARCHY ANALYSIS")
    print(f"{'='*60}")
    
    # Find middle layer with good signal for comparison
    middle_layer = layers_to_analyze[len(layers_to_analyze) // 2]
    
    if middle_layer in direction_comparisons:
        comps = direction_comparisons[middle_layer]
        
        # Expected pattern: Hitler-Fascism should be more similar than Hitler-Harmful
        hitler_fascism = comps.get("hitler_vs_fascism", 0)
        hitler_harmful = comps.get("hitler_vs_harmful_ideology", 0)
        fascism_harmful = comps.get("fascism_vs_harmful_ideology", 0)
        
        print(f"\nAt layer {middle_layer}:")
        print(f"  Hitler <-> Fascism:    {hitler_fascism:.4f}")
        print(f"  Hitler <-> Harmful:    {hitler_harmful:.4f}")
        print(f"  Fascism <-> Harmful:   {fascism_harmful:.4f}")
        
        # Check if hierarchy is preserved
        if hitler_fascism > hitler_harmful:
            print("\n  [OK] Hitler is more similar to Fascism than to general Harmful ideology")
            print("       (Specific -> Abstract pattern preserved)")
        else:
            print("\n  [UNEXPECTED] Hitler is more similar to Harmful than to Fascism")
        
        if fascism_harmful > hitler_harmful:
            print("  [OK] Fascism is more similar to Harmful than Hitler is")
            print("       (Intermediate abstraction level)")
        
        # Compare to neutral baseline
        hitler_neutral = comps.get("hitler_vs_neutral_baseline", 0)
        fascism_neutral = comps.get("fascism_vs_neutral_baseline", 0)
        harmful_neutral = comps.get("harmful_ideology_vs_neutral_baseline", 0)
        
        print(f"\n  Similarity to neutral baseline:")
        print(f"    Hitler:   {hitler_neutral:.4f}")
        print(f"    Fascism:  {fascism_neutral:.4f}")
        print(f"    Harmful:  {harmful_neutral:.4f}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    result = ConceptEvolutionResult(
        concepts=list(CONCEPT_GENERATORS.keys()),
        model_name=model_name,
        timestamp=datetime.now().isoformat(),
        concept_metrics={
            c: {l: asdict(m) for l, m in metrics.items()}
            for c, metrics in concept_metrics.items()
        },
        direction_similarities={
            str(l): comps for l, comps in direction_comparisons.items()
        },
    )
    
    result_file = output_path / f"concept_evolution_{model_name.replace('/', '_')}.json"
    with open(result_file, "w") as f:
        json.dump(asdict(result), f, indent=JSON_INDENT)
    
    print(f"\nResults saved to: {result_file}")
    
    # Generate visualizations
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    visualize_concept_evolution(
        activations_by_concept=activations_by_concept,
        concept_metrics=concept_metrics,
        direction_comparisons=direction_comparisons,
        layers=layers_to_analyze,
        output_dir=output_dir,
        model_name=model_name,
    )
    
    return result

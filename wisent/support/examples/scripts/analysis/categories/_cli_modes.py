"""CLI mode handlers for mixed concept detection: layers, attribution, k-detection."""

import random
from pathlib import Path

import numpy as np

from wisent.core.primitives.models.wisent_model import WisentModel
from wisent.core.utils.config_tools.constants import SEPARATOR_WIDTH_WIDE, SEPARATOR_WIDTH_STANDARD, SEPARATOR_WIDTH_TABLE

from ._data_loading import (
    load_truthfulqa_pairs,
    load_hellaswag_pairs,
    extract_difference_vectors,
)
from ._detection_k import detect_k_concepts
from ._layer_analysis import (
    extract_difference_vectors_all_layers,
    analyze_layer_separability,
)
from ._visualization import (
    visualize_multi_method,
    visualize_layer_analysis,
    attribute_pairs_to_concepts,
    print_concept_attribution,
)
from ._visualization_advanced import visualize_k_concepts


def run_analyze_layers_mode(args):
    """Run multi-layer separability analysis mode."""
    # Multi-layer analysis
    print("=" * SEPARATOR_WIDTH_WIDE)
    print("MULTI-LAYER SEPARABILITY ANALYSIS")
    print("=" * SEPARATOR_WIDTH_WIDE)
    
    vis_output = Path(args.vis_output_dir)
    vis_output.mkdir(parents=True, exist_ok=True)
    
    # Load pairs
    print("\nLoading TruthfulQA pairs...")
    tqa_pairs = load_truthfulqa_pairs(args.n_pairs, args.seed)
    print(f"  Loaded {len(tqa_pairs)} pairs")
    
    print("Loading HellaSwag pairs...")
    hs_pairs = load_hellaswag_pairs(args.n_pairs, args.seed)
    print(f"  Loaded {len(hs_pairs)} pairs")
    
    # Load model
    print(f"\nLoading model: {args.model}")
    model = WisentModel(args.model, device="mps")
    print(f"Model has {model.num_layers} layers")
    
    # Create mixed sample
    mixed_pairs = tqa_pairs + hs_pairs
    random.seed(args.seed)
    random.shuffle(mixed_pairs)
    sources = [p['source'] for p in mixed_pairs]
    
    # Extract activations from ALL layers
    print(f"\nExtracting activations from ALL {model.num_layers} layers...")
    all_layer_diffs = extract_difference_vectors_all_layers(model, mixed_pairs)
    
    # Analyze separability at each layer
    print("\nAnalyzing separability at each layer...")
    layer_results = analyze_layer_separability(all_layer_diffs, sources)
    
    # Print results
    print(f"\n{'Layer':<8} {'Silhouette':<12} {'Dir Sim':<12} {'Purity':<12} {'Sep Score':<12}")
    print("-" * SEPARATOR_WIDTH_TABLE)
    for layer_idx in sorted(layer_results.keys()):
        r = layer_results[layer_idx]
        print(f"{layer_idx:<8} {r['silhouette']:<12.3f} {r['direction_similarity']:<12.3f} {r['cluster_purity']:<12.3f} {r['separability_score']:<12.3f}")
    
    # Find best layer
    best_layer = max(layer_results.keys(), key=lambda l: layer_results[l]['separability_score'])
    print(f"\nBest layer for concept separation: {best_layer}")
    print(f"  - Direction similarity: {layer_results[best_layer]['direction_similarity']:.3f}")
    print(f"  - Cluster purity: {layer_results[best_layer]['cluster_purity']:.3f}")
    
    # Visualize
    visualize_layer_analysis(
        layer_results,
        title=f"Layer-wise Separability: {args.model}",
        output_path=str(vis_output / "layer_analysis.png"),
        show_plot=False
    )
    
    # Also show multi-method projection for best layer
    if args.projection_method == "all":
        print(f"\nGenerating multi-method projections for layer {best_layer}...")
        visualize_multi_method(
            all_layer_diffs[best_layer],
            sources,
            title=f"Layer {best_layer} - Multi-Method Projection",
            output_path=str(vis_output / f"layer_{best_layer}_multi_method.png"),
            show_plot=False
        )
    
    print(f"\nVisualizations saved to: {vis_output}")


def run_attribute_mode(args):
    """Run attribution to trace pairs to concepts."""
    # Run attribution to trace pairs to concepts
    print("=" * SEPARATOR_WIDTH_WIDE)
    print("CONCEPT ATTRIBUTION - Trace Pairs to Detected Concepts")
    print("=" * SEPARATOR_WIDTH_WIDE)
    
    # Load pairs
    print("\nLoading TruthfulQA pairs...")
    tqa_pairs = load_truthfulqa_pairs(args.n_pairs, args.seed)
    print(f"  Loaded {len(tqa_pairs)} pairs")
    
    print("Loading HellaSwag pairs...")
    hs_pairs = load_hellaswag_pairs(args.n_pairs, args.seed)
    print(f"  Loaded {len(hs_pairs)} pairs")
    
    # Load model
    print(f"\nLoading model: {args.model}")
    model = WisentModel(args.model, device="mps")
    
    layer = args.layer if args.layer else model.num_layers // 2
    print(f"Using layer: {layer}")
    
    # Create mixed sample (preserve original pairs list for attribution)
    mixed_pairs = tqa_pairs + hs_pairs
    original_order = list(range(len(mixed_pairs)))
    random.seed(args.seed)
    random.shuffle(mixed_pairs)
    
    # Extract activations
    print(f"\nExtracting activations for {len(mixed_pairs)} pairs...")
    mixed_diffs, mixed_sources = extract_difference_vectors(model, mixed_pairs, layer)
    
    # Run k-concept detection
    print("\nDetecting concepts...")
    detection = detect_k_concepts(mixed_diffs, max_k=args.max_k)
    print(f"  {detection['recommendation']}")
    
    # Run attribution
    print("\nAttributing pairs to concepts...")
    attribution = attribute_pairs_to_concepts(mixed_diffs, mixed_pairs, detection)
    
    # Print results
    print_concept_attribution(attribution, show_samples=True)
    
    # Summary of alignment with true sources
    print(f"\n{'=' * SEPARATOR_WIDTH_WIDE}")
    print("VALIDATION: How well do detected concepts align with true sources?")
    print("=" * SEPARATOR_WIDTH_WIDE)
    
    for concept_id, details in attribution['concept_details'].items():
        sources = details['source_distribution']
        total = details['num_pairs']
        dominant_source = max(sources.keys(), key=lambda k: sources[k])
        purity = sources[dominant_source] / total
        print(f"  Concept {concept_id}: {purity:.1%} purity ({dominant_source})")
        print(f"    Distribution: {sources}")
    


def run_detect_k_mode(args):
    """Run k-concept detection mode."""
    # Run k-concept detection
    print("=" * SEPARATOR_WIDTH_WIDE)
    print("K-CONCEPT DETECTION")
    print("=" * SEPARATOR_WIDTH_WIDE)
    
    vis_output = Path(args.vis_output_dir)
    vis_output.mkdir(parents=True, exist_ok=True)
    
    # Load pairs
    print("\nLoading TruthfulQA pairs...")
    tqa_pairs = load_truthfulqa_pairs(args.n_pairs, args.seed)
    print(f"  Loaded {len(tqa_pairs)} pairs")
    
    print("Loading HellaSwag pairs...")
    hs_pairs = load_hellaswag_pairs(args.n_pairs, args.seed)
    print(f"  Loaded {len(hs_pairs)} pairs")
    
    # Load model
    print(f"\nLoading model: {args.model}")
    model = WisentModel(args.model, device="mps")
    
    layer = args.layer if args.layer else model.num_layers // 2
    print(f"Using layer: {layer}")
    
    # Create mixed sample
    mixed_pairs = tqa_pairs + hs_pairs
    random.seed(args.seed)
    random.shuffle(mixed_pairs)
    
    # Extract activations
    print("\n--- Extracting activations ---")
    
    print("\nMixed sample...")
    mixed_diffs, mixed_sources = extract_difference_vectors(model, mixed_pairs, layer)
    
    print("\nTruthfulQA sample...")
    tqa_diffs, tqa_sources = extract_difference_vectors(model, tqa_pairs, layer)
    
    print("\nHellaSwag sample...")
    hs_diffs, hs_sources = extract_difference_vectors(model, hs_pairs, layer)
    
    # Run k-concept detection
    print("\n--- Running k-concept detection ---")
    
    print("\nAnalyzing MIXED sample...")
    mixed_detection = detect_k_concepts(mixed_diffs, max_k=args.max_k)
    print(f"  {mixed_detection['recommendation']}")
    
    print("\nAnalyzing TruthfulQA sample...")
    tqa_detection = detect_k_concepts(tqa_diffs, max_k=args.max_k)
    print(f"  {tqa_detection['recommendation']}")
    
    print("\nAnalyzing HellaSwag sample...")
    hs_detection = detect_k_concepts(hs_diffs, max_k=args.max_k)
    print(f"  {hs_detection['recommendation']}")
    
    # Summary
    print("\n" + "=" * SEPARATOR_WIDTH_WIDE)
    print("K-CONCEPT DETECTION SUMMARY")
    print("=" * SEPARATOR_WIDTH_WIDE)
    print(f"\n{'Sample':<20} {'Detected':<15} {'Expected':<15} {'Match':<10}")
    print("-" * SEPARATOR_WIDTH_STANDARD)
    print(f"{'Mixed':<20} {mixed_detection['detected_concepts']:<15} {'2':<15} {'YES' if mixed_detection['detected_concepts'] == 2 else 'NO':<10}")
    print(f"{'TruthfulQA':<20} {tqa_detection['detected_concepts']:<15} {'1':<15} {'YES' if tqa_detection['detected_concepts'] == 1 else 'NO':<10}")
    print(f"{'HellaSwag':<20} {hs_detection['detected_concepts']:<15} {'1':<15} {'YES' if hs_detection['detected_concepts'] == 1 else 'NO':<10}")
    
    # Generate visualizations
    print("\n--- Generating visualizations ---")
    
    visualize_k_concepts(
        mixed_diffs, mixed_sources, mixed_detection,
        title="MIXED: TruthfulQA + HellaSwag",
        output_path=str(vis_output / "k_detection_mixed.png"),
        show_plot=False
    )
    
    visualize_k_concepts(
        tqa_diffs, tqa_sources, tqa_detection,
        title="PURE: TruthfulQA Only",
        output_path=str(vis_output / "k_detection_truthfulqa.png"),
        show_plot=False
    )
    
    visualize_k_concepts(
        hs_diffs, hs_sources, hs_detection,
        title="PURE: HellaSwag Only",
        output_path=str(vis_output / "k_detection_hellaswag.png"),
        show_plot=False
    )
    
    print(f"\nVisualizations saved to: {vis_output}")

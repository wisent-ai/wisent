"""Diagnose steering vectors command execution logic."""

import sys
import json
import os
import math

import torch
from wisent.core.utils.device import preferred_dtype


def execute_diagnose_vectors(args):
    """Execute the diagnose-vectors command - analyze existing steering vectors."""
    print(f"\n🔍 Diagnosing steering vectors")
    print(f"   Input file: {args.vectors_file}")

    try:
        # 1. Load vectors from JSON
        if not os.path.exists(args.vectors_file):
            raise FileNotFoundError(f"Vectors file not found: {args.vectors_file}")

        with open(args.vectors_file, 'r') as f:
            data = json.load(f)

        # 2. Basic metadata
        print(f"\n📊 Basic Information:")
        trait_label = data.get('trait_label', 'N/A')
        method = data.get('method', 'N/A')
        normalized = data.get('normalized', 'N/A')
        model = data.get('model', 'N/A')

        print(f"   Trait label: {trait_label}")
        print(f"   Method: {method}")
        print(f"   Normalized: {normalized}")
        if model != 'N/A':
            print(f"   Model: {model}")

        # 3. Vector analysis
        vectors = data.get('vectors', data.get('steering_vectors', {}))

        if not vectors:
            print(f"\n❌ No vectors found in file!")
            sys.exit(1)

        print(f"\n🧠 Vector Analysis:")
        print(f"   Number of layers: {len(vectors)}")
        print(f"   Layers: {sorted([int(k) for k in vectors.keys()])}")

        # Analyze each layer's vector
        layer_stats = []
        for layer_str, vector_list in vectors.items():
            if not isinstance(vector_list, list):
                continue

            # Calculate L2 norm
            l2_norm = math.sqrt(sum(x*x for x in vector_list))

            # Basic statistics
            min_val = min(vector_list)
            max_val = max(vector_list)
            mean_val = sum(vector_list) / len(vector_list)

            # Count zeros
            zero_count = sum(1 for x in vector_list if abs(x) < 1e-9)
            zero_pct = (zero_count / len(vector_list)) * 100

            layer_stats.append({
                'layer': int(layer_str),
                'dim': len(vector_list),
                'l2_norm': l2_norm,
                'min': min_val,
                'max': max_val,
                'mean': mean_val,
                'zeros': zero_count,
                'zero_pct': zero_pct
            })

        # Sort by layer
        layer_stats.sort(key=lambda x: x['layer'])

        # Display layer-by-layer stats
        print(f"\n📈 Layer-by-Layer Statistics:")
        print(f"   {'Layer':<8} {'Dim':<8} {'L2 Norm':<12} {'Min':<10} {'Max':<10} {'Mean':<10} {'Zeros':<8}")
        print(f"   {'-'*78}")

        for stats in layer_stats:
            print(f"   {stats['layer']:<8} {stats['dim']:<8} {stats['l2_norm']:<12.6f} "
                  f"{stats['min']:<10.4f} {stats['max']:<10.4f} {stats['mean']:<10.4f} "
                  f"{stats['zeros']:<8} ({stats['zero_pct']:.1f}%)")

        # Normalization check
        if normalized == True or normalized == 'true':
            print(f"\n✅ Normalization Check:")
            non_unit = [s for s in layer_stats if abs(s['l2_norm'] - 1.0) > 0.01]
            if len(non_unit) == 0:
                print(f"   All vectors have unit L2 norm (≈1.0)")
            else:
                print(f"   ⚠️  {len(non_unit)} vectors do not have unit norm:")
                for s in non_unit:
                    print(f"      Layer {s['layer']}: L2 norm = {s['l2_norm']:.6f}")

        # Quality warnings
        print(f"\n⚠️  Quality Warnings:")
        warnings = []

        for stats in layer_stats:
            if stats['zero_pct'] > 50:
                warnings.append(f"Layer {stats['layer']}: {stats['zero_pct']:.1f}% zeros (may be too sparse)")
            if abs(stats['mean']) > 0.5:
                warnings.append(f"Layer {stats['layer']}: mean = {stats['mean']:.4f} (unusually large)")
            if stats['l2_norm'] < 0.01:
                warnings.append(f"Layer {stats['layer']}: very small L2 norm ({stats['l2_norm']:.6f})")
            if stats['l2_norm'] > 100:
                warnings.append(f"Layer {stats['layer']}: very large L2 norm ({stats['l2_norm']:.2f})")

        if len(warnings) == 0:
            print(f"   No quality issues detected")
        else:
            for warning in warnings:
                print(f"   {warning}")

        # Dimensionality consistency
        dims = [s['dim'] for s in layer_stats]
        if len(set(dims)) == 1:
            print(f"\n✅ Dimensionality: Consistent across all layers ({dims[0]})")
        else:
            print(f"\n⚠️  Dimensionality: Inconsistent across layers")
            for stats in layer_stats:
                print(f"   Layer {stats['layer']}: {stats['dim']}")

        # Show sample vector if requested
        if args.show_sample and len(layer_stats) > 0:
            sample_layer = str(layer_stats[0]['layer'])
            sample_vector = vectors[sample_layer]
            print(f"\n📄 Sample Vector (Layer {sample_layer}, first 10 values):")
            print(f"   {sample_vector[:10]}")

        # Cone structure analysis (if requested)
        if hasattr(args, 'check_cone') and args.check_cone:
            if hasattr(args, 'activations_file') and args.activations_file:
                _run_cone_analysis(
                    args.activations_file, 
                    args.verbose,
                    getattr(args, 'cone_threshold', 0.7),
                    getattr(args, 'cone_directions', 5)
                )
            else:
                print(f"\n⚠️  Cone Analysis: Requires --activations-file with positive/negative activations")
                print(f"   Run with: --activations-file <path> to analyze cone structure")
        
        # Comprehensive geometry analysis (if requested)
        if hasattr(args, 'detect_geometry') and args.detect_geometry:
            if hasattr(args, 'activations_file') and args.activations_file:
                _run_geometry_analysis(
                    args.activations_file,
                    args.verbose,
                    getattr(args, 'cone_directions', 5),
                    getattr(args, 'max_clusters', 5),
                    getattr(args, 'manifold_neighbors', 10),
                )
            else:
                print(f"\n⚠️  Geometry Analysis: Requires --activations-file with positive/negative activations")
                print(f"   Run with: --activations-file <path> to analyze geometry structure")

        print(f"\n✅ Diagnosis complete!\n")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _run_cone_analysis(
    activations_file: str, 
    verbose: bool = False,
    cone_threshold: float = 0.7,
    cone_directions: int = 5,
):
    """Run cone structure analysis on activations."""
    from wisent.core.contrastive_pairs.diagnostics.control_vectors import (
        check_cone_structure,
        ConeAnalysisConfig,
    )
    
    print(f"\n🔺 Cone Structure Analysis")
    print(f"   Activations file: {activations_file}")
    
    try:
        if not os.path.exists(activations_file):
            print(f"   ❌ Activations file not found: {activations_file}")
            return
        
        # Load activations (supports .pt or .json)
        if activations_file.endswith('.pt'):
            activations_data = torch.load(activations_file, weights_only=True)
        else:
            with open(activations_file, 'r') as f:
                activations_data = json.load(f)
        
        # Extract positive and negative activations
        pos_acts = None
        neg_acts = None
        
        if isinstance(activations_data, dict):
            # Format 1: {"positive": [...], "negative": [...]}
            if 'positive' in activations_data and 'negative' in activations_data:
                pos_acts = activations_data['positive']
                neg_acts = activations_data['negative']
            # Format 2: {"pos_activations": [...], "neg_activations": [...]}
            elif 'pos_activations' in activations_data and 'neg_activations' in activations_data:
                pos_acts = activations_data['pos_activations']
                neg_acts = activations_data['neg_activations']
            # Format 3: Per-layer format {"layer_15": {"pos": [...], "neg": [...]}}
            elif any(k.startswith('layer_') for k in activations_data.keys()):
                # Use the first layer found
                for key, layer_data in activations_data.items():
                    if isinstance(layer_data, dict) and 'pos' in layer_data and 'neg' in layer_data:
                        pos_acts = layer_data['pos']
                        neg_acts = layer_data['neg']
                        print(f"   Using layer: {key}")
                        break
        
        if pos_acts is None or neg_acts is None:
            print(f"   ❌ Could not find positive/negative activations in file")
            print(f"   Expected format: {{'positive': [...], 'negative': [...]}}")
            return
        
        # Convert to tensors if needed
        dtype = preferred_dtype()
        if not isinstance(pos_acts, torch.Tensor):
            pos_acts = torch.tensor(pos_acts, dtype=dtype)
        if not isinstance(neg_acts, torch.Tensor):
            neg_acts = torch.tensor(neg_acts, dtype=dtype)
        
        print(f"   Positive samples: {pos_acts.shape[0]}")
        print(f"   Negative samples: {neg_acts.shape[0]}")
        print(f"   Hidden dimension: {pos_acts.shape[1]}")
        
        # Run cone analysis
        config = ConeAnalysisConfig(
            num_directions=cone_directions,
            optimization_steps=100,
            cone_threshold=cone_threshold,
        )
        
        print(f"\n   Running cone analysis...")
        result = check_cone_structure(pos_acts, neg_acts, config)
        
        # Display results
        print(f"\n📊 Cone Analysis Results:")
        
        if result.has_cone_structure:
            print(f"   ✅ CONE STRUCTURE DETECTED")
        else:
            print(f"   ❌ No cone structure (linear subspace is sufficient)")
        
        print(f"\n   Cone Score: {result.cone_score:.3f} (threshold: {config.cone_threshold})")
        print(f"   PCA Explained Variance: {result.pca_explained_variance:.3f}")
        print(f"   Cone Explained Variance: {result.cone_explained_variance:.3f}")
        print(f"   Half-Space Consistency: {result.half_space_consistency:.3f}")
        print(f"   Avg Cosine Similarity: {result.avg_cosine_similarity:.3f}")
        print(f"   Positive Combination Score: {result.positive_combination_score:.3f}")
        print(f"   Directions Found: {result.num_directions_found}")
        
        # Separation scores
        print(f"\n   Per-Direction Separation Scores:")
        for i, score in enumerate(result.separation_scores):
            significance = "***" if abs(score) > 0.3 else "**" if abs(score) > 0.1 else "*" if abs(score) > 0.05 else ""
            print(f"      Direction {i}: {score:.4f} {significance}")
        
        # Interpretation
        print(f"\n📝 Interpretation:")
        if result.has_cone_structure:
            print(f"   - Multiple directions mediate this behavior")
            print(f"   - Consider using PRISM for multi-directional steering")
            print(f"   - CAA may capture only partial behavior")
        else:
            print(f"   - Single direction (CAA) is sufficient")
            print(f"   - Behavior is well-represented by linear subspace")
        
        if verbose:
            print(f"\n   Cosine Similarity Matrix:")
            for i, row in enumerate(result.direction_cosine_similarities):
                print(f"      {[f'{x:.2f}' for x in row]}")
        
    except Exception as e:
        print(f"   ❌ Cone analysis failed: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()


def _run_geometry_analysis(
    activations_file: str,
    verbose: bool = False,
    num_components: int = 5,
    max_clusters: int = 5,
    manifold_neighbors: int = 10,
):
    """Run comprehensive geometry structure analysis on activations."""
    from wisent.core.contrastive_pairs.diagnostics.control_vectors import (
        detect_geometry_structure,
        GeometryAnalysisConfig,
    )
    
    print(f"\n🔷 Comprehensive Geometry Analysis")
    print(f"   Activations file: {activations_file}")
    
    try:
        if not os.path.exists(activations_file):
            print(f"   ❌ Activations file not found: {activations_file}")
            return
        
        # Load activations (supports .pt or .json)
        if activations_file.endswith('.pt'):
            activations_data = torch.load(activations_file, weights_only=True)
        else:
            with open(activations_file, 'r') as f:
                activations_data = json.load(f)
        
        # Extract positive and negative activations
        pos_acts = None
        neg_acts = None
        
        if isinstance(activations_data, dict):
            if 'positive' in activations_data and 'negative' in activations_data:
                pos_acts = activations_data['positive']
                neg_acts = activations_data['negative']
            elif 'pos_activations' in activations_data and 'neg_activations' in activations_data:
                pos_acts = activations_data['pos_activations']
                neg_acts = activations_data['neg_activations']
            elif any(k.startswith('layer_') for k in activations_data.keys()):
                for key, layer_data in activations_data.items():
                    if isinstance(layer_data, dict) and 'pos' in layer_data and 'neg' in layer_data:
                        pos_acts = layer_data['pos']
                        neg_acts = layer_data['neg']
                        print(f"   Using layer: {key}")
                        break
        
        if pos_acts is None or neg_acts is None:
            print(f"   ❌ Could not find positive/negative activations in file")
            return
        
        # Convert to tensors
        dtype = preferred_dtype()
        if not isinstance(pos_acts, torch.Tensor):
            pos_acts = torch.tensor(pos_acts, dtype=dtype)
        if not isinstance(neg_acts, torch.Tensor):
            neg_acts = torch.tensor(neg_acts, dtype=dtype)
        
        print(f"   Positive samples: {pos_acts.shape[0]}")
        print(f"   Negative samples: {neg_acts.shape[0]}")
        print(f"   Hidden dimension: {pos_acts.shape[1]}")
        
        # Run geometry analysis
        config = GeometryAnalysisConfig(
            num_components=num_components,
            max_clusters=max_clusters,
            manifold_neighbors=manifold_neighbors,
        )
        
        print(f"\n   Running geometry analysis...")
        result = detect_geometry_structure(pos_acts, neg_acts, config)
        
        # Display results
        print(f"\n📊 Geometry Analysis Results:")
        print(f"\n   Best Structure: {result.best_structure.value.upper()}")
        print(f"   Best Score: {result.best_score:.3f}")
        
        print(f"\n   All Structure Scores (ranked):")
        print(f"   {'Structure':<12} {'Score':<8} {'Confidence':<10}")
        print(f"   {'-'*32}")
        
        for name, score in sorted(result.all_scores.items(), key=lambda x: x[1].score, reverse=True):
            marker = "***" if score.score > 0.8 else "**" if score.score > 0.6 else "*" if score.score > 0.4 else ""
            print(f"   {name:<12} {score.score:<8.3f} {score.confidence:<10.3f} {marker}")
        
        print(f"\n📝 Recommendation:")
        print(f"   {result.recommendation}")
        
        if verbose:
            print(f"\n   Detailed Structure Analysis:")
            for name, score in result.all_scores.items():
                if score.details:
                    print(f"\n   {name.upper()}:")
                    for key, value in score.details.items():
                        if isinstance(value, float):
                            print(f"      {key}: {value:.4f}")
                        elif isinstance(value, list) and len(value) <= 10:
                            if all(isinstance(v, float) for v in value):
                                print(f"      {key}: [{', '.join(f'{v:.3f}' for v in value)}]")
                            else:
                                print(f"      {key}: {value}")
                        elif isinstance(value, dict):
                            print(f"      {key}:")
                            for k, v in value.items():
                                print(f"         {k}: {v}")
                        else:
                            print(f"      {key}: {value}")
        
    except Exception as e:
        print(f"   ❌ Geometry analysis failed: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()

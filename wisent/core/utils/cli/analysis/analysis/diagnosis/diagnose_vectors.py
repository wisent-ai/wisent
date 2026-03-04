"""Diagnose steering vectors command execution logic."""

import sys
import json
import os
import math

import torch
from wisent.core.utils import preferred_dtype
from wisent.core import constants as _C
from wisent.core.utils.config_tools.constants import NEAR_ZERO_TOL
from wisent.core.utils.cli.analysis.diagnosis.diagnose_vectors_analysis import (
    _run_cone_analysis, _run_geometry_analysis,
)


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
            zero_count = sum(1 for x in vector_list if abs(x) < NEAR_ZERO_TOL)
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
            non_unit = [s for s in layer_stats if abs(s['l2_norm'] - 1.0) > args.diag_norm_tolerance]
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
            if stats['zero_pct'] > args.diag_zero_pct_threshold:
                warnings.append(f"Layer {stats['layer']}: {stats['zero_pct']:.1f}% zeros (may be too sparse)")
            if abs(stats['mean']) > args.diag_mean_magnitude_threshold:
                warnings.append(f"Layer {stats['layer']}: mean = {stats['mean']:.4f} (unusually large)")
            if stats['l2_norm'] < args.diag_norm_min:
                warnings.append(f"Layer {stats['layer']}: very small L2 norm ({stats['l2_norm']:.6f})")
            if stats['l2_norm'] > args.diag_norm_max:
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
            print(f"\n📄 Sample Vector (Layer {sample_layer}, first {_C.DISPLAY_TOP_N_SMALL} values):")
            print(f"   {sample_vector[:_C.DISPLAY_TOP_N_SMALL]}")

        # Cone structure analysis (if requested)
        if hasattr(args, 'check_cone') and args.check_cone:
            if hasattr(args, 'activations_file') and args.activations_file:
                cone_threshold = getattr(args, 'cone_threshold', None)
                if cone_threshold is None:
                    raise ValueError("'cone_threshold' is required. Pass --cone-threshold or run 'wisent optimize-steering auto' first.")
                cone_directions = getattr(args, 'cone_directions', None)
                if cone_directions is None:
                    raise ValueError("'cone_directions' is required. Pass --cone-directions or run 'wisent optimize-steering auto' first.")
                min_cosine_similarity = getattr(args, 'grom_min_cosine_sim', None)
                if min_cosine_similarity is None:
                    raise ValueError("'grom_min_cosine_sim' is required. Pass --grom-min-cosine-sim or run 'wisent optimize-steering auto' first.")
                _run_cone_analysis(
                    args.activations_file,
                    args.verbose,
                    cone_threshold,
                    cone_directions,
                    min_cosine_similarity=min_cosine_similarity,
                )
            else:
                print(f"\n⚠️  Cone Analysis: Requires --activations-file with positive/negative activations")
                print(f"   Run with: --activations-file <path> to analyze cone structure")
        
        # Comprehensive geometry analysis (if requested)
        if hasattr(args, 'detect_geometry') and args.detect_geometry:
            if hasattr(args, 'activations_file') and args.activations_file:
                num_components = getattr(args, 'num_components', None)
                if num_components is None:
                    raise ValueError("'num_components' is required. Pass --num-components or run 'wisent optimize-steering auto' first.")
                max_clusters = getattr(args, 'max_clusters', None)
                if max_clusters is None:
                    raise ValueError("'max_clusters' is required. Pass --max-clusters or run 'wisent optimize-steering auto' first.")
                manifold_neighbors = getattr(args, 'manifold_neighbors', None)
                if manifold_neighbors is None:
                    raise ValueError("'manifold_neighbors' is required. Pass --manifold-neighbors or run 'wisent optimize-steering auto' first.")
                _run_geometry_analysis(
                    args.activations_file,
                    args.verbose,
                    num_components,
                    max_clusters,
                    manifold_neighbors,
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



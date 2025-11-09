"""Diagnose steering vectors command execution logic."""

import sys
import json
import os
import math


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

        print(f"\n✅ Diagnosis complete!\n")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

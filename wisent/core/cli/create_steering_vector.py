"""Create steering vector command execution logic."""

import sys
import json
import os
import time
import torch
from collections import defaultdict


def execute_create_steering_vector(args):
    """Execute the create-steering-vector command - load enriched pairs and create steering vectors."""
    from wisent.core.steering_methods.methods.caa import CAAMethod

    print(f"\n🎯 Creating steering vectors from enriched pairs")
    print(f"   Input file: {args.enriched_pairs_file}")
    print(f"   Method: {args.method}")

    start_time = time.time() if args.timing else None

    try:
        # 1. Load enriched pairs from JSON
        print(f"\n📂 Loading enriched pairs...")
        if not os.path.exists(args.enriched_pairs_file):
            raise FileNotFoundError(f"Enriched pairs file not found: {args.enriched_pairs_file}")

        with open(args.enriched_pairs_file, 'r') as f:
            data = json.load(f)

        # Extract metadata
        trait_label = data.get('trait_label', 'unknown')
        model = data.get('model', 'unknown')
        layers = data.get('layers', [])
        token_aggregation = data.get('token_aggregation', 'unknown')
        pairs_list = data.get('pairs', [])

        print(f"   ✓ Loaded {len(pairs_list)} pairs")
        print(f"   ✓ Model: {model}")
        print(f"   ✓ Layers: {layers}")
        print(f"   ✓ Token aggregation: {token_aggregation}")

        # 2. Organize activations by layer
        print(f"\n📊 Organizing activations by layer...")

        # Structure: {layer_str: {"positive": [tensors], "negative": [tensors]}}
        layer_activations = defaultdict(lambda: {"positive": [], "negative": []})

        for pair in pairs_list:
            # Extract positive activations
            pos_layers = pair['positive_response'].get('layers_activations', {})
            for layer_str, activation_list in pos_layers.items():
                if activation_list is not None:
                    tensor = torch.tensor(activation_list, dtype=torch.float32)
                    layer_activations[layer_str]["positive"].append(tensor)

            # Extract negative activations
            neg_layers = pair['negative_response'].get('layers_activations', {})
            for layer_str, activation_list in neg_layers.items():
                if activation_list is not None:
                    tensor = torch.tensor(activation_list, dtype=torch.float32)
                    layer_activations[layer_str]["negative"].append(tensor)

        available_layers = sorted(layer_activations.keys(), key=int)
        print(f"   ✓ Found activations for {len(available_layers)} layers: {available_layers}")

        # 3. Create steering method instance
        print(f"\n🧠 Initializing {args.method.upper()} steering method...")

        if args.method.lower() == "caa":
            method = CAAMethod(kwargs={"normalize": args.normalize})
        else:
            raise ValueError(f"Unknown method: {args.method}")

        print(f"   ✓ Method initialized (normalize={args.normalize})")

        # 4. Generate steering vectors for each layer
        print(f"\n⚡ Generating steering vectors...")
        steering_vectors = {}

        for layer_str in available_layers:
            pos_list = layer_activations[layer_str]["positive"]
            neg_list = layer_activations[layer_str]["negative"]

            if args.verbose:
                print(f"   Processing layer {layer_str}: {len(pos_list)} positive, {len(neg_list)} negative")

            if not pos_list or not neg_list:
                print(f"   ⚠️  Skipping layer {layer_str}: missing activations")
                continue

            # Generate steering vector for this layer
            vector = method.train_for_layer(pos_list, neg_list)
            steering_vectors[layer_str] = vector.tolist()  # Convert to list for JSON

        print(f"   ✓ Generated {len(steering_vectors)} steering vectors")

        # 5. Save steering vectors (format depends on file extension)
        print(f"\n💾 Saving steering vectors to '{args.output}'...")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)

        if args.output.endswith('.pt'):
            # For .pt format: save single-layer vectors for multi-steer compatibility
            # If multiple layers, save the first one (or could save all and let user specify)
            if len(steering_vectors) == 1:
                layer_str = list(steering_vectors.keys())[0]
                vector_tensor = torch.tensor(steering_vectors[layer_str], dtype=torch.float32)
                torch.save({
                    'steering_vector': vector_tensor,
                    'layer_index': int(layer_str),
                    'trait_label': trait_label,
                    'model': model,
                    'method': args.method,
                    'normalize': args.normalize,
                    'token_aggregation': token_aggregation,
                    'num_pairs': len(pairs_list),
                    # Legacy keys for backward compatibility
                    'vector': vector_tensor,
                    'layer': int(layer_str),
                }, args.output)
                print(f"   ✓ Saved steering vector (layer {layer_str}) to: {args.output}")
            else:
                # Save multiple layers - save each to separate file
                for layer_str in steering_vectors.keys():
                    layer_output = args.output.replace('.pt', f'_layer_{layer_str}.pt')
                    vector_tensor = torch.tensor(steering_vectors[layer_str], dtype=torch.float32)
                    torch.save({
                        'steering_vector': vector_tensor,
                        'layer_index': int(layer_str),
                        'trait_label': trait_label,
                        'model': model,
                        'method': args.method,
                        'normalize': args.normalize,
                        'token_aggregation': token_aggregation,
                        'num_pairs': len(pairs_list),
                        # Legacy keys
                        'vector': vector_tensor,
                        'layer': int(layer_str),
                    }, layer_output)
                    print(f"   ✓ Saved steering vector (layer {layer_str}) to: {layer_output}")
        else:
            # JSON format: save all layers together
            output_data = {
                'trait_label': trait_label,
                'model': model,
                'method': args.method,
                'normalize': args.normalize,
                'token_aggregation': token_aggregation,
                'num_pairs': len(pairs_list),
                'layers': list(steering_vectors.keys()),
                'steering_vectors': steering_vectors,
                'metadata': {
                    'source_file': args.enriched_pairs_file,
                    'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                }
            }

            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)

            print(f"   ✓ Saved steering vectors to: {args.output}")

        # 6. Display statistics
        print(f"\n📈 Steering Vector Statistics:")
        for layer_str in sorted(steering_vectors.keys(), key=int):
            vector = torch.tensor(steering_vectors[layer_str])
            norm = torch.linalg.norm(vector).item()
            print(f"   Layer {layer_str}: dim={len(vector)}, norm={norm:.4f}")

        if args.timing:
            elapsed = time.time() - start_time
            print(f"   ⏱️  Total time: {elapsed:.2f}s")

        print(f"\n✅ Steering vector creation completed successfully!\\n")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

"""Create steering vector command execution logic."""

import sys
import json
import os
import time
import torch
from collections import defaultdict

from wisent.core.errors import SteeringMethodUnknownError, VectorQualityTooLowError
from wisent.core.utils.device import preferred_dtype


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
        dtype = preferred_dtype()

        for pair in pairs_list:
            # Extract positive activations
            pos_layers = pair['positive_response'].get('layers_activations', {})
            for layer_str, activation_list in pos_layers.items():
                if activation_list is not None:
                    tensor = torch.tensor(activation_list, dtype=dtype)
                    layer_activations[layer_str]["positive"].append(tensor)

            # Extract negative activations
            neg_layers = pair['negative_response'].get('layers_activations', {})
            for layer_str, activation_list in neg_layers.items():
                if activation_list is not None:
                    tensor = torch.tensor(activation_list, dtype=dtype)
                    layer_activations[layer_str]["negative"].append(tensor)

        available_layers = sorted(layer_activations.keys(), key=int)
        print(f"   ✓ Found activations for {len(available_layers)} layers: {available_layers}")

        # 3. Create steering method instance
        print(f"\n🧠 Initializing {args.method.upper()} steering method...")

        # Check if we have optimal config from parent pipeline
        optimal_config = getattr(args, '_optimal_config', None)
        
        method_name = args.method.lower()
        
        if method_name == "caa":
            method = CAAMethod(kwargs={"normalize": args.normalize})
            print(f"   ✓ Method initialized (normalize={args.normalize})")
        elif method_name == "prism":
            from wisent.core.steering_methods.methods.prism import PRISMMethod
            prism_params = {
                "num_directions": getattr(args, 'num_directions', 3),
                "auto_num_directions": getattr(args, 'auto_num_directions', False),
                "use_universal_basis_init": getattr(args, 'use_universal_basis_init', False),
            }
            if optimal_config:
                prism_params.update({
                    "num_directions": optimal_config.get("num_directions", prism_params["num_directions"]),
                    "direction_weighting": optimal_config.get("direction_weighting", "primary_only"),
                    "retain_weight": optimal_config.get("retain_weight", 0.0),
                })
                print(f"   Using optimal PRISM params: num_directions={prism_params['num_directions']}, weighting={prism_params.get('direction_weighting', 'primary_only')}")
            if prism_params["auto_num_directions"]:
                print(f"   Using auto_num_directions (Universal Subspace)")
            if prism_params["use_universal_basis_init"]:
                print(f"   Using universal basis initialization")
            method = PRISMMethod(**prism_params)
            print(f"   ✓ PRISM method initialized")
        elif method_name == "pulse":
            from wisent.core.steering_methods.methods.pulse import PULSEMethod
            pulse_params = {}
            if optimal_config:
                pulse_params = {
                    "sensor_layer": optimal_config.get("sensor_layer", -1),
                    "condition_threshold": optimal_config.get("condition_threshold", 0.5),
                    "gate_temperature": optimal_config.get("gate_temperature", 0.5),
                }
                print(f"   Using optimal PULSE params: threshold={pulse_params['condition_threshold']}, temp={pulse_params['gate_temperature']}")
            method = PULSEMethod(**pulse_params)
            print(f"   ✓ PULSE method initialized")
        elif method_name == "titan":
            from wisent.core.steering_methods.methods.titan import TITANMethod
            titan_params = {}
            if optimal_config:
                titan_params = {
                    "num_directions": optimal_config.get("num_directions", 3),
                    "gate_hidden_dim": optimal_config.get("gate_hidden_dim", 64),
                    "intensity_hidden_dim": optimal_config.get("intensity_hidden_dim", 32),
                }
                print(f"   Using optimal TITAN params: num_directions={titan_params['num_directions']}, gate_hidden={titan_params['gate_hidden_dim']}")
            method = TITANMethod(**titan_params)
            print(f"   ✓ TITAN method initialized")
        else:
            raise SteeringMethodUnknownError(method=args.method)

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

        # 4b. Run quality diagnostics if we have enough pairs
        accept_low_quality = getattr(args, 'accept_low_quality_vector', False)
        quality_metadata = None
        
        if len(pairs_list) >= 5:
            try:
                from wisent.core.contrastive_pairs.diagnostics import run_vector_quality_diagnostics
                
                # Use activations from first layer for quality analysis
                first_layer = available_layers[0] if available_layers else None
                if first_layer:
                    pos_tensors = layer_activations[first_layer]["positive"]
                    neg_tensors = layer_activations[first_layer]["negative"]
                    
                    if len(pos_tensors) >= 5 and len(neg_tensors) >= 5:
                        pos_stacked = torch.stack(pos_tensors)
                        neg_stacked = torch.stack(neg_tensors)
                        prompts = [p.get('prompt', '') for p in pairs_list]
                        
                        quality_report, diagnostics_report = run_vector_quality_diagnostics(
                            pos_stacked, neg_stacked, prompts
                        )
                        
                        print(f"\n📊 Vector Quality Analysis:")
                        print(f"   Overall quality: {quality_report.overall_quality.upper()}")
                        if quality_report.convergence_score is not None:
                            print(f"   Convergence: {quality_report.convergence_score:.3f}")
                        if quality_report.cv_score_mean is not None:
                            print(f"   Cross-validation: {quality_report.cv_score_mean:.3f}")
                        if quality_report.snr is not None:
                            print(f"   Signal-to-noise: {quality_report.snr:.2f}")
                        if quality_report.pca_pc1_variance is not None:
                            print(f"   PCA PC1 variance: {quality_report.pca_pc1_variance*100:.1f}%")
                        if quality_report.held_out_transfer is not None:
                            print(f"   Held-out transfer: {quality_report.held_out_transfer:.3f}")
                        if quality_report.cv_classification_accuracy is not None:
                            print(f"   CV classification: {quality_report.cv_classification_accuracy:.3f}")
                        if quality_report.cohens_d is not None:
                            print(f"   Cohen's d: {quality_report.cohens_d:.2f}")
                        
                        # Show issues
                        if diagnostics_report.issues:
                            print(f"\n⚠️  Quality Issues:")
                            for issue in diagnostics_report.issues:
                                marker = "❌" if issue.severity == "critical" else "⚠️"
                                print(f"   {marker} [{issue.severity}] {issue.message}")
                        
                        # Show recommendations
                        if quality_report.recommendations:
                            print(f"\n💡 Recommendations:")
                            for rec in quality_report.recommendations:
                                print(f"   • {rec}")
                        
                        # Store for metadata (convert numpy types to Python types for JSON serialization)
                        quality_metadata = {
                            "overall_quality": quality_report.overall_quality,
                            "convergence_score": float(quality_report.convergence_score) if quality_report.convergence_score is not None else None,
                            "cv_score_mean": float(quality_report.cv_score_mean) if quality_report.cv_score_mean is not None else None,
                            "snr": float(quality_report.snr) if quality_report.snr is not None else None,
                            "pca_pc1_variance": float(quality_report.pca_pc1_variance) if quality_report.pca_pc1_variance is not None else None,
                            "silhouette_score": float(quality_report.silhouette_score) if quality_report.silhouette_score is not None else None,
                            "held_out_transfer": float(quality_report.held_out_transfer) if quality_report.held_out_transfer is not None else None,
                            "cv_classification_accuracy": float(quality_report.cv_classification_accuracy) if quality_report.cv_classification_accuracy is not None else None,
                            "cohens_d": float(quality_report.cohens_d) if quality_report.cohens_d is not None else None,
                            "num_outlier_pairs": len(quality_report.outlier_pairs),
                        }
                        
                        # Raise error if quality is poor and not accepted
                        if diagnostics_report.has_critical_issues and not accept_low_quality:
                            critical_issues = [i for i in diagnostics_report.issues if i.severity == "critical"]
                            reason = "; ".join(i.message for i in critical_issues[:3])
                            raise VectorQualityTooLowError(
                                quality=quality_report.overall_quality,
                                reason=reason,
                                details=quality_metadata
                            )
            except ImportError:
                print(f"\n⚠️  sklearn not available, skipping quality diagnostics")

        # 5. Save steering vectors (format depends on file extension)
        print(f"\n💾 Saving steering vectors to '{args.output}'...")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)

        if args.output.endswith('.pt'):
            # For .pt format: save single-layer vectors for multi-steer compatibility
            # If multiple layers, save the first one (or could save all and let user specify)
            if len(steering_vectors) == 1:
                layer_str = list(steering_vectors.keys())[0]
                vector_tensor = torch.tensor(steering_vectors[layer_str], dtype=dtype)
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
                    vector_tensor = torch.tensor(steering_vectors[layer_str], dtype=dtype)
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
            
            if quality_metadata:
                output_data['vector_quality'] = quality_metadata

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

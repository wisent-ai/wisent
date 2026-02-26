"""
Create steering object command - produces full steering objects with all method-specific components.

Unlike create_steering_vector which flattens everything to simple vectors,
this preserves the full structure: gates, intensity networks, multi-directions, etc.
"""

import sys
import json
import os
import time
import torch
from collections import defaultdict
from datetime import datetime

from wisent.core.steering_methods._steering_object_base import (
    SteeringObjectMetadata,
    BaseSteeringObject,
)
from wisent.core.utils import preferred_dtype
from wisent.core.constants import (
    MLP_DROPOUT,
    MLP_HIDDEN_DIM,
    MLP_LEARNING_RATE,
    MLP_NUM_LAYERS,
    DEFAULT_OPTIMIZATION_STEPS,
    DEFAULT_WEIGHT_DECAY,
    OSTRZE_C,
)



from wisent.core.cli.steering.core.create_steering_helpers import (
    _create_tecza_steering_object,
    _create_tetno_steering_object,
)
from wisent.core.cli.steering.core.create_steering_grom import (
    _create_grom_steering_object,
)


def _parse_layer_spec(spec: str, num_layers: int) -> set:
    """
    Parse layer specification string into a set of layer indices.

    Supports:
    - Single layer: "16"
    - Comma-separated: "12,14,16"
    - Range: "12-18"
    - Mixed: "5,10-15,20"
    """
    layers = set()
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            # Range
            start, end = part.split('-')
            layers.update(range(int(start), int(end) + 1))
        else:
            # Single layer
            layers.add(int(part))
    return layers


def execute_create_steering_object(args):
    """Create a full steering object from enriched pairs."""
    
    print(f"\n🎯 Creating steering object from enriched pairs")
    print(f"   Input file: {args.enriched_pairs_file}")
    print(f"   Method: {args.method}")
    
    start_time = time.time() if getattr(args, 'timing', False) else None
    
    try:
        # 1. Load enriched pairs
        print(f"\n📂 Loading enriched pairs...")
        if not os.path.exists(args.enriched_pairs_file):
            raise FileNotFoundError(f"File not found: {args.enriched_pairs_file}")
        
        with open(args.enriched_pairs_file, 'r') as f:
            data = json.load(f)
        
        trait_label = data.get('trait_label', 'unknown')
        model = data.get('model', 'unknown')
        layers = data.get('layers', [])
        token_aggregation = data.get('token_aggregation', 'unknown')
        extraction_component = data.get('extraction_component', 'residual_stream')
        pairs_list = data.get('pairs', [])
        
        print(f"   ✓ Loaded {len(pairs_list)} pairs")
        print(f"   ✓ Model: {model}")
        print(f"   ✓ Layers: {layers}")
        
        # 2. Organize activations by layer
        print(f"\n📊 Organizing activations...")
        dtype = preferred_dtype()
        layer_activations = defaultdict(lambda: {"positive": [], "negative": []})
        
        for pair in pairs_list:
            pos_layers = pair['positive_response'].get('layers_activations', {})
            for layer_str, activation_list in pos_layers.items():
                if activation_list is not None:
                    tensor = torch.tensor(activation_list, dtype=dtype)
                    layer_activations[layer_str]["positive"].append(tensor)
            
            neg_layers = pair['negative_response'].get('layers_activations', {})
            for layer_str, activation_list in neg_layers.items():
                if activation_list is not None:
                    tensor = torch.tensor(activation_list, dtype=dtype)
                    layer_activations[layer_str]["negative"].append(tensor)
            # Load Q/K projections: Q from negative (source queries), K from positive (target keys)
            for layer_str, q_val in pair['negative_response'].get('q_proj_activations', {}).items():
                if q_val is not None:
                    layer_activations[layer_str].setdefault("q_proj_activations", []).append(
                        torch.tensor(q_val, dtype=dtype))
            for layer_str, k_val in pair['positive_response'].get('k_proj_activations', {}).items():
                if k_val is not None:
                    layer_activations[layer_str].setdefault("k_proj_activations", []).append(
                        torch.tensor(k_val, dtype=dtype))
        all_layers = sorted(layer_activations.keys(), key=lambda x: int(x))
        hidden_dim = layer_activations[all_layers[0]]["positive"][0].shape[-1]

        print(f"   ✓ Found {len(all_layers)} layers, hidden_dim={hidden_dim}")

        # Filter layers if --layer is specified
        if getattr(args, 'layer', None):
            target_layers = _parse_layer_spec(str(args.layer), len(all_layers))
            available_layers = [l for l in all_layers if int(l) in target_layers]
            if not available_layers:
                raise ValueError(f"No matching layers found. Specified: {args.layer}, Available: {all_layers}")
            print(f"   ✓ Filtered to {len(available_layers)} layers: {available_layers}")
        else:
            available_layers = all_layers
        
        # 3. Create metadata
        # Parse category/benchmark from trait_label if possible
        parts = trait_label.split('/')
        category = parts[0] if len(parts) > 1 else 'unknown'
        benchmark = parts[-1]
        
        # Get calibration norms if available
        calibration_norms_raw = data.get('calibration_norms', {})
        calibration_norms = {int(k): float(v) for k, v in calibration_norms_raw.items()}
        
        num_attention_heads = data.get('num_attention_heads')
        num_kv_heads = data.get('num_key_value_heads')
        metadata = SteeringObjectMetadata(
            method=args.method.lower(),
            model_name=model,
            benchmark=benchmark,
            category=category,
            extraction_strategy=token_aggregation,
            num_pairs=len(pairs_list),
            layers=[int(l) for l in available_layers],
            hidden_dim=hidden_dim,
            created_at=datetime.now().isoformat(),
            calibration_norms=calibration_norms,
            extraction_component=extraction_component,
            extra={'num_attention_heads': num_attention_heads, 'num_key_value_heads': num_kv_heads},
        )
        
        # 4. Create steering object based on method
        method_name = args.method.lower()
        print(f"\n🧠 Creating {method_name.upper()} steering object...")
        
        if method_name in ('caa', 'ostrze', 'mlp'):
            steering_obj = _create_simple_steering_object(
                method_name, metadata, layer_activations, available_layers, args
            )
        elif method_name == 'tecza':
            steering_obj = _create_tecza_steering_object(
                metadata, layer_activations, available_layers, args
            )
        elif method_name == 'tetno':
            steering_obj = _create_tetno_steering_object(
                metadata, layer_activations, available_layers, args
            )
        elif method_name == 'grom':
            steering_obj = _create_grom_steering_object(
                metadata, layer_activations, available_layers, args
            )
        elif method_name == 'nurt':
            from wisent.core.cli.steering.core.create_nurt import _create_nurt_steering_object
            steering_obj = _create_nurt_steering_object(
                metadata, layer_activations, available_layers, args
            )
        elif method_name == 'szlak':
            from wisent.core.steering_methods.methods.szlak.create import _create_szlak_steering_object
            steering_obj = _create_szlak_steering_object(
                metadata, layer_activations, available_layers, args
            )
        elif method_name == 'wicher':
            from wisent.core.steering_methods.methods.wicher.create import _create_wicher_steering_object
            steering_obj = _create_wicher_steering_object(
                metadata, layer_activations, available_layers, args
            )
        elif method_name == 'przelom':
            from wisent.core.steering_methods.methods.przelom.create import _create_przelom_steering_object
            steering_obj = _create_przelom_steering_object(
                metadata, layer_activations, available_layers, args
            )
        else:
            raise ValueError(f"Unknown method: {args.method}")
        
        # 5. Save steering object
        print(f"\n💾 Saving steering object to '{args.output}'...")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
        steering_obj.save(args.output)
        print(f"   ✓ Saved steering object")
        
        # 6. Summary
        print(f"\n📈 Steering Object Summary:")
        print(f"   Method: {steering_obj.method_name}")
        print(f"   Layers: {metadata.layers}")
        print(f"   Hidden dim: {metadata.hidden_dim}")
        print(f"   Num pairs: {metadata.num_pairs}")
        
        if start_time:
            print(f"   ⏱️  Time: {time.time() - start_time:.2f}s")
        
        print(f"\n✅ Steering object created successfully!\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        raise


def _create_simple_steering_object(
    method_name: str,
    metadata: SteeringObjectMetadata,
    layer_activations: dict,
    available_layers: list,
    args,
) -> BaseSteeringObject:
    """Create CAA, Ostrze, or MLP steering object."""
    
    # Get method class
    if method_name == 'caa':
        from wisent.core.steering_methods.methods.caa import CAAMethod
        from wisent.core.steering_methods._steering_object_simple import CAASteeringObject
        method = CAAMethod(normalize=getattr(args, 'normalize', True))
        obj_class = CAASteeringObject
    elif method_name == 'ostrze':
        from wisent.core.steering_methods.methods.ostrze import OstrzeMethod
        from wisent.core.steering_methods._steering_object_simple import OstrzeSteeringObject
        method = OstrzeMethod(
            normalize=getattr(args, 'normalize', True),
            C=getattr(args, 'ostrze_C', OSTRZE_C),
        )
        obj_class = OstrzeSteeringObject
    elif method_name == 'mlp':
        from wisent.core.steering_methods.methods.mlp import MLPMethod
        from wisent.core.steering_methods._steering_object_simple import MLPSteeringObject
        method = MLPMethod(
            normalize=getattr(args, 'normalize', True),
            hidden_dim=getattr(args, 'mlp_hidden_dim', MLP_HIDDEN_DIM),
            num_layers=getattr(args, 'mlp_num_layers', MLP_NUM_LAYERS),
            dropout=getattr(args, 'mlp_dropout', MLP_DROPOUT),
            epochs=getattr(args, 'mlp_epochs', DEFAULT_OPTIMIZATION_STEPS),
            learning_rate=getattr(args, 'mlp_learning_rate', MLP_LEARNING_RATE),
            weight_decay=getattr(args, 'mlp_weight_decay', DEFAULT_WEIGHT_DECAY),
        )
        obj_class = MLPSteeringObject
    else:
        raise ValueError(f"Unknown simple method: {method_name}")
    
    # Train vectors for each layer
    vectors = {}
    for layer_str in available_layers:
        pos_list = layer_activations[layer_str]["positive"]
        neg_list = layer_activations[layer_str]["negative"]
        
        if not pos_list or not neg_list:
            continue
        
        vector = method.train_for_layer(pos_list, neg_list)
        vectors[int(layer_str)] = vector
        print(f"   Layer {layer_str}: norm={vector.norm().item():.4f}")
    
    return obj_class(metadata=metadata, vectors=vectors)



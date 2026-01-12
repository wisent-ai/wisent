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

from wisent.core.steering_methods.steering_object import (
    SteeringObjectMetadata,
    CAASteeringObject,
    HyperplaneSteeringObject,
    MLPSteeringObject,
    PRISMSteeringObject,
    PULSESteeringObject,
    TITANSteeringObject,
    BaseSteeringObject,
)
from wisent.core.utils.device import preferred_dtype


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
        
        available_layers = sorted(layer_activations.keys(), key=lambda x: int(x))
        hidden_dim = layer_activations[available_layers[0]]["positive"][0].shape[-1]
        
        print(f"   ✓ Found {len(available_layers)} layers, hidden_dim={hidden_dim}")
        
        # 3. Create metadata
        # Parse category/benchmark from trait_label if possible
        parts = trait_label.split('/')
        category = parts[0] if len(parts) > 1 else 'unknown'
        benchmark = parts[-1]
        
        # Get calibration norms if available
        calibration_norms_raw = data.get('calibration_norms', {})
        calibration_norms = {int(k): float(v) for k, v in calibration_norms_raw.items()}
        
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
        )
        
        # 4. Create steering object based on method
        method_name = args.method.lower()
        print(f"\n🧠 Creating {method_name.upper()} steering object...")
        
        if method_name in ('caa', 'hyperplane', 'mlp'):
            steering_obj = _create_simple_steering_object(
                method_name, metadata, layer_activations, available_layers, args
            )
        elif method_name == 'prism':
            steering_obj = _create_prism_steering_object(
                metadata, layer_activations, available_layers, args
            )
        elif method_name == 'pulse':
            steering_obj = _create_pulse_steering_object(
                metadata, layer_activations, available_layers, args
            )
        elif method_name == 'titan':
            steering_obj = _create_titan_steering_object(
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
        sys.exit(1)


def _create_simple_steering_object(
    method_name: str,
    metadata: SteeringObjectMetadata,
    layer_activations: dict,
    available_layers: list,
    args,
) -> BaseSteeringObject:
    """Create CAA, Hyperplane, or MLP steering object."""
    
    # Get method class
    if method_name == 'caa':
        from wisent.core.steering_methods.methods.caa import CAAMethod
        method = CAAMethod(normalize=getattr(args, 'normalize', True))
        obj_class = CAASteeringObject
    elif method_name == 'hyperplane':
        from wisent.core.steering_methods.methods.hyperplane import HyperplaneMethod
        method = HyperplaneMethod(
            normalize=getattr(args, 'normalize', True),
            max_iter=getattr(args, 'hyperplane_max_iter', 1000),
            C=getattr(args, 'hyperplane_C', 1.0),
        )
        obj_class = HyperplaneSteeringObject
    elif method_name == 'mlp':
        from wisent.core.steering_methods.methods.mlp import MLPMethod
        method = MLPMethod(
            normalize=getattr(args, 'normalize', True),
            hidden_dim=getattr(args, 'mlp_hidden_dim', 256),
            num_layers=getattr(args, 'mlp_num_layers', 2),
            dropout=getattr(args, 'mlp_dropout', 0.1),
            epochs=getattr(args, 'mlp_epochs', 100),
            learning_rate=getattr(args, 'mlp_learning_rate', 0.001),
            weight_decay=getattr(args, 'mlp_weight_decay', 0.01),
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


def _create_prism_steering_object(
    metadata: SteeringObjectMetadata,
    layer_activations: dict,
    available_layers: list,
    args,
) -> PRISMSteeringObject:
    """Create PRISM steering object with multiple directions."""
    from wisent.core.steering_methods.methods.prism import PRISMMethod
    
    num_directions = getattr(args, 'prism_num_directions', 3)
    
    method = PRISMMethod(
        num_directions=num_directions,
        optimization_steps=getattr(args, 'prism_optimization_steps', 100),
        learning_rate=getattr(args, 'prism_learning_rate', 0.01),
        normalize=getattr(args, 'normalize', True),
    )
    
    directions = {}
    direction_weights = {}
    
    for layer_str in available_layers:
        pos_list = layer_activations[layer_str]["positive"]
        neg_list = layer_activations[layer_str]["negative"]
        
        if not pos_list or not neg_list:
            continue
        
        # Stack activations
        pos_tensor = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
        neg_tensor = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)
        
        # Train directions
        layer_dirs, meta = method._train_layer_directions(pos_tensor, neg_tensor, layer_str)
        
        layer_int = int(layer_str)
        directions[layer_int] = layer_dirs
        # Equal weights by default
        direction_weights[layer_int] = torch.ones(layer_dirs.shape[0]) / layer_dirs.shape[0]
        
        print(f"   Layer {layer_str}: {layer_dirs.shape[0]} directions, avg_cosine={meta.get('avg_cosine_similarity', 0):.3f}")
    
    return PRISMSteeringObject(
        metadata=metadata,
        directions=directions,
        direction_weights=direction_weights,
        primary_only=False,
    )


def _create_pulse_steering_object(
    metadata: SteeringObjectMetadata,
    layer_activations: dict,
    available_layers: list,
    args,
) -> PULSESteeringObject:
    """Create PULSE steering object with conditional gating."""
    from wisent.core.steering_methods.methods.pulse import PULSEMethod
    
    # Determine sensor layer (default: 75% through network)
    num_layers = len(available_layers)
    sensor_layer_idx = getattr(args, 'pulse_sensor_layer', None)
    if sensor_layer_idx is None:
        sensor_layer_idx = int(num_layers * 0.75)
    sensor_layer = int(available_layers[min(sensor_layer_idx, num_layers - 1)])
    
    method = PULSEMethod(
        sensor_layer=sensor_layer,
        condition_threshold=getattr(args, 'pulse_condition_threshold', 0.5),
        gate_temperature=getattr(args, 'pulse_gate_temperature', 0.1),
        learn_threshold=getattr(args, 'pulse_learn_threshold', True),
        normalize=getattr(args, 'normalize', True),
    )
    
    # Train behavior vectors
    behavior_vectors = {}
    layer_scales = {}
    
    for layer_str in available_layers:
        pos_list = layer_activations[layer_str]["positive"]
        neg_list = layer_activations[layer_str]["negative"]
        
        if not pos_list or not neg_list:
            continue
        
        pos_tensor = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
        neg_tensor = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)
        
        # CAA for behavior vector
        behavior_vec = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
        behavior_vec = torch.nn.functional.normalize(behavior_vec, dim=0)
        
        layer_int = int(layer_str)
        behavior_vectors[layer_int] = behavior_vec
        
        # Compute layer scale based on separation
        pos_proj = (pos_tensor * behavior_vec).sum(dim=1).mean()
        neg_proj = (neg_tensor * behavior_vec).sum(dim=1).mean()
        separation = (pos_proj - neg_proj).item()
        layer_scales[layer_int] = max(0.1, separation)
        
        print(f"   Layer {layer_str}: separation={separation:.3f}")
    
    # Normalize scales
    if layer_scales:
        mean_scale = sum(layer_scales.values()) / len(layer_scales)
        if mean_scale > 0:
            layer_scales = {k: v / mean_scale for k, v in layer_scales.items()}
    
    # Train condition vector at sensor layer
    sensor_layer_str = str(sensor_layer)
    pos_sensor = torch.stack([t.detach().float().reshape(-1) for t in layer_activations[sensor_layer_str]["positive"]], dim=0)
    neg_sensor = torch.stack([t.detach().float().reshape(-1) for t in layer_activations[sensor_layer_str]["negative"]], dim=0)
    
    # Initialize with CAA direction
    condition_vec = pos_sensor.mean(dim=0) - neg_sensor.mean(dim=0)
    condition_vec = torch.nn.functional.normalize(condition_vec, dim=0)
    
    # Optimize condition vector
    condition_vec = condition_vec.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([condition_vec], lr=0.01)
    
    for _ in range(50):
        optimizer.zero_grad()
        c_norm = torch.nn.functional.normalize(condition_vec, dim=0)
        pos_norm = torch.nn.functional.normalize(pos_sensor, dim=1)
        neg_norm = torch.nn.functional.normalize(neg_sensor, dim=1)
        
        pos_sim = (pos_norm * c_norm).sum(dim=1)
        neg_sim = (neg_norm * c_norm).sum(dim=1)
        
        loss = torch.relu(0.5 - pos_sim).mean() + torch.relu(neg_sim + 0.5).mean()
        loss.backward()
        optimizer.step()
    
    condition_vec = torch.nn.functional.normalize(condition_vec.detach(), dim=0)
    
    # Find optimal threshold
    pos_norm = torch.nn.functional.normalize(pos_sensor, dim=1)
    neg_norm = torch.nn.functional.normalize(neg_sensor, dim=1)
    pos_sims = (pos_norm * condition_vec).sum(dim=1)
    neg_sims = (neg_norm * condition_vec).sum(dim=1)
    
    best_threshold, best_acc = 0.5, 0.0
    for t in torch.linspace(pos_sims.min(), pos_sims.max(), 20):
        acc = ((pos_sims > t).float().mean() + (neg_sims <= t).float().mean()) / 2
        if acc > best_acc:
            best_acc, best_threshold = acc.item(), t.item()
    
    print(f"   Sensor layer: {sensor_layer}")
    print(f"   Optimal threshold: {best_threshold:.3f} (accuracy: {best_acc:.3f})")
    
    return PULSESteeringObject(
        metadata=metadata,
        behavior_vectors=behavior_vectors,
        condition_vector=condition_vec,
        sensor_layer=sensor_layer,
        threshold=best_threshold,
        layer_scales=layer_scales,
        gate_temperature=getattr(args, 'pulse_gate_temperature', 0.1),
    )


def _create_titan_steering_object(
    metadata: SteeringObjectMetadata,
    layer_activations: dict,
    available_layers: list,
    args,
) -> TITANSteeringObject:
    """Create TITAN steering object with learned networks."""
    from wisent.core.steering_methods.methods.titan import TITANMethod
    from wisent.core.steering_methods.steering_object import TITANGateNetwork, TITANIntensityNetwork
    
    num_directions = getattr(args, 'titan_num_directions', 5)
    hidden_dim = metadata.hidden_dim
    num_layers = len(available_layers)
    
    # Determine sensor layer
    sensor_layer_idx = getattr(args, 'titan_sensor_layer', None)
    if sensor_layer_idx is None:
        sensor_layer_idx = int(num_layers * 0.75)
    sensor_layer = int(available_layers[min(sensor_layer_idx, num_layers - 1)])
    
    gate_hidden_dim = getattr(args, 'titan_gate_hidden_dim', None)
    if gate_hidden_dim is None:
        gate_hidden_dim = max(32, min(512, hidden_dim // 16))
    intensity_hidden_dim = getattr(args, 'titan_intensity_hidden_dim', None)
    if intensity_hidden_dim is None:
        intensity_hidden_dim = max(16, min(256, hidden_dim // 32))
    max_alpha = getattr(args, 'titan_max_alpha', None)
    if max_alpha is None:
        max_alpha = 3.0
    
    # Initialize networks
    gate_network = TITANGateNetwork(hidden_dim, gate_hidden_dim)
    intensity_network = TITANIntensityNetwork(hidden_dim, num_layers, intensity_hidden_dim, max_alpha)
    
    # Prepare data
    layer_order = [int(l) for l in available_layers]
    
    # Initialize directions and weights
    directions = {}
    direction_weights = {}
    
    # Collect all data
    all_pos = {}
    all_neg = {}
    
    for layer_str in available_layers:
        pos_list = layer_activations[layer_str]["positive"]
        neg_list = layer_activations[layer_str]["negative"]
        
        if not pos_list or not neg_list:
            continue
        
        pos_tensor = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
        neg_tensor = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)
        
        layer_int = int(layer_str)
        all_pos[layer_int] = pos_tensor
        all_neg[layer_int] = neg_tensor
        
        # Initialize directions with CAA + perturbations
        caa_dir = torch.nn.functional.normalize(pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0), dim=0)
        dirs = torch.randn(num_directions, hidden_dim)
        dirs[0] = caa_dir
        for i in range(1, num_directions):
            dirs[i] = torch.nn.functional.normalize(caa_dir + torch.randn(hidden_dim) * 0.3, dim=0)
        
        directions[layer_int] = torch.nn.Parameter(dirs)
        direction_weights[layer_int] = torch.nn.Parameter(torch.zeros(num_directions))
    
    # Joint optimization
    all_params = list(directions.values()) + list(direction_weights.values())
    all_params.extend(gate_network.parameters())
    all_params.extend(intensity_network.parameters())
    
    optimizer = torch.optim.AdamW(all_params, lr=0.005, weight_decay=0.01)
    
    sensor_pos = all_pos[sensor_layer]
    sensor_neg = all_neg[sensor_layer]
    
    print(f"   Training TITAN ({num_directions} directions, {len(layer_order)} layers)...")
    
    for step in range(200):
        optimizer.zero_grad()
        
        total_loss = torch.tensor(0.0)
        
        # Gate loss
        pos_gate = gate_network(sensor_pos)
        neg_gate = gate_network(sensor_neg)
        gate_loss = torch.relu(0.5 - pos_gate).mean() + torch.relu(neg_gate - 0.5).mean()
        total_loss = total_loss + gate_loss
        
        # Per-layer losses
        for i, layer in enumerate(layer_order):
            if layer not in directions:
                continue
            
            dirs = torch.nn.functional.normalize(directions[layer], dim=1)
            weights = torch.softmax(direction_weights[layer], dim=0)
            effective_dir = (weights.unsqueeze(-1) * dirs).sum(dim=0)
            
            pos_data = all_pos[layer]
            neg_data = all_neg[layer]
            
            # Behavior loss
            pos_proj = (pos_data * effective_dir).sum(dim=1)
            behavior_loss = torch.relu(1.0 - pos_proj).mean()
            
            # Retain loss
            neg_proj = (neg_data * effective_dir).sum(dim=1).abs()
            retain_loss = neg_proj.mean() * 0.2
            
            total_loss = total_loss + behavior_loss + retain_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()
        
        if step % 50 == 0:
            print(f"      Step {step}: loss={total_loss.item():.4f}")
    
    # Finalize directions
    final_directions = {}
    final_weights = {}
    
    for layer in layer_order:
        if layer in directions:
            final_directions[layer] = torch.nn.functional.normalize(directions[layer].detach(), dim=1)
            final_weights[layer] = torch.softmax(direction_weights[layer].detach(), dim=0)
    
    print(f"   Sensor layer: {sensor_layer}")
    print(f"   Final gate accuracy: pos={pos_gate.mean().item():.3f}, neg={neg_gate.mean().item():.3f}")
    
    return TITANSteeringObject(
        metadata=metadata,
        directions=final_directions,
        direction_weights=final_weights,
        gate_network=gate_network,
        intensity_network=intensity_network,
        layer_order=layer_order,
        gate_temperature=getattr(args, 'titan_gate_temperature', 0.5),
        max_alpha=max_alpha,
    )

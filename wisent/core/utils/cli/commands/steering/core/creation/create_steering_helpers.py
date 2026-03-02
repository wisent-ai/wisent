"""TECZA and TETNO steering object creation helpers."""
from __future__ import annotations
import torch

from wisent.core import constants as _C
from wisent.core.utils.config_tools.constants import (
    DEFAULT_SCORE,
    TECZA_LEARNING_RATE,
    TECZA_NUM_DIRECTIONS,
    DEFAULT_OPTIMIZATION_STEPS,
    TETNO_CONDITION_THRESHOLD,
    TETNO_GATE_TEMPERATURE,
    TECZA_LEARNING_RATE,
    TETNO_THRESHOLD_SEARCH_STEPS,
    THRESHOLD_SEARCH_INIT_VALUE,
)


def _create_tecza_steering_object(
    metadata: SteeringObjectMetadata,
    layer_activations: dict,
    available_layers: list,
    args,
) -> TECZASteeringObject:
    """Create TECZA steering object with multiple directions."""
    from wisent.core.control.steering_methods.methods.advanced import TECZAMethod
    
    num_directions = getattr(args, 'tecza_num_directions', TECZA_NUM_DIRECTIONS)
    
    method = TECZAMethod(
        num_directions=num_directions,
        optimization_steps=getattr(args, 'tecza_optimization_steps', DEFAULT_OPTIMIZATION_STEPS),
        learning_rate=getattr(args, 'tecza_learning_rate', TECZA_LEARNING_RATE),
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
        
        print(f"   Layer {layer_str}: {layer_dirs.shape[0]} directions, avg_cosine={meta.get('avg_cosine_similarity', DEFAULT_SCORE):.3f}")
    
    from wisent.core.control.steering_methods._steering_object_advanced import TECZASteeringObject
    return TECZASteeringObject(
        metadata=metadata,
        directions=directions,
        direction_weights=direction_weights,
        primary_only=False,
    )


def _create_tetno_steering_object(
    metadata: SteeringObjectMetadata,
    layer_activations: dict,
    available_layers: list,
    args,
) -> TETNOSteeringObject:
    """Create TETNO steering object with conditional gating."""
    from wisent.core.control.steering_methods.methods.advanced import TETNOMethod
    
    # Determine sensor layer — use last available layer if not specified
    num_layers = len(available_layers)
    sensor_layer_idx = getattr(args, 'tetno_sensor_layer', None)
    if sensor_layer_idx is None:
        sensor_layer_idx = num_layers - 1
    sensor_layer = int(available_layers[min(sensor_layer_idx, num_layers - 1)])
    
    method = TETNOMethod(
        sensor_layer=sensor_layer,
        condition_threshold=getattr(args, 'tetno_condition_threshold', TETNO_CONDITION_THRESHOLD),
        gate_temperature=getattr(args, 'tetno_gate_temperature', TETNO_GATE_TEMPERATURE),
        learn_threshold=getattr(args, 'tetno_learn_threshold', True),
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
        layer_scales[layer_int] = max(_C.TETNO_MIN_LAYER_SCALE, separation)
        
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
    optimizer = torch.optim.Adam([condition_vec], lr=TECZA_LEARNING_RATE)
    
    for _ in range(_C.TETNO_CONDITION_VEC_OPT_ITERS):
        optimizer.zero_grad()
        c_norm = torch.nn.functional.normalize(condition_vec, dim=0)
        pos_norm = torch.nn.functional.normalize(pos_sensor, dim=1)
        neg_norm = torch.nn.functional.normalize(neg_sensor, dim=1)
        
        pos_sim = (pos_norm * c_norm).sum(dim=1)
        neg_sim = (neg_norm * c_norm).sum(dim=1)
        
        loss = torch.relu(_C.STEERING_LOSS_MARGIN - pos_sim).mean() + torch.relu(neg_sim + _C.STEERING_LOSS_MARGIN).mean()
        loss.backward()
        optimizer.step()
    
    condition_vec = torch.nn.functional.normalize(condition_vec.detach(), dim=0)
    
    # Find optimal threshold
    pos_norm = torch.nn.functional.normalize(pos_sensor, dim=1)
    neg_norm = torch.nn.functional.normalize(neg_sensor, dim=1)
    pos_sims = (pos_norm * condition_vec).sum(dim=1)
    neg_sims = (neg_norm * condition_vec).sum(dim=1)
    
    best_threshold, best_acc = THRESHOLD_SEARCH_INIT_VALUE, 0.0
    for t in torch.linspace(pos_sims.min(), pos_sims.max(), TETNO_THRESHOLD_SEARCH_STEPS):
        acc = ((pos_sims > t).float().mean() + (neg_sims <= t).float().mean()) / 2
        if acc > best_acc:
            best_acc, best_threshold = acc.item(), t.item()
    
    print(f"   Sensor layer: {sensor_layer}")
    print(f"   Optimal threshold: {best_threshold:.3f} (accuracy: {best_acc:.3f})")
    
    from wisent.core.control.steering_methods._steering_object_advanced import TETNOSteeringObject
    return TETNOSteeringObject(
        metadata=metadata,
        behavior_vectors=behavior_vectors,
        condition_vector=condition_vec,
        sensor_layer=sensor_layer,
        threshold=best_threshold,
        layer_scales=layer_scales,
        gate_temperature=getattr(args, 'tetno_gate_temperature', TETNO_GATE_TEMPERATURE),
    )



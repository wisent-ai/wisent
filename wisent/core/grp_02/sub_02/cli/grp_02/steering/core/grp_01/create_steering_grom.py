"""GROM steering object creation helper."""
from __future__ import annotations
import torch


def _create_grom_steering_object(
    metadata: SteeringObjectMetadata,
    layer_activations: dict,
    available_layers: list,
    args,
) -> GROMSteeringObject:
    """Create GROM steering object with learned networks."""
    from wisent.core.steering_methods.methods.grom import GROMMethod
    from wisent.core.steering_methods.steering_object import GROMGateNetwork, GROMIntensityNetwork
    
    num_directions = getattr(args, 'grom_num_directions', 5)
    hidden_dim = metadata.hidden_dim
    num_layers = len(available_layers)
    
    # Determine sensor layer
    sensor_layer_idx = getattr(args, 'grom_sensor_layer', None)
    if sensor_layer_idx is None:
        sensor_layer_idx = int(num_layers * 0.75)
    sensor_layer = int(available_layers[min(sensor_layer_idx, num_layers - 1)])
    
    gate_hidden_dim = getattr(args, 'grom_gate_hidden_dim', None)
    if gate_hidden_dim is None:
        gate_hidden_dim = max(32, min(512, hidden_dim // 16))
    intensity_hidden_dim = getattr(args, 'grom_intensity_hidden_dim', None)
    if intensity_hidden_dim is None:
        intensity_hidden_dim = max(16, min(256, hidden_dim // 32))
    max_alpha = getattr(args, 'grom_max_alpha', None)
    if max_alpha is None:
        max_alpha = 3.0
    
    # Initialize networks
    gate_network = GROMGateNetwork(hidden_dim, gate_hidden_dim)
    intensity_network = GROMIntensityNetwork(hidden_dim, num_layers, intensity_hidden_dim, max_alpha)
    
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
    
    print(f"   Training GROM ({num_directions} directions, {len(layer_order)} layers)...")
    
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
    
    return GROMSteeringObject(
        metadata=metadata,
        directions=final_directions,
        direction_weights=final_weights,
        gate_network=gate_network,
        intensity_network=intensity_network,
        layer_order=layer_order,
        gate_temperature=getattr(args, 'grom_gate_temperature', 0.5),
        max_alpha=max_alpha,
    )

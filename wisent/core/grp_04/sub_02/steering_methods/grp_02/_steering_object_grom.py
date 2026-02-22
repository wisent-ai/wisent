"""GROM steering object with neural gate and intensity networks."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from wisent.core.steering_methods._steering_object_base import (
    BaseSteeringObject, SteeringObjectMetadata, LayerName,
)

class GROMGateNetwork(nn.Module):
    """Serializable gate network for GROM."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, h: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
        if h.dim() == 1:
            h = h.unsqueeze(0)
        logit = self.net(h).squeeze(-1)
        return torch.sigmoid(logit / temperature)


class GROMIntensityNetwork(nn.Module):
    """Serializable intensity network for GROM."""
    
    def __init__(self, input_dim: int, num_layers: int, hidden_dim: int = 64, max_alpha: float = 3.0):
        super().__init__()
        self.max_alpha = max_alpha
        self.num_layers = num_layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_layers),
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() == 1:
            h = h.unsqueeze(0)
        raw = self.net(h)
        return torch.sigmoid(raw) * self.max_alpha


class GROMSteeringObject(BaseSteeringObject):
    """
    GROM steering object with learned gating and intensity networks.
    
    Includes:
    - Multi-directional manifold per layer
    - Learned gate network
    - Learned intensity network
    - Direction weights
    """
    
    method_name = "grom"
    
    def __init__(
        self,
        metadata: SteeringObjectMetadata,
        directions: Dict[int, torch.Tensor],  # layer -> [num_directions, hidden_dim]
        direction_weights: Dict[int, torch.Tensor],  # layer -> [num_directions]
        gate_network: Optional[GROMGateNetwork],
        intensity_network: GROMIntensityNetwork,
        layer_order: List[int],
        gate_temperature: float = 0.5,
        max_alpha: float = 3.0,
    ):
        super().__init__(metadata)
        self.directions = directions
        self.direction_weights = direction_weights
        self.gate_network = gate_network
        self.intensity_network = intensity_network
        self.layer_order = layer_order
        self.gate_temperature = gate_temperature
        self.max_alpha = max_alpha
        
        # Create layer index mapping
        self._layer_to_idx = {layer: i for i, layer in enumerate(layer_order)}
    
    def get_steering_vector(self, layer: int, hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        if layer not in self.directions:
            raise KeyError(f"No directions for layer {layer}")
        
        dirs = self.directions[layer]
        weights = self.direction_weights.get(layer)
        
        if weights is None:
            return dirs[0]
        
        # Weighted combination
        weighted = (weights.unsqueeze(-1) * dirs).sum(dim=0)
        return F.normalize(weighted, dim=-1)
    
    def compute_gate(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if self.gate_network is None:
            batch_size = hidden_state.shape[0] if hidden_state.dim() > 1 else 1
            return torch.ones(batch_size, device=hidden_state.device, dtype=hidden_state.dtype)
        
        # Always use float32 for network computation (MPS compatibility)
        self.gate_network = self.gate_network.to(device=hidden_state.device, dtype=torch.float32)
        h_float = hidden_state.float()
        result = self.gate_network(h_float, self.gate_temperature)
        return result.to(hidden_state.dtype)
    
    def compute_intensity(self, hidden_state: torch.Tensor, layer: int) -> torch.Tensor:
        if layer not in self._layer_to_idx:
            batch_size = hidden_state.shape[0] if hidden_state.dim() > 1 else 1
            return torch.ones(batch_size, device=hidden_state.device, dtype=hidden_state.dtype)
        
        # Always use float32 for network computation (MPS compatibility)
        self.intensity_network = self.intensity_network.to(device=hidden_state.device, dtype=torch.float32)
        h_float = hidden_state.float()
        intensities = self.intensity_network(h_float)
        layer_idx = self._layer_to_idx[layer]
        return intensities[:, layer_idx].to(hidden_state.dtype)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'method': self.method_name,
            'metadata': {
                'method': self.metadata.method,
                'model_name': self.metadata.model_name,
                'benchmark': self.metadata.benchmark,
                'category': self.metadata.category,
                'extraction_strategy': self.metadata.extraction_strategy,
                'num_pairs': self.metadata.num_pairs,
                'layers': self.metadata.layers,
                'hidden_dim': self.metadata.hidden_dim,
                'created_at': self.metadata.created_at,
                'extra': self.metadata.extra,
                'extraction_component': self.metadata.extraction_component,
            },
            'directions': {str(k): v for k, v in self.directions.items()},
            'direction_weights': {str(k): v for k, v in self.direction_weights.items()},
            'gate_network_state': self.gate_network.state_dict() if self.gate_network else None,
            'gate_network_config': {
                'input_dim': self.metadata.hidden_dim,
                'hidden_dim': self.gate_network.net[0].out_features,  # Infer from first layer
            } if self.gate_network else None,
            'intensity_network_state': self.intensity_network.state_dict(),
            'intensity_network_config': {
                'input_dim': self.metadata.hidden_dim,
                'num_layers': len(self.layer_order),
                'hidden_dim': self.intensity_network.net[0].out_features,  # Infer from first layer
                'max_alpha': self.max_alpha,
            },
            'layer_order': self.layer_order,
            'gate_temperature': self.gate_temperature,
            'max_alpha': self.max_alpha,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GROMSteeringObject":
        meta_data = data['metadata']
        metadata = SteeringObjectMetadata(
            method=meta_data['method'],
            model_name=meta_data['model_name'],
            benchmark=meta_data['benchmark'],
            category=meta_data['category'],
            extraction_strategy=meta_data['extraction_strategy'],
            num_pairs=meta_data['num_pairs'],
            layers=meta_data['layers'],
            hidden_dim=meta_data['hidden_dim'],
            created_at=meta_data.get('created_at', ''),
            extra=meta_data.get('extra', {}),
            extraction_component=meta_data.get('extraction_component', 'residual_stream'),
        )
        
        def to_tensor(v):
            return torch.tensor(v) if isinstance(v, list) else v
        
        directions = {int(k): to_tensor(v) for k, v in data['directions'].items()}
        direction_weights = {int(k): to_tensor(v) for k, v in data['direction_weights'].items()}
        
        # Helper to convert state dict values from lists to tensors
        def convert_state_dict(state_dict):
            return {k: torch.tensor(v) if isinstance(v, list) else v for k, v in state_dict.items()}
        
        # Reconstruct gate network
        gate_network = None
        if data.get('gate_network_state') and data.get('gate_network_config'):
            config = data['gate_network_config']
            gate_network = GROMGateNetwork(config['input_dim'], config.get('hidden_dim', 128))
            gate_network.load_state_dict(convert_state_dict(data['gate_network_state']))
        
        # Reconstruct intensity network
        int_config = data['intensity_network_config']
        intensity_network = GROMIntensityNetwork(
            int_config['input_dim'],
            int_config['num_layers'],
            int_config.get('hidden_dim', 64),
            int_config.get('max_alpha', 3.0),
        )
        intensity_network.load_state_dict(convert_state_dict(data['intensity_network_state']))
        
        return cls(
            metadata=metadata,
            directions=directions,
            direction_weights=direction_weights,
            gate_network=gate_network,
            intensity_network=intensity_network,
            layer_order=data['layer_order'],
            gate_temperature=data.get('gate_temperature', 0.5),
            max_alpha=data.get('max_alpha', 3.0),
        )



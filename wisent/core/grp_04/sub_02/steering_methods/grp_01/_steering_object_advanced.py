"""Advanced steering objects (TECZA, TETNO)."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch
from wisent.core.steering_methods._steering_object_base import (
    BaseSteeringObject, SteeringObjectMetadata, LayerName,
)
from wisent.core.constants import TETNO_GATE_TEMPERATURE, DEFAULT_LAYER_WEIGHT

class TECZASteeringObject(BaseSteeringObject):
    """
    TECZA steering object with multiple directions per layer.
    
    Stores all discovered directions and their weights,
    allowing flexible combination strategies.
    """
    
    method_name = "tecza"
    
    def __init__(
        self,
        metadata: SteeringObjectMetadata,
        directions: Dict[int, torch.Tensor],  # layer -> [num_directions, hidden_dim]
        direction_weights: Optional[Dict[int, torch.Tensor]] = None,  # layer -> [num_directions]
        primary_only: bool = True,
    ):
        super().__init__(metadata)
        self.directions = directions
        self.direction_weights = direction_weights or {}
        self.primary_only = primary_only
    
    def get_steering_vector(self, layer: int, hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        if layer not in self.directions:
            raise KeyError(f"No directions for layer {layer}")
        
        dirs = self.directions[layer]
        
        if self.primary_only or layer not in self.direction_weights:
            return dirs[0]  # Primary direction
        
        # Weighted combination
        weights = self.direction_weights[layer]
        weighted = (weights.unsqueeze(-1) * dirs).sum(dim=0)
        return F.normalize(weighted, dim=-1)
    
    def compute_gate(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_state.shape[0] if hidden_state.dim() > 1 else 1
        return torch.ones(batch_size, device=hidden_state.device, dtype=hidden_state.dtype)
    
    def compute_intensity(self, hidden_state: torch.Tensor, layer: int) -> torch.Tensor:
        batch_size = hidden_state.shape[0] if hidden_state.dim() > 1 else 1
        return torch.ones(batch_size, device=hidden_state.device, dtype=hidden_state.dtype)
    
    def get_all_directions(self, layer: int) -> torch.Tensor:
        """Get all directions for a layer."""
        return self.directions[layer]
    
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
            'primary_only': self.primary_only,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TECZASteeringObject":
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
        direction_weights = {int(k): to_tensor(v) for k, v in data.get('direction_weights', {}).items()}
        
        return cls(
            metadata=metadata,
            directions=directions,
            direction_weights=direction_weights,
            primary_only=data.get('primary_only', True),
        )


class TETNOSteeringObject(BaseSteeringObject):
    """
    TETNO steering object with conditional gating.
    
    Includes:
    - Behavior vectors per layer
    - Condition vector at sensor layer
    - Learned threshold for gating
    - Per-layer scaling factors
    """
    
    method_name = "tetno"
    
    def __init__(
        self,
        metadata: SteeringObjectMetadata,
        behavior_vectors: Dict[int, torch.Tensor],
        condition_vector: torch.Tensor,
        sensor_layer: int,
        threshold: float,
        layer_scales: Dict[int, float],
        gate_temperature: float = TETNO_GATE_TEMPERATURE,
    ):
        super().__init__(metadata)
        self.behavior_vectors = behavior_vectors
        self.condition_vector = condition_vector
        self.sensor_layer = sensor_layer
        self.threshold = threshold
        self.layer_scales = layer_scales
        self.gate_temperature = gate_temperature
    
    def get_steering_vector(self, layer: int, hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        if layer not in self.behavior_vectors:
            raise KeyError(f"No behavior vector for layer {layer}")
        return self.behavior_vectors[layer]
    
    def compute_gate(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Compute gate based on similarity to condition vector."""
        h_norm = F.normalize(hidden_state, dim=-1)
        c_norm = F.normalize(self.condition_vector.to(device=hidden_state.device, dtype=hidden_state.dtype), dim=-1)
        
        similarity = (h_norm * c_norm).sum(dim=-1)
        gate = torch.sigmoid((similarity - self.threshold) / self.gate_temperature)
        return gate
    
    def compute_intensity(self, hidden_state: torch.Tensor, layer: int) -> torch.Tensor:
        batch_size = hidden_state.shape[0] if hidden_state.dim() > 1 else 1
        scale = self.layer_scales.get(layer, DEFAULT_LAYER_WEIGHT)
        return torch.full((batch_size,), scale, device=hidden_state.device, dtype=hidden_state.dtype)
    
    def should_steer(self, hidden_state: torch.Tensor) -> bool:
        """Hard decision on whether to steer."""
        gate = self.compute_gate(hidden_state)
        return gate.mean().item() > 0.5
    
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
            'behavior_vectors': {str(k): v for k, v in self.behavior_vectors.items()},
            'condition_vector': self.condition_vector,
            'sensor_layer': self.sensor_layer,
            'threshold': self.threshold,
            'layer_scales': {str(k): v for k, v in self.layer_scales.items()},
            'gate_temperature': self.gate_temperature,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TETNOSteeringObject":
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
        
        behavior_vectors = {int(k): to_tensor(v) for k, v in data['behavior_vectors'].items()}
        condition_vector = to_tensor(data['condition_vector'])
        layer_scales = {int(k): float(v) for k, v in data['layer_scales'].items()}
        
        return cls(
            metadata=metadata,
            behavior_vectors=behavior_vectors,
            condition_vector=condition_vector,
            sensor_layer=data['sensor_layer'],
            threshold=data['threshold'],
            layer_scales=layer_scales,
            gate_temperature=data.get('gate_temperature', TETNO_GATE_TEMPERATURE),
        )



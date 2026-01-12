"""
Unified Steering Objects for all steering methods.

Each steering method produces a SteeringObject that contains:
- The steering vectors/directions
- Method-specific components (gates, networks, thresholds)
- Metadata about training
- Methods to apply steering at inference time

This preserves the full structure of complex methods like PULSE and TITAN
rather than flattening everything to simple vectors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


LayerName = str


@dataclass
class SteeringObjectMetadata:
    """Common metadata for all steering objects."""
    method: str
    model_name: str
    benchmark: str
    category: str
    extraction_strategy: str
    num_pairs: int
    layers: List[int]
    hidden_dim: int
    created_at: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    # Calibration data: average hidden state norm per layer
    calibration_norms: Dict[int, float] = field(default_factory=dict)
    
    def get_calibrated_strength(self, layer: int, target_percentage: float = 1.0) -> float:
        """
        Compute calibrated strength for a layer based on hidden state norms.
        
        Args:
            layer: Layer index
            target_percentage: Target steering magnitude as percentage of hidden state norm (default 100%)
            
        Returns:
            Calibrated strength value
        """
        if layer not in self.calibration_norms or self.calibration_norms[layer] == 0:
            # No calibration data, return a reasonable default
            return target_percentage * 100  # Assume norm ~100 for typical models
        
        return target_percentage * self.calibration_norms[layer]


class BaseSteeringObject(ABC):
    """
    Base class for all steering objects.
    
    A steering object encapsulates everything needed to apply steering at inference:
    - Vectors/directions
    - Gating logic (when to steer)
    - Intensity computation (how much to steer)
    """
    
    method_name: str = "base"
    
    def __init__(self, metadata: SteeringObjectMetadata):
        self.metadata = metadata
    
    @abstractmethod
    def get_steering_vector(self, layer: int, hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get the steering vector for a layer, optionally conditioned on hidden state."""
        pass
    
    @abstractmethod
    def compute_gate(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Compute gate value (0-1) determining whether to apply steering."""
        pass
    
    @abstractmethod
    def compute_intensity(self, hidden_state: torch.Tensor, layer: int) -> torch.Tensor:
        """Compute steering intensity for a layer."""
        pass
    
    def get_calibrated_strength(self, target_percentage: float = 1.0) -> float:
        """
        Get auto-calibrated strength based on hidden state norms.
        
        Args:
            target_percentage: Target steering magnitude as percentage of hidden state norm (default 100%)
            
        Returns:
            Calibrated strength to use with apply_steering
        """
        if not self.metadata.calibration_norms:
            # No calibration data - return reasonable default for normalized vectors
            return target_percentage * 100
        
        # Use average calibration norm across layers
        avg_norm = sum(self.metadata.calibration_norms.values()) / len(self.metadata.calibration_norms)
        return target_percentage * avg_norm
    
    def apply_steering(
        self,
        hidden_state: torch.Tensor,
        layer: int,
        base_strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply steering to hidden state.
        
        Args:
            hidden_state: [batch, seq, hidden] or [seq, hidden] or [hidden]
            layer: Layer index
            base_strength: Base strength multiplier (use get_calibrated_strength() for auto-calibration)
            
        Returns:
            Steered hidden state
        """
        original_shape = hidden_state.shape
        
        # Flatten to [batch, hidden] for gate computation
        if hidden_state.dim() == 1:
            h_flat = hidden_state.unsqueeze(0)
        elif hidden_state.dim() == 2:
            h_flat = hidden_state.mean(dim=0, keepdim=True)  # avg over seq
        else:
            h_flat = hidden_state.mean(dim=1)  # [batch, hidden]
        
        gate = self.compute_gate(h_flat)
        intensity = self.compute_intensity(h_flat, layer)
        direction = self.get_steering_vector(layer, h_flat)
        
        # Ensure direction matches hidden state device/dtype
        direction = direction.to(hidden_state.device, hidden_state.dtype)
        
        # Compute steering delta
        scale = gate * intensity * base_strength
        
        # Broadcast direction to match hidden_state shape
        if hidden_state.dim() == 1:
            delta = scale.squeeze() * direction
        elif hidden_state.dim() == 2:
            delta = scale.unsqueeze(-1) * direction.unsqueeze(0)
        else:
            delta = scale.unsqueeze(-1).unsqueeze(-1) * direction.unsqueeze(0).unsqueeze(0)
        
        return hidden_state + delta
    
    def to_steering_plan(self, scale: float = 1.0, normalize: bool = True) -> "SteeringPlan":
        """
        Convert this steering object to a SteeringPlan for use with WisentModel.
        
        This allows steering objects to be used with the existing model infrastructure.
        Complex methods (PULSE, TITAN) are simplified to their primary vectors.
        
        Args:
            scale: Base scale for all vectors
            normalize: Whether to normalize vectors
            
        Returns:
            SteeringPlan compatible with WisentModel.apply_steering()
        """
        from wisent.core.models.core.atoms import SteeringPlan, SteeringVector
        
        layers_dict = {}
        for layer in self.metadata.layers:
            try:
                vec = self.get_steering_vector(layer)
                layers_dict[str(layer)] = SteeringVector(
                    vector=vec,
                    scale=scale,
                    normalize=normalize,
                    layer_description=f"{self.method_name}_{self.metadata.benchmark}",
                )
            except (KeyError, IndexError):
                continue
        
        return SteeringPlan(
            layers=layers_dict,
            layers_description=[f"{self.method_name}_{self.metadata.benchmark}"],
        )
    
    def to_raw_activation_map(self) -> Dict[str, torch.Tensor]:
        """
        Convert to RawActivationMap format (layer_name -> tensor).
        
        This is useful for integrating with existing code that expects this format.
        """
        raw = {}
        for layer in self.metadata.layers:
            try:
                raw[str(layer)] = self.get_steering_vector(layer)
            except (KeyError, IndexError):
                continue
        return raw
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for saving."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseSteeringObject":
        """Deserialize from dictionary."""
        pass
    
    def save(self, path: str):
        """Save steering object to file."""
        data = self.to_dict()
        
        if path.endswith('.pt'):
            torch.save(data, path)
        else:
            # Convert tensors to lists for JSON
            def convert(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert(v) for v in obj]
                return obj
            
            with open(path, 'w') as f:
                json.dump(convert(data), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "BaseSteeringObject":
        """Load steering object from file."""
        if path.endswith('.pt'):
            data = torch.load(path, map_location='cpu')
        else:
            with open(path, 'r') as f:
                data = json.load(f)
        
        method = data.get('method', 'caa')
        
        # Dispatch to appropriate class
        if method == 'caa':
            return CAASteeringObject.from_dict(data)
        elif method == 'hyperplane':
            return HyperplaneSteeringObject.from_dict(data)
        elif method == 'mlp':
            return MLPSteeringObject.from_dict(data)
        elif method == 'prism':
            return PRISMSteeringObject.from_dict(data)
        elif method == 'pulse':
            return PULSESteeringObject.from_dict(data)
        elif method == 'titan':
            return TITANSteeringObject.from_dict(data)
        else:
            raise ValueError(f"Unknown steering method: {method}")


class SimpleSteeringObject(BaseSteeringObject):
    """
    Simple steering object for methods that produce single vector per layer.
    Used by: CAA, Hyperplane, MLP
    
    Always steers (gate=1.0) with fixed intensity.
    """
    
    def __init__(
        self,
        metadata: SteeringObjectMetadata,
        vectors: Dict[int, torch.Tensor],
        default_intensity: float = 1.0,
    ):
        super().__init__(metadata)
        self.vectors = vectors
        self.default_intensity = default_intensity
    
    def get_steering_vector(self, layer: int, hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        if layer not in self.vectors:
            raise KeyError(f"No steering vector for layer {layer}")
        return self.vectors[layer]
    
    def compute_gate(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # Always steer
        batch_size = hidden_state.shape[0] if hidden_state.dim() > 1 else 1
        return torch.ones(batch_size, device=hidden_state.device)
    
    def compute_intensity(self, hidden_state: torch.Tensor, layer: int) -> torch.Tensor:
        batch_size = hidden_state.shape[0] if hidden_state.dim() > 1 else 1
        return torch.full((batch_size,), self.default_intensity, device=hidden_state.device)
    
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
                'calibration_norms': {str(k): v for k, v in self.metadata.calibration_norms.items()},
            },
            'vectors': {str(k): v for k, v in self.vectors.items()},
            'default_intensity': self.default_intensity,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimpleSteeringObject":
        meta_data = data['metadata']
        calibration_norms_raw = meta_data.get('calibration_norms', {})
        calibration_norms = {int(k): float(v) for k, v in calibration_norms_raw.items()}
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
            calibration_norms=calibration_norms,
        )
        
        vectors = {}
        for k, v in data['vectors'].items():
            if isinstance(v, list):
                vectors[int(k)] = torch.tensor(v)
            else:
                vectors[int(k)] = v
        
        return cls(
            metadata=metadata,
            vectors=vectors,
            default_intensity=data.get('default_intensity', 1.0),
        )


class CAASteeringObject(SimpleSteeringObject):
    method_name = "caa"


class HyperplaneSteeringObject(SimpleSteeringObject):
    method_name = "hyperplane"


class MLPSteeringObject(SimpleSteeringObject):
    method_name = "mlp"


class PRISMSteeringObject(BaseSteeringObject):
    """
    PRISM steering object with multiple directions per layer.
    
    Stores all discovered directions and their weights,
    allowing flexible combination strategies.
    """
    
    method_name = "prism"
    
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
        return torch.ones(batch_size, device=hidden_state.device)
    
    def compute_intensity(self, hidden_state: torch.Tensor, layer: int) -> torch.Tensor:
        batch_size = hidden_state.shape[0] if hidden_state.dim() > 1 else 1
        return torch.ones(batch_size, device=hidden_state.device)
    
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
            },
            'directions': {str(k): v for k, v in self.directions.items()},
            'direction_weights': {str(k): v for k, v in self.direction_weights.items()},
            'primary_only': self.primary_only,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PRISMSteeringObject":
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


class PULSESteeringObject(BaseSteeringObject):
    """
    PULSE steering object with conditional gating.
    
    Includes:
    - Behavior vectors per layer
    - Condition vector at sensor layer
    - Learned threshold for gating
    - Per-layer scaling factors
    """
    
    method_name = "pulse"
    
    def __init__(
        self,
        metadata: SteeringObjectMetadata,
        behavior_vectors: Dict[int, torch.Tensor],
        condition_vector: torch.Tensor,
        sensor_layer: int,
        threshold: float,
        layer_scales: Dict[int, float],
        gate_temperature: float = 0.1,
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
        c_norm = F.normalize(self.condition_vector.to(hidden_state.device), dim=-1)
        
        similarity = (h_norm * c_norm).sum(dim=-1)
        gate = torch.sigmoid((similarity - self.threshold) / self.gate_temperature)
        return gate
    
    def compute_intensity(self, hidden_state: torch.Tensor, layer: int) -> torch.Tensor:
        batch_size = hidden_state.shape[0] if hidden_state.dim() > 1 else 1
        scale = self.layer_scales.get(layer, 1.0)
        return torch.full((batch_size,), scale, device=hidden_state.device)
    
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
            },
            'behavior_vectors': {str(k): v for k, v in self.behavior_vectors.items()},
            'condition_vector': self.condition_vector,
            'sensor_layer': self.sensor_layer,
            'threshold': self.threshold,
            'layer_scales': {str(k): v for k, v in self.layer_scales.items()},
            'gate_temperature': self.gate_temperature,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PULSESteeringObject":
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
            gate_temperature=data.get('gate_temperature', 0.1),
        )


class TITANGateNetwork(nn.Module):
    """Serializable gate network for TITAN."""
    
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


class TITANIntensityNetwork(nn.Module):
    """Serializable intensity network for TITAN."""
    
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


class TITANSteeringObject(BaseSteeringObject):
    """
    TITAN steering object with learned gating and intensity networks.
    
    Includes:
    - Multi-directional manifold per layer
    - Learned gate network
    - Learned intensity network
    - Direction weights
    """
    
    method_name = "titan"
    
    def __init__(
        self,
        metadata: SteeringObjectMetadata,
        directions: Dict[int, torch.Tensor],  # layer -> [num_directions, hidden_dim]
        direction_weights: Dict[int, torch.Tensor],  # layer -> [num_directions]
        gate_network: Optional[TITANGateNetwork],
        intensity_network: TITANIntensityNetwork,
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
    def from_dict(cls, data: Dict[str, Any]) -> "TITANSteeringObject":
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
            gate_network = TITANGateNetwork(config['input_dim'], config.get('hidden_dim', 128))
            gate_network.load_state_dict(convert_state_dict(data['gate_network_state']))
        
        # Reconstruct intensity network
        int_config = data['intensity_network_config']
        intensity_network = TITANIntensityNetwork(
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


# Factory function
def create_steering_object(
    method: str,
    metadata: SteeringObjectMetadata,
    **kwargs,
) -> BaseSteeringObject:
    """Factory to create appropriate steering object for a method."""
    if method == 'caa':
        return CAASteeringObject(metadata, **kwargs)
    elif method == 'hyperplane':
        return HyperplaneSteeringObject(metadata, **kwargs)
    elif method == 'mlp':
        return MLPSteeringObject(metadata, **kwargs)
    elif method == 'prism':
        return PRISMSteeringObject(metadata, **kwargs)
    elif method == 'pulse':
        return PULSESteeringObject(metadata, **kwargs)
    elif method == 'titan':
        return TITANSteeringObject(metadata, **kwargs)
    else:
        raise ValueError(f"Unknown steering method: {method}")


def load_steering_object(path: str) -> BaseSteeringObject:
    """
    Load a steering object from file.
    
    Convenience function that calls BaseSteeringObject.load().
    
    Args:
        path: Path to the steering object file (.pt or .json)
        
    Returns:
        The loaded steering object
    """
    return BaseSteeringObject.load(path)

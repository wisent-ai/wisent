"""Simple steering objects (CAA, Ostrze, MLP)."""
from __future__ import annotations
from typing import Any, Dict, Optional
import torch
from wisent.core.steering_methods._steering_object_base import (
    BaseSteeringObject, SteeringObjectMetadata, LayerName,
)
from wisent.core.constants import DEFAULT_STRENGTH

class SimpleSteeringObject(BaseSteeringObject):
    """
    Simple steering object for methods that produce single vector per layer.
    Used by: CAA, Ostrze, MLP
    
    Always steers (gate=1.0) with fixed intensity.
    """
    
    def __init__(
        self,
        metadata: SteeringObjectMetadata,
        vectors: Dict[int, torch.Tensor],
        default_intensity: float = DEFAULT_STRENGTH,
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
        return torch.ones(batch_size, device=hidden_state.device, dtype=hidden_state.dtype)
    
    def compute_intensity(self, hidden_state: torch.Tensor, layer: int) -> torch.Tensor:
        batch_size = hidden_state.shape[0] if hidden_state.dim() > 1 else 1
        return torch.full((batch_size,), self.default_intensity, device=hidden_state.device, dtype=hidden_state.dtype)
    
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
                'extraction_component': self.metadata.extraction_component,
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
            extraction_component=meta_data.get('extraction_component', 'residual_stream'),
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
            default_intensity=data.get('default_intensity', DEFAULT_STRENGTH),
        )


class CAASteeringObject(SimpleSteeringObject):
    method_name = "caa"


class OstrzeSteeringObject(SimpleSteeringObject):
    method_name = "ostrze"


class MLPSteeringObject(SimpleSteeringObject):
    method_name = "mlp"


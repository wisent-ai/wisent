"""
ZapisSteeringObject — KV cache steering at inference time.

Stores per-layer steering vectors and injects them into the KV cache
via steer_cache(). No forward hooks are needed during generation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import torch

from wisent.core.control.steering_methods.configs.optimal import get_optimal
from wisent.core.control.steering_methods.steering_object import (
    BaseSteeringObject,
    SteeringObjectMetadata,
)
from wisent.core.utils.config_tools.constants import (
    LOG_EPS,
    INDEX_LAST,
    INDEX_FIRST,
    NDIM_VECTOR,
    ZAPIS_OFFSET_TOKEN_DEFAULT,
)

__all__ = ["ZapisSteeringObject"]

# Gate and intensity return value for one-shot methods
_UNIT = torch.tensor(True).float()


class ZapisSteeringObject(BaseSteeringObject):
    """Steering object that injects directions into KV cache."""

    method_name = "zapis"

    def __init__(
        self,
        metadata: SteeringObjectMetadata,
        vectors: Dict[int, torch.Tensor],
        c_keys: float,
        c_values: float,
        offset_token: str = ZAPIS_OFFSET_TOKEN_DEFAULT,
    ):
        super().__init__(metadata)
        self.vectors = vectors
        self.c_keys = c_keys
        self.c_values = c_values
        self.offset_token = offset_token

    def get_steering_vector(
        self, layer: int, hidden_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return the steering vector for a layer."""
        if layer not in self.vectors:
            raise KeyError(f"No steering vector for layer {layer}")
        vec = self.vectors[layer].float()
        norm = torch.linalg.norm(vec)
        if norm > LOG_EPS:
            vec = vec / norm
        return vec

    def compute_gate(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Always-on gate (one-shot method, no gating needed)."""
        batch_size = hidden_state.shape[INDEX_FIRST] if hidden_state.dim() > NDIM_VECTOR else NDIM_VECTOR
        return torch.ones(
            batch_size, device=hidden_state.device, dtype=hidden_state.dtype
        )

    def compute_intensity(
        self, hidden_state: torch.Tensor, layer: int
    ) -> torch.Tensor:
        """Constant intensity (strength via c_keys/c_values)."""
        batch_size = hidden_state.shape[INDEX_FIRST] if hidden_state.dim() > NDIM_VECTOR else NDIM_VECTOR
        return torch.ones(
            batch_size, device=hidden_state.device, dtype=hidden_state.dtype
        )

    def steer_cache(
        self,
        past_key_values,
        application_idx: int = INDEX_LAST,
    ) -> None:
        """
        Inject steering vectors into a DynamicCache in-place.

        For each layer with a stored vector, reshapes the vector to
        match the KV cache head structure and adds it (scaled by
        c_keys / c_values) to the key and value caches at the
        specified sequence position.

        Args:
            past_key_values: transformers DynamicCache object
            application_idx: sequence position to modify (default: last token)
        """
        for layer_idx, vec in self.vectors.items():
            if layer_idx >= len(past_key_values.key_cache):
                continue

            key_cache = past_key_values.key_cache[layer_idx]
            value_cache = past_key_values.value_cache[layer_idx]

            # key_cache shape: [batch, num_heads, seq_len, head_dim]
            num_heads = key_cache.shape[NDIM_VECTOR]
            head_dim = key_cache.shape[INDEX_LAST]

            # Reshape flat vector [hidden_dim] -> [num_heads, head_dim]
            vec_dev = vec.to(key_cache.device, key_cache.dtype)
            vec_kv = vec_dev.view(num_heads, head_dim)

            # Apply to keys: K* = K + c_k * S
            key_cache[:, :, application_idx, :] += self.c_keys * vec_kv

            # Apply to values: V* = V + c_v * S
            value_cache[:, :, application_idx, :] += self.c_values * vec_kv

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to dictionary."""
        meta = self.metadata
        return {
            "method": self.method_name,
            "metadata": {
                "method": meta.method,
                "model_name": meta.model_name,
                "benchmark": meta.benchmark,
                "category": meta.category,
                "extraction_strategy": meta.extraction_strategy,
                "num_pairs": meta.num_pairs,
                "layers": meta.layers,
                "hidden_dim": meta.hidden_dim,
                "created_at": meta.created_at,
                "extra": meta.extra,
                "calibration_norms": {
                    str(k): v for k, v in meta.calibration_norms.items()
                },
                "extraction_component": meta.extraction_component,
            },
            "vectors": {str(k): v for k, v in self.vectors.items()},
            "c_keys": self.c_keys,
            "c_values": self.c_values,
            "offset_token": self.offset_token,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ZapisSteeringObject":
        """Deserialise from dictionary."""
        meta_data = data["metadata"]
        cal_raw = meta_data.get("calibration_norms", {})
        calibration_norms = {int(k): float(v) for k, v in cal_raw.items()}
        metadata = SteeringObjectMetadata(
            method=meta_data["method"],
            model_name=meta_data["model_name"],
            benchmark=meta_data["benchmark"],
            category=meta_data["category"],
            extraction_strategy=meta_data["extraction_strategy"],
            num_pairs=meta_data["num_pairs"],
            layers=meta_data["layers"],
            hidden_dim=meta_data["hidden_dim"],
            created_at=meta_data.get("created_at", ""),
            extra=meta_data.get("extra", {}),
            calibration_norms=calibration_norms,
            extraction_component=meta_data.get(
                "extraction_component", get_optimal("extraction_component")
            ),
        )

        def to_tensor(v):
            return torch.tensor(v) if isinstance(v, list) else v

        vectors = {
            int(k): to_tensor(v) for k, v in data["vectors"].items()
        }

        return cls(
            metadata=metadata,
            vectors=vectors,
            c_keys=data["c_keys"],
            c_values=data["c_values"],
            offset_token=data.get("offset_token", ZAPIS_OFFSET_TOKEN_DEFAULT),
        )

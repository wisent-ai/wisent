"""
SzlakSteeringObject — steering object for geodesic optimal transport.

At inference, finds K nearest training source points via torch.cdist,
computes inverse-distance-weighted average of their precomputed
displacements, and applies the interpolated displacement to the hidden state.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F

from wisent.core.control.steering_methods.steering_object import (
    BaseSteeringObject,
    SteeringObjectMetadata,
)
from wisent.core.utils.config_tools.constants import NORM_EPS, DEFAULT_STRENGTH, SZLAK_INFERENCE_K

__all__ = ["SzlakSteeringObject"]


class SzlakSteeringObject(BaseSteeringObject):
    """Steering object that applies geodesic OT-derived displacements."""

    method_name = "szlak"

    def __init__(
        self,
        metadata: SteeringObjectMetadata,
        source_points: Dict[int, torch.Tensor],
        displacements: Dict[int, torch.Tensor],
        inference_k: int = SZLAK_INFERENCE_K,
    ):
        super().__init__(metadata)
        self.source_points = source_points
        self.displacements = displacements
        self.inference_k = inference_k

    def get_steering_vector(
        self, layer: int, hidden_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compatibility: return mean displacement direction for layer."""
        if layer not in self.displacements:
            raise KeyError(f"No displacement data for layer {layer}")
        mean_disp = self.displacements[layer].mean(dim=0)
        return F.normalize(mean_disp.float(), p=2, dim=-1)

    def compute_gate(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Always returns 1.0 — displacement naturally handles magnitude."""
        batch_size = hidden_state.shape[0] if hidden_state.dim() > 1 else 1
        return torch.ones(batch_size, device=hidden_state.device, dtype=hidden_state.dtype)

    def compute_intensity(self, hidden_state: torch.Tensor, layer: int) -> torch.Tensor:
        """Always returns 1.0 — strength controlled by base_strength."""
        batch_size = hidden_state.shape[0] if hidden_state.dim() > 1 else 1
        return torch.ones(batch_size, device=hidden_state.device, dtype=hidden_state.dtype)

    def apply_steering(
        self, hidden_state: torch.Tensor, layer: int, base_strength: float = DEFAULT_STRENGTH,
    ) -> torch.Tensor:
        """
        Apply geodesic OT steering via NN-lookup + interpolated displacement.

        1. Find K nearest source points via torch.cdist
        2. Inverse-distance-weighted average of their displacements
        3. Apply: h_new = h + strength * displacement
        """
        if layer not in self.source_points:
            return hidden_state

        src = self.source_points[layer].float()
        disp = self.displacements[layer].float()
        original_shape = hidden_state.shape
        original_dtype = hidden_state.dtype

        if hidden_state.dim() == 1:
            h_2d = hidden_state.unsqueeze(0)
        elif hidden_state.dim() == 2:
            h_2d = hidden_state
        else:
            b, s, hd = hidden_state.shape
            h_2d = hidden_state.reshape(b * s, hd)

        h_float = h_2d.float()
        src_dev = src.to(h_float.device)
        disp_dev = disp.to(h_float.device)

        # Compute distances to all source points [batch, N_src]
        dists = torch.cdist(h_float, src_dev)  # [B, N_src]

        # Select K nearest
        K = min(self.inference_k, src_dev.shape[0])
        topk_dists, topk_idx = torch.topk(dists, K, dim=1, largest=False)

        # Inverse-distance weights (add small epsilon to avoid div-by-zero)
        eps = NORM_EPS
        inv_dists = 1.0 / (topk_dists + eps)
        weights = inv_dists / inv_dists.sum(dim=1, keepdim=True)  # [B, K]

        # Gather corresponding displacements
        batch_size = h_float.shape[0]
        topk_disps = disp_dev[topk_idx.reshape(-1)].reshape(batch_size, K, -1)

        # Weighted average displacement [B, D]
        weighted_disp = (weights.unsqueeze(-1) * topk_disps).sum(dim=1)

        h_new = h_float + base_strength * weighted_disp
        h_new = h_new.to(original_dtype)

        if hidden_state.dim() == 1:
            return h_new.squeeze(0)
        elif hidden_state.dim() == 3:
            return h_new.reshape(original_shape)
        return h_new

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for saving."""
        meta = self.metadata
        return {
            "method": self.method_name,
            "metadata": {
                "method": meta.method, "model_name": meta.model_name,
                "benchmark": meta.benchmark, "category": meta.category,
                "extraction_strategy": meta.extraction_strategy,
                "num_pairs": meta.num_pairs, "layers": meta.layers,
                "hidden_dim": meta.hidden_dim,
                "created_at": meta.created_at, "extra": meta.extra,
                "calibration_norms": {str(k): v for k, v in meta.calibration_norms.items()},
                "extraction_component": meta.extraction_component,
            },
            "source_points": {str(k): v for k, v in self.source_points.items()},
            "displacements": {str(k): v for k, v in self.displacements.items()},
            "inference_k": self.inference_k,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SzlakSteeringObject":
        """Deserialize from dictionary."""
        meta_data = data["metadata"]
        cal_raw = meta_data.get("calibration_norms", {})
        calibration_norms = {int(k): float(v) for k, v in cal_raw.items()}
        metadata = SteeringObjectMetadata(
            method=meta_data["method"], model_name=meta_data["model_name"],
            benchmark=meta_data["benchmark"], category=meta_data["category"],
            extraction_strategy=meta_data["extraction_strategy"],
            num_pairs=meta_data["num_pairs"], layers=meta_data["layers"],
            hidden_dim=meta_data["hidden_dim"],
            created_at=meta_data.get("created_at", ""),
            extra=meta_data.get("extra", {}),
            calibration_norms=calibration_norms,
            extraction_component=meta_data.get("extraction_component", "residual_stream"),
        )

        def to_tensor(v):
            return torch.tensor(v) if isinstance(v, list) else v

        source_points = {int(k): to_tensor(v) for k, v in data["source_points"].items()}
        disps = {int(k): to_tensor(v) for k, v in data["displacements"].items()}

        return cls(
            metadata=metadata,
            source_points=source_points,
            displacements=disps,
            inference_k=data.get("inference_k", SZLAK_INFERENCE_K),
        )

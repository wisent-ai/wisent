"""
NurtSteeringObject - steering object that applies flow-based transport.

Instead of the standard h + alpha * direction, this does:
1. Project hidden state to concept subspace
2. Compute orthogonal complement
3. Euler integrate through learned velocity field
4. Reconstruct from subspace preserving orthogonal complement
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F

from wisent.core.steering_methods.steering_object import (
    BaseSteeringObject,
    SteeringObjectMetadata,
)
from .flow_network import FlowVelocityNetwork, euler_integrate
from .subspace import project_to_subspace, reconstruct_from_subspace

__all__ = ["NurtSteeringObject"]


class NurtSteeringObject(BaseSteeringObject):
    """Steering object that applies flow-based transport in concept subspace."""

    method_name = "nurt"

    def __init__(
        self,
        metadata: SteeringObjectMetadata,
        flow_networks: Dict[int, FlowVelocityNetwork],
        concept_bases: Dict[int, torch.Tensor],
        mean_neg: Dict[int, torch.Tensor],
        mean_pos: Dict[int, torch.Tensor],
        num_integration_steps: int = 4,
        t_max: float = 1.0,
        layer_variance: Optional[Dict[int, float]] = None,
    ):
        super().__init__(metadata)
        self.flow_networks = flow_networks
        self.concept_bases = concept_bases
        self.mean_neg = mean_neg
        self.mean_pos = mean_pos
        self.num_integration_steps = num_integration_steps
        self.t_max = t_max
        # Per-layer variance explained from SVD — used to weight transport
        self.layer_variance = layer_variance or {}
        # Precompute normalized weights so sum = 1.0
        total_var = sum(self.layer_variance.values()) if self.layer_variance else 0.0
        if total_var > 0:
            self._layer_weights = {k: v / total_var for k, v in self.layer_variance.items()}
        else:
            n = len(self.flow_networks)
            self._layer_weights = {k: 1.0 / max(n, 1) for k in self.flow_networks}

    def get_steering_vector(
        self, layer: int, hidden_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compatibility: compute mean transport direction in full space."""
        if layer not in self.concept_bases:
            raise KeyError(f"No concept basis for layer {layer}")
        Vh = self.concept_bases[layer]
        diff = self.mean_pos[layer] - self.mean_neg[layer]
        transport = diff.float() @ Vh.float()
        return F.normalize(transport, p=2, dim=-1)

    def compute_gate(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Flow naturally handles no-op via t_max=0; always returns 1.0."""
        batch_size = hidden_state.shape[0] if hidden_state.dim() > 1 else 1
        return torch.ones(batch_size, device=hidden_state.device, dtype=hidden_state.dtype)

    def compute_intensity(self, hidden_state: torch.Tensor, layer: int) -> torch.Tensor:
        """Strength is controlled via t_max, not intensity; always returns 1.0."""
        batch_size = hidden_state.shape[0] if hidden_state.dim() > 1 else 1
        return torch.ones(batch_size, device=hidden_state.device, dtype=hidden_state.dtype)

    def apply_steering(
        self, hidden_state: torch.Tensor, layer: int, base_strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply flow-based steering: project -> integrate -> reconstruct.

        The orthogonal complement is perfectly preserved.
        Strength controls effective t_max: t_eff = t_max * base_strength.
        """
        if layer not in self.flow_networks:
            return hidden_state

        network = self.flow_networks[layer]
        Vh = self.concept_bases[layer]
        original_shape = hidden_state.shape
        original_dtype = hidden_state.dtype

        # Flatten to 2D for processing
        if hidden_state.dim() == 1:
            h_2d = hidden_state.unsqueeze(0)
        elif hidden_state.dim() == 2:
            h_2d = hidden_state
        else:
            # [batch, seq, hidden] -> [batch*seq, hidden]
            b, s, hd = hidden_state.shape
            h_2d = hidden_state.reshape(b * s, hd)

        h_float = h_2d.float()
        Vh_f = Vh.float().to(h_float.device)
        network = network.to(h_float.device)
        network.eval()

        # Project to concept subspace
        z = project_to_subspace(h_float, Vh_f)

        # Euler integrate through velocity field
        # Per-layer weight from variance_explained: layers with stronger concept
        # signal get more transport. Weights sum to 1.0 across all layers.
        layer_w = self._layer_weights.get(layer, 1.0 / max(len(self.flow_networks), 1))
        t_eff = self.t_max * base_strength * layer_w
        if t_eff > 0:
            z_new = euler_integrate(network, z, t_max=t_eff, num_steps=self.num_integration_steps)
        else:
            z_new = z

        # Reconstruct preserving orthogonal complement
        h_new = reconstruct_from_subspace(z_new, h_float, Vh_f)
        h_new = h_new.to(original_dtype)

        # Restore original shape
        if hidden_state.dim() == 1:
            return h_new.squeeze(0)
        elif hidden_state.dim() == 3:
            return h_new.reshape(original_shape)
        return h_new

    def compute_mean_transport(self, base_strength: float = 1.0) -> Dict[int, torch.Tensor]:
        """
        Compute mean transport direction per layer in full hidden space.

        Uses the stored mean_neg + flow networks to compute the average
        displacement that concept flow applies. Returns vectors in the
        full hidden_dim space (not concept subspace).

        Returns:
            Dict mapping layer_idx -> delta vector [hidden_dim].
        """
        deltas: Dict[int, torch.Tensor] = {}
        for layer, network in self.flow_networks.items():
            Vh = self.concept_bases[layer].float()
            z_neg = self.mean_neg[layer].float()
            if z_neg.dim() == 1:
                z_neg = z_neg.unsqueeze(0)

            network_dev = network.to(z_neg.device)
            network_dev.eval()

            layer_w = self._layer_weights.get(
                layer, 1.0 / max(len(self.flow_networks), 1),
            )
            t_eff = self.t_max * base_strength * layer_w

            if t_eff > 0:
                z_new = euler_integrate(
                    network_dev, z_neg, t_max=t_eff,
                    num_steps=self.num_integration_steps,
                )
            else:
                z_new = z_neg

            delta_concept = z_new - z_neg          # [1, k]
            delta_full = delta_concept @ Vh         # [1, hidden_dim]
            deltas[layer] = delta_full.squeeze(0)

        return deltas

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for saving."""
        meta = self.metadata
        flow_net_data = {}
        for layer, net in self.flow_networks.items():
            flow_net_data[str(layer)] = {
                "state_dict": net.state_dict(),
                "config": {"concept_dim": net.concept_dim, "hidden_dim": net.hidden_dim},
            }
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
            "flow_networks": flow_net_data,
            "concept_bases": {str(k): v for k, v in self.concept_bases.items()},
            "mean_neg": {str(k): v for k, v in self.mean_neg.items()},
            "mean_pos": {str(k): v for k, v in self.mean_pos.items()},
            "num_integration_steps": self.num_integration_steps,
            "t_max": self.t_max,
            "layer_variance": {str(k): v for k, v in self.layer_variance.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NurtSteeringObject":
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

        # Reconstruct flow networks
        flow_networks = {}
        for layer_str, net_data in data["flow_networks"].items():
            cfg = net_data["config"]
            net = FlowVelocityNetwork(cfg["concept_dim"], cfg["hidden_dim"])
            sd = net_data["state_dict"]
            sd = {k: torch.tensor(v) if isinstance(v, list) else v for k, v in sd.items()}
            net.load_state_dict(sd)
            net.eval()
            flow_networks[int(layer_str)] = net

        concept_bases = {int(k): to_tensor(v) for k, v in data["concept_bases"].items()}
        mean_neg = {int(k): to_tensor(v) for k, v in data["mean_neg"].items()}
        mean_pos = {int(k): to_tensor(v) for k, v in data["mean_pos"].items()}

        lv_raw = data.get("layer_variance", {})
        layer_variance = {int(k): float(v) for k, v in lv_raw.items()}

        return cls(
            metadata=metadata,
            flow_networks=flow_networks,
            concept_bases=concept_bases,
            mean_neg=mean_neg,
            mean_pos=mean_pos,
            num_integration_steps=data.get("num_integration_steps", 4),
            t_max=data.get("t_max", 1.0),
            layer_variance=layer_variance,
        )

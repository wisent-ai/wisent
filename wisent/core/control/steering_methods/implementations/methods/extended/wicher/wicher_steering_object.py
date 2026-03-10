"""
WicherSteeringObject — inference-time subspace-projected Broyden steering.

``apply_steering()`` runs Broyden iterations in the SVD concept subspace.
No lm_head weight matrix is required.

Stores per layer: concept_direction [D], concept_basis [k,D],
component_variances [k], layer_variance (float).
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F

from wisent.core.control.steering_methods.configs.optimal import get_optimal
from wisent.core.control.steering_methods.steering_object import (
    BaseSteeringObject,
    SteeringObjectMetadata,
)
from .solvers import get_solver_fn
from wisent.core.utils.config_tools.constants import (
    NORM_EPS,
    WICHER_DEFAULT_SOLVER,
    INDEX_FIRST,
    NDIM_VECTOR,
    NDIM_MATRIX,
    NDIM_BATCH_SEQ,
)
from wisent.core.utils.infra_tools.errors import InsufficientDataError

__all__ = ["WicherSteeringObject"]


class WicherSteeringObject(BaseSteeringObject):
    """Steering object that applies subspace-projected Broyden steps."""

    method_name = "wicher"

    def __init__(
        self,
        metadata: SteeringObjectMetadata,
        concept_directions: Dict[int, torch.Tensor],
        concept_bases: Dict[int, torch.Tensor],
        component_variances: Dict[int, torch.Tensor],
        num_steps: int,
        alpha: float,
        eta: float,
        beta: float,
        alpha_decay: float,
        layer_variance: Optional[Dict[int, float]] = None,
        solver: str = WICHER_DEFAULT_SOLVER,
    ):
        super().__init__(metadata)
        self.concept_directions = concept_directions
        self.concept_bases = concept_bases
        self.component_variances = component_variances
        self.num_steps = num_steps
        self.alpha = alpha
        self.eta = eta
        self.beta = beta
        self.alpha_decay = alpha_decay
        self.layer_variance = layer_variance or {}
        self.solver = solver
        self._solver_fn = get_solver_fn(solver)

        self._variance_weights: Dict[int, float] = {}
        if self.layer_variance:
            total = sum(self.layer_variance.values())
            if total > 0:
                n_layers = len(self.layer_variance)
                for layer, var in self.layer_variance.items():
                    self._variance_weights[layer] = (var / total) * n_layers

    def get_steering_vector(
        self, layer: int, hidden_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compatibility: return normalised concept direction for layer."""
        if layer not in self.concept_directions:
            raise KeyError(f"No concept direction for layer {layer}")
        return F.normalize(
            self.concept_directions[layer].float(), p=2, dim=-1
        )

    def compute_gate(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_state.shape[0] if hidden_state.dim() > 1 else 1
        return torch.ones(
            batch_size, device=hidden_state.device, dtype=hidden_state.dtype
        )

    def compute_intensity(
        self, hidden_state: torch.Tensor, layer: int
    ) -> torch.Tensor:
        batch_size = hidden_state.shape[0] if hidden_state.dim() > 1 else 1
        return torch.ones(
            batch_size, device=hidden_state.device, dtype=hidden_state.dtype
        )

    def apply_steering(
        self,
        hidden_state: torch.Tensor,
        layer: int,
        base_strength: float,
    ) -> torch.Tensor:
        """Apply WICHER steering with layer variance weighting."""
        if layer not in self.concept_directions:
            return hidden_state
        if layer not in self.concept_bases:
            return hidden_state

        concept_dir = self.concept_directions[layer].float()
        original_shape = hidden_state.shape
        original_dtype = hidden_state.dtype

        effective_strength = base_strength

        return self._apply_solver(
            hidden_state, concept_dir, effective_strength,
            layer, original_shape, original_dtype,
        )

    def _apply_solver(
        self,
        hidden_state: torch.Tensor,
        concept_dir: torch.Tensor,
        base_strength: float,
        layer: int,
        original_shape: torch.Size,
        original_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Solver iteration in SVD concept subspace."""
        cd = concept_dir.to(hidden_state.device)
        cd = cd / cd.norm().clamp(min=NORM_EPS)
        basis = self.concept_bases[layer].float().to(hidden_state.device)
        comp_var = self.component_variances[layer].float().to(
            hidden_state.device
        )

        if hidden_state.dim() == NDIM_VECTOR:
            h = hidden_state.unsqueeze(INDEX_FIRST).float()
        elif hidden_state.dim() == NDIM_MATRIX:
            h = hidden_state.float()
        else:
            b, s, hd = hidden_state.shape
            h = hidden_state.reshape(b * s, hd).float()

        h_new = self._solver_fn(
            h, cd * base_strength,
            concept_basis=basis,
            component_variances=comp_var,
            num_steps=self.num_steps,
            alpha=self.alpha,
            eta=self.eta,
            beta=self.beta,
            alpha_decay=self.alpha_decay,
        )
        h_new = h_new.to(original_dtype)

        if hidden_state.dim() == NDIM_VECTOR:
            return h_new.squeeze(INDEX_FIRST)
        if hidden_state.dim() == NDIM_BATCH_SEQ:
            return h_new.reshape(original_shape)
        return h_new

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
            "concept_directions": {
                str(k): v for k, v in self.concept_directions.items()
            },
            "concept_bases": {
                str(k): v for k, v in self.concept_bases.items()
            },
            "component_variances": {
                str(k): v for k, v in self.component_variances.items()
            },
            "num_steps": self.num_steps,
            "alpha": self.alpha,
            "eta": self.eta,
            "beta": self.beta,
            "alpha_decay": self.alpha_decay,
            "solver": self.solver,
            "layer_variance": {
                str(k): v for k, v in self.layer_variance.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WicherSteeringObject":
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

        concept_dirs = {
            int(k): to_tensor(v)
            for k, v in data["concept_directions"].items()
        }
        concept_bases = {
            int(k): to_tensor(v)
            for k, v in data.get("concept_bases", {}).items()
        }
        comp_variances = {
            int(k): to_tensor(v)
            for k, v in data.get("component_variances", {}).items()
        }

        lv_raw = data.get("layer_variance", {})
        layer_variance = {int(k): float(v) for k, v in lv_raw.items()}

        _REQUIRED_KEYS = ["num_steps", "alpha", "eta", "beta", "alpha_decay"]
        for key in _REQUIRED_KEYS:
            if data.get(key) is None:
                raise InsufficientDataError(
                    reason=f"Missing '{key}' in saved data. Re-train the steering object.",
                )

        return cls(
            metadata=metadata,
            concept_directions=concept_dirs,
            concept_bases=concept_bases,
            component_variances=comp_variances,
            num_steps=data["num_steps"],
            alpha=data["alpha"],
            eta=data["eta"],
            beta=data["beta"],
            alpha_decay=data["alpha_decay"],
            layer_variance=layer_variance,
            solver=data.get("solver", WICHER_DEFAULT_SOLVER),
        )

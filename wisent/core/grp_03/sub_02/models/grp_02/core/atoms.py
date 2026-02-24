from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import torch
from typing import Mapping

from wisent.core.errors import InvalidValueError, InvalidRangeError
from wisent.core.utils import preferred_dtype

# Re-export from helpers
from wisent.core.constants import LOG_EPS, DEFAULT_BASE_STRENGTH
from wisent.core.models.core._atoms_helpers import (
    HookHandleGroup,
    TopLogits,
    GenerationStats,
)

if TYPE_CHECKING:
    from wisent.core.activations.core.atoms import RawActivationMap


__all__ = [
    "SteeringVector",
    "SteeringPlan",
    "HookHandleGroup",
    "TopLogits",
    "GenerationStats",
]


@dataclass(slots=True)
class SteeringVector:
    """
    Single steering vector added to a layer's residual stream (output).

    arguments:
        vector: tensor whose last dim == hidden_size.
        scale:  scalar coefficient (alpha) multiplied before adding.
        normalize: L2-normalize the vector before applying 'scale'.
        layer_description: human-readable description.
    """
    vector: torch.Tensor
    scale: float = DEFAULT_BASE_STRENGTH
    normalize: bool = False
    layer_description: str = ""

    def materialize(self, like: torch.Tensor) -> torch.Tensor:
        """Broadcast + cast the vector so it's addable to 'like' ([B, T, H])."""
        v = self.vector
        if self.normalize and torch.is_floating_point(v):
            denom = torch.linalg.vector_norm(v.float(), dim=-1, keepdim=True).clamp_min(LOG_EPS)
            v = v / denom

        if v.dim() == 1:
            v = v.view(1, 1, -1)
        elif v.dim() == 2:
            v = v.view(1, *v.shape)
        elif v.dim() == 3:
            pass
        else:
            raise InvalidValueError(
                param_name="steering vector shape",
                actual=tuple(v.shape),
                expected="[H], [1,H], [1,1,H], or [B,T,H]"
            )
        return v.to(dtype=like.dtype, device=like.device) * float(self.scale)


@dataclass(slots=True)
class SteeringPlan:
    """Plan for applying steering vectors to multiple layers."""
    layers: dict[str, SteeringVector] = field(default_factory=dict)
    layers_description: list[str] = field(default_factory=list)

    @classmethod
    def from_raw(
        cls,
        raw: Sequence[RawActivationMap] | RawActivationMap | None,
        layers_description: list[str] | None = None,
        scale: float = DEFAULT_BASE_STRENGTH,
        normalize: bool = False,
        weights: Sequence[float] | None = None,
        expected_hidden_size: int | None = None,
    ) -> SteeringPlan:
        """Build a SteeringPlan by merging one or more RawActivationMap(s)."""
        maps = cls._coerce_sequence(raw)
        if layers_description is None:
            layers_description = [f"steering_{i}" for i in range(len(maps))]
        if len(layers_description) != len(maps):
            raise InvalidValueError(param_name="layers_description length", actual=len(layers_description), expected=f"{len(maps)} (number of maps)")
        if not maps:
            plan = cls(layers={}, layers_description=layers_description)
            if expected_hidden_size is not None:
                plan.validate_hidden_size(expected_hidden_size)
            return plan
        w = cls._normalize_weights(len(maps), weights)
        conv = cls._convert_maps(maps)
        order = cls._collect_layer_order(conv)
        out_layers = cls._build_layers(
            layer_order=order, converted_maps=conv, weights=w,
            layers_description=layers_description, scale=scale, normalize=normalize)
        plan = cls(layers=out_layers, layers_description=list(layers_description))
        if expected_hidden_size is not None:
            plan.validate_hidden_size(expected_hidden_size)
        return plan

    def validate_hidden_size(self, hidden_size: int) -> None:
        """Ensure all steering vectors have the specified hidden size."""
        for layer, sv in self.layers.items():
            if sv.vector.shape[-1] != hidden_size:
                raise InvalidValueError(
                    param_name=f"steering vector hidden_size at layer {layer}",
                    actual=sv.vector.shape[-1], expected=hidden_size)

    def is_empty(self) -> bool:
        return not self.layers

    @staticmethod
    def _as_tensor(x: torch.Tensor | float | int) -> torch.Tensor:
        return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)

    @staticmethod
    def _normalize_weights(n: int, weights: Sequence[float] | None) -> torch.Tensor:
        """Return a length-n float32 tensor of weights that sums to 1."""
        if n < 0:
            raise InvalidRangeError(param_name="n", actual=n, min_val=0)
        dtype = preferred_dtype()
        if n == 0:
            return torch.empty(0, dtype=dtype)
        if weights is None:
            return torch.full((n,), 1.0 / n, dtype=dtype)
        w = torch.as_tensor(weights, dtype=dtype)
        if w.numel() != n:
            raise InvalidValueError(param_name="weights length", actual=w.numel(), expected=f"{n} (number of activation maps)")
        s = float(w.sum())
        if abs(s) < LOG_EPS:
            raise InvalidValueError(param_name="weights sum", actual=s, expected="non-zero value for normalization")
        return w / s

    @staticmethod
    def _coerce_sequence(raw):
        if raw is None:
            return []
        if isinstance(raw, Mapping):
            return [raw]
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            return [r or {} for r in raw]
        raise TypeError("raw must be a Mapping[str, Tensor|None], a sequence of them, or None.")

    @classmethod
    def _convert_maps(cls, maps):
        out = []
        for mapping in maps:
            conv = {}
            for k, v in mapping.items():
                if v is None:
                    continue
                conv[str(k)] = cls._as_tensor(v)
            out.append(conv)
        return out

    @staticmethod
    def _collect_layer_order(converted_maps):
        return list(dict.fromkeys(k for m in converted_maps for k in m.keys()))

    @staticmethod
    def _combine_for_layer(layer, converted_maps, weights, layers_description):
        combined = None
        hidden_size = None
        desc_parts = []
        for i, m in enumerate(converted_maps):
            v = m.get(layer)
            if v is None:
                continue
            last_dim = v.shape[-1]
            if hidden_size is None:
                hidden_size = last_dim
            elif last_dim != hidden_size:
                raise InvalidValueError(param_name=f"hidden size at layer {layer}",
                                       actual=last_dim, expected=f"{hidden_size} (consistent across maps)")
            scaled_v = v * float(weights[i])
            if combined is None:
                combined = scaled_v.clone()
            else:
                combined.add_(scaled_v)
            desc = layers_description[i]
            if desc not in desc_parts:
                desc_parts.append(desc)
        return combined, " + ".join(desc_parts)

    @classmethod
    def _build_layers(cls, layer_order, converted_maps, weights, layers_description, scale, normalize):
        out = {}
        for layer in layer_order:
            combined, desc = cls._combine_for_layer(
                layer=layer, converted_maps=converted_maps,
                weights=weights, layers_description=layers_description)
            if combined is None:
                continue
            out[layer] = SteeringVector(
                vector=combined, scale=scale, normalize=normalize, layer_description=desc)
        return out

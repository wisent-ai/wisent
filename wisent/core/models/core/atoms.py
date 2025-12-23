from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import torch
from typing import Mapping

from wisent.core.errors import InvalidValueError, InvalidRangeError
from wisent.core.utils.device import preferred_dtype

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
        vector: tensor whose last dim == hidden_size. Shape may be [H], [1,H], [1,1,H] or [B,T,H].
        scale:  scalar coefficient (alpha) multiplied before adding.
        normalize: L2-normalize the vector (safe + epsilon) before applying 'scale'.
        layer_description: human-readable description of the steering vector. Like "toxic", "biased", etc. 

    example:
        >>> sv = SteeringVector(
        ...     torch.randn(4096),
        ...     scale=0.8,
        ...     normalize=True,
        ...     layer_description="toxic"
        ... )
    """
    vector: torch.Tensor
    scale: float = 1.0
    normalize: bool = False
    layer_description: str = ""

    def materialize(self, like: torch.Tensor) -> torch.Tensor:
        """
        Broadcast + cast the vector so it's addable to 'like' ([B, T, H]).
        Returns a tensor on like.device and like.dtype.

        returns:
            Broadcast + cast the vector so it's addable to 'like' ([B, T, H]).

        raises:
            ValueError: if the vector shape is incompatible.
        """
        v = self.vector
        if self.normalize and torch.is_floating_point(v):
            denom = torch.linalg.vector_norm(v.float(), dim=-1, keepdim=True).clamp_min(1e-12)
            v = v / denom

        if v.dim() == 1:       # [H] -> [1,1,H]
            v = v.view(1, 1, -1)
        elif v.dim() == 2:     # [1,H] -> [1,1,H]  or [B,H] -> [1,B,H] (still broadcastable)
            v = v.view(1, *v.shape)
        elif v.dim() == 3:     # [B,T,H] fine
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
    """
    Plan for applying steering vectors to multiple layers. It supports linear
    combinations of multiple steering layers (for the same llm layer).

    attributes:
        layers:
            dict of layer_name -> SteeringVector to apply at that layer.
        layers_description:
            descriptions corresponding to each RawActivationMap, for example
            "toxic", "biased", etc. These are used to build combined
            per-layer descriptions.
    """
    layers: dict[str, SteeringVector] = field(default_factory=dict)
    layers_description: list[str] = field(default_factory=list)

    @classmethod
    def from_raw(
        cls,
        raw: Sequence[RawActivationMap] | RawActivationMap | None,
        layers_description: list[str] | None = None,
        scale: float = 1.0,
        normalize: bool = False,
        weights: Sequence[float] | None = None,
        expected_hidden_size: int | None = None,
    ) -> SteeringPlan:
        """
        Build a SteeringPlan by merging one or more RawActivationMap(s).
        Each RawActivationMap is: layer_name (str) -> torch.Tensor (or None to skip). 
        Each RawActivationMap corresponds to one description in layers_description.
        The final steering vector at each layer is a weighted sum of the
        contributions from each RawActivationMap.

        arguments:
            raw:
                One or more RawActivationMap(s) to combine
            layers_description:
                Descriptions corresponding to each RawActivationMap, for example
                "toxic", "biased", etc. These are used to build combined
                per-layer descriptions.
            scale:
                Scalar coefficient (alpha) applied to all steering vectors.
            normalize:
                Whether to L2-normalize each steering vector before applying 'scale'.
            weights:
                Optional weights for each RawActivationMap when combining.
                If None, uniform weights are used. Length must match number of maps.
            expected_hidden_size:
                If provided, validate that all steering vectors have this hidden size.
        """
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
            layer_order=order,
            converted_maps=conv,
            weights=w,
            layers_description=layers_description,
            scale=scale,
            normalize=normalize,
        )

        plan = cls(layers=out_layers, layers_description=list(layers_description))

        if expected_hidden_size is not None:
            plan.validate_hidden_size(expected_hidden_size)
        return plan

    def validate_hidden_size(self, hidden_size: int) -> None:
        """
        Ensure all steering vectors have the specified hidden size.
        
        arguments:
            hidden_size: expected hidden size (last dim of steering vectors).

        raises:
            ValueError: if any steering vector has a mismatched hidden size.
        """
        for layer, sv in self.layers.items():
            if sv.vector.shape[-1] != hidden_size:
                raise InvalidValueError(
                    param_name=f"steering vector hidden_size at layer {layer}",
                    actual=sv.vector.shape[-1],
                    expected=hidden_size
                )

    def is_empty(self) -> bool:
        """True if there are no layers."""
        return not self.layers

    @staticmethod
    def _as_tensor(x: torch.Tensor | float | int) -> torch.Tensor:
        return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)

    @staticmethod
    def _normalize_weights(n: int, weights: Sequence[float] | None) -> torch.Tensor:
        """
        Return a length-n float32 tensor of weights that sums to 1.
        If weights is None, use uniform weights. Raises on length mismatch or zero-sum.

        arguments:
            n:
                number of activation maps (must be non-negative).
            weights:
                optional sequence of weights (length n) to normalize. If None, uniform weights are used.
        
        returns:
            A torch.Tensor of shape (n,) with float32 weights summing to 1.
        
        raises:
            ValueError: if n < 0, or if weights length != n, or if weights sum to 0.
        
        example:
            >>> SteeringPlan._normalize_weights(3, [0.2, 0.3, 0.5])
            tensor([0.2000, 0.3000, 0.5000])
            >>> SteeringPlan._normalize_weights(2, None)
            tensor([0.5000, 0.5000])
            >>> SteeringPlan._normalize_weights(0, None)
            tensor([])
        """
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
        if abs(s) < 1e-12:
            raise InvalidValueError(param_name="weights sum", actual=s, expected="non-zero value for normalization")
        return w / s

    @staticmethod
    def _coerce_sequence(
        raw: Sequence[RawActivationMap] | RawActivationMap | None,
    ) -> list[RawActivationMap]:
        """
        Normalize input into a list[RawActivationMap].

        arguments:        
            raw: A raw activation map or a sequence of them.

        returns:
            A list of RawActivationMap.

        raises:
            TypeError: if raw is not a Mapping or a sequence of them.
        
        """
        if raw is None:
            return []
        if isinstance(raw, Mapping):
            return [raw]
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            return [r or {} for r in raw]
        raise TypeError(
            "raw must be a Mapping[str, Tensor|None], a sequence of them, or None."
        )

    @classmethod
    def _convert_maps(cls, maps: list[RawActivationMap]) -> list[dict[str, torch.Tensor]]:
        """
        Convert values to tensors and drop None entries early.

        arguments:
            maps: list of RawActivationMap to convert.

        returns:
            A list of dicts mapping layer names to torch.Tensors.

        raises:
            None
        """
        out: list[dict[str, torch.Tensor]] = []
        for mapping in maps:
            conv: dict[str, torch.Tensor] = {}
            for k, v in mapping.items():
                if v is None:
                    continue
                conv[str(k)] = cls._as_tensor(v)
            out.append(conv)
        return out

    @staticmethod
    def _collect_layer_order(converted_maps: list[dict[str, torch.Tensor]]) -> list[str]:
        """
        First-seen layer order across all maps.

        arguments:
            converted_maps: list of dicts mapping layer names to torch.Tensors.

        returns:
            A list of layer names in first-seen order.

        example:
            >>> maps = [
            ...     {"layer1": torch.randn(4), "layer2": torch.randn(4)},
            ...     {"layer2": torch.randn(4), "layer3": torch.randn(4)},
            ...     {"layer1": torch.randn(4), "layer4": torch.randn(4)},
            ... ]
            >>> SteeringPlan._collect_layer_order(maps)
            ['layer1', 'layer2', 'layer3', 'layer4']
        """
        return list(dict.fromkeys(k for m in converted_maps for k in m.keys()))

    @staticmethod
    def _combine_for_layer(
        layer: str,
        converted_maps: list[dict[str, torch.Tensor]],
        weights: torch.Tensor,
        layers_description: Sequence[str],
    ) -> tuple[torch.Tensor | None, str]:
        """
        Combine weighted vectors for a single layer and build a combined description.

        arguments:
            layer:
                the layer name to combine.
            converted_maps:
                list of dicts mapping layer names to torch.Tensors.
            weights:
                tensor of shape (len(converted_maps),) with float32 weights summing to 1.
            layers_description:
                descriptions corresponding to each converted_map.

        returns:
            A tuple containing the combined tensor (or None) and the description string.

        raises:
            ValueError: if hidden sizes mismatch across maps for this layer.
        
        example:
            >>> maps = [
            ...     {"layer1": torch.tensor([1.0, 2.0]), "layer2": torch.tensor([3.0, 4.0])},
            ...     {"layer2": torch.tensor([5.0, 6.0]), "layer3": torch.tensor([7.0, 8.0])},
            ... ]
            >>> weights = torch.tensor([0.4, 0.6])
            >>> descs = ["toxic", "biased"]
            >>> combined, desc = SteeringPlan._combine_for_layer("layer2", maps, weights, descs)
            >>> print(combined)  # tensor([4.2, 5.2])
            >>> print(desc)      # "toxic + biased"
        """
        combined: torch.Tensor | None = None
        hidden_size: int | None = None
        desc_parts: list[str] = []

        for i, m in enumerate(converted_maps):
            v = m.get(layer)
            if v is None:
                continue

            last_dim = v.shape[-1]
            if hidden_size is None:
                hidden_size = last_dim
            elif last_dim != hidden_size:
                raise InvalidValueError(
                    param_name=f"hidden size at layer {layer}",
                    actual=last_dim,
                    expected=f"{hidden_size} (consistent across maps)"
                )

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
    def _build_layers(
        cls,
        layer_order: list[str],
        converted_maps: list[dict[str, torch.Tensor]],
        weights: torch.Tensor,
        layers_description: Sequence[str],
        scale: float,
        normalize: bool,
    ) -> dict[str, SteeringVector]:
        """
        Iterate over layer_order, combine per-layer contributions, and
        construct SteeringVector objects.

        arguments:
            layer_order:
              list of layer names in first-seen order.
            converted_maps:
               list of dicts mapping layer names to torch.Tensors.
            weights:
               tensor of shape (len(converted_maps),) with float32 weights summing to 1.
            layers_description:
               descriptions corresponding to each converted_map.
        """
        out: dict[str, SteeringVector] = {}
        for layer in layer_order:
            combined, desc = cls._combine_for_layer(
                layer=layer,
                converted_maps=converted_maps,
                weights=weights,
                layers_description=layers_description,
            )
            if combined is None:
                continue
            out[layer] = SteeringVector(
                vector=combined,
                scale=scale,
                normalize=normalize,
                layer_description=desc,
            )
        return out

class HookHandleGroup:
    """
    Manage a set of torch hooks to ensure clean detach.
    """
    def __init__(self) -> None:
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def add(self, handle: torch.utils.hooks.RemovableHandle) -> None:
        self._handles.append(handle)

    def remove_all(self) -> None:
        while self._handles:
            h = self._handles.pop()
            try:
                h.remove()
            except Exception:
                pass


@dataclass(slots=True)
class TopLogits:
    """
    Info for a generated step.

    attributes:
        token_id: 
            chosen token id at this step.
        logit: 
            raw logit for that token.
        prob: 
            softmax probability for that token.
        topk_ids/topk_probs:
            optional top-k for analysis/visualization.
    """
    token_id: int
    logit: float
    prob: float
    topk_ids: list[int] | None = None
    topk_probs: list[float] | None = None


@dataclass(slots=True)
class GenerationStats:
    """
    Per-sequence stats for a generation call.

    attributes:
        tokens:
            the generated token ids (excluding the prompt).
        per_step: 
            optional list of TopLogits, one per generated step.
    """
    tokens: list[int]
    per_step: list[TopLogits] | None = None

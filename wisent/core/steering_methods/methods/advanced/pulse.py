"""
PULSE - Probabilistic Uncertainty-guided Layer Steering Engine.

A layer-adaptive conditional steering method that combines:
- Condition-based gating (from CAST paper)
- Uncertainty-guided intensity modulation (from DAC paper)
- Multi-layer steering with learned per-layer scaling

Key innovations:
1. Sensor layer detects when steering should activate via condition vector
2. Uncertainty (entropy/KL) determines steering intensity
3. Steering applied to configurable layer range with per-layer scaling
4. Supports multiple behaviors with independent conditions

Based on insights from:
- "Dynamic Activation Composition for Multifaceted Steering" (DAC)
- "Conditional Activation Steering" (CAST)
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F

from wisent.core.steering_methods.core.atoms import BaseSteeringMethod
from wisent.core.activations.core.atoms import LayerActivations, RawActivationMap, LayerName
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.errors import InsufficientDataError

__all__ = [
    "PULSEMethod",
    "PULSEConfig",
    "PULSEResult",
]


@dataclass
class PULSEConfig:
    """Configuration for PULSE steering method."""
    
    # Layer configuration
    sensor_layer: Optional[int] = None
    """Layer index where condition gating is computed. If None, auto-computed from num_layers."""
    
    steering_layers: Optional[List[int]] = None
    """Layer indices where steering is applied. If None, auto-computed from num_layers."""
    
    num_layers: Optional[int] = None
    """Total layers in the model. Used to auto-compute steering_layers and sensor_layer."""
    
    per_layer_scaling: bool = True
    """Whether to learn/use different scaling per layer."""
    
    def resolve_layers(self, num_layers: int) -> None:
        """Resolve steering_layers and sensor_layer based on model's num_layers."""
        self.num_layers = num_layers
        if self.sensor_layer is None:
            # 75% through the network
            self.sensor_layer = int(num_layers * 0.75)
        if self.steering_layers is None:
            # Middle to late layers (50% to 85% of network)
            start = int(num_layers * 0.5)
            end = int(num_layers * 0.85)
            self.steering_layers = list(range(start, end))
    
    # Condition gating
    condition_threshold: float = 0.5
    """Threshold for condition activation (0-1)."""
    
    gate_temperature: float = 0.1
    """Temperature for sigmoid gating (lower = sharper)."""
    
    learn_threshold: bool = True
    """Whether to learn optimal threshold via grid search."""
    
    # Uncertainty-guided intensity
    use_entropy_scaling: bool = True
    """Enable entropy-based intensity modulation."""
    
    entropy_floor: float = 0.5
    """Minimum entropy to trigger scaling (below = no steering)."""
    
    entropy_ceiling: float = 2.0
    """Entropy at which max_alpha is reached."""
    
    max_alpha: float = 2.0
    """Maximum steering strength."""
    
    # Training
    optimization_steps: int = 100
    """Steps for condition vector optimization."""
    
    learning_rate: float = 0.01
    """Learning rate for optimization."""
    
    use_caa_init: bool = True
    """Initialize behavior vectors using CAA."""
    
    normalize: bool = True
    """L2-normalize vectors."""
    
    # Threshold search
    threshold_search_steps: int = 20
    """Number of threshold values to try in grid search."""


@dataclass
class PULSEResult:
    """Result containing PULSE steering components."""
    
    behavior_vectors: Dict[LayerName, torch.Tensor]
    """Per-layer behavior/steering vectors."""
    
    condition_vector: torch.Tensor
    """Condition vector for gating (at sensor layer)."""
    
    layer_scales: Dict[LayerName, float]
    """Per-layer scaling factors."""
    
    optimal_threshold: float
    """Learned or configured threshold."""
    
    metadata: Dict[str, Any]
    """Training metadata and diagnostics."""
    
    def get_behavior_vector(self, layer: LayerName) -> Optional[torch.Tensor]:
        """Get behavior vector for a specific layer."""
        return self.behavior_vectors.get(layer)
    
    def get_layer_scale(self, layer: LayerName) -> float:
        """Get scaling factor for a layer."""
        return self.layer_scales.get(layer, 1.0)
    
    def should_steer(self, hidden_state: torch.Tensor, threshold: Optional[float] = None) -> Tuple[bool, float]:
        """
        Determine if steering should activate based on condition.
        
        Args:
            hidden_state: Hidden state at sensor layer [batch, seq, hidden] or [hidden]
            threshold: Override threshold (uses optimal_threshold if None)
            
        Returns:
            Tuple of (should_steer: bool, gate_value: float)
        """
        thresh = threshold if threshold is not None else self.optimal_threshold
        
        # Flatten to [hidden] if needed
        if hidden_state.dim() > 1:
            hidden_state = hidden_state.reshape(-1, hidden_state.shape[-1]).mean(dim=0)
        
        # Cosine similarity with condition vector
        h_norm = F.normalize(hidden_state, p=2, dim=-1)
        c_norm = F.normalize(self.condition_vector, p=2, dim=-1)
        similarity = (h_norm * c_norm).sum()
        
        return similarity.item() > thresh, similarity.item()
    
    def compute_gate(self, hidden_state: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """
        Compute soft gate value for steering.
        
        Args:
            hidden_state: Hidden state at sensor layer
            temperature: Sigmoid temperature
            
        Returns:
            Gate value in [0, 1]
        """
        if hidden_state.dim() > 1:
            hidden_state = hidden_state.reshape(-1, hidden_state.shape[-1]).mean(dim=0)
        
        h_norm = F.normalize(hidden_state, p=2, dim=-1)
        c_norm = F.normalize(self.condition_vector, p=2, dim=-1)
        similarity = (h_norm * c_norm).sum()
        
        gate = torch.sigmoid((similarity - self.optimal_threshold) / temperature)
        return gate


class PULSEMethod(BaseSteeringMethod):
    """
    PULSE - Probabilistic Uncertainty-guided Layer Steering Engine.
    
    A layer-adaptive conditional steering method that:
    - Detects when to steer via condition vector at sensor layer
    - Modulates intensity based on model uncertainty
    - Applies steering across multiple layers with per-layer scaling
    
    Usage:
        method = PULSEMethod(sensor_layer=15, steering_layers=[12,13,14,15,16,17,18])
        result = method.train_pulse(behavior_pairs, condition_pairs)
        
        # At inference:
        gate = result.compute_gate(hidden_at_sensor)
        for layer in steering_layers:
            h[layer] += gate * intensity * result.layer_scales[layer] * result.behavior_vectors[layer]
    """
    
    name = "pulse"
    description = "Layer-adaptive conditional steering with uncertainty-guided intensity"
    
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # steering_layers and sensor_layer default to None - resolved at training time
        # based on actual num_layers in the model
        self.config = PULSEConfig(
            sensor_layer=kwargs.get("sensor_layer", None),  # Auto-resolve from num_layers
            steering_layers=kwargs.get("steering_layers", None),  # Auto-resolve from num_layers
            num_layers=kwargs.get("num_layers", None),
            per_layer_scaling=kwargs.get("per_layer_scaling", True),
            condition_threshold=kwargs.get("condition_threshold", 0.5),
            gate_temperature=kwargs.get("gate_temperature", 0.1),
            learn_threshold=kwargs.get("learn_threshold", True),
            use_entropy_scaling=kwargs.get("use_entropy_scaling", True),
            entropy_floor=kwargs.get("entropy_floor", 0.5),
            entropy_ceiling=kwargs.get("entropy_ceiling", 2.0),
            max_alpha=kwargs.get("max_alpha", 2.0),
            optimization_steps=kwargs.get("optimization_steps", 100),
            learning_rate=kwargs.get("learning_rate", 0.01),
            use_caa_init=kwargs.get("use_caa_init", True),
            normalize=kwargs.get("normalize", True),
            threshold_search_steps=kwargs.get("threshold_search_steps", 20),
        )
        self._training_logs: List[Dict[str, float]] = []
    
    def train(self, pair_set: ContrastivePairSet) -> LayerActivations:
        """
        Train PULSE from contrastive pairs (simplified interface).
        
        Uses the same pairs for both behavior and condition training.
        For full control, use train_pulse() instead.
        
        Args:
            pair_set: ContrastivePairSet with collected activations.
            
        Returns:
            LayerActivations with behavior vectors (for backward compatibility).
        """
        result = self.train_pulse(pair_set)
        
        # Return behavior vectors as LayerActivations
        dtype = self.kwargs.get("dtype", None)
        return LayerActivations(result.behavior_vectors, dtype=dtype)
    
    def train_pulse(
        self,
        behavior_pairs: ContrastivePairSet,
        condition_pairs: Optional[ContrastivePairSet] = None,
    ) -> PULSEResult:
        """
        Full PULSE training with separate behavior and condition pairs.
        
        Args:
            behavior_pairs: Pairs for training behavior vectors (what to steer)
            condition_pairs: Pairs for training condition vector (when to steer)
                            If None, uses behavior_pairs for both.
        
        Returns:
            PULSEResult with all trained components.
        """
        if condition_pairs is None:
            condition_pairs = behavior_pairs
        
        # Detect num_layers from available data and resolve config
        buckets = self._collect_from_set(behavior_pairs)
        if buckets:
            max_layer_idx = 0
            for layer_name in buckets.keys():
                try:
                    layer_idx = int(str(layer_name).split("_")[-1])
                    max_layer_idx = max(max_layer_idx, layer_idx)
                except (ValueError, IndexError):
                    pass
            detected_num_layers = max_layer_idx + 1
            if self.config.steering_layers is None or self.config.sensor_layer is None:
                self.config.resolve_layers(detected_num_layers)
        
        # 1. Train behavior vectors for steering layers
        behavior_vectors = self._train_behavior_vectors(behavior_pairs)
        
        if not behavior_vectors:
            raise InsufficientDataError(reason="No behavior vectors could be trained")
        
        # 2. Train condition vector at sensor layer
        condition_vector = self._train_condition_vector(condition_pairs)
        
        # 3. Learn per-layer scaling
        layer_scales = self._compute_layer_scales(behavior_vectors, behavior_pairs)
        
        # 4. Optionally learn optimal threshold
        if self.config.learn_threshold:
            optimal_threshold = self._learn_threshold(
                condition_vector, condition_pairs
            )
        else:
            optimal_threshold = self.config.condition_threshold
        
        return PULSEResult(
            behavior_vectors=behavior_vectors,
            condition_vector=condition_vector,
            layer_scales=layer_scales,
            optimal_threshold=optimal_threshold,
            metadata={
                "config": self.config.__dict__,
                "num_steering_layers": len(behavior_vectors),
                "training_logs": self._training_logs,
            }
        )
    
    def _train_behavior_vectors(
        self,
        pair_set: ContrastivePairSet,
    ) -> Dict[LayerName, torch.Tensor]:
        """Train behavior vectors using CAA for each steering layer."""
        buckets = self._collect_from_set(pair_set)
        
        behavior_vectors: Dict[LayerName, torch.Tensor] = {}
        steering_layer_names = set()
        
        # Map layer indices to layer names
        for layer_name in buckets.keys():
            try:
                layer_idx = int(layer_name.split("_")[-1]) if "_" in str(layer_name) else int(layer_name)
                if layer_idx in self.config.steering_layers:
                    steering_layer_names.add(layer_name)
            except (ValueError, AttributeError):
                # If we can't parse, include all layers
                steering_layer_names.add(layer_name)
        
        # If no matching layers found, use all available
        if not steering_layer_names:
            steering_layer_names = set(buckets.keys())
        
        for layer_name in steering_layer_names:
            if layer_name not in buckets:
                continue
                
            pos_list, neg_list = buckets[layer_name]
            if not pos_list or not neg_list:
                continue
            
            # Stack activations
            pos_tensor = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
            neg_tensor = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)
            
            # CAA: mean difference
            behavior_vec = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
            
            if self.config.normalize:
                behavior_vec = F.normalize(behavior_vec, p=2, dim=0)
            
            behavior_vectors[layer_name] = behavior_vec
            
            self._training_logs.append({
                "phase": "behavior",
                "layer": str(layer_name),
                "pos_samples": len(pos_list),
                "neg_samples": len(neg_list),
                "vector_norm": behavior_vec.norm().item(),
            })
        
        return behavior_vectors
    
    def _train_condition_vector(
        self,
        pair_set: ContrastivePairSet,
    ) -> torch.Tensor:
        """
        Train condition vector at sensor layer.
        
        The condition vector detects when the behavior (e.g., refusal) is present.
        High similarity with condition vector → steering should activate.
        """
        buckets = self._collect_from_set(pair_set)
        
        # Find sensor layer
        sensor_layer_name = None
        for layer_name in buckets.keys():
            try:
                layer_idx = int(layer_name.split("_")[-1]) if "_" in str(layer_name) else int(layer_name)
                if layer_idx == self.config.sensor_layer:
                    sensor_layer_name = layer_name
                    break
            except (ValueError, AttributeError):
                continue
        
        # Fallback to first available layer if sensor not found
        if sensor_layer_name is None:
            sensor_layer_name = list(buckets.keys())[0] if buckets else None
        
        if sensor_layer_name is None:
            raise InsufficientDataError(reason="No activations found for condition training")
        
        pos_list, neg_list = buckets[sensor_layer_name]
        if not pos_list or not neg_list:
            raise InsufficientDataError(reason="Empty activations at sensor layer")
        
        # Stack activations
        pos_tensor = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
        neg_tensor = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)
        
        if self.config.use_caa_init:
            # Initialize with CAA direction
            condition_vec = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
        else:
            condition_vec = torch.randn(pos_tensor.shape[1])
        
        # Optimize condition vector to maximize separation
        condition_vec = self._optimize_condition_vector(
            condition_vec, pos_tensor, neg_tensor
        )
        
        if self.config.normalize:
            condition_vec = F.normalize(condition_vec, p=2, dim=0)
        
        self._training_logs.append({
            "phase": "condition",
            "sensor_layer": str(sensor_layer_name),
            "pos_samples": len(pos_list),
            "neg_samples": len(neg_list),
            "vector_norm": condition_vec.norm().item(),
        })
        
        return condition_vec
    
    def _optimize_condition_vector(
        self,
        init_vec: torch.Tensor,
        pos_tensor: torch.Tensor,
        neg_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optimize condition vector to maximize separation between pos and neg.
        
        Goal: pos examples should have high similarity, neg examples low similarity.
        """
        condition_vec = init_vec.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([condition_vec], lr=self.config.learning_rate)
        
        best_vec = condition_vec.detach().clone()
        best_separation = -float('inf')
        
        for step in range(self.config.optimization_steps):
            optimizer.zero_grad()
            
            # Normalize for cosine similarity
            c_norm = F.normalize(condition_vec, p=2, dim=0)
            pos_norm = F.normalize(pos_tensor, p=2, dim=1)
            neg_norm = F.normalize(neg_tensor, p=2, dim=1)
            
            # Cosine similarities
            pos_sim = (pos_norm * c_norm).sum(dim=1)  # [N_pos]
            neg_sim = (neg_norm * c_norm).sum(dim=1)  # [N_neg]
            
            # Loss: maximize pos_sim, minimize neg_sim
            # Equivalent to minimizing -pos_sim + neg_sim
            margin = 0.5
            pos_loss = F.relu(margin - pos_sim).mean()
            neg_loss = F.relu(neg_sim + margin).mean()
            
            loss = pos_loss + neg_loss
            
            loss.backward()
            optimizer.step()
            
            # Track best
            separation = pos_sim.mean() - neg_sim.mean()
            if separation.item() > best_separation:
                best_separation = separation.item()
                best_vec = condition_vec.detach().clone()
            
            if step % 20 == 0:
                self._training_logs.append({
                    "phase": "condition_opt",
                    "step": step,
                    "loss": loss.item(),
                    "pos_sim_mean": pos_sim.mean().item(),
                    "neg_sim_mean": neg_sim.mean().item(),
                    "separation": separation.item(),
                })
        
        return best_vec
    
    def _compute_layer_scales(
        self,
        behavior_vectors: Dict[LayerName, torch.Tensor],
        pair_set: ContrastivePairSet,
    ) -> Dict[LayerName, float]:
        """
        Compute per-layer scaling factors.
        
        Uses the separation score at each layer to determine relative importance.
        """
        if not self.config.per_layer_scaling:
            return {layer: 1.0 for layer in behavior_vectors.keys()}
        
        buckets = self._collect_from_set(pair_set)
        layer_scores: Dict[LayerName, float] = {}
        
        for layer_name, vec in behavior_vectors.items():
            if layer_name not in buckets:
                layer_scores[layer_name] = 1.0
                continue
            
            pos_list, neg_list = buckets[layer_name]
            if not pos_list or not neg_list:
                layer_scores[layer_name] = 1.0
                continue
            
            pos_tensor = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
            neg_tensor = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)
            
            # Compute separation score
            vec_norm = F.normalize(vec, p=2, dim=0)
            pos_proj = (pos_tensor * vec_norm).sum(dim=1).mean()
            neg_proj = (neg_tensor * vec_norm).sum(dim=1).mean()
            separation = (pos_proj - neg_proj).item()
            
            layer_scores[layer_name] = max(0.1, separation)  # Floor at 0.1
        
        # Normalize scales to have mean 1.0
        if layer_scores:
            mean_score = sum(layer_scores.values()) / len(layer_scores)
            if mean_score > 0:
                layer_scores = {k: v / mean_score for k, v in layer_scores.items()}
        
        return layer_scores
    
    def _learn_threshold(
        self,
        condition_vector: torch.Tensor,
        pair_set: ContrastivePairSet,
    ) -> float:
        """
        Learn optimal threshold via grid search.
        
        Finds threshold that best separates pos (should steer) from neg (shouldn't).
        """
        buckets = self._collect_from_set(pair_set)
        
        # Find sensor layer activations
        sensor_layer_name = None
        for layer_name in buckets.keys():
            try:
                layer_idx = int(layer_name.split("_")[-1]) if "_" in str(layer_name) else int(layer_name)
                if layer_idx == self.config.sensor_layer:
                    sensor_layer_name = layer_name
                    break
            except (ValueError, AttributeError):
                continue
        
        if sensor_layer_name is None:
            return self.config.condition_threshold
        
        pos_list, neg_list = buckets[sensor_layer_name]
        if not pos_list or not neg_list:
            return self.config.condition_threshold
        
        pos_tensor = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
        neg_tensor = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)
        
        # Compute similarities
        c_norm = F.normalize(condition_vector, p=2, dim=0)
        pos_norm = F.normalize(pos_tensor, p=2, dim=1)
        neg_norm = F.normalize(neg_tensor, p=2, dim=1)
        
        pos_sims = (pos_norm * c_norm).sum(dim=1)
        neg_sims = (neg_norm * c_norm).sum(dim=1)
        
        # Grid search for best threshold
        min_sim = min(pos_sims.min().item(), neg_sims.min().item())
        max_sim = max(pos_sims.max().item(), neg_sims.max().item())
        
        best_threshold = self.config.condition_threshold
        best_accuracy = 0.0
        
        for i in range(self.config.threshold_search_steps):
            threshold = min_sim + (max_sim - min_sim) * i / self.config.threshold_search_steps
            
            # Accuracy: pos should be above threshold, neg below
            pos_correct = (pos_sims > threshold).float().mean().item()
            neg_correct = (neg_sims <= threshold).float().mean().item()
            accuracy = (pos_correct + neg_correct) / 2
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        self._training_logs.append({
            "phase": "threshold_search",
            "best_threshold": best_threshold,
            "best_accuracy": best_accuracy,
            "pos_sim_range": [pos_sims.min().item(), pos_sims.max().item()],
            "neg_sim_range": [neg_sims.min().item(), neg_sims.max().item()],
        })
        
        return best_threshold
    
    def _collect_from_set(
        self, pair_set: ContrastivePairSet
    ) -> Dict[LayerName, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """Build {layer_name: ([pos tensors...], [neg tensors...])} from pairs."""
        from collections import defaultdict
        
        buckets: Dict[LayerName, Tuple[List[torch.Tensor], List[torch.Tensor]]] = defaultdict(lambda: ([], []))
        
        for pair in pair_set.pairs:
            pos_la = getattr(pair.positive_response, "layers_activations", None)
            neg_la = getattr(pair.negative_response, "layers_activations", None)
            
            if pos_la is None or neg_la is None:
                continue
            
            layer_names = set(pos_la.to_dict().keys()) | set(neg_la.to_dict().keys())
            for layer in layer_names:
                p = pos_la.to_dict().get(layer, None) if pos_la is not None else None
                n = neg_la.to_dict().get(layer, None) if neg_la is not None else None
                if isinstance(p, torch.Tensor) and isinstance(n, torch.Tensor):
                    buckets[layer][0].append(p)
                    buckets[layer][1].append(n)
        
        return buckets
    
    def get_training_logs(self) -> List[Dict[str, Any]]:
        """Return training logs."""
        return self._training_logs


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy from logits.
    
    Args:
        logits: Raw logits [batch, vocab] or [vocab]
        
    Returns:
        Entropy value(s)
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def compute_intensity_from_entropy(
    entropy: torch.Tensor,
    floor: float = 0.5,
    ceiling: float = 2.0,
    max_alpha: float = 2.0,
) -> torch.Tensor:
    """
    Compute steering intensity from entropy.
    
    Higher entropy → higher intensity (model is uncertain → steer more)
    
    Args:
        entropy: Entropy value(s)
        floor: Min entropy (below = 0 intensity)
        ceiling: Max entropy (above = max intensity)
        max_alpha: Maximum steering strength
        
    Returns:
        Intensity value(s) in [0, max_alpha]
    """
    # Normalize to [0, 1] range
    normalized = (entropy - floor) / (ceiling - floor)
    normalized = torch.clamp(normalized, 0.0, 1.0)
    
    # Scale to [0, max_alpha]
    return normalized * max_alpha

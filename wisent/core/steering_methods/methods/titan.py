"""
TITAN - Total Integrated Targeted Activation Navigation.

The most powerful steering method that jointly optimizes:
- WHAT to steer (multi-directional manifold discovery)
- WHEN to steer (learned gating network)
- WHERE to steer (learned layer importance)
- HOW MUCH to steer (input-dependent intensity prediction)

Combines insights from:
- PRISM: Gradient-optimized multi-directional discovery
- PULSE: Conditional gating and layer-adaptive steering
- DAC/CAST: Dynamic intensity modulation

Key innovations:
1. Learned gating head (MLP) instead of fixed cosine similarity
2. Per-input intensity prediction instead of global strength
3. Direction weighting within the manifold
4. Joint end-to-end optimization of all components
5. Sparse layer activation for minimal side effects
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F

from wisent.core.steering_methods.core.atoms import BaseSteeringMethod
from wisent.core.activations.core.atoms import LayerActivations, RawActivationMap, LayerName
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.errors import InsufficientDataError

__all__ = [
    "TITANMethod",
    "TITANConfig",
    "TITANResult",
    "GatingNetwork",
    "IntensityNetwork",
]


@dataclass
class TITANConfig:
    """Configuration for TITAN steering method."""
    
    # Manifold configuration
    num_directions: int = 5
    """Number of directions per layer in the steering manifold."""
    
    # Layer configuration  
    steering_layers: List[int] = field(default_factory=lambda: [10, 11, 12, 13, 14, 15, 16, 17, 18])
    """Layer indices where steering can be applied."""
    
    sensor_layer: int = 15
    """Primary layer for gating decisions."""
    
    # Network architecture
    gate_hidden_dim: int = 128
    """Hidden dimension for gating network."""
    
    intensity_hidden_dim: int = 64
    """Hidden dimension for intensity network."""
    
    # Training
    optimization_steps: int = 200
    """Total optimization steps."""
    
    learning_rate: float = 0.005
    """Learning rate for all components."""
    
    warmup_steps: int = 20
    """Steps to warmup (train manifold only before adding networks)."""
    
    # Loss weights
    behavior_weight: float = 1.0
    """Weight for behavior effectiveness loss."""
    
    retain_weight: float = 0.2
    """Weight for retain loss (minimize side effects)."""
    
    sparse_weight: float = 0.05
    """Weight for sparsity loss (encourage sparse layer activation)."""
    
    smooth_weight: float = 0.02
    """Weight for smoothness loss (penalize abrupt intensity changes)."""
    
    independence_weight: float = 0.03
    """Weight for direction independence loss."""
    
    # Constraints
    max_alpha: float = 3.0
    """Maximum steering intensity."""
    
    gate_temperature: float = 0.5
    """Temperature for gate sigmoid."""
    
    min_cosine_similarity: float = 0.2
    """Minimum cosine similarity between directions."""
    
    max_cosine_similarity: float = 0.9
    """Maximum cosine similarity (avoid redundancy)."""
    
    # Initialization
    use_caa_init: bool = True
    """Initialize primary direction with CAA."""
    
    normalize: bool = True
    """L2-normalize directions."""


class GatingNetwork(nn.Module):
    """
    Learned gating network that predicts whether steering should activate.
    
    Takes hidden states from sensor layer, outputs gate value in [0, 1].
    """
    
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
        """
        Predict gate value.
        
        Args:
            h: Hidden state [batch, hidden_dim] or [hidden_dim]
            temperature: Sigmoid temperature
            
        Returns:
            Gate value in [0, 1]
        """
        if h.dim() == 1:
            h = h.unsqueeze(0)
        logit = self.net(h).squeeze(-1)
        return torch.sigmoid(logit / temperature)


class IntensityNetwork(nn.Module):
    """
    Learned intensity network that predicts per-layer steering strength.
    
    Takes hidden states, outputs intensity values for each steering layer.
    """
    
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
        """
        Predict per-layer intensity.
        
        Args:
            h: Hidden state [batch, hidden_dim] or [hidden_dim]
            
        Returns:
            Intensity values [batch, num_layers] in [0, max_alpha]
        """
        if h.dim() == 1:
            h = h.unsqueeze(0)
        raw = self.net(h)
        return torch.sigmoid(raw) * self.max_alpha


class DirectionWeightNetwork(nn.Module):
    """
    Learned network that predicts weights for combining directions in manifold.
    """
    
    def __init__(self, input_dim: int, num_directions: int, hidden_dim: int = 32):
        super().__init__()
        self.num_directions = num_directions
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_directions),
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Predict direction weights.
        
        Args:
            h: Hidden state [batch, hidden_dim] or [hidden_dim]
            
        Returns:
            Softmax weights [batch, num_directions]
        """
        if h.dim() == 1:
            h = h.unsqueeze(0)
        logits = self.net(h)
        return F.softmax(logits, dim=-1)


@dataclass
class TITANResult:
    """Result containing all TITAN components."""
    
    # Manifold
    directions: Dict[LayerName, torch.Tensor]
    """Per-layer directions [num_directions, hidden_dim]."""
    
    # Networks
    gate_network: GatingNetwork
    """Learned gating network."""
    
    intensity_network: IntensityNetwork
    """Learned intensity prediction network."""
    
    direction_weights: Dict[LayerName, torch.Tensor]
    """Learned static direction weights per layer [num_directions]."""
    
    # Layer mapping
    layer_order: List[LayerName]
    """Ordered list of layer names (for intensity network output indexing)."""
    
    # Metadata
    metadata: Dict[str, Any]
    """Training metadata and diagnostics."""
    
    def get_effective_direction(self, layer: LayerName, h: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get the effective steering direction for a layer.
        
        If h is provided, uses direction weight network (if available).
        Otherwise uses static learned weights.
        
        Args:
            layer: Layer name
            h: Optional hidden state for dynamic weighting
            
        Returns:
            Weighted direction vector [hidden_dim]
        """
        dirs = self.directions[layer]  # [K, H]
        weights = self.direction_weights[layer]  # [K]
        
        # Weighted sum of directions
        effective = (weights.unsqueeze(-1) * dirs).sum(dim=0)  # [H]
        return F.normalize(effective, p=2, dim=-1)
    
    def predict_gate(self, h: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
        """Predict gate value from hidden state."""
        return self.gate_network(h, temperature)
    
    def predict_intensity(self, h: torch.Tensor) -> Dict[LayerName, torch.Tensor]:
        """
        Predict per-layer intensity from hidden state.
        
        Returns dict mapping layer name to intensity value.
        """
        intensities = self.intensity_network(h)  # [batch, num_layers]
        if intensities.dim() == 1:
            intensities = intensities.unsqueeze(0)
        
        result = {}
        for i, layer in enumerate(self.layer_order):
            result[layer] = intensities[:, i]
        return result
    
    def apply_steering(
        self,
        activations: Dict[LayerName, torch.Tensor],
        sensor_hidden: torch.Tensor,
    ) -> Dict[LayerName, torch.Tensor]:
        """
        Apply full TITAN steering to activations.
        
        Args:
            activations: Dict of layer activations
            sensor_hidden: Hidden state at sensor layer for gating/intensity
            
        Returns:
            Steered activations
        """
        # Get gate and intensities
        gate = self.predict_gate(sensor_hidden)
        intensities = self.predict_intensity(sensor_hidden)
        
        steered = {}
        for layer, h in activations.items():
            if layer in self.directions:
                direction = self.get_effective_direction(layer)
                intensity = intensities.get(layer, torch.ones(1))
                
                # Apply steering: h' = h + gate * intensity * direction
                if h.dim() == 1:
                    steered[layer] = h + gate.squeeze() * intensity.squeeze() * direction
                else:
                    steered[layer] = h + gate.unsqueeze(-1) * intensity.unsqueeze(-1) * direction
            else:
                steered[layer] = h
        
        return steered


class TITANMethod(BaseSteeringMethod):
    """
    TITAN - Total Integrated Targeted Activation Navigation.
    
    The most powerful steering method combining:
    - Multi-directional manifold discovery (PRISM-style)
    - Learned gating network (improved PULSE)
    - Per-input intensity prediction
    - Direction weighting within manifold
    - Joint end-to-end optimization
    
    Usage:
        method = TITANMethod(num_directions=5, optimization_steps=200)
        result = method.train_titan(pair_set)
        
        # At inference:
        steered = result.apply_steering(activations, sensor_hidden)
    """
    
    name = "titan"
    description = "Total Integrated Targeted Activation Navigation - joint optimization of manifold, gating, and intensity"
    
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = TITANConfig(
            num_directions=kwargs.get("num_directions", 5),
            steering_layers=kwargs.get("steering_layers", [10, 11, 12, 13, 14, 15, 16, 17, 18]),
            sensor_layer=kwargs.get("sensor_layer", 15),
            gate_hidden_dim=kwargs.get("gate_hidden_dim", 128),
            intensity_hidden_dim=kwargs.get("intensity_hidden_dim", 64),
            optimization_steps=kwargs.get("optimization_steps", 200),
            learning_rate=kwargs.get("learning_rate", 0.005),
            warmup_steps=kwargs.get("warmup_steps", 20),
            behavior_weight=kwargs.get("behavior_weight", 1.0),
            retain_weight=kwargs.get("retain_weight", 0.2),
            sparse_weight=kwargs.get("sparse_weight", 0.05),
            smooth_weight=kwargs.get("smooth_weight", 0.02),
            independence_weight=kwargs.get("independence_weight", 0.03),
            max_alpha=kwargs.get("max_alpha", 3.0),
            gate_temperature=kwargs.get("gate_temperature", 0.5),
            min_cosine_similarity=kwargs.get("min_cosine_similarity", 0.2),
            max_cosine_similarity=kwargs.get("max_cosine_similarity", 0.9),
            use_caa_init=kwargs.get("use_caa_init", True),
            normalize=kwargs.get("normalize", True),
        )
        self._training_logs: List[Dict[str, float]] = []
    
    def train(self, pair_set: ContrastivePairSet) -> LayerActivations:
        """
        Train TITAN (simplified interface for compatibility).
        
        Returns primary directions as LayerActivations.
        """
        result = self.train_titan(pair_set)
        
        # Return effective directions
        primary_map: RawActivationMap = {}
        for layer in result.directions:
            primary_map[layer] = result.get_effective_direction(layer)
        
        dtype = self.kwargs.get("dtype", None)
        agg = self.kwargs.get("activation_aggregation_strategy", None)
        return LayerActivations(primary_map, activation_aggregation_strategy=agg, dtype=dtype)
    
    def train_titan(self, pair_set: ContrastivePairSet) -> TITANResult:
        """
        Full TITAN training with all components.
        
        Args:
            pair_set: ContrastivePairSet with collected activations.
            
        Returns:
            TITANResult with manifold, networks, and metadata.
        """
        # Collect activations
        buckets = self._collect_from_set(pair_set)
        
        if not buckets:
            raise InsufficientDataError(reason="No valid activation pairs found")
        
        # Filter to steering layers and determine hidden dim
        layer_names = []
        hidden_dim = None
        
        for layer_name in sorted(buckets.keys()):
            pos_list, neg_list = buckets[layer_name]
            if not pos_list or not neg_list:
                continue
            
            # Check if layer matches steering_layers config
            try:
                layer_idx = int(str(layer_name).split("_")[-1])
                if layer_idx not in self.config.steering_layers:
                    continue
            except (ValueError, IndexError):
                pass  # Include if can't parse
            
            layer_names.append(layer_name)
            if hidden_dim is None:
                hidden_dim = pos_list[0].reshape(-1).shape[0]
        
        if not layer_names or hidden_dim is None:
            raise InsufficientDataError(reason="No valid steering layers found")
        
        num_layers = len(layer_names)
        
        # Initialize components
        directions = self._initialize_directions(buckets, layer_names, hidden_dim)
        gate_network = GatingNetwork(hidden_dim, self.config.gate_hidden_dim)
        intensity_network = IntensityNetwork(
            hidden_dim, num_layers, 
            self.config.intensity_hidden_dim, 
            self.config.max_alpha
        )
        direction_weights = {
            layer: torch.ones(self.config.num_directions) / self.config.num_directions
            for layer in layer_names
        }
        
        # Make direction weights trainable
        direction_weight_params = {
            layer: nn.Parameter(torch.zeros(self.config.num_directions))
            for layer in layer_names
        }
        
        # Prepare data tensors
        data = self._prepare_data_tensors(buckets, layer_names)
        
        # Find sensor layer
        sensor_layer = self._find_sensor_layer(layer_names)
        
        # Joint optimization
        directions, gate_network, intensity_network, direction_weights = self._joint_optimization(
            directions=directions,
            gate_network=gate_network,
            intensity_network=intensity_network,
            direction_weight_params=direction_weight_params,
            data=data,
            layer_names=layer_names,
            sensor_layer=sensor_layer,
        )
        
        return TITANResult(
            directions=directions,
            gate_network=gate_network,
            intensity_network=intensity_network,
            direction_weights=direction_weights,
            layer_order=layer_names,
            metadata={
                "config": self.config.__dict__,
                "num_layers": num_layers,
                "hidden_dim": hidden_dim,
                "sensor_layer": sensor_layer,
                "training_logs": self._training_logs,
            }
        )
    
    def _initialize_directions(
        self,
        buckets: Dict[LayerName, Tuple[List[torch.Tensor], List[torch.Tensor]]],
        layer_names: List[LayerName],
        hidden_dim: int,
    ) -> Dict[LayerName, torch.Tensor]:
        """Initialize direction manifold for each layer."""
        directions = {}
        K = self.config.num_directions
        
        for layer in layer_names:
            pos_list, neg_list = buckets[layer]
            
            pos_tensor = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
            neg_tensor = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)
            
            # Initialize directions
            dirs = torch.randn(K, hidden_dim)
            
            if self.config.use_caa_init:
                # First direction: CAA
                caa_dir = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
                caa_dir = F.normalize(caa_dir, p=2, dim=0)
                dirs[0] = caa_dir
                
                # Others: perturbations
                for i in range(1, K):
                    noise = torch.randn(hidden_dim) * 0.3
                    dirs[i] = F.normalize(caa_dir + noise, p=2, dim=0)
            else:
                dirs = F.normalize(dirs, p=2, dim=1)
            
            directions[layer] = dirs
        
        return directions
    
    def _prepare_data_tensors(
        self,
        buckets: Dict[LayerName, Tuple[List[torch.Tensor], List[torch.Tensor]]],
        layer_names: List[LayerName],
    ) -> Dict[str, Dict[LayerName, torch.Tensor]]:
        """Prepare stacked tensors for training."""
        data = {"pos": {}, "neg": {}}
        
        for layer in layer_names:
            pos_list, neg_list = buckets[layer]
            data["pos"][layer] = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
            data["neg"][layer] = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)
        
        return data
    
    def _find_sensor_layer(self, layer_names: List[LayerName]) -> LayerName:
        """Find the sensor layer from available layers."""
        for layer in layer_names:
            try:
                layer_idx = int(str(layer).split("_")[-1])
                if layer_idx == self.config.sensor_layer:
                    return layer
            except (ValueError, IndexError):
                continue
        
        # Fallback to middle layer
        return layer_names[len(layer_names) // 2]
    
    def _joint_optimization(
        self,
        directions: Dict[LayerName, torch.Tensor],
        gate_network: GatingNetwork,
        intensity_network: IntensityNetwork,
        direction_weight_params: Dict[LayerName, nn.Parameter],
        data: Dict[str, Dict[LayerName, torch.Tensor]],
        layer_names: List[LayerName],
        sensor_layer: LayerName,
    ) -> Tuple[Dict[LayerName, torch.Tensor], GatingNetwork, IntensityNetwork, Dict[LayerName, torch.Tensor]]:
        """
        Joint end-to-end optimization of all TITAN components.
        """
        # Make directions trainable
        direction_params = {layer: nn.Parameter(dirs.clone()) for layer, dirs in directions.items()}
        
        # Collect all parameters
        all_params = []
        all_params.extend(direction_params.values())
        all_params.extend(gate_network.parameters())
        all_params.extend(intensity_network.parameters())
        all_params.extend(direction_weight_params.values())
        
        optimizer = torch.optim.AdamW(all_params, lr=self.config.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.optimization_steps, eta_min=self.config.learning_rate * 0.1
        )
        
        best_loss = float('inf')
        best_state = None
        
        for step in range(self.config.optimization_steps):
            optimizer.zero_grad()
            
            # Compute effective directions (weighted sum)
            effective_dirs = {}
            for layer in layer_names:
                weights = F.softmax(direction_weight_params[layer], dim=0)
                dirs = direction_params[layer]
                dirs_norm = F.normalize(dirs, p=2, dim=1)
                effective_dirs[layer] = (weights.unsqueeze(-1) * dirs_norm).sum(dim=0)
            
            # Get sensor layer data
            pos_sensor = data["pos"][sensor_layer]
            neg_sensor = data["neg"][sensor_layer]
            
            # Predict gates
            pos_gate = gate_network(pos_sensor, self.config.gate_temperature)
            neg_gate = gate_network(neg_sensor, self.config.gate_temperature)
            
            # Predict intensities
            pos_intensity = intensity_network(pos_sensor)  # [N_pos, num_layers]
            neg_intensity = intensity_network(neg_sensor)  # [N_neg, num_layers]
            
            # Compute losses
            loss, loss_components = self._compute_titan_loss(
                direction_params=direction_params,
                effective_dirs=effective_dirs,
                pos_gate=pos_gate,
                neg_gate=neg_gate,
                pos_intensity=pos_intensity,
                neg_intensity=neg_intensity,
                data=data,
                layer_names=layer_names,
                step=step,
            )
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Apply constraints to directions
            with torch.no_grad():
                for layer in layer_names:
                    direction_params[layer].data = self._apply_direction_constraints(
                        direction_params[layer].data
                    )
            
            # Track best
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {
                    "directions": {l: p.detach().clone() for l, p in direction_params.items()},
                    "gate_network": {k: v.detach().clone() for k, v in gate_network.state_dict().items()},
                    "intensity_network": {k: v.detach().clone() for k, v in intensity_network.state_dict().items()},
                    "direction_weights": {l: F.softmax(p.detach().clone(), dim=0) for l, p in direction_weight_params.items()},
                }
            
            # Log
            if step % 20 == 0 or step == self.config.optimization_steps - 1:
                self._training_logs.append({
                    "step": step,
                    "total_loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    **{k: v.item() for k, v in loss_components.items()},
                    "pos_gate_mean": pos_gate.mean().item(),
                    "neg_gate_mean": neg_gate.mean().item(),
                    "pos_intensity_mean": pos_intensity.mean().item(),
                    "neg_intensity_mean": neg_intensity.mean().item(),
                })
        
        # Restore best state
        if best_state is not None:
            final_directions = best_state["directions"]
            gate_network.load_state_dict(best_state["gate_network"])
            intensity_network.load_state_dict(best_state["intensity_network"])
            final_weights = best_state["direction_weights"]
        else:
            final_directions = {l: p.detach() for l, p in direction_params.items()}
            final_weights = {l: F.softmax(p.detach(), dim=0) for l, p in direction_weight_params.items()}
        
        # Final normalization
        if self.config.normalize:
            final_directions = {l: F.normalize(d, p=2, dim=1) for l, d in final_directions.items()}
        
        return final_directions, gate_network, intensity_network, final_weights
    
    def _compute_titan_loss(
        self,
        direction_params: Dict[LayerName, nn.Parameter],
        effective_dirs: Dict[LayerName, torch.Tensor],
        pos_gate: torch.Tensor,
        neg_gate: torch.Tensor,
        pos_intensity: torch.Tensor,
        neg_intensity: torch.Tensor,
        data: Dict[str, Dict[LayerName, torch.Tensor]],
        layer_names: List[LayerName],
        step: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the full TITAN loss.
        
        Components:
        1. Behavior loss: Steering should be effective on positives
        2. Retain loss: Minimal effect on negatives
        3. Sparse loss: Encourage sparse layer activation
        4. Smooth loss: Penalize intensity variance across layers
        5. Independence loss: Directions should be independent
        6. Gate loss: Gate should discriminate pos from neg
        """
        loss_components = {}
        
        # 1. Behavior loss - steering effectiveness on positives
        behavior_loss = torch.tensor(0.0)
        for i, layer in enumerate(layer_names):
            pos_data = data["pos"][layer]
            eff_dir = effective_dirs[layer]
            
            # Projection of positive samples onto direction
            pos_proj = (pos_data * eff_dir).sum(dim=1)
            
            # We want high projection (margin loss)
            margin = 1.0
            behavior_loss = behavior_loss + F.relu(margin - pos_proj).mean()
        
        behavior_loss = behavior_loss / len(layer_names)
        loss_components["behavior"] = behavior_loss
        
        # 2. Retain loss - minimal effect on negatives
        retain_loss = torch.tensor(0.0)
        for i, layer in enumerate(layer_names):
            neg_data = data["neg"][layer]
            eff_dir = effective_dirs[layer]
            
            # We want LOW projection on negatives
            neg_proj = (neg_data * eff_dir).sum(dim=1).abs()
            retain_loss = retain_loss + neg_proj.mean()
        
        retain_loss = retain_loss / len(layer_names)
        loss_components["retain"] = retain_loss
        
        # 3. Sparse loss - encourage sparse layer activation
        # Penalize uniform intensity distribution
        pos_intensity_norm = pos_intensity / (pos_intensity.sum(dim=1, keepdim=True) + 1e-8)
        sparse_loss = -torch.mean(torch.sum(pos_intensity_norm * torch.log(pos_intensity_norm + 1e-8), dim=1))
        sparse_loss = -sparse_loss  # We want LOW entropy (sparse)
        loss_components["sparse"] = sparse_loss
        
        # 4. Smooth loss - penalize abrupt intensity changes
        if pos_intensity.shape[1] > 1:
            intensity_diff = (pos_intensity[:, 1:] - pos_intensity[:, :-1]).abs()
            smooth_loss = intensity_diff.mean()
        else:
            smooth_loss = torch.tensor(0.0)
        loss_components["smooth"] = smooth_loss
        
        # 5. Independence loss - directions within manifold
        independence_loss = torch.tensor(0.0)
        for layer in layer_names:
            dirs = direction_params[layer]
            dirs_norm = F.normalize(dirs, p=2, dim=1)
            K = dirs_norm.shape[0]
            
            if K > 1:
                cos_sim = dirs_norm @ dirs_norm.T
                mask = 1 - torch.eye(K, device=cos_sim.device)
                
                # Penalize too high or too low similarity
                too_similar = F.relu(cos_sim - self.config.max_cosine_similarity)
                too_different = F.relu(self.config.min_cosine_similarity - cos_sim)
                independence_loss = independence_loss + ((too_similar + too_different) * mask).mean()
        
        independence_loss = independence_loss / len(layer_names)
        loss_components["independence"] = independence_loss
        
        # 6. Gate discrimination loss
        # Pos should have high gate, neg should have low gate
        gate_loss = F.relu(0.5 - pos_gate).mean() + F.relu(neg_gate - 0.5).mean()
        loss_components["gate"] = gate_loss
        
        # Combine losses with warmup
        if step < self.config.warmup_steps:
            # Warmup: focus on manifold + basic gate
            total_loss = (
                self.config.behavior_weight * behavior_loss +
                self.config.retain_weight * retain_loss +
                self.config.independence_weight * independence_loss +
                0.5 * gate_loss
            )
        else:
            # Full training
            total_loss = (
                self.config.behavior_weight * behavior_loss +
                self.config.retain_weight * retain_loss +
                self.config.sparse_weight * sparse_loss +
                self.config.smooth_weight * smooth_loss +
                self.config.independence_weight * independence_loss +
                gate_loss
            )
        
        return total_loss, loss_components
    
    def _apply_direction_constraints(self, directions: torch.Tensor) -> torch.Tensor:
        """Apply constraints to direction manifold."""
        # Normalize
        directions = F.normalize(directions, p=2, dim=1)
        
        # Cone constraint: all directions in same half-space as first
        if directions.shape[0] > 1:
            primary = directions[0:1]
            for i in range(1, directions.shape[0]):
                cos_sim = (directions[i:i+1] * primary).sum()
                if cos_sim < 0:
                    directions[i] = -directions[i]
        
        return directions
    
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

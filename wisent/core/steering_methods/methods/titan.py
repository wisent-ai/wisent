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
    "GeometryAdaptation",
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
    steering_layers: Optional[List[int]] = None
    """Layer indices where steering can be applied. If None, auto-computed from num_layers."""
    
    sensor_layer: Optional[int] = None
    """Primary layer for gating decisions. If None, auto-computed from num_layers."""
    
    num_layers: Optional[int] = None
    """Total layers in the model. Used to auto-compute steering_layers and sensor_layer."""
    
    def resolve_layers(self, num_layers: int) -> None:
        """Resolve steering_layers and sensor_layer based on model's num_layers."""
        self.num_layers = num_layers
        if self.sensor_layer is None:
            # 75% through the network
            self.sensor_layer = int(num_layers * 0.75)
        if self.steering_layers is None:
            # Middle to late layers (50% to 90% of network)
            start = int(num_layers * 0.5)
            end = int(num_layers * 0.9)
            self.steering_layers = list(range(start, end))
    
    # Network architecture
    gate_hidden_dim: Optional[int] = None
    """Hidden dimension for gating network. If None, auto-computed as hidden_dim // 16."""
    
    intensity_hidden_dim: Optional[int] = None
    """Hidden dimension for intensity network. If None, auto-computed as hidden_dim // 32."""
    
    def resolve_network_dims(self, hidden_dim: int) -> None:
        """Resolve network dimensions based on model's hidden dimension."""
        if self.gate_hidden_dim is None:
            # Scale with model size, but clamp to reasonable range [32, 512]
            self.gate_hidden_dim = max(32, min(512, hidden_dim // 16))
        if self.intensity_hidden_dim is None:
            # Scale with model size, but clamp to reasonable range [16, 256]
            self.intensity_hidden_dim = max(16, min(256, hidden_dim // 32))
    
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
    
    # Geometry-adaptive configuration
    adapt_to_geometry: bool = True
    """Whether to analyze geometry and adapt configuration."""
    
    geometry_analysis_layer: Optional[int] = None
    """Layer to use for geometry analysis. If None, uses sensor_layer."""
    
    linear_threshold: float = 0.8
    """If linear score > threshold, simplify to single direction."""
    
    skip_gating_if_linear: bool = True
    """Skip gating network if structure is clearly linear."""
    
    auto_num_directions: bool = True
    """Automatically determine num_directions based on geometry."""


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
class GeometryAdaptation:
    """Results from geometry analysis and adaptations made."""
    
    detected_structure: str
    """Primary detected structure type (linear, cone, manifold, etc.)."""
    
    structure_scores: Dict[str, float]
    """Scores for all structure types."""
    
    adaptations_made: List[str]
    """List of adaptations applied based on geometry."""
    
    original_num_directions: int
    """Originally configured num_directions."""
    
    adapted_num_directions: int
    """Num directions after adaptation."""
    
    gating_enabled: bool
    """Whether gating network is enabled."""
    
    recommendation: str
    """Steering method recommendation based on geometry."""


@dataclass
class TITANResult:
    """Result containing all TITAN components."""
    
    # Manifold
    directions: Dict[LayerName, torch.Tensor]
    """Per-layer directions [num_directions, hidden_dim]."""
    
    # Networks
    gate_network: Optional[GatingNetwork]
    """Learned gating network (None if disabled due to linear structure)."""
    
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
    
    # Geometry analysis
    geometry_adaptation: Optional[GeometryAdaptation] = None
    """Results from geometry analysis (if adapt_to_geometry was True)."""
    
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
        if self.gate_network is None:
            # No gating - always return 1.0 (always steer)
            if h.dim() == 1:
                return torch.ones(1, device=h.device)
            return torch.ones(h.shape[0], device=h.device)
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
        # steering_layers and sensor_layer default to None - resolved at training time
        # based on actual num_layers in the model
        self.config = TITANConfig(
            num_directions=kwargs.get("num_directions", 5),
            steering_layers=kwargs.get("steering_layers", None),  # Auto-resolve from num_layers
            sensor_layer=kwargs.get("sensor_layer", None),  # Auto-resolve from num_layers
            num_layers=kwargs.get("num_layers", None),
            gate_hidden_dim=kwargs.get("gate_hidden_dim", None),  # Auto-resolve from hidden_dim
            intensity_hidden_dim=kwargs.get("intensity_hidden_dim", None),  # Auto-resolve from hidden_dim
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
        return LayerActivations(primary_map, dtype=dtype)
    
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
        
        # Detect num_layers from available data if not set
        # Find max layer index to determine model size
        max_layer_idx = 0
        for layer_name in buckets.keys():
            try:
                layer_idx = int(str(layer_name).split("_")[-1])
                max_layer_idx = max(max_layer_idx, layer_idx)
            except (ValueError, IndexError):
                pass
        
        # Resolve steering_layers and sensor_layer based on detected num_layers
        detected_num_layers = max_layer_idx + 1  # layers are 0-indexed
        if self.config.steering_layers is None or self.config.sensor_layer is None:
            self.config.resolve_layers(detected_num_layers)
        
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
        
        # Resolve network dimensions based on actual hidden_dim
        if self.config.gate_hidden_dim is None or self.config.intensity_hidden_dim is None:
            self.config.resolve_network_dims(hidden_dim)
        
        num_layers = len(layer_names)
        
        # Geometry analysis and adaptation
        geometry_adaptation = None
        effective_num_directions = self.config.num_directions
        enable_gating = True
        
        if self.config.adapt_to_geometry:
            geometry_adaptation = self._analyze_and_adapt_geometry(
                buckets, layer_names, hidden_dim
            )
            effective_num_directions = geometry_adaptation.adapted_num_directions
            enable_gating = geometry_adaptation.gating_enabled
        
        # Initialize components with adapted configuration
        directions = self._initialize_directions(
            buckets, layer_names, hidden_dim, 
            num_directions=effective_num_directions
        )
        
        gate_network: Optional[GatingNetwork] = None
        if enable_gating:
            gate_network = GatingNetwork(hidden_dim, self.config.gate_hidden_dim)
        
        intensity_network = IntensityNetwork(
            hidden_dim, num_layers, 
            self.config.intensity_hidden_dim, 
            self.config.max_alpha
        )
        direction_weights = {
            layer: torch.ones(effective_num_directions) / effective_num_directions
            for layer in layer_names
        }
        
        # Make direction weights trainable
        direction_weight_params = {
            layer: nn.Parameter(torch.zeros(effective_num_directions))
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
            enable_gating=enable_gating,
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
                "effective_num_directions": effective_num_directions,
                "gating_enabled": enable_gating,
            },
            geometry_adaptation=geometry_adaptation,
        )
    
    def _analyze_and_adapt_geometry(
        self,
        buckets: Dict[LayerName, Tuple[List[torch.Tensor], List[torch.Tensor]]],
        layer_names: List[LayerName],
        hidden_dim: int,
    ) -> GeometryAdaptation:
        """
        Analyze geometry of activations and adapt TITAN configuration.
        
        Returns:
            GeometryAdaptation with detected structure and adaptations made.
        """
        from wisent.core.contrastive_pairs.diagnostics.control_vectors import (
            detect_geometry_structure,
            GeometryAnalysisConfig,
        )
        
        # Find the layer to analyze
        analysis_layer_idx = self.config.geometry_analysis_layer or self.config.sensor_layer
        analysis_layer = None
        for layer in layer_names:
            try:
                idx = int(str(layer).split("_")[-1])
                if idx == analysis_layer_idx:
                    analysis_layer = layer
                    break
            except (ValueError, IndexError):
                continue
        
        if analysis_layer is None:
            analysis_layer = layer_names[len(layer_names) // 2]
        
        # Get activations for analysis
        pos_list, neg_list = buckets[analysis_layer]
        pos_tensor = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
        neg_tensor = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)
        
        # Run geometry detection
        geo_config = GeometryAnalysisConfig(
            num_components=self.config.num_directions,
            max_clusters=5,
            manifold_neighbors=min(10, len(pos_list) - 1),
        )
        geo_result = detect_geometry_structure(pos_tensor, neg_tensor, geo_config)
        
        # Extract scores
        structure_scores = {
            name: score.score for name, score in geo_result.all_scores.items()
        }
        detected_structure = geo_result.best_structure.value
        
        # Determine adaptations
        adaptations = []
        original_num_directions = self.config.num_directions
        adapted_num_directions = original_num_directions
        gating_enabled = True
        
        linear_score = structure_scores.get("linear", 0)
        cone_score = structure_scores.get("cone", 0)
        manifold_score = structure_scores.get("manifold", 0)
        
        # Adaptation 1: Simplify if linear
        if linear_score > self.config.linear_threshold:
            if self.config.auto_num_directions:
                adapted_num_directions = 1
                adaptations.append(f"Reduced num_directions to 1 (linear score={linear_score:.2f})")
            
            if self.config.skip_gating_if_linear:
                gating_enabled = False
                adaptations.append("Disabled gating network (linear structure)")
        
        # Adaptation 2: Adjust directions based on cone structure
        elif cone_score > 0.7 and self.config.auto_num_directions:
            # Cone structure benefits from multiple directions
            cone_details = geo_result.all_scores.get("cone")
            if cone_details and hasattr(cone_details, "details"):
                sig_dirs = cone_details.details.get("significant_directions", 3)
                adapted_num_directions = max(2, min(sig_dirs + 1, 7))
                if adapted_num_directions != original_num_directions:
                    adaptations.append(
                        f"Adjusted num_directions to {adapted_num_directions} based on cone structure"
                    )
        
        # Adaptation 3: Increase directions for manifold/orthogonal
        elif (manifold_score > 0.8 or structure_scores.get("orthogonal", 0) > 0.7):
            if self.config.auto_num_directions and adapted_num_directions < 5:
                adapted_num_directions = 5
                adaptations.append(
                    f"Increased num_directions to 5 for manifold/orthogonal structure"
                )
        
        if not adaptations:
            adaptations.append("No adaptations needed - using default configuration")
        
        return GeometryAdaptation(
            detected_structure=detected_structure,
            structure_scores=structure_scores,
            adaptations_made=adaptations,
            original_num_directions=original_num_directions,
            adapted_num_directions=adapted_num_directions,
            gating_enabled=gating_enabled,
            recommendation=geo_result.recommendation,
        )
    
    def _initialize_directions(
        self,
        buckets: Dict[LayerName, Tuple[List[torch.Tensor], List[torch.Tensor]]],
        layer_names: List[LayerName],
        hidden_dim: int,
        num_directions: Optional[int] = None,
    ) -> Dict[LayerName, torch.Tensor]:
        """Initialize direction manifold for each layer."""
        directions = {}
        K = num_directions if num_directions is not None else self.config.num_directions
        
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
        gate_network: Optional[GatingNetwork],
        intensity_network: IntensityNetwork,
        direction_weight_params: Dict[LayerName, nn.Parameter],
        data: Dict[str, Dict[LayerName, torch.Tensor]],
        layer_names: List[LayerName],
        sensor_layer: LayerName,
        enable_gating: bool = True,
    ) -> Tuple[Dict[LayerName, torch.Tensor], Optional[GatingNetwork], IntensityNetwork, Dict[LayerName, torch.Tensor]]:
        """
        Joint end-to-end optimization of all TITAN components.
        """
        # Make directions trainable
        direction_params = {layer: nn.Parameter(dirs.clone()) for layer, dirs in directions.items()}
        
        # Collect all parameters
        all_params = []
        all_params.extend(direction_params.values())
        if gate_network is not None:
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
            
            # Predict gates (or use constant 1.0 if gating disabled)
            if gate_network is not None:
                pos_gate = gate_network(pos_sensor, self.config.gate_temperature)
                neg_gate = gate_network(neg_sensor, self.config.gate_temperature)
            else:
                pos_gate = torch.ones(pos_sensor.shape[0], device=pos_sensor.device)
                neg_gate = torch.ones(neg_sensor.shape[0], device=neg_sensor.device)
            
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
                    "gate_network": {k: v.detach().clone() for k, v in gate_network.state_dict().items()} if gate_network is not None else None,
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
            if gate_network is not None and best_state["gate_network"] is not None:
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

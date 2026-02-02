"""
PRISM - Projected Representations for Independent Steering Manifolds.

A gradient-optimized multi-directional steering method that discovers multiple
refusal directions per layer, forming a coherent steering manifold.

Based on insights from:
- "The Geometry of Refusal in Large Language Models" (Wollschläger et al., 2025)
- "SOM Directions are Better than One" (Piras et al., 2025)

Key innovations:
1. Gradient-based direction optimization (not just difference-in-means)
2. Multiple directions per layer that form a coherent manifold
3. Representational independence constraint (soft, not strict orthogonality)
4. Retain loss to minimize side effects on harmless queries
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from wisent.core.steering_methods.core.atoms import BaseSteeringMethod
from wisent.core.activations.core.atoms import LayerActivations, RawActivationMap, LayerName
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.errors import InsufficientDataError

__all__ = [
    "PRISMMethod",
    "PRISMConfig",
    "MultiDirectionResult",
]


@dataclass
class PRISMConfig:
    """Configuration for PRISM steering method."""
    
    num_directions: int = 3
    """Number of directions to discover per layer. Set to 'auto' or -1 for automatic."""
    
    auto_num_directions: bool = False
    """Automatically determine num_directions based on explained variance."""
    
    variance_threshold: float = 0.80
    """Target cumulative variance for auto num_directions."""
    
    marginal_threshold: float = 0.05
    """Minimum marginal variance for adding another direction."""
    
    max_directions: int = 10
    """Maximum directions when using auto num_directions."""
    
    optimization_steps: int = 100
    """Number of gradient descent steps for direction optimization."""
    
    learning_rate: float = 0.01
    """Learning rate for direction optimization."""
    
    retain_weight: float = 0.1
    """Weight for retain loss (preserving behavior on negative/harmless examples)."""
    
    independence_weight: float = 0.05
    """Weight for representational independence loss between directions."""
    
    ablation_weight: float = 1.0
    """Weight for ablation loss (making model comply with harmful after ablation)."""
    
    addition_weight: float = 1.0
    """Weight for addition loss (making model refuse harmless after addition)."""
    
    normalize: bool = True
    """Whether to L2-normalize the final directions."""
    
    use_caa_init: bool = True
    """Whether to initialize first direction using CAA (difference-in-means)."""
    
    use_universal_basis_init: bool = False
    """Whether to initialize from universal subspace basis if available."""
    
    cone_constraint: bool = True
    """Whether to constrain directions to form a polyhedral cone (all positive combinations)."""
    
    min_cosine_similarity: float = 0.3
    """Minimum cosine similarity between directions (they should be related, not orthogonal)."""
    
    max_cosine_similarity: float = 0.95
    """Maximum cosine similarity (avoid redundant directions)."""


@dataclass
class MultiDirectionResult:
    """Result containing multiple steering directions per layer."""
    
    directions: Dict[LayerName, torch.Tensor]
    """Per-layer directions tensor of shape [num_directions, hidden_dim]."""
    
    metadata: Dict[str, Any]
    """Training metadata including losses and diagnostics."""
    
    def get_primary_direction(self, layer: LayerName) -> torch.Tensor:
        """Get the primary (first/strongest) direction for a layer."""
        return self.directions[layer][0]
    
    def get_all_directions(self, layer: LayerName) -> torch.Tensor:
        """Get all directions for a layer as [num_directions, hidden_dim]."""
        return self.directions[layer]
    
    def to_single_direction_map(self) -> Dict[LayerName, torch.Tensor]:
        """Convert to single-direction format (for backward compatibility)."""
        return {layer: dirs[0] for layer, dirs in self.directions.items()}


class PRISMMethod(BaseSteeringMethod):
    """
    PRISM - Projected Representations for Independent Steering Manifolds.
    
    Discovers multiple steering directions per layer using gradient-based
    optimization with representational independence constraints.
    
    Unlike CAA which computes a single direction via difference-in-means,
    PRISM finds k directions that:
    - Each mediate the target behavior (e.g., refusal)
    - Are related but not redundant (controlled cosine similarity)
    - Form a coherent manifold when ablated together
    - Minimize side effects via retain loss
    
    Usage:
        method = PRISMMethod(num_directions=3, optimization_steps=100)
        result = method.train(pair_set)
        # result.directions contains [num_directions, hidden_dim] per layer
    """
    
    name = "prism"
    description = "Gradient-optimized multi-directional steering via projected representations"
    
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        
        # Handle auto num_directions
        num_dirs = kwargs.get("num_directions", 3)
        auto_num = kwargs.get("auto_num_directions", False)
        if num_dirs == "auto" or num_dirs == -1:
            auto_num = True
            num_dirs = 3  # Will be overridden during training
        
        self.config = PRISMConfig(
            num_directions=num_dirs,
            auto_num_directions=auto_num,
            variance_threshold=kwargs.get("variance_threshold", 0.80),
            marginal_threshold=kwargs.get("marginal_threshold", 0.05),
            max_directions=kwargs.get("max_directions", 10),
            optimization_steps=kwargs.get("optimization_steps", 100),
            learning_rate=kwargs.get("learning_rate", 0.01),
            retain_weight=kwargs.get("retain_weight", 0.1),
            independence_weight=kwargs.get("independence_weight", 0.05),
            ablation_weight=kwargs.get("ablation_weight", 1.0),
            addition_weight=kwargs.get("addition_weight", 1.0),
            normalize=kwargs.get("normalize", True),
            use_caa_init=kwargs.get("use_caa_init", True),
            use_universal_basis_init=kwargs.get("use_universal_basis_init", False),
            cone_constraint=kwargs.get("cone_constraint", True),
            min_cosine_similarity=kwargs.get("min_cosine_similarity", 0.3),
            max_cosine_similarity=kwargs.get("max_cosine_similarity", 0.95),
        )
        self._training_logs: List[Dict[str, float]] = []
    
    def train_for_layer(self, pos_list: List[torch.Tensor], neg_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Train a PRISM steering vector for a single layer.
        
        This provides compatibility with the PerLayerBaseSteeringMethod interface
        used by CLI tools.
        
        Arguments:
            pos_list: List of positive activation tensors
            neg_list: List of negative activation tensors
            
        Returns:
            Primary steering direction as a tensor
        """
        if not pos_list or not neg_list:
            raise InsufficientDataError(reason="Need both positive and negative activations")
        
        # Stack activations into tensors
        pos_tensor = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
        neg_tensor = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)
        
        # Train directions using the internal method
        directions, _meta = self._train_layer_directions(pos_tensor, neg_tensor, "layer")
        
        # Return primary direction (first one)
        primary = directions[0]
        
        # Normalize if configured
        if self.config.normalize:
            primary = F.normalize(primary, dim=-1)
        
        return primary
    
    def train(self, pair_set: ContrastivePairSet) -> LayerActivations:
        """
        Train PRISM directions from contrastive pairs.
        
        For backward compatibility with the base interface, returns LayerActivations
        with the PRIMARY direction per layer. Use train_multi() to get all directions.
        
        Arguments:
            pair_set: ContrastivePairSet with collected activations.
            
        Returns:
            LayerActivations with one (primary) steering vector per layer.
        """
        multi_result = self.train_multi(pair_set)
        
        # Return primary directions for backward compatibility
        primary_map: RawActivationMap = multi_result.to_single_direction_map()
        
        dtype = self.kwargs.get("dtype", None)
        return LayerActivations(primary_map, dtype=dtype)
    
    def train_multi(self, pair_set: ContrastivePairSet) -> MultiDirectionResult:
        """
        Train multiple PRISM directions from contrastive pairs.
        
        This is the full PRISM interface that returns all discovered directions.
        
        Arguments:
            pair_set: ContrastivePairSet with collected activations.
            
        Returns:
            MultiDirectionResult with all directions per layer and metadata.
        """
        # Collect activations by layer
        buckets = self._collect_from_set(pair_set)
        
        if not buckets:
            raise InsufficientDataError(reason="No valid activation pairs found in pair_set")
        
        all_directions: Dict[LayerName, torch.Tensor] = {}
        layer_metadata: Dict[LayerName, Dict[str, Any]] = {}
        
        for layer_name, (pos_list, neg_list) in sorted(buckets.items(), key=lambda kv: (len(kv[0]), kv[0])):
            if not pos_list or not neg_list:
                continue
            
            # Stack activations
            pos_tensor = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
            neg_tensor = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)
            
            # Train directions for this layer
            directions, meta = self._train_layer_directions(pos_tensor, neg_tensor, layer_name)
            
            all_directions[layer_name] = directions
            layer_metadata[layer_name] = meta
        
        return MultiDirectionResult(
            directions=all_directions,
            metadata={
                "config": self.config.__dict__,
                "num_layers": len(all_directions),
                "layer_metadata": layer_metadata,
                "training_logs": self._training_logs,
            }
        )
    
    def _train_layer_directions(
        self,
        pos_tensor: torch.Tensor,
        neg_tensor: torch.Tensor,
        layer_name: str,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Train multiple directions for a single layer.
        
        Arguments:
            pos_tensor: Positive activations [N_pos, H]
            neg_tensor: Negative activations [N_neg, H]
            layer_name: Name of the layer (for logging)
            
        Returns:
            Tuple of (directions [K, H], metadata dict)
        """
        hidden_dim = pos_tensor.shape[1]
        
        # Determine num_directions (auto or fixed)
        if self.config.auto_num_directions:
            from wisent.core.universal_subspace import compute_optimal_num_directions
            num_dirs, auto_details = compute_optimal_num_directions(
                pos_tensor, neg_tensor,
                variance_threshold=self.config.variance_threshold,
                marginal_threshold=self.config.marginal_threshold,
                max_directions=self.config.max_directions,
                min_directions=1,
            )
        else:
            num_dirs = self.config.num_directions
            auto_details = None
        
        # Initialize directions (ensure same device as input)
        device = pos_tensor.device
        directions = self._initialize_directions(pos_tensor, neg_tensor, hidden_dim, num_dirs)
        directions = directions.to(device)
        directions.requires_grad_(True)
        
        # Optimizer
        optimizer = torch.optim.Adam([directions], lr=self.config.learning_rate)
        
        # Training loop
        best_directions = directions.detach().clone()
        best_loss = float('inf')
        
        for step in range(self.config.optimization_steps):
            optimizer.zero_grad()
            
            # Compute losses
            loss, loss_components = self._compute_prism_loss(
                directions, pos_tensor, neg_tensor
            )
            
            # Backward and step
            loss.backward()
            optimizer.step()
            
            # Apply constraints
            with torch.no_grad():
                directions.data = self._apply_constraints(directions.data)
            
            # Track best
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_directions = directions.detach().clone()
            
            # Log
            if step % 10 == 0 or step == self.config.optimization_steps - 1:
                self._training_logs.append({
                    "layer": layer_name,
                    "step": step,
                    "total_loss": loss.item(),
                    **{k: v.item() for k, v in loss_components.items()}
                })
        
        # Final normalization if requested
        if self.config.normalize:
            best_directions = F.normalize(best_directions, p=2, dim=1)
        
        # Compute final metadata
        metadata = self._compute_direction_metadata(best_directions, pos_tensor, neg_tensor)
        metadata["final_loss"] = best_loss
        metadata["num_directions"] = num_dirs
        if auto_details is not None:
            metadata["auto_num_directions"] = auto_details
        
        return best_directions, metadata
    
    def _initialize_directions(
        self,
        pos_tensor: torch.Tensor,
        neg_tensor: torch.Tensor,
        hidden_dim: int,
        num_dirs: int,
    ) -> torch.Tensor:
        """
        Initialize directions using one of several strategies:
        1. Universal basis (if enabled and available)
        2. CAA + perturbations (default)
        3. Random initialization
        """
        # Try universal basis initialization first
        if self.config.use_universal_basis_init:
            try:
                from wisent.core.universal_subspace import initialize_from_universal_basis
                directions = initialize_from_universal_basis(
                    hidden_dim=hidden_dim,
                    num_directions=num_dirs,
                    noise_scale=0.1,
                )
                return directions
            except Exception:
                pass  # Fall through to other methods
        
        device = pos_tensor.device
        directions = torch.randn(num_dirs, hidden_dim, device=device)
        
        if self.config.use_caa_init:
            # First direction: CAA (difference-in-means)
            caa_dir = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
            caa_dir = F.normalize(caa_dir, p=2, dim=0)
            directions[0] = caa_dir
            
            # Initialize others as perturbations of CAA direction
            for i in range(1, num_dirs):
                noise = torch.randn(hidden_dim, device=device) * 0.3
                perturbed = caa_dir + noise
                directions[i] = F.normalize(perturbed, p=2, dim=0)
        else:
            # Random initialization, but ensure reasonable cosine similarities
            directions = F.normalize(directions, p=2, dim=1)
        
        return directions
    
    def _compute_prism_loss(
        self,
        directions: torch.Tensor,
        pos_tensor: torch.Tensor,
        neg_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the full PRISM loss.
        
        Components:
        1. Separation loss: directions should separate positive from negative
        2. Independence loss: directions should be representationally independent
        3. Similarity constraint: directions should be related (not orthogonal)
        4. Retain loss: minimize impact on negative (harmless) examples
        """
        loss_components = {}
        
        # Normalize directions for computation
        dirs_normalized = F.normalize(directions, p=2, dim=1)  # [K, H]
        
        # 1. Separation loss - each direction should separate pos from neg
        # Project activations onto directions
        pos_proj = pos_tensor @ dirs_normalized.T  # [N_pos, K]
        neg_proj = neg_tensor @ dirs_normalized.T  # [N_neg, K]
        
        # We want: pos_proj > neg_proj (positive examples project higher)
        # Loss: max(0, margin - (pos_mean - neg_mean))
        margin = 1.0
        pos_mean = pos_proj.mean(dim=0)  # [K]
        neg_mean = neg_proj.mean(dim=0)  # [K]
        separation = pos_mean - neg_mean
        separation_loss = F.relu(margin - separation).mean()
        loss_components["separation"] = separation_loss
        
        # 2. Independence loss - directions shouldn't interfere under ablation
        # Soft constraint: penalize very high correlations in how they affect activations
        K = directions.shape[0]
        if K > 1:
            # Compute pairwise effect correlation
            effects = []
            for i in range(K):
                # Effect of direction i on positive examples
                effect_i = (pos_tensor * dirs_normalized[i:i+1]).sum(dim=1)  # [N_pos]
                effects.append(effect_i)
            effects = torch.stack(effects, dim=1)  # [N_pos, K]
            
            # Correlation matrix of effects
            effects_centered = effects - effects.mean(dim=0, keepdim=True)
            effects_std = effects_centered.std(dim=0, keepdim=True) + 1e-8
            effects_norm = effects_centered / effects_std
            corr_matrix = (effects_norm.T @ effects_norm) / effects_norm.shape[0]
            
            # Penalize high off-diagonal correlations
            mask = 1 - torch.eye(K, device=corr_matrix.device)
            independence_loss = (corr_matrix.abs() * mask).mean()
            loss_components["independence"] = independence_loss
        else:
            independence_loss = torch.tensor(0.0)
            loss_components["independence"] = independence_loss
        
        # 3. Similarity constraint - directions should be related (part of same manifold)
        if K > 1:
            # Cosine similarity between directions
            cos_sim = dirs_normalized @ dirs_normalized.T  # [K, K]
            mask = 1 - torch.eye(K, device=cos_sim.device)
            off_diag = cos_sim * mask
            
            # Penalize if too dissimilar (< min) or too similar (> max)
            too_dissimilar = F.relu(self.config.min_cosine_similarity - off_diag)
            too_similar = F.relu(off_diag - self.config.max_cosine_similarity)
            similarity_loss = (too_dissimilar + too_similar).sum() / (K * (K - 1) + 1e-8)
            loss_components["similarity"] = similarity_loss
        else:
            similarity_loss = torch.tensor(0.0)
            loss_components["similarity"] = similarity_loss
        
        # 4. Retain loss - minimize change to negative (harmless) examples
        # After ablating direction, negative examples should stay similar
        neg_norms = neg_tensor.norm(dim=1, keepdim=True)  # [N_neg, 1]
        neg_directions = F.normalize(neg_tensor, p=2, dim=1)  # [N_neg, H]
        
        # Simulate ablation effect
        ablation_effect = torch.zeros_like(neg_tensor)
        for i in range(K):
            proj_scalar = (neg_tensor * dirs_normalized[i:i+1]).sum(dim=1, keepdim=True)
            ablation_effect += proj_scalar * dirs_normalized[i:i+1]
        
        # Retain loss: how much the negative examples change
        retain_loss = (ablation_effect.norm(dim=1) / (neg_norms.squeeze() + 1e-8)).mean()
        loss_components["retain"] = retain_loss
        
        # Combine losses
        total_loss = (
            self.config.ablation_weight * separation_loss +
            self.config.independence_weight * independence_loss +
            self.config.addition_weight * similarity_loss +
            self.config.retain_weight * retain_loss
        )
        
        return total_loss, loss_components
    
    def _apply_constraints(self, directions: torch.Tensor) -> torch.Tensor:
        """
        Apply constraints to directions after gradient step.
        """
        # Normalize
        directions = F.normalize(directions, p=2, dim=1)
        
        # Cone constraint: ensure all directions have positive correlation with first
        if self.config.cone_constraint and directions.shape[0] > 1:
            primary = directions[0:1]  # [1, H]
            for i in range(1, directions.shape[0]):
                cos_sim = (directions[i:i+1] * primary).sum()
                if cos_sim < 0:
                    # Flip direction to be in same half-space
                    directions[i] = -directions[i]
        
        return directions
    
    def _compute_direction_metadata(
        self,
        directions: torch.Tensor,
        pos_tensor: torch.Tensor,
        neg_tensor: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Compute metadata about the discovered directions.
        """
        K = directions.shape[0]
        dirs_normalized = F.normalize(directions, p=2, dim=1)
        
        # Pairwise cosine similarities
        cos_sim_matrix = dirs_normalized @ dirs_normalized.T
        
        # Per-direction separation scores
        pos_proj = pos_tensor @ dirs_normalized.T
        neg_proj = neg_tensor @ dirs_normalized.T
        separation_scores = (pos_proj.mean(dim=0) - neg_proj.mean(dim=0)).tolist()
        
        # Average cosine similarity (off-diagonal)
        if K > 1:
            mask = 1 - torch.eye(K)
            avg_cos_sim = (cos_sim_matrix * mask).sum() / (K * (K - 1))
        else:
            avg_cos_sim = 1.0
        
        return {
            "num_directions": K,
            "separation_scores": separation_scores,
            "avg_cosine_similarity": float(avg_cos_sim),
            "cosine_similarity_matrix": cos_sim_matrix.tolist(),
            "direction_norms": directions.norm(dim=1).tolist(),
        }
    
    def _collect_from_set(
        self, pair_set: ContrastivePairSet
    ) -> Dict[LayerName, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """
        Build {layer_name: ([pos tensors...], [neg tensors...])} by iterating pairs.
        """
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
    
    def get_training_logs(self) -> List[Dict[str, float]]:
        """Return the training logs from the last train() call."""
        return self._training_logs

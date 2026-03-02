"""TECZA direction training and loss computation."""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations, RawActivationMap, LayerName
from wisent.core.control.steering_methods.methods.advanced._tecza_types import TECZAConfig
from wisent.core.utils.config_tools.constants import NORM_EPS, TECZA_SEPARATION_MARGIN, TECZA_PERTURBATION_SCALE, TECZA_UNIVERSAL_BASIS_NOISE, TECZA_LOGGING_INTERVAL

class TECZATrainingMixin:
    """Mixin: direction training, initialization, and loss."""

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
            from wisent.core.control.steering_core.core.universal_subspace import compute_optimal_num_directions
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
            loss, loss_components = self._compute_tecza_loss(
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
            if step % TECZA_LOGGING_INTERVAL == 0 or step == self.config.optimization_steps - 1:
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
                from wisent.core.control.steering_core.core.universal_subspace import initialize_from_universal_basis
                directions = initialize_from_universal_basis(
                    hidden_dim=hidden_dim,
                    num_directions=num_dirs,
                    noise_scale=TECZA_UNIVERSAL_BASIS_NOISE,
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
                noise = torch.randn(hidden_dim, device=device) * TECZA_PERTURBATION_SCALE
                perturbed = caa_dir + noise
                directions[i] = F.normalize(perturbed, p=2, dim=0)
        else:
            # Random initialization, but ensure reasonable cosine similarities
            directions = F.normalize(directions, p=2, dim=1)
        
        return directions
    
    def _compute_tecza_loss(
        self,
        directions: torch.Tensor,
        pos_tensor: torch.Tensor,
        neg_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the full TECZA loss.
        
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
        margin = TECZA_SEPARATION_MARGIN
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
            effects_std = effects_centered.std(dim=0, keepdim=True) + NORM_EPS
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
            similarity_loss = (too_dissimilar + too_similar).sum() / (K * (K - 1) + NORM_EPS)
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
        retain_loss = (ablation_effect.norm(dim=1) / (neg_norms.squeeze() + NORM_EPS)).mean()
        loss_components["retain"] = retain_loss
        
        # Combine losses
        total_loss = (
            self.config.ablation_weight * separation_loss +
            self.config.independence_weight * independence_loss +
            self.config.addition_weight * similarity_loss +
            self.config.retain_weight * retain_loss
        )
        
        return total_loss, loss_components
    

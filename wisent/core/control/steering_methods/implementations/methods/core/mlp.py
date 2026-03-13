"""
MLP-based steering using adversarial gradient direction.

Trains an MLP classifier to separate positive from negative activations,
then extracts the gradient direction that maximally changes classification.
This captures non-linear decision boundaries while producing a linear steering vector.
"""

from __future__ import annotations

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from wisent.core.control.steering_methods.core.atoms import PerLayerBaseSteeringMethod
from wisent.core.control.steering_methods.configs.optimal import get_optimal
from wisent.core.utils.infra_tools.errors import InsufficientDataError
from wisent.core import constants as _C

__all__ = ["MLPMethod"]


def _require(name: str, kwargs: dict):
    """Raise ValueError if a required hyperparameter is missing."""
    if name not in kwargs:
        raise ValueError(
            f"Parameter '{name}' is required. "
            f"Run 'wisent optimize-steering auto' first, or pass it explicitly."
        )
    return kwargs[name]


class MLPClassifier(nn.Module):
    """Simple MLP for classifying activations."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        *,
        gating_hidden_dim_divisor: int,
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        last_layer_idx = num_layers - _C.COMBO_OFFSET
        for i in range(num_layers):
            next_dim = hidden_dim if i < last_layer_idx else hidden_dim // gating_hidden_dim_divisor
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.LayerNorm(next_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = next_dim
        
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPMethod(PerLayerBaseSteeringMethod):
    """
    MLP-based steering using adversarial gradient direction.
    
    Trains a non-linear MLP classifier to separate positive from negative
    activations, then computes the gradient direction that maximally changes
    the classification. This captures non-linear structure while producing
    a usable linear steering vector.
    
    The key insight: even if the true decision boundary is non-linear,
    the locally optimal steering direction at any point is still a vector.
    By averaging gradients across samples, we get a robust steering direction.
    """
    
    name = "mlp"
    description = "MLP-based steering using adversarial gradient direction from trained classifier"
    
    def train_for_layer(self, pos_list: List[torch.Tensor], neg_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Train MLP classifier and extract adversarial steering direction.
        
        Args:
            pos_list: List of positive activation tensors
            neg_list: List of negative activation tensors
            
        Returns:
            Normalized steering vector
        """
        if not pos_list or not neg_list:
            raise InsufficientDataError(reason="Both positive and negative lists must be non-empty")
        
        # Stack and prepare data
        pos = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
        neg = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)
        
        hidden_dim = pos.shape[1]
        
        # Combine into dataset
        X = torch.cat([pos, neg], dim=0)
        y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))], dim=0)
        
        # Shuffle
        perm = torch.randperm(len(X))
        X, y = X[perm], y[perm]
        
        # Get hyperparameters (all required — no silent defaults)
        mlp_hidden = int(_require("hidden_dim", self.kwargs))
        mlp_input_divisor = int(_require("mlp_input_divisor", self.kwargs))
        mlp_hidden = min(mlp_hidden, hidden_dim // mlp_input_divisor)
        mlp_layers = int(_require("num_layers", self.kwargs))
        dropout = float(_require("dropout", self.kwargs))
        epochs = int(_require("epochs", self.kwargs))
        lr = float(_require("learning_rate", self.kwargs))
        weight_decay = float(_require("weight_decay", self.kwargs))
        early_stop_tol = float(_require("early_stop_tol", self.kwargs))
        gating_divisor = int(_require("gating_hidden_dim_divisor", self.kwargs))

        # Initialize MLP
        mlp = MLPClassifier(
            input_dim=hidden_dim,
            hidden_dim=mlp_hidden,
            num_layers=mlp_layers,
            dropout=dropout,
            gating_hidden_dim_divisor=gating_divisor,
        )
        
        # Train classifier
        optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        mlp.train()
        best_loss = float('inf')
        patience_counter = _C.RECURSION_INITIAL_DEPTH
        patience = int(_require("mlp_early_stopping_patience", self.kwargs))

        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = mlp(X)
            loss = F.binary_cross_entropy_with_logits(logits, y)

            if not torch.isfinite(loss):
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=float(_C.COMBO_OFFSET))
            optimizer.step()
            scheduler.step()

            current_loss = loss.item()
            if current_loss < best_loss - early_stop_tol:
                best_loss = current_loss
                patience_counter = _C.RECURSION_INITIAL_DEPTH
            else:
                patience_counter += _C.COMBO_OFFSET
                if patience_counter >= patience:
                    break

        # Extract adversarial gradient direction
        mlp.eval()
        steering_vector = self._extract_adversarial_direction(mlp, pos, neg)

        # Fall back to CAA direction if MLP produced NaN
        if not torch.isfinite(steering_vector).all():
            steering_vector = pos.mean(dim=_C.RECURSION_INITIAL_DEPTH) - neg.mean(dim=_C.RECURSION_INITIAL_DEPTH)

        # Normalize if requested
        if self.kwargs.get("normalize", get_optimal("normalize")):
            steering_vector = F.normalize(steering_vector, dim=_C.RECURSION_INITIAL_DEPTH)

        return steering_vector
    
    def _extract_adversarial_direction(
        self,
        mlp: MLPClassifier,
        pos: torch.Tensor,
        neg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract the adversarial direction from trained MLP.
        
        Computes gradient of classifier output w.r.t. inputs,
        averaged across samples. The direction points from negative to positive.
        """
        gradients = []
        
        # Get gradients for negative samples (direction to make them positive)
        for i in range(len(neg)):
            x = neg[i:i+1].clone().requires_grad_(True)
            output = mlp(x)
            output.backward()
            gradients.append(x.grad.squeeze(0).clone())
        
        # Average gradients
        avg_gradient = torch.stack(gradients).mean(dim=0)
        
        # Gradient points in direction of increasing classifier output (toward positive)
        return avg_gradient

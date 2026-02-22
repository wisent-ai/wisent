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

from wisent.core.steering_methods.core.atoms import PerLayerBaseSteeringMethod
from wisent.core.errors import InsufficientDataError

__all__ = ["MLPMethod"]


class MLPClassifier(nn.Module):
    """Simple MLP for classifying activations."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else hidden_dim // 2
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
        
        # Get hyperparameters
        mlp_hidden = int(self.kwargs.get("hidden_dim", min(256, hidden_dim // 4)))
        mlp_layers = int(self.kwargs.get("num_layers", 2))
        dropout = float(self.kwargs.get("dropout", 0.1))
        epochs = int(self.kwargs.get("epochs", 100))
        lr = float(self.kwargs.get("learning_rate", 0.001))
        weight_decay = float(self.kwargs.get("weight_decay", 0.01))
        
        # Initialize MLP
        mlp = MLPClassifier(
            input_dim=hidden_dim,
            hidden_dim=mlp_hidden,
            num_layers=mlp_layers,
            dropout=dropout,
        )
        
        # Train classifier
        optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        mlp.train()
        best_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = mlp(X)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Early stopping
            if loss.item() < best_loss - 1e-4:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # Extract adversarial gradient direction
        mlp.eval()
        steering_vector = self._extract_adversarial_direction(mlp, pos, neg)
        
        # Normalize if requested
        if self.kwargs.get("normalize", True):
            steering_vector = F.normalize(steering_vector, dim=0)
        
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

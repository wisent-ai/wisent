"""Sparse Autoencoder model and training loop."""

import torch
import numpy as np
from typing import Tuple
from wisent.core.utils.config_tools.constants import NORM_EPS, L1_DEFAULT_COEF, SAE_L1_COEF_DEFAULT, SAE_N_EPOCHS_DEFAULT, SAE_BATCH_SIZE_DEFAULT, SAE_HIDDEN_DIM_MULTIPLIER


# =============================================================================
# SPARSE AUTOENCODER FOR FEATURE ANALYSIS
# =============================================================================

class SparseAutoencoder(torch.nn.Module):
    """
    Sparse Autoencoder for finding interpretable features in activations.
    
    Architecture: input -> encoder (with ReLU) -> sparse features -> decoder -> reconstruction
    
    The encoder learns an overcomplete basis where each feature ideally represents
    one interpretable concept. Sparsity is enforced via L1 penalty.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,  # Usually 4x-8x input_dim for overcomplete
        l1_coef: float = L1_DEFAULT_COEF,
        tied_weights: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l1_coef = l1_coef
        self.tied_weights = tied_weights
        
        # Encoder: input -> hidden (sparse features)
        self.encoder = torch.nn.Linear(input_dim, hidden_dim, bias=True)
        
        # Decoder: hidden -> input (reconstruction)
        if tied_weights:
            # Tied weights: decoder weight = encoder weight transposed
            self.decoder_bias = torch.nn.Parameter(torch.zeros(input_dim))
        else:
            self.decoder = torch.nn.Linear(hidden_dim, input_dim, bias=True)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Kaiming initialization for encoder
        torch.nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.encoder.bias)
        
        if not self.tied_weights:
            torch.nn.init.kaiming_uniform_(self.decoder.weight, nonlinearity='linear')
            torch.nn.init.zeros_(self.decoder.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse feature activations."""
        return torch.relu(self.encoder(x))
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to input space."""
        if self.tied_weights:
            return torch.mm(features, self.encoder.weight) + self.decoder_bias
        else:
            return self.decoder(features)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (reconstruction, sparse_features)."""
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features
    
    def loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute total loss = reconstruction_loss + l1_coef * sparsity_loss."""
        reconstruction, features = self.forward(x)
        
        # MSE reconstruction loss
        recon_loss = torch.nn.functional.mse_loss(reconstruction, x)
        
        # L1 sparsity penalty on feature activations
        sparsity_loss = features.abs().mean()
        
        total_loss = recon_loss + self.l1_coef * sparsity_loss
        
        return total_loss, recon_loss, sparsity_loss


def train_sparse_autoencoder(
    activations: torch.Tensor,
    device: str,
    hidden_dim: int = None,
    l1_coef: float = SAE_L1_COEF_DEFAULT,
    n_epochs: int = SAE_N_EPOCHS_DEFAULT,
    batch_size: int = SAE_BATCH_SIZE_DEFAULT,
    lr: float = L1_DEFAULT_COEF,
    verbose: bool = True,
) -> SparseAutoencoder:
    """
    Train a sparse autoencoder on activations.
    
    Args:
        activations: Tensor of shape (n_samples, input_dim)
        hidden_dim: Number of features to learn (default: 4x input_dim)
        l1_coef: L1 sparsity coefficient
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        device: Device to train on
        verbose: Print training progress
    
    Returns:
        Trained SparseAutoencoder
    """
    input_dim = activations.shape[1]
    if hidden_dim is None:
        hidden_dim = input_dim * SAE_HIDDEN_DIM_MULTIPLIER  # overcomplete
    
    # Move data to device
    activations = activations.to(device).float()
    
    # Normalize activations (important for SAE training)
    mean = activations.mean(dim=0, keepdim=True)
    std = activations.std(dim=0, keepdim=True) + NORM_EPS
    activations_norm = (activations - mean) / std
    
    # Create SAE
    sae = SparseAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        l1_coef=l1_coef,
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # Training loop
    n_samples = activations_norm.shape[0]
    
    for epoch in range(n_epochs):
        # Shuffle data
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_sparsity = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = perm[i:i+batch_size]
            batch = activations_norm[batch_idx]
            
            optimizer.zero_grad()
            loss, recon_loss, sparsity_loss = sae.loss(batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_sparsity += sparsity_loss.item()
            n_batches += 1
        
        scheduler.step()
        
        if verbose and (epoch + 1) % 100 == 0:
            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon / n_batches
            avg_sparse = epoch_sparsity / n_batches
            
            # Compute sparsity stats
            with torch.no_grad():
                features = sae.encode(activations_norm)
                active_frac = (features > 0).float().mean().item()
                avg_active = (features > 0).sum(dim=1).float().mean().item()
            
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f} "
                  f"(recon={avg_recon:.4f}, sparse={avg_sparse:.4f}), "
                  f"active_features={avg_active:.1f}/{hidden_dim} ({active_frac*100:.1f}%)")
    
    # Store normalization parameters
    sae.mean = mean
    sae.std = std
    
    return sae

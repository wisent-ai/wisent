"""
Helper functions for Wisent-Guard
"""

import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from sklearn.metrics.pairwise import cosine_similarity

def ensure_dir(directory: str) -> None:
    """
    Make sure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def cosine_sim(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Cosine similarity value
    """
    try:
        # For MPS or CUDA tensors, explicitly move to CPU first
        if isinstance(v1, torch.Tensor):
            if v1.device.type in ['mps', 'cuda']:
                v1 = v1.detach().cpu()
            v1 = v1.detach().numpy()
            
        if isinstance(v2, torch.Tensor):
            if v2.device.type in ['mps', 'cuda']:
                v2 = v2.detach().cpu()
            v2 = v2.detach().numpy()
            
        # Reshape if needed
        if len(v1.shape) > 1:
            v1 = v1.reshape(1, -1)
        if len(v2.shape) > 1:
            v2 = v2.reshape(1, -1)
        
        # Check for NaN or Inf values that could cause issues
        if np.isnan(v1).any() or np.isnan(v2).any() or np.isinf(v1).any() or np.isinf(v2).any():
            print("Warning: NaN or Inf values detected in vectors")
            # Replace NaN/Inf with zeros
            v1 = np.nan_to_num(v1)
            v2 = np.nan_to_num(v2)
            
        # Verify shapes match for comparison
        if v1.shape[1] != v2.shape[1]:
            # Truncate to the smaller dimension
            min_dim = min(v1.shape[1], v2.shape[1])
            v1 = v1[:, :min_dim]
            v2 = v2[:, :min_dim]
            
        return float(cosine_similarity(v1, v2)[0][0])
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        # Return a low similarity value to be safe
        return 0.0

def normalize_vector(vector: torch.Tensor) -> torch.Tensor:
    """
    Normalize a vector to unit length.
    
    Args:
        vector: Input vector
        
    Returns:
        Normalized vector
    """
    try:
        # Calculate norm on the same device as the vector
        device = vector.device
        
        # Handle small or zero vectors gracefully
        epsilon = 1e-10
        norm = torch.norm(vector, p=2)
        
        # Check for NaN/Inf
        if torch.isnan(norm) or torch.isinf(norm) or norm < epsilon:
            # Return original vector if it can't be normalized properly
            print("Warning: Vector has very small norm, could not normalize properly")
            return vector
        
        # Perform normalization on the same device
        normalized = vector / norm
        
        # Check result
        if torch.isnan(normalized).any() or torch.isinf(normalized).any():
            print("Warning: Normalized vector contains NaN or Inf values")
            # Fallback to detaching and moving to CPU
            cpu_vector = vector.detach().cpu()
            cpu_norm = torch.norm(cpu_vector, p=2)
            if cpu_norm > epsilon:
                normalized = cpu_vector / cpu_norm
                # Move back to original device
                return normalized.to(device)
            else:
                return vector
        
        return normalized
    except Exception as e:
        print(f"Error normalizing vector: {e}")
        return vector

def calculate_average_vector(vectors: List[torch.Tensor]) -> torch.Tensor:
    """
    Calculate the average of a list of vectors.
    
    Args:
        vectors: List of vectors to average
        
    Returns:
        Average vector
    """
    if not vectors:
        raise ValueError("Empty list of vectors provided")
    
    try:
        # Ensure all tensors are on the same device, preferring CPU for stability
        cpu_vectors = []
        for vector in vectors:
            if vector.device.type != 'cpu':
                cpu_vectors.append(vector.detach().cpu())
            else:
                cpu_vectors.append(vector)
        
        # Stack tensors and calculate mean
        stacked = torch.stack(cpu_vectors)
        return torch.mean(stacked, dim=0)
    except Exception as e:
        print(f"Error calculating average vector: {e}")
        # Fallback to a simpler implementation
        result = None
        for vector in vectors:
            if result is None:
                result = vector.detach().cpu()
            else:
                result += vector.detach().cpu()
        
        if result is not None:
            return result / len(vectors)
        else:
            raise ValueError("Failed to calculate average vector")

def get_layer_name(model_type: str, layer_idx: int) -> str:
    """
    Get the appropriate layer name based on model type and layer index.
    
    Args:
        model_type: Type of model ('opt', 'llama', 'gpt2', etc.)
        layer_idx: Layer index
        
    Returns:
        Layer name for the specific model architecture
    """
    layer_name_map = {
        "opt": f"model.decoder.layers.{layer_idx}",
        "llama": f"model.layers.{layer_idx}",
        "gpt2": f"transformer.h.{layer_idx}",
        "gpt_neox": f"gpt_neox.layers.{layer_idx}",
        "gptj": f"transformer.h.{layer_idx}",
        "t5": f"encoder.block.{layer_idx}",
        "bart": f"model.encoder.layers.{layer_idx}",
    }
    
    return layer_name_map.get(model_type.lower(), f"layers.{layer_idx}") 
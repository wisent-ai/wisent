"""Steering vector compression via universal subspace."""
from __future__ import annotations
import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
from pathlib import Path
import torch
import torch.nn.functional as F
from wisent.core.utils.cli.cli_logger import setup_logger, bind
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerName
from wisent.core.utils.config_tools.constants import ZERO_THRESHOLD

_LOG = setup_logger(__name__)

@dataclass
class UniversalBasis:
    """Universal basis for steering vector compression."""
    
    components: torch.Tensor
    """Principal components [n_components, hidden_dim]."""
    
    mean: torch.Tensor
    """Mean vector used for centering [hidden_dim]."""
    
    singular_values: torch.Tensor
    """Singular values for each component."""
    
    explained_variance_ratio: torch.Tensor
    """Variance explained by each component."""
    
    n_components: int
    """Number of components."""
    
    hidden_dim: int
    """Hidden dimension of original vectors."""
    
    source_info: Dict[str, Any] = field(default_factory=dict)
    """Information about how basis was computed."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "components": self.components.cpu().tolist(),
            "mean": self.mean.cpu().tolist(),
            "singular_values": self.singular_values.cpu().tolist(),
            "explained_variance_ratio": self.explained_variance_ratio.cpu().tolist(),
            "n_components": self.n_components,
            "hidden_dim": self.hidden_dim,
            "source_info": self.source_info,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UniversalBasis":
        """Load from dictionary."""
        return cls(
            components=torch.tensor(data["components"]),
            mean=torch.tensor(data["mean"]),
            singular_values=torch.tensor(data["singular_values"]),
            explained_variance_ratio=torch.tensor(data["explained_variance_ratio"]),
            n_components=data["n_components"],
            hidden_dim=data["hidden_dim"],
            source_info=data.get("source_info", {}),
        )


def compute_universal_basis(
    vectors: List[torch.Tensor] | Dict[LayerName, torch.Tensor],
    n_components: int = None,
    normalize: bool = True,
) -> UniversalBasis:
    """
    Compute a universal basis from a collection of steering vectors.

    Args:
        vectors: Collection of steering vectors
        n_components: Number of basis components
        normalize: Whether to normalize vectors before PCA

    Returns:
        UniversalBasis that can be used for compression/initialization
    """
    if n_components is None:
        raise ValueError("n_components is required")
    log = bind(_LOG)

    # Convert to matrix
    if isinstance(vectors, dict):
        vector_list = [v.detach().float().reshape(-1) for v in vectors.values() if v is not None]
    else:
        vector_list = [v.detach().float().reshape(-1) for v in vectors]
    
    matrix = torch.stack(vector_list, dim=0)
    
    if normalize:
        matrix = F.normalize(matrix, p=2, dim=1)
    
    # Compute mean
    mean = matrix.mean(dim=0)
    centered = matrix - mean
    
    # SVD
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    
    # Keep top k components
    k = min(n_components, len(S))
    components = Vh[:k]
    singular_values = S[:k]
    
    # Explained variance
    total_var = (S ** 2).sum()
    explained_var_ratio = (S[:k] ** 2) / total_var
    
    log.info(
        "Computed universal basis",
        extra={
            "n_vectors": len(vector_list),
            "n_components": k,
            "total_variance_explained": explained_var_ratio.sum().item(),
        }
    )
    
    return UniversalBasis(
        components=components,
        mean=mean,
        singular_values=singular_values,
        explained_variance_ratio=explained_var_ratio,
        n_components=k,
        hidden_dim=matrix.shape[1],
        source_info={
            "n_source_vectors": len(vector_list),
            "total_variance_explained": explained_var_ratio.sum().item(),
        }
    )


def compress_steering_vectors(
    vectors: Dict[LayerName, torch.Tensor],
    basis: UniversalBasis,
) -> Dict[LayerName, torch.Tensor]:
    """
    Compress steering vectors to coefficients in universal basis.
    
    Instead of storing [hidden_dim] floats per layer, stores [n_components] coefficients.
    Memory reduction: hidden_dim / n_components (e.g., 4096/16 = 256x).
    
    Args:
        vectors: Dict mapping layer names to steering vectors
        basis: Universal basis for compression
        
    Returns:
        Dict mapping layer names to coefficient vectors [n_components]
    """
    compressed = {}
    
    for layer, vec in vectors.items():
        if vec is None:
            continue
        
        vec_flat = vec.detach().float().reshape(-1)
        
        # Center using basis mean
        centered = vec_flat - basis.mean.to(vec_flat.device)
        
        # Project onto basis components
        coefficients = centered @ basis.components.T.to(vec_flat.device)
        
        compressed[layer] = coefficients
    
    return compressed


def decompress_steering_vectors(
    coefficients: Dict[LayerName, torch.Tensor],
    basis: UniversalBasis,
) -> Dict[LayerName, torch.Tensor]:
    """
    Decompress steering vectors from coefficients.
    
    Args:
        coefficients: Dict mapping layer names to coefficient vectors
        basis: Universal basis used for compression
        
    Returns:
        Dict mapping layer names to reconstructed steering vectors
    """
    decompressed = {}
    
    for layer, coeff in coefficients.items():
        if coeff is None:
            continue
        
        coeff = coeff.to(basis.components.device)
        
        # Reconstruct from basis
        reconstructed = coeff @ basis.components + basis.mean
        
        decompressed[layer] = reconstructed
    
    return decompressed


def save_compressed_vectors(
    coefficients: Dict[LayerName, torch.Tensor],
    basis: UniversalBasis,
    path: str,
) -> None:
    """Save compressed vectors and basis to file."""
    data = {
        "coefficients": {str(k): v.cpu().tolist() for k, v in coefficients.items()},
        "basis": basis.to_dict(),
        "format_version": "1.0",
    }
    
    with open(path, "w") as f:
        json.dump(data, f)


def load_compressed_vectors(path: str) -> Tuple[Dict[LayerName, torch.Tensor], UniversalBasis]:
    """Load compressed vectors and basis from file."""
    with open(path) as f:
        data = json.load(f)
    
    coefficients = {k: torch.tensor(v) for k, v in data["coefficients"].items()}
    basis = UniversalBasis.from_dict(data["basis"])
    
    return coefficients, basis


# =============================================================================
# 3. AUTO NUM_DIRECTIONS BASED ON EXPLAINED VARIANCE
# =============================================================================

def explained_variance_analysis(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    *,
    max_components: int,
) -> Tuple[List[float], List[float]]:
    """
    Analyze how many directions are needed to explain the behavioral difference.

    Args:
        pos_activations: Positive example activations [N_pos, hidden_dim]
        neg_activations: Negative example activations [N_neg, hidden_dim]
        max_components: Maximum components to analyze

    Returns:
        Tuple of (individual_variance, cumulative_variance) lists
    """
    n_min = min(pos_activations.shape[0], neg_activations.shape[0])
    diff = pos_activations[:n_min] - neg_activations[:n_min]
    
    # Center
    centered = diff - diff.mean(dim=0, keepdim=True)
    
    # SVD
    try:
        _, S, _ = torch.linalg.svd(centered, full_matrices=False)
    except Exception:
        return [1.0], [1.0]
    
    # Variance ratios
    total_var = (S ** 2).sum()
    if total_var < ZERO_THRESHOLD:
        return [1.0], [1.0]
    
    k = min(max_components, len(S))
    individual = ((S[:k] ** 2) / total_var).tolist()
    cumulative = ((S[:k] ** 2).cumsum(0) / total_var).tolist()
    
    return individual, cumulative


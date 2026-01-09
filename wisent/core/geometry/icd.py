"""
Intrinsic Concept Dimensionality (ICD) metric.

ICD measures the effective rank of difference vectors - how many
independent directions are needed to represent the concept.
"""

import torch
import numpy as np
from typing import Dict


def compute_icd(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute Intrinsic Concept Dimensionality (ICD) of difference vectors.
    
    ICD measures effective rank - how many independent directions are needed
    to represent the concept. Low ICD = concentrated signal, high ICD = diffuse/noise.
    
    Args:
        pos_activations: [N, hidden_dim] positive activations
        neg_activations: [N, hidden_dim] negative activations
        
    Returns:
        Dict with:
            - icd: Intrinsic Concept Dimensionality (effective rank)
            - top1_variance: Fraction of variance explained by top direction
            - top5_variance: Fraction explained by top 5 directions
            - n_samples: Number of samples used
    """
    n = min(len(pos_activations), len(neg_activations))
    if n < 5:
        return {"icd": 0.0, "top1_variance": 0.0, "top5_variance": 0.0, "n_samples": n}
    
    diff_vectors = (pos_activations[:n] - neg_activations[:n]).float().cpu().numpy()
    diff_vectors = diff_vectors.astype(np.float64)
    
    try:
        U, S, Vh = np.linalg.svd(diff_vectors, full_matrices=False)
        S = S[S > 1e-10]
        
        if len(S) == 0:
            return {"icd": 0.0, "top1_variance": 0.0, "top5_variance": 0.0, "n_samples": n}
        
        icd = float((S.sum() ** 2) / (S ** 2).sum())
        
        total_var = (S ** 2).sum()
        top1_var = float((S[0] ** 2) / total_var) if total_var > 0 else 0.0
        top5_var = float((S[:5] ** 2).sum() / total_var) if total_var > 0 else 0.0
        
        return {
            "icd": icd,
            "top1_variance": top1_var,
            "top5_variance": top5_var,
            "n_samples": n,
        }
    except Exception:
        return {"icd": 0.0, "top1_variance": 0.0, "top5_variance": 0.0, "n_samples": n}

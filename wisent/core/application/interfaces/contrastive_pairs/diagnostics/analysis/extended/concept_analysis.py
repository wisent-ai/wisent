"""
Concept Analysis: ICD and Concept Decomposition

This module provides tools for analyzing whether a contrastive dataset contains
a single concept or multiple mixed concepts.

Key metrics:
- ICD (Intrinsic Concept Dimensionality): Effective rank of difference vectors
- Spectral decomposition: Recover individual concepts from mixed data
- Concept correlation: Measure similarity between concept directions

Theory reference: Zwiad paper, Section "A Theory of Concepts in Activation Space"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from wisent.core.constants import (
    ZERO_THRESHOLD, DEFAULT_RANDOM_SEED, LINEARITY_N_INIT,
    CONCEPT_PCA_COMPONENTS, CONCEPT_K_MAX,
    CLUSTERING_MEANINGFUL_THRESHOLD,
)


@dataclass
class ConceptAnalysisResult:
    """Results from concept analysis."""
    # Intrinsic Concept Dimensionality
    icd: float  # Effective rank of difference vectors
    
    # Eigenvalue spectrum
    top_singular_values: List[float]  # Top-k singular values
    top_variance_explained: List[float]  # Cumulative variance explained
    eigenvalue_gap: float  # Gap between 1st and 2nd singular value
    
    # Clustering analysis
    num_concepts_detected: int  # Best k by silhouette
    silhouette_scores: Dict[int, float]  # k -> silhouette score
    cluster_sizes: List[int]  # Sizes of detected clusters
    
    # Concept directions (if decomposed)
    concept_directions: Optional[List[np.ndarray]] = None
    concept_correlations: Optional[Dict[str, float]] = None
    
    # Interpretation
    is_single_concept: bool = True
    confidence: str = "unknown"  # "high", "medium", "low"
    interpretation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "icd": self.icd,
            "top_singular_values": self.top_singular_values,
            "top_variance_explained": self.top_variance_explained,
            "eigenvalue_gap": self.eigenvalue_gap,
            "num_concepts_detected": self.num_concepts_detected,
            "silhouette_scores": self.silhouette_scores,
            "cluster_sizes": self.cluster_sizes,
            "is_single_concept": self.is_single_concept,
            "confidence": self.confidence,
            "interpretation": self.interpretation,
        }


def compute_icd(diff_vectors: np.ndarray) -> float:
    """
    Compute Intrinsic Concept Dimensionality (ICD).
    
    ICD is the effective rank of the difference vector matrix, measuring
    how many independent directions are needed to represent the data.
    
    ICD = (sum of singular values)^2 / (sum of squared singular values)
    
    Interpretation:
    - ICD ≈ 1: Single dominant concept direction
    - ICD ≈ k: Approximately k equally-weighted concept directions
    - ICD large: No coherent concept structure (noise-dominated)
    
    Args:
        diff_vectors: [n_samples, hidden_dim] difference vectors (pos - neg)
        
    Returns:
        ICD value (effective rank)
    """
    if len(diff_vectors) < 2:
        return 1.0
    
    # Ensure float64 for numerical stability
    diff_vectors = np.asarray(diff_vectors, dtype=np.float64)
    
    # SVD
    try:
        U, S, Vh = np.linalg.svd(diff_vectors, full_matrices=False)
    except np.linalg.LinAlgError:
        return float(diff_vectors.shape[1])
    
    # Filter near-zero singular values
    S = S[S > ZERO_THRESHOLD]
    
    if len(S) == 0:
        return 0.0
    
    # Effective rank formula
    icd = (S.sum() ** 2) / (S ** 2).sum()
    
    return float(icd)


def compute_eigenvalue_spectrum(
    diff_vectors: np.ndarray,
    n_components: int = CONCEPT_PCA_COMPONENTS,
) -> Tuple[List[float], List[float], float]:
    """
    Compute eigenvalue spectrum of difference vectors.
    
    Args:
        diff_vectors: [n_samples, hidden_dim] difference vectors
        n_components: Number of components to return
        
    Returns:
        Tuple of (singular_values, cumulative_variance_explained, gap_1_2)
    """
    diff_vectors = np.asarray(diff_vectors, dtype=np.float64)
    
    try:
        U, S, Vh = np.linalg.svd(diff_vectors, full_matrices=False)
    except np.linalg.LinAlgError:
        return [], [], 0.0
    
    # Keep top n_components
    S = S[:min(n_components, len(S))]
    
    # Cumulative variance explained
    total_var = (S ** 2).sum()
    if total_var > 0:
        cum_var = np.cumsum(S ** 2) / total_var
    else:
        cum_var = np.zeros_like(S)
    
    # Gap between 1st and 2nd singular value
    gap = float(S[0] - S[1]) if len(S) > 1 else 0.0
    
    return S.tolist(), cum_var.tolist(), gap


def decompose_concepts(
    diff_vectors: np.ndarray,
    k_max: int = CONCEPT_K_MAX,
    method: str = "kmeans",
) -> Tuple[int, np.ndarray, List[np.ndarray], Dict[int, float]]:
    """
    Decompose mixed concepts using clustering on difference vectors.
    
    This implements Algorithm 2 from the Zwiad paper:
    1. Normalize difference vectors
    2. Try different k values
    3. Select best k by silhouette score
    4. Return cluster assignments and directions
    
    Args:
        diff_vectors: [n_samples, hidden_dim] difference vectors
        k_max: Maximum number of concepts to consider
        method: Clustering method ("kmeans" or "spectral")
        
    Returns:
        Tuple of:
        - k_detected: Estimated number of concepts
        - labels: Cluster assignments [n_samples]
        - directions: List of concept directions (normalized)
        - silhouette_scores: Dict mapping k -> silhouette score
    """
    n_samples = len(diff_vectors)
    diff_vectors = np.asarray(diff_vectors, dtype=np.float64)
    
    if n_samples < 4:
        direction = diff_vectors.mean(axis=0)
        norm = np.linalg.norm(direction)
        if norm > ZERO_THRESHOLD:
            direction = direction / norm
        return 1, np.zeros(n_samples, dtype=int), [direction], {1: 0.0}
    
    # Normalize difference vectors for clustering
    norms = np.linalg.norm(diff_vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, ZERO_THRESHOLD)
    diff_normalized = diff_vectors / norms
    
    # Try different k values
    silhouette_scores = {}
    best_k = 1
    best_score = -1
    best_labels = np.zeros(n_samples, dtype=int)
    
    for k in range(2, min(k_max + 1, n_samples // 2)):
        try:
            if method == "kmeans":
                clusterer = KMeans(n_clusters=k, random_state=DEFAULT_RANDOM_SEED, n_init=LINEARITY_N_INIT)
            else:
                from sklearn.cluster import SpectralClustering
                clusterer = SpectralClustering(n_clusters=k, random_state=DEFAULT_RANDOM_SEED)
            
            labels = clusterer.fit_predict(diff_normalized)
            
            if len(np.unique(labels)) < 2:
                continue
            
            score = silhouette_score(diff_normalized, labels)
            silhouette_scores[k] = score
            
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
                
        except Exception:
            continue
    
    # If no good clustering found, return single concept
    if best_score < CLUSTERING_MEANINGFUL_THRESHOLD:
        direction = diff_vectors.mean(axis=0)
        norm = np.linalg.norm(direction)
        if norm > ZERO_THRESHOLD:
            direction = direction / norm
        return 1, np.zeros(n_samples, dtype=int), [direction], silhouette_scores
    
    # Compute directions for each cluster
    directions = []
    for i in range(best_k):
        mask = best_labels == i
        if mask.sum() > 0:
            direction = diff_vectors[mask].mean(axis=0)
            norm = np.linalg.norm(direction)
            if norm > ZERO_THRESHOLD:
                direction = direction / norm
            directions.append(direction)
    
    return best_k, best_labels, directions, silhouette_scores




# Re-exports from split module
from wisent.core.contrastive_pairs.diagnostics.analysis._concept_analysis_part2 import (
    compute_concept_correlations,
    analyze_concepts,
    analyze_concept_interference,
)

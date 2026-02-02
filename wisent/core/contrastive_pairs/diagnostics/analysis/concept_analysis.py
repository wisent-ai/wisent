"""
Concept Analysis: ICD and Concept Decomposition

This module provides tools for analyzing whether a contrastive dataset contains
a single concept or multiple mixed concepts.

Key metrics:
- ICD (Intrinsic Concept Dimensionality): Effective rank of difference vectors
- Spectral decomposition: Recover individual concepts from mixed data
- Concept correlation: Measure similarity between concept directions

Theory reference: RepScan paper, Section "A Theory of Concepts in Activation Space"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA


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
    S = S[S > 1e-10]
    
    if len(S) == 0:
        return 0.0
    
    # Effective rank formula
    icd = (S.sum() ** 2) / (S ** 2).sum()
    
    return float(icd)


def compute_eigenvalue_spectrum(
    diff_vectors: np.ndarray,
    n_components: int = 20,
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
    k_max: int = 10,
    method: str = "kmeans",
) -> Tuple[int, np.ndarray, List[np.ndarray], Dict[int, float]]:
    """
    Decompose mixed concepts using clustering on difference vectors.
    
    This implements Algorithm 2 from the RepScan paper:
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
        if norm > 1e-10:
            direction = direction / norm
        return 1, np.zeros(n_samples, dtype=int), [direction], {1: 0.0}
    
    # Normalize difference vectors for clustering
    norms = np.linalg.norm(diff_vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    diff_normalized = diff_vectors / norms
    
    # Try different k values
    silhouette_scores = {}
    best_k = 1
    best_score = -1
    best_labels = np.zeros(n_samples, dtype=int)
    
    for k in range(2, min(k_max + 1, n_samples // 2)):
        try:
            if method == "kmeans":
                clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
            else:
                from sklearn.cluster import SpectralClustering
                clusterer = SpectralClustering(n_clusters=k, random_state=42)
            
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
    if best_score < 0.1:  # Threshold for meaningful clustering
        direction = diff_vectors.mean(axis=0)
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction = direction / norm
        return 1, np.zeros(n_samples, dtype=int), [direction], silhouette_scores
    
    # Compute directions for each cluster
    directions = []
    for i in range(best_k):
        mask = best_labels == i
        if mask.sum() > 0:
            direction = diff_vectors[mask].mean(axis=0)
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
            directions.append(direction)
    
    return best_k, best_labels, directions, silhouette_scores


def compute_concept_correlations(
    directions: List[np.ndarray],
    names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute pairwise correlations between concept directions.
    
    Args:
        directions: List of normalized concept directions
        names: Optional names for concepts (for dict keys)
        
    Returns:
        Dict mapping "concept_i vs concept_j" -> correlation
    """
    n = len(directions)
    if names is None:
        names = [f"concept_{i}" for i in range(n)]
    
    correlations = {}
    for i in range(n):
        for j in range(i + 1, n):
            corr = float(np.dot(directions[i], directions[j]))
            key = f"{names[i]} vs {names[j]}"
            correlations[key] = corr
    
    return correlations


def analyze_concepts(
    pos_activations: np.ndarray,
    neg_activations: np.ndarray,
    k_max: int = 10,
) -> ConceptAnalysisResult:
    """
    Full concept analysis on contrastive activations.
    
    This is the main entry point for concept analysis. It computes:
    - ICD (Intrinsic Concept Dimensionality)
    - Eigenvalue spectrum
    - Concept decomposition via clustering
    - Interpretation of results
    
    Args:
        pos_activations: [n_samples, hidden_dim] positive class activations
        neg_activations: [n_samples, hidden_dim] negative class activations
        k_max: Maximum number of concepts to detect
        
    Returns:
        ConceptAnalysisResult with all metrics
    """
    # Ensure numpy arrays
    if isinstance(pos_activations, torch.Tensor):
        pos_activations = pos_activations.float().cpu().numpy()
    if isinstance(neg_activations, torch.Tensor):
        neg_activations = neg_activations.float().cpu().numpy()
    
    pos_activations = np.asarray(pos_activations, dtype=np.float64)
    neg_activations = np.asarray(neg_activations, dtype=np.float64)
    
    # Compute difference vectors
    n = min(len(pos_activations), len(neg_activations))
    diff_vectors = pos_activations[:n] - neg_activations[:n]
    
    # Compute ICD
    icd = compute_icd(diff_vectors)
    
    # Compute eigenvalue spectrum
    singular_values, cum_variance, gap = compute_eigenvalue_spectrum(diff_vectors)
    
    # Decompose concepts
    k_detected, labels, directions, silhouette_scores = decompose_concepts(
        diff_vectors, k_max=k_max
    )
    
    # Compute cluster sizes
    cluster_sizes = []
    for i in range(k_detected):
        cluster_sizes.append(int((labels == i).sum()))
    
    # Compute concept correlations if multiple concepts
    concept_correlations = None
    if k_detected > 1:
        concept_correlations = compute_concept_correlations(directions)
    
    # Interpret results
    is_single_concept = True
    confidence = "low"
    interpretation = ""
    
    if icd < 20:
        is_single_concept = True
        confidence = "high"
        interpretation = f"ICD={icd:.1f} indicates a single dominant concept direction."
    elif icd < 100:
        # Check clustering quality
        best_sil = max(silhouette_scores.values()) if silhouette_scores else 0
        if best_sil > 0.3 and k_detected > 1:
            is_single_concept = False
            confidence = "medium"
            interpretation = f"ICD={icd:.1f} with silhouette={best_sil:.2f} suggests {k_detected} distinct concepts."
        else:
            is_single_concept = True
            confidence = "medium"
            interpretation = f"ICD={icd:.1f} is moderate but clustering is weak. Likely single concept with noise."
    else:
        # High ICD
        best_sil = max(silhouette_scores.values()) if silhouette_scores else 0
        if best_sil > 0.2 and k_detected > 1:
            is_single_concept = False
            confidence = "high"
            interpretation = f"ICD={icd:.1f} is high, suggesting {k_detected}+ mixed concepts or high noise."
        else:
            is_single_concept = False
            confidence = "low"
            interpretation = f"ICD={icd:.1f} is very high. Data may contain many concepts or be noise-dominated."
    
    return ConceptAnalysisResult(
        icd=icd,
        top_singular_values=singular_values[:10],
        top_variance_explained=cum_variance[:10] if cum_variance else [],
        eigenvalue_gap=gap,
        num_concepts_detected=k_detected,
        silhouette_scores=silhouette_scores,
        cluster_sizes=cluster_sizes,
        concept_directions=directions,
        concept_correlations=concept_correlations,
        is_single_concept=is_single_concept,
        confidence=confidence,
        interpretation=interpretation,
    )


def analyze_concept_interference(
    activations_a: Tuple[np.ndarray, np.ndarray],
    activations_b: Tuple[np.ndarray, np.ndarray],
    name_a: str = "concept_A",
    name_b: str = "concept_B",
) -> Dict[str, Any]:
    """
    Analyze interference between two concepts.
    
    Tests whether steering on concept A would affect concept B by:
    1. Computing direction correlation
    2. Training probe on A, testing on B (cross-probe accuracy)
    
    Args:
        activations_a: (pos_acts, neg_acts) for concept A
        activations_b: (pos_acts, neg_acts) for concept B
        name_a: Name for concept A
        name_b: Name for concept B
        
    Returns:
        Dict with correlation, cross-probe accuracy, and interpretation
    """
    from sklearn.linear_model import LogisticRegression
    
    pos_a, neg_a = activations_a
    pos_b, neg_b = activations_b
    
    # Convert to numpy
    if isinstance(pos_a, torch.Tensor):
        pos_a = pos_a.float().cpu().numpy()
        neg_a = neg_a.float().cpu().numpy()
        pos_b = pos_b.float().cpu().numpy()
        neg_b = neg_b.float().cpu().numpy()
    
    # Compute directions
    n_a = min(len(pos_a), len(neg_a))
    n_b = min(len(pos_b), len(neg_b))
    
    dir_a = (pos_a[:n_a] - neg_a[:n_a]).mean(axis=0)
    dir_b = (pos_b[:n_b] - neg_b[:n_b]).mean(axis=0)
    
    # Normalize
    dir_a = dir_a / (np.linalg.norm(dir_a) + 1e-10)
    dir_b = dir_b / (np.linalg.norm(dir_b) + 1e-10)
    
    # Correlation
    correlation = float(np.dot(dir_a, dir_b))
    
    # Cross-probe: train on A, test on B
    X_train = np.vstack([pos_a[:n_a], neg_a[:n_a]])
    y_train = np.array([1] * n_a + [0] * n_a)
    
    X_test = np.vstack([pos_b[:n_b], neg_b[:n_b]])
    y_test = np.array([1] * n_b + [0] * n_b)
    
    probe = LogisticRegression(max_iter=1000)
    probe.fit(X_train, y_train)
    
    cross_accuracy_a_to_b = float(probe.score(X_test, y_test))
    
    # Train on B, test on A
    probe_b = LogisticRegression(max_iter=1000)
    probe_b.fit(X_test, y_test)
    cross_accuracy_b_to_a = float(probe_b.score(X_train, y_train))
    
    # Interpretation
    if abs(correlation) > 0.5:
        interference = "high"
        interpretation = f"High correlation ({correlation:.2f}) means steering {name_a} will significantly affect {name_b}."
    elif abs(correlation) > 0.2:
        interference = "medium"
        interpretation = f"Moderate correlation ({correlation:.2f}) means some interference between concepts."
    else:
        interference = "low"
        interpretation = f"Low correlation ({correlation:.2f}) means concepts are nearly independent."
    
    return {
        "correlation": correlation,
        f"cross_probe_{name_a}_to_{name_b}": cross_accuracy_a_to_b,
        f"cross_probe_{name_b}_to_{name_a}": cross_accuracy_b_to_a,
        "interference_level": interference,
        "interpretation": interpretation,
    }

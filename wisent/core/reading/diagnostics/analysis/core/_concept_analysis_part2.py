"""Concept analysis: correlations, main analysis, and interference detection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

from wisent.core.utils.config_tools.constants import (
    ZERO_THRESHOLD, CONCEPT_K_MAX, CONCEPT_TOP_SINGULAR_VALUES,
    ICD_SINGLE_CONCEPT_THRESHOLD, ICD_MODERATE_THRESHOLD,
    SILHOUETTE_MULTI_CONCEPT_THRESHOLD, SILHOUETTE_WEAK_MULTI_THRESHOLD,
    CONCEPT_CORRELATION_HIGH, CONCEPT_CORRELATION_MODERATE,
)
from wisent.core.primitives.contrastive_pairs.diagnostics.analysis.concept_analysis import (
    ConceptAnalysisResult,
    compute_icd,
    compute_eigenvalue_spectrum,
    decompose_concepts,
)

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
    k_max: int = CONCEPT_K_MAX,
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
    
    if icd < ICD_SINGLE_CONCEPT_THRESHOLD:
        is_single_concept = True
        confidence = "high"
        interpretation = f"ICD={icd:.1f} indicates a single dominant concept direction."
    elif icd < ICD_MODERATE_THRESHOLD:
        # Check clustering quality
        best_sil = max(silhouette_scores.values()) if silhouette_scores else 0
        if best_sil > SILHOUETTE_MULTI_CONCEPT_THRESHOLD and k_detected > 1:
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
        if best_sil > SILHOUETTE_WEAK_MULTI_THRESHOLD and k_detected > 1:
            is_single_concept = False
            confidence = "high"
            interpretation = f"ICD={icd:.1f} is high, suggesting {k_detected}+ mixed concepts or high noise."
        else:
            is_single_concept = False
            confidence = "low"
            interpretation = f"ICD={icd:.1f} is very high. Data may contain many concepts or be noise-dominated."
    
    return ConceptAnalysisResult(
        icd=icd,
        top_singular_values=singular_values[:CONCEPT_TOP_SINGULAR_VALUES],
        top_variance_explained=cum_variance[:CONCEPT_TOP_SINGULAR_VALUES] if cum_variance else [],
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
    dir_a = dir_a / (np.linalg.norm(dir_a) + ZERO_THRESHOLD)
    dir_b = dir_b / (np.linalg.norm(dir_b) + ZERO_THRESHOLD)
    
    # Correlation
    correlation = float(np.dot(dir_a, dir_b))
    
    # Cross-probe: train on A, test on B
    X_train = np.vstack([pos_a[:n_a], neg_a[:n_a]])
    y_train = np.array([1] * n_a + [0] * n_a)
    
    X_test = np.vstack([pos_b[:n_b], neg_b[:n_b]])
    y_test = np.array([1] * n_b + [0] * n_b)
    
    probe = LogisticRegression()
    probe.fit(X_train, y_train)
    
    cross_accuracy_a_to_b = float(probe.score(X_test, y_test))
    
    # Train on B, test on A
    probe_b = LogisticRegression()
    probe_b.fit(X_test, y_test)
    cross_accuracy_b_to_a = float(probe_b.score(X_train, y_train))
    
    # Interpretation
    if abs(correlation) > CONCEPT_CORRELATION_HIGH:
        interference = "high"
        interpretation = f"High correlation ({correlation:.2f}) means steering {name_a} will significantly affect {name_b}."
    elif abs(correlation) > CONCEPT_CORRELATION_MODERATE:
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

"""Single-sample multi-concept detection algorithm."""

from typing import Dict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from wisent.core.utils.config_tools.constants import (
    ZERO_THRESHOLD, DEFAULT_RANDOM_SEED,
)


def detect_multiple_concepts_single_sample(
    diff_vectors: np.ndarray,
    n_bootstrap: int,
    seed: int = DEFAULT_RANDOM_SEED,
    *,
    kmeans_n_init_small: int,
    kmeans_n_init_medium: int,
    concept_pca_variance_90: float,
    concept_bic_strong_threshold: float,
    concept_silhouette_strong: float,
    concept_silhouette_moderate: float,
    concept_direction_distinct: float,
    concept_direction_moderate: float,
    concept_stability_threshold: float,
    linearity_n_init: int,
) -> Dict:
    """
    Detect if a SINGLE sample contains multiple concepts.
    
    This is the key function for answering: "Given just this data,
    does it contain multiple distinct concepts?"
    
    Method: Use ABSOLUTE thresholds calibrated from known single-concept data.
    The key insight is that for truly single-concept data:
    - BIC difference (1-component vs 2-component) should be NEGATIVE (1 is better)
    - Silhouette for k=2 should be LOW (< 0.1, no natural clusters)
    - Cross-cluster prediction should be HIGH (clusters don't separate classes)
    
    For mixed concepts:
    - BIC difference should be POSITIVE (2 components fit better)
    - Silhouette should be HIGHER (natural clusters exist)
    - Cross-cluster prediction should show cluster-class correlation
    
    Returns:
        Dictionary with detection results and confidence
    """
    print("  Computing metrics...")
    from sklearn.mixture import GaussianMixture
    
    n = len(diff_vectors)
    
    # 1. BIC Analysis - Does 2-component GMM fit better than 1-component?
    # Project to top PCs first (more robust)
    n_pcs = min(10, n - 1)
    pca = PCA(n_components=n_pcs)
    proj = pca.fit_transform(diff_vectors)
    
    gmm1 = GaussianMixture(n_components=1, random_state=seed, n_init=kmeans_n_init_medium)
    gmm1.fit(proj)
    gmm2 = GaussianMixture(n_components=2, random_state=seed, n_init=kmeans_n_init_medium)
    gmm2.fit(proj)
    gmm3 = GaussianMixture(n_components=3, random_state=seed, n_init=kmeans_n_init_medium)
    gmm3.fit(proj)
    
    bic_1 = gmm1.bic(proj)
    bic_2 = gmm2.bic(proj)
    bic_3 = gmm3.bic(proj)
    
    # BIC difference: positive means 2 is better
    bic_diff_2v1 = bic_1 - bic_2
    bic_diff_3v2 = bic_2 - bic_3
    
    # 2. Cluster Stability - Do different clustering runs agree?
    cluster_agreements = []
    for i in range(5):
        km1 = KMeans(n_clusters=2, random_state=seed + i, n_init=kmeans_n_init_small)
        km2 = KMeans(n_clusters=2, random_state=seed + i + 100, n_init=kmeans_n_init_small)
        labels1 = km1.fit_predict(diff_vectors)
        labels2 = km2.fit_predict(diff_vectors)
        # Compute agreement (accounting for label flipping)
        agreement1 = np.mean(labels1 == labels2)
        agreement2 = np.mean(labels1 == (1 - labels2))
        cluster_agreements.append(max(agreement1, agreement2))
    cluster_stability = np.mean(cluster_agreements)
    
    # 3. Silhouette Score
    km = KMeans(n_clusters=2, random_state=seed, n_init=linearity_n_init)
    cluster_labels = km.fit_predict(diff_vectors)
    silhouette = silhouette_score(diff_vectors, cluster_labels)
    
    # 4. Direction Variance Test
    # Split data by clusters and compute directions for each
    # If concepts are different, directions should be different
    cluster_0_mask = cluster_labels == 0
    cluster_1_mask = cluster_labels == 1
    
    if cluster_0_mask.sum() >= 5 and cluster_1_mask.sum() >= 5:
        dir_0 = diff_vectors[cluster_0_mask].mean(axis=0)
        dir_1 = diff_vectors[cluster_1_mask].mean(axis=0)
        dir_0 = dir_0 / (np.linalg.norm(dir_0) + ZERO_THRESHOLD)
        dir_1 = dir_1 / (np.linalg.norm(dir_1) + ZERO_THRESHOLD)
        cluster_direction_similarity = np.abs(np.dot(dir_0, dir_1))
    else:
        cluster_direction_similarity = 1.0  # Can't compute, assume same
    
    # 5. Eigenvalue Analysis
    pca_full = PCA(n_components=min(20, n-1))
    pca_full.fit(diff_vectors)
    evs = pca_full.explained_variance_ratio_
    
    # Effective dimensionality (how many PCs needed to explain 90% variance)
    cumsum = np.cumsum(evs)
    effective_dim = np.searchsorted(cumsum, concept_pca_variance_90) + 1
    
    # Ratio of second to first eigenvalue
    ev_ratio = evs[1] / evs[0] if len(evs) > 1 else 0
    
    # ===== DECISION LOGIC =====
    # Use calibrated thresholds based on our experiment
    
    evidence_for_multiple = 0
    evidence_details = []
    
    # BIC: If 2-component is substantially better (diff > 10), suggests multiple concepts
    if bic_diff_2v1 > concept_bic_strong_threshold:
        evidence_for_multiple += 2  # Strong evidence
        evidence_details.append(f"BIC strongly favors 2 components (diff={bic_diff_2v1:.1f})")
    elif bic_diff_2v1 > 0:
        evidence_for_multiple += 1  # Weak evidence
        evidence_details.append(f"BIC slightly favors 2 components (diff={bic_diff_2v1:.1f})")
    else:
        evidence_details.append(f"BIC favors 1 component (diff={bic_diff_2v1:.1f})")
    
    # Silhouette: If k=2 gives good clustering (> 0.1), suggests structure
    if silhouette > concept_silhouette_strong:
        evidence_for_multiple += 2
        evidence_details.append(f"Strong cluster structure (silhouette={silhouette:.3f})")
    elif silhouette > concept_silhouette_moderate:
        evidence_for_multiple += 1
        evidence_details.append(f"Moderate cluster structure (silhouette={silhouette:.3f})")
    else:
        evidence_details.append(f"Weak cluster structure (silhouette={silhouette:.3f})")
    
    # Direction similarity: If clusters have VERY different directions (< 0.2), strongly suggests different concepts
    # Note: Even single-concept data can have ~0.25-0.3 similarity due to noise
    # Truly mixed concepts (like TruthfulQA + HellaSwag) show ~0.1 similarity
    if cluster_direction_similarity < concept_direction_distinct:
        evidence_for_multiple += 3  # Very strong evidence - this is the key discriminator
        evidence_details.append(f"Clusters have VERY different directions (sim={cluster_direction_similarity:.3f}) - STRONG signal for multiple concepts")
    elif cluster_direction_similarity < concept_direction_moderate:
        evidence_for_multiple += 1
        evidence_details.append(f"Clusters have somewhat different directions (sim={cluster_direction_similarity:.3f})")
    else:
        evidence_details.append(f"Clusters have similar directions (sim={cluster_direction_similarity:.3f}) - typical for single concept")
    
    # Cluster stability: High stability (> 0.8) with good silhouette suggests real structure
    if cluster_stability > concept_stability_threshold and silhouette > 0.1:
        evidence_for_multiple += 1
        evidence_details.append(f"Clusters are stable (stability={cluster_stability:.3f})")
    
    # Determine verdict
    # The key discriminator is cluster_direction_similarity < concept_direction_distinct
    # This alone is strong evidence for multiple concepts
    has_strong_direction_signal = cluster_direction_similarity < concept_direction_distinct
    
    if has_strong_direction_signal:
        verdict = "MULTIPLE_CONCEPTS"
        confidence = "high"
    elif evidence_for_multiple >= 5:
        verdict = "MULTIPLE_CONCEPTS"
        confidence = "medium"
    elif evidence_for_multiple >= 3:
        verdict = "POSSIBLY_MULTIPLE"
        confidence = "low"
    else:
        verdict = "SINGLE_CONCEPT"
        confidence = "high" if evidence_for_multiple <= 1 else "medium"
    
    return {
        "verdict": verdict,
        "confidence": confidence,
        "evidence_score": evidence_for_multiple,
        "max_evidence": 7,  # Maximum possible score
        "metrics": {
            "bic_diff_2v1": float(bic_diff_2v1),
            "bic_diff_3v2": float(bic_diff_3v2),
            "silhouette_k2": float(silhouette),
            "cluster_stability": float(cluster_stability),
            "cluster_direction_similarity": float(cluster_direction_similarity),
            "eigenvalue_ratio": float(ev_ratio),
            "effective_dimensionality": int(effective_dim),
        },
        "evidence_details": evidence_details,
        "interpretation": f"{verdict} ({confidence} confidence). Evidence score: {evidence_for_multiple}/7",
    }


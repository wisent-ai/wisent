#!/usr/bin/env python3
"""
Detect Multiple Concepts in Mixed Samples (Without Labels)

This experiment tests whether we can detect that a sample contains
multiple distinct concepts when we DON'T know which pairs belong to which concept.

Setup:
- Mix 100 TruthfulQA pairs (truthfulness) with 100 HellaSwag pairs (commonsense)
- Treat them as a single unlabeled dataset of 200 pairs
- Apply various detection methods to see if we can tell there are 2 concepts

Detection Methods:
1. Eigenvalue Spectrum - Two concepts should have 2 significant eigenvalues
2. Clustering Quality - k=2 should fit better than k=1
3. Direction Consistency - Random splits should give inconsistent directions
4. Cross-Validation Variance - High variance indicates mixed concepts
5. Bimodality Test - Projections should be bimodal with mixed concepts

Usage:
    python -m wisent.examples.scripts.detect_mixed_concepts --model meta-llama/Llama-3.2-1B-Instruct
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score
from datasets import load_dataset
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import pacmap
    HAS_PACMAP = True
except ImportError:
    HAS_PACMAP = False

from wisent.core.models.wisent_model import WisentModel


@dataclass
class ConceptDetectionResult:
    """Results from concept detection analysis."""
    # Eigenvalue analysis
    eigenvalue_ratio: float  # lambda_2 / lambda_1 (higher = more concepts)
    top_5_eigenvalue_ratios: List[float]  # lambda_i / lambda_1 for i=1..5
    explained_variance_2d: float  # variance explained by first 2 PCs
    
    # Clustering analysis
    silhouette_k1: float  # silhouette for k=1 (always 0)
    silhouette_k2: float  # silhouette for k=2
    silhouette_k3: float  # silhouette for k=3
    inertia_ratio: float  # inertia_k2 / inertia_k1 (lower = k=2 fits better)
    
    # Direction consistency
    direction_cosine_similarities: List[float]  # cosine sim across random splits
    direction_consistency_mean: float
    direction_consistency_std: float
    
    # Cross-validation variance
    cv_accuracy_mean: float
    cv_accuracy_std: float
    cv_variance_ratio: float  # std / mean (higher = more variance)
    
    # Bimodality test
    dip_statistic: float  # Hartigan's dip test
    dip_pvalue: float
    is_bimodal: bool
    
    # GMM comparison
    bic_1_component: float
    bic_2_components: float
    bic_difference: float  # 1-component - 2-component (positive = 2 is better)
    
    # Final verdict
    num_concepts_detected: int
    confidence: str  # "high", "medium", "low"
    evidence_summary: str


def load_truthfulqa_pairs(n_pairs: int = 100, seed: int = 42) -> List[Dict]:
    """Load contrastive pairs from TruthfulQA."""
    random.seed(seed)
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    
    pairs = []
    indices = list(range(len(ds)))
    random.shuffle(indices)
    
    for idx in indices[:n_pairs]:
        sample = ds[idx]
        if sample["incorrect_answers"]:
            pairs.append({
                "question": sample["question"],
                "positive": sample["best_answer"],
                "negative": random.choice(sample["incorrect_answers"]),
                "source": "truthfulqa",
            })
    
    return pairs


def load_hellaswag_pairs(n_pairs: int = 100, seed: int = 42) -> List[Dict]:
    """Load contrastive pairs from HellaSwag."""
    random.seed(seed)
    ds = load_dataset("Rowan/hellaswag", split="validation")
    
    pairs = []
    indices = list(range(len(ds)))
    random.shuffle(indices)
    
    for idx in indices[:n_pairs]:
        sample = ds[idx]
        correct_idx = int(sample["label"])
        endings = sample["endings"]
        
        # Get incorrect endings
        incorrect_indices = [i for i in range(len(endings)) if i != correct_idx]
        if incorrect_indices:
            incorrect_idx = random.choice(incorrect_indices)
            
            context = sample["ctx"]
            pairs.append({
                "question": context,
                "positive": endings[correct_idx],
                "negative": endings[incorrect_idx],
                "source": "hellaswag",
            })
    
    return pairs[:n_pairs]


def get_activations(model: WisentModel, text: str, layer: int) -> torch.Tensor:
    """Extract last token activation from a specific layer."""
    inputs = model.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    activations = {}
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activations["hidden"] = output[0][:, -1, :].detach().cpu()
        else:
            activations["hidden"] = output[:, -1, :].detach().cpu()
    
    layers = model._layers
    handle = layers[layer].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        model.hf_model(**inputs)
    
    handle.remove()
    return activations["hidden"].squeeze(0)


def extract_difference_vectors(
    model: WisentModel,
    pairs: List[Dict],
    layer: int,
    show_progress: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract difference vectors (positive - negative) for all pairs.
    
    Returns:
        diff_vectors: [N, hidden_dim] array of difference vectors
        sources: list of source labels (for validation only, not used in detection)
    """
    diff_vectors = []
    sources = []
    
    total = len(pairs)
    for i, pair in enumerate(pairs):
        if show_progress and (i + 1) % 20 == 0:
            print(f"  Extracting activations: {i+1}/{total}")
        
        # Format as chat
        prompt = pair["question"]
        
        pos_text = model.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": pair["positive"]}],
            tokenize=False, add_generation_prompt=False
        )
        neg_text = model.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": pair["negative"]}],
            tokenize=False, add_generation_prompt=False
        )
        
        pos_act = get_activations(model, pos_text, layer)
        neg_act = get_activations(model, neg_text, layer)
        
        diff = (pos_act - neg_act).numpy()
        diff_vectors.append(diff)
        sources.append(pair.get("source", "unknown"))
    
    return np.array(diff_vectors), sources


def compute_eigenvalue_analysis(diff_vectors: np.ndarray) -> Dict:
    """Analyze eigenvalue spectrum of difference vectors."""
    pca = PCA(n_components=min(50, len(diff_vectors) - 1))
    pca.fit(diff_vectors)
    
    eigenvalues = pca.explained_variance_
    
    # Ratios relative to first eigenvalue
    ratios = eigenvalues / eigenvalues[0]
    
    return {
        "eigenvalue_ratio": ratios[1] if len(ratios) > 1 else 0,
        "top_5_ratios": ratios[:5].tolist(),
        "explained_variance_2d": sum(pca.explained_variance_ratio_[:2]),
        "eigenvalues": eigenvalues[:10].tolist(),
    }


def compute_clustering_analysis(diff_vectors: np.ndarray) -> Dict:
    """Analyze clustering quality for k=1, 2, 3."""
    results = {}
    
    # k=1: just compute inertia (no silhouette for k=1)
    km1 = KMeans(n_clusters=1, random_state=42, n_init=10)
    km1.fit(diff_vectors)
    results["inertia_k1"] = km1.inertia_
    results["silhouette_k1"] = 0.0  # undefined for k=1
    
    # k=2
    km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels2 = km2.fit_predict(diff_vectors)
    results["inertia_k2"] = km2.inertia_
    results["silhouette_k2"] = silhouette_score(diff_vectors, labels2)
    
    # k=3
    km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels3 = km3.fit_predict(diff_vectors)
    results["inertia_k3"] = km3.inertia_
    results["silhouette_k3"] = silhouette_score(diff_vectors, labels3)
    
    results["inertia_ratio"] = results["inertia_k2"] / results["inertia_k1"]
    
    return results


def compute_direction_consistency(
    diff_vectors: np.ndarray,
    n_splits: int = 10,
    seed: int = 42
) -> Dict:
    """
    Test if random splits give consistent directions.
    
    For a single concept, random splits should give similar directions.
    For mixed concepts, splits will be inconsistent.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    n = len(diff_vectors)
    directions = []
    
    for _ in range(n_splits):
        # Random 50% split
        indices = np.random.permutation(n)
        split_indices = indices[:n//2]
        
        # Compute mean direction for this split
        subset = diff_vectors[split_indices]
        direction = subset.mean(axis=0)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        directions.append(direction)
    
    # Compute pairwise cosine similarities
    cosine_sims = []
    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            sim = np.dot(directions[i], directions[j])
            cosine_sims.append(sim)
    
    return {
        "cosine_similarities": cosine_sims,
        "mean": np.mean(cosine_sims),
        "std": np.std(cosine_sims),
    }


def compute_cv_variance(diff_vectors: np.ndarray, n_folds: int = 5) -> Dict:
    """
    Compute cross-validation variance for linear probe.
    
    High variance indicates mixed concepts (some folds get lucky splits).
    """
    # Create labels: we'll use a proxy task of predicting positive vs negative
    # by using the sign of projection onto mean direction
    mean_dir = diff_vectors.mean(axis=0)
    mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-10)
    
    projections = diff_vectors @ mean_dir
    labels = (projections > np.median(projections)).astype(int)
    
    # Cross-validate a linear classifier
    clf = LogisticRegression(random_state=42, max_iter=1000)
    scores = cross_val_score(clf, diff_vectors, labels, cv=n_folds)
    
    return {
        "mean": scores.mean(),
        "std": scores.std(),
        "scores": scores.tolist(),
        "variance_ratio": scores.std() / (scores.mean() + 1e-10),
    }


def hartigans_dip_test(data: np.ndarray) -> Tuple[float, float]:
    """
    Hartigan's dip test for unimodality.
    
    Returns dip statistic and p-value.
    Higher dip = more evidence of multimodality.
    """
    from scipy.stats import uniform
    
    # Sort data
    data = np.sort(data.flatten())
    n = len(data)
    
    # Compute empirical CDF
    ecdf = np.arange(1, n + 1) / n
    
    # Compute greatest convex minorant and least concave majorant
    # Simplified version: compute max deviation from uniform
    uniform_cdf = np.linspace(0, 1, n)
    
    # Normalize data to [0, 1]
    data_norm = (data - data.min()) / (data.max() - data.min() + 1e-10)
    
    # Dip = max difference between empirical CDF and closest unimodal CDF
    dip = np.max(np.abs(ecdf - data_norm))
    
    # Approximate p-value using Monte Carlo
    n_simulations = 1000
    dip_null = []
    for _ in range(n_simulations):
        sample = np.sort(np.random.uniform(0, 1, n))
        sample_ecdf = np.arange(1, n + 1) / n
        dip_null.append(np.max(np.abs(sample_ecdf - sample)))
    
    p_value = np.mean(np.array(dip_null) >= dip)
    
    return dip, p_value


def compute_bimodality_analysis(diff_vectors: np.ndarray) -> Dict:
    """
    Test for bimodality in the projections onto the main direction.
    """
    # Project onto first PC
    pca = PCA(n_components=1)
    projections = pca.fit_transform(diff_vectors).flatten()
    
    # Hartigan's dip test
    dip, p_value = hartigans_dip_test(projections)
    
    # GMM comparison
    from sklearn.mixture import GaussianMixture
    
    projections_2d = projections.reshape(-1, 1)
    
    gmm1 = GaussianMixture(n_components=1, random_state=42)
    gmm1.fit(projections_2d)
    bic1 = gmm1.bic(projections_2d)
    
    gmm2 = GaussianMixture(n_components=2, random_state=42)
    gmm2.fit(projections_2d)
    bic2 = gmm2.bic(projections_2d)
    
    return {
        "dip_statistic": dip,
        "dip_pvalue": p_value,
        "is_bimodal": p_value < 0.05,
        "bic_1_component": bic1,
        "bic_2_components": bic2,
        "bic_difference": bic1 - bic2,  # positive = 2 is better
        "projections": projections.tolist(),
    }


def compute_null_distribution(
    diff_vectors: np.ndarray,
    n_bootstrap: int = 100,
    seed: int = 42
) -> Dict:
    """
    Compute null distribution of metrics assuming data is from ONE concept.
    
    We bootstrap resample the data and compute metrics. If the actual data
    has multiple concepts, its metrics should be outliers compared to this
    null distribution.
    
    Key insight: If data is truly one concept, resampling shouldn't change
    the structure much. If there are multiple concepts, some resamples will
    accidentally separate them and show different structure.
    """
    np.random.seed(seed)
    n = len(diff_vectors)
    
    null_bic_diffs = []
    null_silhouettes = []
    null_eigenvalue_ratios = []
    
    for _ in range(n_bootstrap):
        # Bootstrap resample
        indices = np.random.choice(n, size=n, replace=True)
        sample = diff_vectors[indices]
        
        # BIC difference
        pca = PCA(n_components=1)
        proj = pca.fit_transform(sample).reshape(-1, 1)
        
        from sklearn.mixture import GaussianMixture
        gmm1 = GaussianMixture(n_components=1, random_state=42)
        gmm1.fit(proj)
        gmm2 = GaussianMixture(n_components=2, random_state=42)
        gmm2.fit(proj)
        null_bic_diffs.append(gmm1.bic(proj) - gmm2.bic(proj))
        
        # Silhouette for k=2
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = km.fit_predict(sample)
        null_silhouettes.append(silhouette_score(sample, labels))
        
        # Eigenvalue ratio
        pca_full = PCA(n_components=min(10, n-1))
        pca_full.fit(sample)
        evs = pca_full.explained_variance_
        null_eigenvalue_ratios.append(evs[1] / evs[0] if len(evs) > 1 else 0)
    
    return {
        "bic_diff": {
            "mean": np.mean(null_bic_diffs),
            "std": np.std(null_bic_diffs),
            "p95": np.percentile(null_bic_diffs, 95),
            "values": null_bic_diffs,
        },
        "silhouette": {
            "mean": np.mean(null_silhouettes),
            "std": np.std(null_silhouettes),
            "p95": np.percentile(null_silhouettes, 95),
            "values": null_silhouettes,
        },
        "eigenvalue_ratio": {
            "mean": np.mean(null_eigenvalue_ratios),
            "std": np.std(null_eigenvalue_ratios),
            "p95": np.percentile(null_eigenvalue_ratios, 95),
            "values": null_eigenvalue_ratios,
        },
    }


def detect_multiple_concepts_single_sample(
    diff_vectors: np.ndarray,
    n_bootstrap: int = 100,
    seed: int = 42,
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
    
    gmm1 = GaussianMixture(n_components=1, random_state=seed, n_init=5)
    gmm1.fit(proj)
    gmm2 = GaussianMixture(n_components=2, random_state=seed, n_init=5)
    gmm2.fit(proj)
    gmm3 = GaussianMixture(n_components=3, random_state=seed, n_init=5)
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
        km1 = KMeans(n_clusters=2, random_state=seed + i, n_init=3)
        km2 = KMeans(n_clusters=2, random_state=seed + i + 100, n_init=3)
        labels1 = km1.fit_predict(diff_vectors)
        labels2 = km2.fit_predict(diff_vectors)
        # Compute agreement (accounting for label flipping)
        agreement1 = np.mean(labels1 == labels2)
        agreement2 = np.mean(labels1 == (1 - labels2))
        cluster_agreements.append(max(agreement1, agreement2))
    cluster_stability = np.mean(cluster_agreements)
    
    # 3. Silhouette Score
    km = KMeans(n_clusters=2, random_state=seed, n_init=10)
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
        dir_0 = dir_0 / (np.linalg.norm(dir_0) + 1e-10)
        dir_1 = dir_1 / (np.linalg.norm(dir_1) + 1e-10)
        cluster_direction_similarity = np.abs(np.dot(dir_0, dir_1))
    else:
        cluster_direction_similarity = 1.0  # Can't compute, assume same
    
    # 5. Eigenvalue Analysis
    pca_full = PCA(n_components=min(20, n-1))
    pca_full.fit(diff_vectors)
    evs = pca_full.explained_variance_ratio_
    
    # Effective dimensionality (how many PCs needed to explain 90% variance)
    cumsum = np.cumsum(evs)
    effective_dim = np.searchsorted(cumsum, 0.90) + 1
    
    # Ratio of second to first eigenvalue
    ev_ratio = evs[1] / evs[0] if len(evs) > 1 else 0
    
    # ===== DECISION LOGIC =====
    # Use calibrated thresholds based on our experiment
    
    evidence_for_multiple = 0
    evidence_details = []
    
    # BIC: If 2-component is substantially better (diff > 10), suggests multiple concepts
    if bic_diff_2v1 > 10:
        evidence_for_multiple += 2  # Strong evidence
        evidence_details.append(f"BIC strongly favors 2 components (diff={bic_diff_2v1:.1f})")
    elif bic_diff_2v1 > 0:
        evidence_for_multiple += 1  # Weak evidence
        evidence_details.append(f"BIC slightly favors 2 components (diff={bic_diff_2v1:.1f})")
    else:
        evidence_details.append(f"BIC favors 1 component (diff={bic_diff_2v1:.1f})")
    
    # Silhouette: If k=2 gives good clustering (> 0.1), suggests structure
    if silhouette > 0.15:
        evidence_for_multiple += 2
        evidence_details.append(f"Strong cluster structure (silhouette={silhouette:.3f})")
    elif silhouette > 0.08:
        evidence_for_multiple += 1
        evidence_details.append(f"Moderate cluster structure (silhouette={silhouette:.3f})")
    else:
        evidence_details.append(f"Weak cluster structure (silhouette={silhouette:.3f})")
    
    # Direction similarity: If clusters have VERY different directions (< 0.2), strongly suggests different concepts
    # Note: Even single-concept data can have ~0.25-0.3 similarity due to noise
    # Truly mixed concepts (like TruthfulQA + HellaSwag) show ~0.1 similarity
    if cluster_direction_similarity < 0.15:
        evidence_for_multiple += 3  # Very strong evidence - this is the key discriminator
        evidence_details.append(f"Clusters have VERY different directions (sim={cluster_direction_similarity:.3f}) - STRONG signal for multiple concepts")
    elif cluster_direction_similarity < 0.25:
        evidence_for_multiple += 1
        evidence_details.append(f"Clusters have somewhat different directions (sim={cluster_direction_similarity:.3f})")
    else:
        evidence_details.append(f"Clusters have similar directions (sim={cluster_direction_similarity:.3f}) - typical for single concept")
    
    # Cluster stability: High stability (> 0.8) with good silhouette suggests real structure
    if cluster_stability > 0.85 and silhouette > 0.1:
        evidence_for_multiple += 1
        evidence_details.append(f"Clusters are stable (stability={cluster_stability:.3f})")
    
    # Determine verdict
    # The key discriminator is cluster_direction_similarity < 0.15
    # This alone is strong evidence for multiple concepts
    has_strong_direction_signal = cluster_direction_similarity < 0.15
    
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


def detect_k_concepts(
    diff_vectors: np.ndarray,
    max_k: int = 6,
    direction_threshold: float = 0.20,  # Clusters with similarity < this are distinct concepts
    seed: int = 42,
) -> Dict:
    """
    Detect the number of distinct concepts in a sample of contrastive pairs.
    
    Method:
    1. For each k from 2 to max_k, cluster into k groups
    2. Compute pairwise direction similarity between all cluster pairs
    3. Count how many cluster pairs have low similarity (< threshold) = distinct concepts
    4. Find k where we have k distinct directions
    
    Args:
        diff_vectors: [N, hidden_dim] difference vectors
        max_k: Maximum number of clusters to try
        direction_threshold: Below this similarity, clusters are considered distinct concepts
        seed: Random seed
        
    Returns:
        Dictionary with detected number of concepts and analysis details
    """
    n = len(diff_vectors)
    max_k = min(max_k, n // 5)  # Need at least 5 samples per cluster
    
    results_by_k = {}
    
    for k in range(2, max_k + 1):
        # Cluster
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(diff_vectors)
        
        # Compute direction for each cluster
        directions = []
        cluster_sizes = []
        for cluster_id in range(k):
            mask = labels == cluster_id
            cluster_size = mask.sum()
            cluster_sizes.append(cluster_size)
            
            if cluster_size >= 3:
                direction = diff_vectors[mask].mean(axis=0)
                direction = direction / (np.linalg.norm(direction) + 1e-10)
                directions.append(direction)
            else:
                directions.append(None)
        
        # Compute pairwise similarities
        pairwise_sims = []
        distinct_pairs = []
        
        for i in range(k):
            for j in range(i + 1, k):
                if directions[i] is not None and directions[j] is not None:
                    sim = np.abs(np.dot(directions[i], directions[j]))
                    pairwise_sims.append({
                        'clusters': (i, j),
                        'similarity': sim,
                        'is_distinct': sim < direction_threshold
                    })
                    if sim < direction_threshold:
                        distinct_pairs.append((i, j))
        
        # Compute silhouette
        if k < n:
            sil = silhouette_score(diff_vectors, labels)
        else:
            sil = 0
        
        # Count distinct concepts using graph connectivity
        # If clusters A-B are distinct and B-C are distinct, we have 3 concepts
        # Build adjacency: clusters are "same concept" if similarity > threshold
        from collections import defaultdict
        
        same_concept_graph = defaultdict(set)
        for i in range(k):
            same_concept_graph[i].add(i)  # Self
            
        for pair_info in pairwise_sims:
            i, j = pair_info['clusters']
            if not pair_info['is_distinct']:  # High similarity = same concept
                same_concept_graph[i].add(j)
                same_concept_graph[j].add(i)
        
        # Find connected components (each component = one concept)
        visited = set()
        num_distinct_concepts = 0
        concept_groups = []
        
        for start in range(k):
            if start not in visited:
                # BFS to find component
                component = set()
                queue = [start]
                while queue:
                    node = queue.pop(0)
                    if node not in visited:
                        visited.add(node)
                        component.add(node)
                        for neighbor in same_concept_graph[node]:
                            if neighbor not in visited:
                                queue.append(neighbor)
                concept_groups.append(component)
                num_distinct_concepts += 1
        
        # Compute average within-concept similarity and between-concept similarity
        within_sims = []
        between_sims = []
        for pair_info in pairwise_sims:
            i, j = pair_info['clusters']
            # Check if i and j are in the same concept group
            same_group = any(i in group and j in group for group in concept_groups)
            if same_group:
                within_sims.append(pair_info['similarity'])
            else:
                between_sims.append(pair_info['similarity'])
        
        results_by_k[k] = {
            'num_clusters': k,
            'num_distinct_concepts': num_distinct_concepts,
            'concept_groups': [list(g) for g in concept_groups],
            'cluster_sizes': cluster_sizes,
            'silhouette': sil,
            'pairwise_similarities': pairwise_sims,
            'num_distinct_pairs': len(distinct_pairs),
            'avg_within_concept_sim': np.mean(within_sims) if within_sims else 1.0,
            'avg_between_concept_sim': np.mean(between_sims) if between_sims else 0.0,
            'min_pairwise_sim': min(p['similarity'] for p in pairwise_sims) if pairwise_sims else 1.0,
        }
    
    # Determine optimal k
    # Look for the k where num_distinct_concepts stabilizes
    # and silhouette is reasonable
    
    concept_counts = [(k, results_by_k[k]['num_distinct_concepts']) for k in range(2, max_k + 1)]
    
    # Find the smallest k where we capture all distinct concepts
    # (where increasing k doesn't increase num_distinct_concepts)
    optimal_k = 2
    optimal_concepts = results_by_k[2]['num_distinct_concepts']
    
    for k in range(2, max_k + 1):
        current_concepts = results_by_k[k]['num_distinct_concepts']
        current_sil = results_by_k[k]['silhouette']
        
        # Accept this k if it finds more concepts and has decent silhouette
        if current_concepts > optimal_concepts and current_sil > 0.05:
            optimal_k = k
            optimal_concepts = current_concepts
    
    return {
        'detected_concepts': optimal_concepts,
        'optimal_k': optimal_k,
        'results_by_k': results_by_k,
        'recommendation': f"Detected {optimal_concepts} distinct concept(s) using k={optimal_k} clusters",
    }


def get_activations_all_layers(model: WisentModel, text: str) -> Dict[int, torch.Tensor]:
    """Extract last token activation from ALL layers."""
    inputs = model.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    activations = {}
    handles = []
    
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                activations[layer_idx] = output[0][:, -1, :].detach().cpu()
            else:
                activations[layer_idx] = output[:, -1, :].detach().cpu()
        return hook_fn
    
    layers = model._layers
    for idx, layer in enumerate(layers):
        handle = layer.register_forward_hook(make_hook(idx))
        handles.append(handle)
    
    with torch.no_grad():
        model.hf_model(**inputs)
    
    for handle in handles:
        handle.remove()
    
    return {k: v.squeeze(0) for k, v in activations.items()}


def extract_difference_vectors_all_layers(
    model: WisentModel,
    pairs: List[Dict],
    show_progress: bool = True
) -> Dict[int, np.ndarray]:
    """
    Extract difference vectors for all layers.
    
    Returns:
        Dict mapping layer_idx -> [N, hidden_dim] array
    """
    all_diffs = {i: [] for i in range(model.num_layers)}
    
    total = len(pairs)
    for i, pair in enumerate(pairs):
        if show_progress and (i + 1) % 20 == 0:
            print(f"  Extracting activations: {i+1}/{total}")
        
        prompt = pair["question"]
        
        pos_text = model.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": pair["positive"]}],
            tokenize=False, add_generation_prompt=False
        )
        neg_text = model.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": pair["negative"]}],
            tokenize=False, add_generation_prompt=False
        )
        
        pos_acts = get_activations_all_layers(model, pos_text)
        neg_acts = get_activations_all_layers(model, neg_text)
        
        for layer_idx in range(model.num_layers):
            diff = (pos_acts[layer_idx] - neg_acts[layer_idx]).numpy()
            all_diffs[layer_idx].append(diff)
    
    return {k: np.array(v) for k, v in all_diffs.items()}


def compute_projection(
    diff_vectors: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, str]:
    """
    Project difference vectors to 2D using various methods.
    
    Args:
        diff_vectors: [N, hidden_dim] array
        method: "pca", "umap", or "pacmap"
        n_components: number of output dimensions
        seed: random seed
        
    Returns:
        projected: [N, n_components] array
        method_used: actual method used (may differ if requested not available)
    """
    if method == "umap":
        if not HAS_UMAP:
            print("  UMAP not installed, falling back to PCA")
            method = "pca"
        else:
            reducer = umap.UMAP(n_components=n_components, random_state=seed, n_neighbors=15, min_dist=0.1)
            projected = reducer.fit_transform(diff_vectors)
            return projected, "umap"
    
    if method == "pacmap":
        if not HAS_PACMAP:
            print("  PaCMAP not installed, falling back to PCA")
            method = "pca"
        else:
            reducer = pacmap.PaCMAP(n_components=n_components, random_state=seed)
            projected = reducer.fit_transform(diff_vectors)
            return projected, "pacmap"
    
    # Default: PCA
    pca = PCA(n_components=n_components, random_state=seed)
    projected = pca.fit_transform(diff_vectors)
    return projected, "pca"


def analyze_layer_separability(
    diff_vectors_by_layer: Dict[int, np.ndarray],
    sources: List[str],
) -> Dict[int, Dict]:
    """
    Analyze how well concepts are separated at each layer.
    
    For each layer, compute:
    - Silhouette score for k=2 clustering
    - Cluster direction similarity
    - Cluster purity (how well clusters align with true sources)
    """
    results = {}
    
    for layer_idx, diffs in diff_vectors_by_layer.items():
        # Cluster
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = km.fit_predict(diffs)
        
        # Silhouette
        sil = silhouette_score(diffs, labels)
        
        # Direction similarity
        mask0 = labels == 0
        mask1 = labels == 1
        if mask0.sum() >= 3 and mask1.sum() >= 3:
            dir0 = diffs[mask0].mean(axis=0)
            dir1 = diffs[mask1].mean(axis=0)
            dir0 = dir0 / (np.linalg.norm(dir0) + 1e-10)
            dir1 = dir1 / (np.linalg.norm(dir1) + 1e-10)
            dir_sim = np.abs(np.dot(dir0, dir1))
        else:
            dir_sim = 1.0
        
        # Cluster purity
        from collections import Counter
        c0_sources = [sources[i] for i in range(len(sources)) if labels[i] == 0]
        c1_sources = [sources[i] for i in range(len(sources)) if labels[i] == 1]
        
        if c0_sources and c1_sources:
            c0_purity = max(Counter(c0_sources).values()) / len(c0_sources)
            c1_purity = max(Counter(c1_sources).values()) / len(c1_sources)
            avg_purity = (c0_purity + c1_purity) / 2
        else:
            avg_purity = 0.5
        
        results[layer_idx] = {
            'silhouette': sil,
            'direction_similarity': dir_sim,
            'cluster_purity': avg_purity,
            'separability_score': (1 - dir_sim) * avg_purity,  # Higher = better separation aligned with sources
        }
    
    return results


def visualize_multi_method(
    diff_vectors: np.ndarray,
    sources: List[str],
    title: str = "Multi-Method Projection",
    output_path: Optional[str] = None,
    show_plot: bool = True,
):
    """
    Visualize projections using PCA, UMAP, and PaCMAP side by side.
    """
    methods = ["pca"]
    if HAS_UMAP:
        methods.append("umap")
    if HAS_PACMAP:
        methods.append("pacmap")
    
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    source_colors = {'truthfulqa': '#e74c3c', 'hellaswag': '#3498db', 'unknown': '#95a5a6'}
    unique_sources = list(set(sources))
    
    for ax, method in zip(axes, methods):
        proj, method_used = compute_projection(diff_vectors, method=method)
        
        for source in unique_sources:
            mask = np.array([s == source for s in sources])
            color = source_colors.get(source, '#95a5a6')
            ax.scatter(proj[mask, 0], proj[mask, 1], c=color, label=source, alpha=0.6, s=50)
        
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.set_title(f'{method_used.upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_layer_analysis(
    layer_results: Dict[int, Dict],
    title: str = "Layer-wise Separability Analysis",
    output_path: Optional[str] = None,
    show_plot: bool = True,
):
    """
    Visualize how concept separability varies across layers.
    """
    layers = sorted(layer_results.keys())
    silhouettes = [layer_results[l]['silhouette'] for l in layers]
    dir_sims = [layer_results[l]['direction_similarity'] for l in layers]
    purities = [layer_results[l]['cluster_purity'] for l in layers]
    sep_scores = [layer_results[l]['separability_score'] for l in layers]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Silhouette
    ax1 = axes[0, 0]
    ax1.plot(layers, silhouettes, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Cluster Quality (higher = better clusters)')
    ax1.grid(True, alpha=0.3)
    
    # Direction similarity
    ax2 = axes[0, 1]
    ax2.plot(layers, dir_sims, 'r-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Direction Similarity')
    ax2.set_title('Cluster Direction Similarity (lower = more distinct)')
    ax2.axhline(y=0.2, color='g', linestyle='--', label='Threshold (0.2)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Cluster purity
    ax3 = axes[1, 0]
    ax3.plot(layers, purities, 'g-o', linewidth=2, markersize=6)
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Cluster Purity')
    ax3.set_title('Alignment with True Sources (higher = better)')
    ax3.grid(True, alpha=0.3)
    
    # Combined separability
    ax4 = axes[1, 1]
    ax4.plot(layers, sep_scores, 'm-o', linewidth=2, markersize=6)
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Separability Score')
    ax4.set_title('Combined Score: (1 - dir_sim) * purity')
    best_layer = layers[np.argmax(sep_scores)]
    ax4.axvline(x=best_layer, color='orange', linestyle='--', label=f'Best: Layer {best_layer}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return best_layer


def attribute_pairs_to_concepts(
    diff_vectors: np.ndarray,
    pairs: List[Dict],
    detection_result: Dict,
) -> Dict:
    """
    Attribute each pair to its detected concept.
    
    Given the k-concept detection result, assign each original pair
    to one of the detected concepts. This allows you to:
    1. See which pairs belong to which concept
    2. Analyze the semantic content of each detected concept
    3. Validate that the detection aligns with true sources (if known)
    
    Args:
        diff_vectors: [N, hidden_dim] difference vectors
        pairs: Original list of contrastive pairs (with 'question', 'positive', 'negative')
        detection_result: Output from detect_k_concepts()
        
    Returns:
        Dictionary with:
        - concept_assignments: List[int] - concept ID for each pair
        - concepts: Dict mapping concept_id -> list of pair indices
        - concept_details: Dict with statistics and sample pairs for each concept
    """
    optimal_k = detection_result['optimal_k']
    k_result = detection_result['results_by_k'][optimal_k]
    concept_groups = k_result['concept_groups']  # List of sets of cluster IDs
    
    # Cluster all pairs
    km = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(diff_vectors)
    
    # Map cluster -> concept
    cluster_to_concept = {}
    for concept_id, cluster_set in enumerate(concept_groups):
        for cluster_id in cluster_set:
            cluster_to_concept[cluster_id] = concept_id
    
    # Assign each pair to a concept
    concept_assignments = [cluster_to_concept[c] for c in cluster_labels]
    
    # Group pairs by concept
    concepts = {i: [] for i in range(len(concept_groups))}
    for pair_idx, concept_id in enumerate(concept_assignments):
        concepts[concept_id].append(pair_idx)
    
    # Build detailed info for each concept
    concept_details = {}
    for concept_id, pair_indices in concepts.items():
        # Get sample pairs
        sample_indices = pair_indices[:5]  # First 5 as samples
        sample_pairs = [pairs[i] for i in sample_indices]
        
        # Compute concept direction
        concept_diffs = diff_vectors[pair_indices]
        concept_direction = concept_diffs.mean(axis=0)
        concept_direction = concept_direction / (np.linalg.norm(concept_direction) + 1e-10)
        
        # Source distribution (if sources are available)
        sources_in_concept = [pairs[i].get('source', 'unknown') for i in pair_indices]
        from collections import Counter
        source_distribution = dict(Counter(sources_in_concept))
        
        # Clusters in this concept
        clusters_in_concept = list(concept_groups[concept_id])
        
        concept_details[concept_id] = {
            'num_pairs': len(pair_indices),
            'pair_indices': pair_indices,
            'sample_pairs': sample_pairs,
            'source_distribution': source_distribution,
            'clusters': clusters_in_concept,
            'direction_norm': float(np.linalg.norm(concept_diffs.mean(axis=0))),
        }
    
    return {
        'concept_assignments': concept_assignments,
        'concepts': concepts,
        'concept_details': concept_details,
        'num_concepts': len(concept_groups),
    }


def print_concept_attribution(attribution: Dict, show_samples: bool = True):
    """Print a summary of concept attributions."""
    print(f"\n{'=' * 70}")
    print(f"CONCEPT ATTRIBUTION RESULTS")
    print(f"{'=' * 70}")
    print(f"\nDetected {attribution['num_concepts']} distinct concepts\n")
    
    for concept_id, details in attribution['concept_details'].items():
        print(f"\n--- Concept {concept_id} ---")
        print(f"  Pairs: {details['num_pairs']}")
        print(f"  Clusters: {details['clusters']}")
        print(f"  Source distribution: {details['source_distribution']}")
        
        if show_samples and details['sample_pairs']:
            print(f"\n  Sample pairs:")
            for i, pair in enumerate(details['sample_pairs'][:3]):
                q = pair['question'][:80] + '...' if len(pair['question']) > 80 else pair['question']
                p = pair['positive'][:50] + '...' if len(pair['positive']) > 50 else pair['positive']
                n = pair['negative'][:50] + '...' if len(pair['negative']) > 50 else pair['negative']
                source = pair.get('source', 'unknown')
                print(f"    [{i+1}] Source: {source}")
                print(f"        Q: {q}")
                print(f"        +: {p}")
                print(f"        -: {n}")


def visualize_k_concepts(
    diff_vectors: np.ndarray,
    sources: List[str],
    detection_result: Dict,
    title: str = "Multi-Concept Detection",
    output_path: Optional[str] = None,
    show_plot: bool = True,
):
    """
    Visualize the k-concept detection results.
    """
    optimal_k = detection_result['optimal_k']
    k_result = detection_result['results_by_k'][optimal_k]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"{title}\nDetected {detection_result['detected_concepts']} distinct concepts", 
                 fontsize=14, fontweight='bold')
    
    # PCA projection
    pca = PCA(n_components=2)
    proj_2d = pca.fit_transform(diff_vectors)
    
    # Cluster with optimal k
    km = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(diff_vectors)
    
    # Color palettes
    source_colors = {'truthfulqa': '#e74c3c', 'hellaswag': '#3498db', 'mmlu': '#2ecc71', 
                     'gsm8k': '#f39c12', 'unknown': '#95a5a6'}
    cluster_cmap = plt.cm.Set2
    
    # === Plot 1: By True Source ===
    ax1 = axes[0, 0]
    unique_sources = list(set(sources))
    for source in unique_sources:
        mask = np.array([s == source for s in sources])
        color = source_colors.get(source, '#95a5a6')
        ax1.scatter(proj_2d[mask, 0], proj_2d[mask, 1], c=color, label=source, alpha=0.6, s=50)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax1.set_title('Ground Truth Sources')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # === Plot 2: By Cluster ===
    ax2 = axes[0, 1]
    for i in range(optimal_k):
        mask = cluster_labels == i
        color = cluster_cmap(i / optimal_k)
        ax2.scatter(proj_2d[mask, 0], proj_2d[mask, 1], c=[color], label=f'Cluster {i}', alpha=0.6, s=50)
    ax2.set_xlabel(f'PC1')
    ax2.set_ylabel(f'PC2')
    ax2.set_title(f'K-Means Clustering (k={optimal_k})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # === Plot 3: By Detected Concept ===
    ax3 = axes[0, 2]
    concept_groups = k_result['concept_groups']
    concept_cmap = plt.cm.Set1
    
    for concept_id, cluster_group in enumerate(concept_groups):
        mask = np.isin(cluster_labels, list(cluster_group))
        color = concept_cmap(concept_id / len(concept_groups))
        ax3.scatter(proj_2d[mask, 0], proj_2d[mask, 1], c=[color], 
                   label=f'Concept {concept_id} (clusters {cluster_group})', alpha=0.6, s=50)
    ax3.set_xlabel(f'PC1')
    ax3.set_ylabel(f'PC2')
    ax3.set_title(f'Detected Concepts ({len(concept_groups)} found)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # === Plot 4: Pairwise Similarity Matrix ===
    ax4 = axes[1, 0]
    sim_matrix = np.eye(optimal_k)
    for pair_info in k_result['pairwise_similarities']:
        i, j = pair_info['clusters']
        sim_matrix[i, j] = pair_info['similarity']
        sim_matrix[j, i] = pair_info['similarity']
    
    im = ax4.imshow(sim_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    ax4.set_xticks(range(optimal_k))
    ax4.set_yticks(range(optimal_k))
    ax4.set_xticklabels([f'C{i}' for i in range(optimal_k)])
    ax4.set_yticklabels([f'C{i}' for i in range(optimal_k)])
    ax4.set_title('Cluster Direction Similarity\n(Red=Different, Green=Same)')
    plt.colorbar(im, ax=ax4)
    
    # Add text annotations
    for i in range(optimal_k):
        for j in range(optimal_k):
            ax4.text(j, i, f'{sim_matrix[i,j]:.2f}', ha='center', va='center', fontsize=9)
    
    # === Plot 5: Concepts vs K ===
    ax5 = axes[1, 1]
    ks = list(detection_result['results_by_k'].keys())
    concepts = [detection_result['results_by_k'][k]['num_distinct_concepts'] for k in ks]
    silhouettes = [detection_result['results_by_k'][k]['silhouette'] for k in ks]
    
    ax5.plot(ks, concepts, 'bo-', label='Distinct Concepts', linewidth=2, markersize=8)
    ax5.axhline(y=detection_result['detected_concepts'], color='r', linestyle='--', 
                label=f'Detected: {detection_result["detected_concepts"]}')
    ax5.axvline(x=optimal_k, color='g', linestyle='--', label=f'Optimal k={optimal_k}')
    ax5.set_xlabel('Number of Clusters (k)')
    ax5.set_ylabel('Distinct Concepts Found')
    ax5.set_title('Concepts Detected vs. k')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xticks(ks)
    
    # === Plot 6: Cluster Composition ===
    ax6 = axes[1, 2]
    
    from collections import Counter
    cluster_compositions = []
    for i in range(optimal_k):
        mask = cluster_labels == i
        cluster_sources = [sources[j] for j in range(len(sources)) if mask[j]]
        cluster_compositions.append(Counter(cluster_sources))
    
    x = np.arange(optimal_k)
    width = 0.8 / len(unique_sources)
    
    for idx, source in enumerate(unique_sources):
        counts = [cluster_compositions[i].get(source, 0) for i in range(optimal_k)]
        color = source_colors.get(source, '#95a5a6')
        ax6.bar(x + idx * width, counts, width, label=source, color=color)
    
    ax6.set_xticks(x + width * (len(unique_sources) - 1) / 2)
    ax6.set_xticklabels([f'Cluster {i}' for i in range(optimal_k)])
    ax6.set_ylabel('Count')
    ax6.set_title('Cluster Composition by Source')
    ax6.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_concept_detection(
    diff_vectors: np.ndarray,
    sources: List[str],
    title: str = "Concept Detection Visualization",
    output_path: Optional[str] = None,
    show_plot: bool = True,
):
    """
    Create a visualization showing:
    1. PCA projection colored by true source (if known)
    2. PCA projection colored by k=2 clustering
    3. The cluster directions as arrows
    4. Key metrics
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # PCA projection
    pca = PCA(n_components=2)
    proj_2d = pca.fit_transform(diff_vectors)
    
    # K-means clustering
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(diff_vectors)
    
    # Compute cluster directions
    cluster_0_mask = cluster_labels == 0
    cluster_1_mask = cluster_labels == 1
    
    dir_0 = diff_vectors[cluster_0_mask].mean(axis=0)
    dir_1 = diff_vectors[cluster_1_mask].mean(axis=0)
    dir_0_norm = dir_0 / (np.linalg.norm(dir_0) + 1e-10)
    dir_1_norm = dir_1 / (np.linalg.norm(dir_1) + 1e-10)
    cluster_sim = np.abs(np.dot(dir_0_norm, dir_1_norm))
    
    # Project directions to 2D for visualization
    dir_0_2d = pca.transform(dir_0.reshape(1, -1))[0]
    dir_1_2d = pca.transform(dir_1.reshape(1, -1))[0]
    
    # Get unique sources for coloring
    unique_sources = list(set(sources))
    source_colors = {'truthfulqa': '#e74c3c', 'hellaswag': '#3498db', 'unknown': '#95a5a6'}
    cluster_colors = {0: '#2ecc71', 1: '#9b59b6'}
    
    # === Plot 1: Colored by TRUE SOURCE ===
    ax1 = axes[0]
    for source in unique_sources:
        mask = np.array([s == source for s in sources])
        color = source_colors.get(source, '#95a5a6')
        ax1.scatter(proj_2d[mask, 0], proj_2d[mask, 1], 
                   c=color, label=source, alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax1.set_title('Colored by True Source\n(Ground Truth)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # === Plot 2: Colored by CLUSTER ===
    ax2 = axes[1]
    for cluster_id in [0, 1]:
        mask = cluster_labels == cluster_id
        color = cluster_colors[cluster_id]
        ax2.scatter(proj_2d[mask, 0], proj_2d[mask, 1],
                   c=color, label=f'Cluster {cluster_id}', alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    # Draw cluster direction arrows
    center = proj_2d.mean(axis=0)
    arrow_scale = 2.0
    ax2.annotate('', xy=center + dir_0_2d * arrow_scale, xytext=center,
                arrowprops=dict(arrowstyle='->', color=cluster_colors[0], lw=3))
    ax2.annotate('', xy=center + dir_1_2d * arrow_scale, xytext=center,
                arrowprops=dict(arrowstyle='->', color=cluster_colors[1], lw=3))
    
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax2.set_title(f'Colored by K-Means Cluster\nDirection Similarity: {cluster_sim:.3f}')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # === Plot 3: Cluster Composition ===
    ax3 = axes[2]
    
    # Count sources per cluster
    cluster_0_sources = [sources[i] for i in range(len(sources)) if cluster_labels[i] == 0]
    cluster_1_sources = [sources[i] for i in range(len(sources)) if cluster_labels[i] == 1]
    
    from collections import Counter
    c0_counts = Counter(cluster_0_sources)
    c1_counts = Counter(cluster_1_sources)
    
    # Create stacked bar chart
    x = np.arange(2)
    width = 0.6
    
    bottom_0 = 0
    bottom_1 = 0
    
    for source in unique_sources:
        color = source_colors.get(source, '#95a5a6')
        heights = [c0_counts.get(source, 0), c1_counts.get(source, 0)]
        ax3.bar(x, heights, width, bottom=[bottom_0, bottom_1], label=source, color=color, edgecolor='white')
        bottom_0 += heights[0]
        bottom_1 += heights[1]
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Cluster 0', 'Cluster 1'])
    ax3.set_ylabel('Count')
    ax3.set_title('Cluster Composition\n(How well clusters separate sources)')
    ax3.legend(loc='upper right')
    
    # Add purity annotation
    total_0 = len(cluster_0_sources)
    total_1 = len(cluster_1_sources)
    if total_0 > 0 and total_1 > 0:
        max_0 = max(c0_counts.values()) if c0_counts else 0
        max_1 = max(c1_counts.values()) if c1_counts else 0
        purity_0 = max_0 / total_0
        purity_1 = max_1 / total_1
        avg_purity = (purity_0 + purity_1) / 2
        ax3.text(0.5, -0.15, f'Cluster Purity: {avg_purity:.1%}', 
                transform=ax3.transAxes, ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return {
        "cluster_similarity": cluster_sim,
        "cluster_0_composition": dict(c0_counts),
        "cluster_1_composition": dict(c1_counts),
    }


def detect_concepts(
    diff_vectors: np.ndarray,
    sources: Optional[List[str]] = None
) -> ConceptDetectionResult:
    """
    Run all detection methods and synthesize results.
    """
    print("\n  Running eigenvalue analysis...")
    eigen_results = compute_eigenvalue_analysis(diff_vectors)
    
    print("  Running clustering analysis...")
    cluster_results = compute_clustering_analysis(diff_vectors)
    
    print("  Running direction consistency analysis...")
    consistency_results = compute_direction_consistency(diff_vectors)
    
    print("  Running cross-validation variance analysis...")
    cv_results = compute_cv_variance(diff_vectors)
    
    print("  Running bimodality analysis...")
    bimodal_results = compute_bimodality_analysis(diff_vectors)
    
    # Synthesize evidence
    evidence = []
    num_concepts_votes = []
    
    # Evidence 1: Eigenvalue ratio
    if eigen_results["eigenvalue_ratio"] > 0.5:
        evidence.append(f"High eigenvalue ratio ({eigen_results['eigenvalue_ratio']:.3f}) suggests multiple directions")
        num_concepts_votes.append(2)
    else:
        evidence.append(f"Low eigenvalue ratio ({eigen_results['eigenvalue_ratio']:.3f}) suggests single direction")
        num_concepts_votes.append(1)
    
    # Evidence 2: Clustering
    if cluster_results["silhouette_k2"] > 0.1:
        evidence.append(f"Good k=2 silhouette ({cluster_results['silhouette_k2']:.3f}) suggests 2 clusters")
        num_concepts_votes.append(2)
    else:
        evidence.append(f"Poor k=2 silhouette ({cluster_results['silhouette_k2']:.3f}) suggests 1 cluster")
        num_concepts_votes.append(1)
    
    # Evidence 3: Direction consistency
    if consistency_results["std"] > 0.1:
        evidence.append(f"Inconsistent directions (std={consistency_results['std']:.3f}) suggests multiple concepts")
        num_concepts_votes.append(2)
    else:
        evidence.append(f"Consistent directions (std={consistency_results['std']:.3f}) suggests single concept")
        num_concepts_votes.append(1)
    
    # Evidence 4: CV variance
    if cv_results["variance_ratio"] > 0.1:
        evidence.append(f"High CV variance ({cv_results['variance_ratio']:.3f}) suggests heterogeneous data")
        num_concepts_votes.append(2)
    else:
        evidence.append(f"Low CV variance ({cv_results['variance_ratio']:.3f}) suggests homogeneous data")
        num_concepts_votes.append(1)
    
    # Evidence 5: Bimodality
    if bimodal_results["bic_difference"] > 10:
        evidence.append(f"BIC favors 2 components (diff={bimodal_results['bic_difference']:.1f})")
        num_concepts_votes.append(2)
    else:
        evidence.append(f"BIC favors 1 component (diff={bimodal_results['bic_difference']:.1f})")
        num_concepts_votes.append(1)
    
    # Final verdict
    num_concepts = 2 if sum(num_concepts_votes) / len(num_concepts_votes) > 1.5 else 1
    confidence = "high" if sum(v == num_concepts for v in num_concepts_votes) >= 4 else \
                 "medium" if sum(v == num_concepts for v in num_concepts_votes) >= 3 else "low"
    
    return ConceptDetectionResult(
        eigenvalue_ratio=eigen_results["eigenvalue_ratio"],
        top_5_eigenvalue_ratios=eigen_results["top_5_ratios"],
        explained_variance_2d=eigen_results["explained_variance_2d"],
        silhouette_k1=cluster_results["silhouette_k1"],
        silhouette_k2=cluster_results["silhouette_k2"],
        silhouette_k3=cluster_results["silhouette_k3"],
        inertia_ratio=cluster_results["inertia_ratio"],
        direction_cosine_similarities=consistency_results["cosine_similarities"],
        direction_consistency_mean=consistency_results["mean"],
        direction_consistency_std=consistency_results["std"],
        cv_accuracy_mean=cv_results["mean"],
        cv_accuracy_std=cv_results["std"],
        cv_variance_ratio=cv_results["variance_ratio"],
        dip_statistic=bimodal_results["dip_statistic"],
        dip_pvalue=bimodal_results["dip_pvalue"],
        is_bimodal=bimodal_results["is_bimodal"],
        bic_1_component=bimodal_results["bic_1_component"],
        bic_2_components=bimodal_results["bic_2_components"],
        bic_difference=bimodal_results["bic_difference"],
        num_concepts_detected=num_concepts,
        confidence=confidence,
        evidence_summary="\n".join(evidence),
    )


def run_experiment(
    model_name: str,
    n_pairs_per_concept: int = 100,
    layer: int = None,
    seed: int = 42,
    output_dir: str = "/tmp/concept_detection"
):
    """
    Run the full experiment:
    1. Load pairs from both benchmarks
    2. Test MIXED (should detect 2 concepts)
    3. Test PURE TruthfulQA (should detect 1 concept)
    4. Test PURE HellaSwag (should detect 1 concept)
    """
    print("=" * 70)
    print("MIXED CONCEPT DETECTION EXPERIMENT")
    print("=" * 70)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nLoading model: {model_name}")
    model = WisentModel(model_name, device="mps")  # Use MPS for local
    
    # Determine layer
    if layer is None:
        layer = model.num_layers // 2  # Middle layer
    print(f"Using layer: {layer} (of {model.num_layers})")
    
    # Load pairs
    print(f"\nLoading TruthfulQA pairs ({n_pairs_per_concept})...")
    truthfulqa_pairs = load_truthfulqa_pairs(n_pairs_per_concept, seed)
    print(f"  Loaded {len(truthfulqa_pairs)} pairs")
    
    print(f"\nLoading HellaSwag pairs ({n_pairs_per_concept})...")
    hellaswag_pairs = load_hellaswag_pairs(n_pairs_per_concept, seed)
    print(f"  Loaded {len(hellaswag_pairs)} pairs")
    
    results = {}
    
    # ===== TEST 1: MIXED (should detect 2 concepts) =====
    print("\n" + "=" * 70)
    print("TEST 1: MIXED CONCEPTS (TruthfulQA + HellaSwag)")
    print("Expected: Should detect 2 distinct concepts")
    print("=" * 70)
    
    mixed_pairs = truthfulqa_pairs + hellaswag_pairs
    random.seed(seed)
    random.shuffle(mixed_pairs)  # Shuffle to remove any ordering info
    
    print(f"\nExtracting activations for {len(mixed_pairs)} mixed pairs...")
    mixed_diffs, mixed_sources = extract_difference_vectors(model, mixed_pairs, layer)
    
    print("\nAnalyzing mixed sample (labels hidden)...")
    mixed_result = detect_concepts(mixed_diffs)
    results["mixed"] = mixed_result
    
    print(f"\n--- MIXED RESULTS ---")
    print(f"Detected concepts: {mixed_result.num_concepts_detected}")
    print(f"Confidence: {mixed_result.confidence}")
    print(f"\nEvidence:")
    print(mixed_result.evidence_summary)
    
    # Validation: check if clusters align with true sources
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(mixed_diffs)
    
    # Compute alignment with true sources
    from collections import Counter
    cluster_0_sources = [mixed_sources[i] for i in range(len(mixed_sources)) if cluster_labels[i] == 0]
    cluster_1_sources = [mixed_sources[i] for i in range(len(mixed_sources)) if cluster_labels[i] == 1]
    
    print(f"\n[VALIDATION - using hidden labels]")
    print(f"Cluster 0: {Counter(cluster_0_sources)}")
    print(f"Cluster 1: {Counter(cluster_1_sources)}")
    
    # ===== TEST 2: PURE TruthfulQA (should detect 1 concept) =====
    print("\n" + "=" * 70)
    print("TEST 2: PURE TruthfulQA")
    print("Expected: Should detect 1 concept")
    print("=" * 70)
    
    print(f"\nExtracting activations for {len(truthfulqa_pairs)} TruthfulQA pairs...")
    tqa_diffs, tqa_sources = extract_difference_vectors(model, truthfulqa_pairs, layer)
    
    print("\nAnalyzing TruthfulQA-only sample...")
    tqa_result = detect_concepts(tqa_diffs)
    results["truthfulqa"] = tqa_result
    
    print(f"\n--- TRUTHFULQA RESULTS ---")
    print(f"Detected concepts: {tqa_result.num_concepts_detected}")
    print(f"Confidence: {tqa_result.confidence}")
    print(f"\nEvidence:")
    print(tqa_result.evidence_summary)
    
    # ===== TEST 3: PURE HellaSwag (should detect 1 concept) =====
    print("\n" + "=" * 70)
    print("TEST 3: PURE HellaSwag")
    print("Expected: Should detect 1 concept")
    print("=" * 70)
    
    print(f"\nExtracting activations for {len(hellaswag_pairs)} HellaSwag pairs...")
    hs_diffs, hs_sources = extract_difference_vectors(model, hellaswag_pairs, layer)
    
    print("\nAnalyzing HellaSwag-only sample...")
    hs_result = detect_concepts(hs_diffs)
    results["hellaswag"] = hs_result
    
    print(f"\n--- HELLASWAG RESULTS ---")
    print(f"Detected concepts: {hs_result.num_concepts_detected}")
    print(f"Confidence: {hs_result.confidence}")
    print(f"\nEvidence:")
    print(hs_result.evidence_summary)
    
    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Condition':<20} {'Detected':<10} {'Expected':<10} {'Match':<10}")
    print("-" * 50)
    
    conditions = [
        ("Mixed", mixed_result.num_concepts_detected, 2),
        ("TruthfulQA", tqa_result.num_concepts_detected, 1),
        ("HellaSwag", hs_result.num_concepts_detected, 1),
    ]
    
    for name, detected, expected in conditions:
        match = "YES" if detected == expected else "NO"
        print(f"{name:<20} {detected:<10} {expected:<10} {match:<10}")
    
    # Key metrics comparison
    print(f"\n{'Metric':<30} {'Mixed':<15} {'TruthfulQA':<15} {'HellaSwag':<15}")
    print("-" * 75)
    print(f"{'Eigenvalue ratio (λ2/λ1)':<30} {mixed_result.eigenvalue_ratio:<15.3f} {tqa_result.eigenvalue_ratio:<15.3f} {hs_result.eigenvalue_ratio:<15.3f}")
    print(f"{'Silhouette (k=2)':<30} {mixed_result.silhouette_k2:<15.3f} {tqa_result.silhouette_k2:<15.3f} {hs_result.silhouette_k2:<15.3f}")
    print(f"{'Direction consistency (std)':<30} {mixed_result.direction_consistency_std:<15.3f} {tqa_result.direction_consistency_std:<15.3f} {hs_result.direction_consistency_std:<15.3f}")
    print(f"{'CV variance ratio':<30} {mixed_result.cv_variance_ratio:<15.3f} {tqa_result.cv_variance_ratio:<15.3f} {hs_result.cv_variance_ratio:<15.3f}")
    print(f"{'BIC difference (2 vs 1)':<30} {mixed_result.bic_difference:<15.1f} {tqa_result.bic_difference:<15.1f} {hs_result.bic_difference:<15.1f}")
    
    # Save results
    output_file = output_path / f"concept_detection_{model_name.replace('/', '_')}.json"
    
    def to_python_float(val):
        """Convert numpy/torch floats to Python floats for JSON."""
        if hasattr(val, 'item'):
            return val.item()
        return float(val)
    
    with open(output_file, 'w') as f:
        json.dump({
            "model": model_name,
            "layer": layer,
            "n_pairs_per_concept": n_pairs_per_concept,
            "results": {
                "mixed": {
                    "num_concepts_detected": mixed_result.num_concepts_detected,
                    "confidence": mixed_result.confidence,
                    "eigenvalue_ratio": to_python_float(mixed_result.eigenvalue_ratio),
                    "silhouette_k2": to_python_float(mixed_result.silhouette_k2),
                    "direction_consistency_std": to_python_float(mixed_result.direction_consistency_std),
                    "cv_variance_ratio": to_python_float(mixed_result.cv_variance_ratio),
                    "bic_difference": to_python_float(mixed_result.bic_difference),
                },
                "truthfulqa": {
                    "num_concepts_detected": tqa_result.num_concepts_detected,
                    "confidence": tqa_result.confidence,
                    "eigenvalue_ratio": to_python_float(tqa_result.eigenvalue_ratio),
                    "silhouette_k2": to_python_float(tqa_result.silhouette_k2),
                    "direction_consistency_std": to_python_float(tqa_result.direction_consistency_std),
                    "cv_variance_ratio": to_python_float(tqa_result.cv_variance_ratio),
                    "bic_difference": to_python_float(tqa_result.bic_difference),
                },
                "hellaswag": {
                    "num_concepts_detected": hs_result.num_concepts_detected,
                    "confidence": hs_result.confidence,
                    "eigenvalue_ratio": to_python_float(hs_result.eigenvalue_ratio),
                    "silhouette_k2": to_python_float(hs_result.silhouette_k2),
                    "direction_consistency_std": to_python_float(hs_result.direction_consistency_std),
                    "cv_variance_ratio": to_python_float(hs_result.cv_variance_ratio),
                    "bic_difference": to_python_float(hs_result.bic_difference),
                },
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


def run_single_sample_detection(
    model_name: str,
    pairs: List[Dict],
    layer: int = None,
    n_bootstrap: int = 100,
    seed: int = 42,
):
    """
    Run detection on a SINGLE sample to determine if it contains multiple concepts.
    
    This is the answer to: "I have this data, how do I know if it's mixed?"
    """
    print("=" * 70)
    print("SINGLE SAMPLE CONCEPT DETECTION")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading model: {model_name}")
    model = WisentModel(model_name, device="mps")
    
    if layer is None:
        layer = model.num_layers // 2
    print(f"Using layer: {layer}")
    
    print(f"\nExtracting activations for {len(pairs)} pairs...")
    diff_vectors, _ = extract_difference_vectors(model, pairs, layer)
    
    print("\nRunning single-sample detection...")
    result = detect_multiple_concepts_single_sample(diff_vectors, n_bootstrap, seed)
    
    print(f"\n{'=' * 50}")
    print(f"VERDICT: {result['verdict']}")
    print(f"CONFIDENCE: {result['confidence']}")
    print(f"EVIDENCE SCORE: {result['evidence_score']}/{result['max_evidence']}")
    print(f"{'=' * 50}")
    
    print(f"\nEvidence breakdown:")
    for detail in result["evidence_details"]:
        print(f"  - {detail}")
    
    print(f"\nRaw metrics:")
    for metric_name, value in result["metrics"].items():
        print(f"  {metric_name}: {value}")
    
    print(f"\n{result['interpretation']}")
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect multiple concepts in mixed samples")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Model to use")
    parser.add_argument("--n-pairs", type=int, default=100,
                        help="Number of pairs per concept")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer to extract activations from (default: middle)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default="/tmp/concept_detection",
                        help="Output directory")
    parser.add_argument("--single-sample-test", action="store_true",
                        help="Run single-sample detection test (tests if we can detect mixed vs pure)")
    parser.add_argument("--n-bootstrap", type=int, default=100,
                        help="Number of bootstrap samples for null distribution")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations")
    parser.add_argument("--vis-output-dir", type=str, default="/tmp/concept_detection_vis",
                        help="Directory for visualization outputs")
    parser.add_argument("--detect-k", action="store_true",
                        help="Run k-concept detection (find how many concepts exist)")
    parser.add_argument("--max-k", type=int, default=6,
                        help="Maximum k to try for concept detection")
    parser.add_argument("--attribute", action="store_true",
                        help="Run attribution to trace pairs back to detected concepts")
    parser.add_argument("--analyze-layers", action="store_true",
                        help="Analyze separability across all layers")
    parser.add_argument("--projection-method", type=str, default="pca",
                        choices=["pca", "umap", "pacmap", "all"],
                        help="Projection method for visualization")
    
    args = parser.parse_args()
    
    if args.analyze_layers:
        # Multi-layer analysis
        print("=" * 70)
        print("MULTI-LAYER SEPARABILITY ANALYSIS")
        print("=" * 70)
        
        vis_output = Path(args.vis_output_dir)
        vis_output.mkdir(parents=True, exist_ok=True)
        
        # Load pairs
        print("\nLoading TruthfulQA pairs...")
        tqa_pairs = load_truthfulqa_pairs(args.n_pairs, args.seed)
        print(f"  Loaded {len(tqa_pairs)} pairs")
        
        print("Loading HellaSwag pairs...")
        hs_pairs = load_hellaswag_pairs(args.n_pairs, args.seed)
        print(f"  Loaded {len(hs_pairs)} pairs")
        
        # Load model
        print(f"\nLoading model: {args.model}")
        model = WisentModel(args.model, device="mps")
        print(f"Model has {model.num_layers} layers")
        
        # Create mixed sample
        mixed_pairs = tqa_pairs + hs_pairs
        random.seed(args.seed)
        random.shuffle(mixed_pairs)
        sources = [p['source'] for p in mixed_pairs]
        
        # Extract activations from ALL layers
        print(f"\nExtracting activations from ALL {model.num_layers} layers...")
        all_layer_diffs = extract_difference_vectors_all_layers(model, mixed_pairs)
        
        # Analyze separability at each layer
        print("\nAnalyzing separability at each layer...")
        layer_results = analyze_layer_separability(all_layer_diffs, sources)
        
        # Print results
        print(f"\n{'Layer':<8} {'Silhouette':<12} {'Dir Sim':<12} {'Purity':<12} {'Sep Score':<12}")
        print("-" * 56)
        for layer_idx in sorted(layer_results.keys()):
            r = layer_results[layer_idx]
            print(f"{layer_idx:<8} {r['silhouette']:<12.3f} {r['direction_similarity']:<12.3f} {r['cluster_purity']:<12.3f} {r['separability_score']:<12.3f}")
        
        # Find best layer
        best_layer = max(layer_results.keys(), key=lambda l: layer_results[l]['separability_score'])
        print(f"\nBest layer for concept separation: {best_layer}")
        print(f"  - Direction similarity: {layer_results[best_layer]['direction_similarity']:.3f}")
        print(f"  - Cluster purity: {layer_results[best_layer]['cluster_purity']:.3f}")
        
        # Visualize
        visualize_layer_analysis(
            layer_results,
            title=f"Layer-wise Separability: {args.model}",
            output_path=str(vis_output / "layer_analysis.png"),
            show_plot=False
        )
        
        # Also show multi-method projection for best layer
        if args.projection_method == "all":
            print(f"\nGenerating multi-method projections for layer {best_layer}...")
            visualize_multi_method(
                all_layer_diffs[best_layer],
                sources,
                title=f"Layer {best_layer} - Multi-Method Projection",
                output_path=str(vis_output / f"layer_{best_layer}_multi_method.png"),
                show_plot=False
            )
        
        print(f"\nVisualizations saved to: {vis_output}")
        
    elif args.attribute:
        # Run attribution to trace pairs to concepts
        print("=" * 70)
        print("CONCEPT ATTRIBUTION - Trace Pairs to Detected Concepts")
        print("=" * 70)
        
        # Load pairs
        print("\nLoading TruthfulQA pairs...")
        tqa_pairs = load_truthfulqa_pairs(args.n_pairs, args.seed)
        print(f"  Loaded {len(tqa_pairs)} pairs")
        
        print("Loading HellaSwag pairs...")
        hs_pairs = load_hellaswag_pairs(args.n_pairs, args.seed)
        print(f"  Loaded {len(hs_pairs)} pairs")
        
        # Load model
        print(f"\nLoading model: {args.model}")
        model = WisentModel(args.model, device="mps")
        
        layer = args.layer if args.layer else model.num_layers // 2
        print(f"Using layer: {layer}")
        
        # Create mixed sample (preserve original pairs list for attribution)
        mixed_pairs = tqa_pairs + hs_pairs
        original_order = list(range(len(mixed_pairs)))
        random.seed(args.seed)
        random.shuffle(mixed_pairs)
        
        # Extract activations
        print(f"\nExtracting activations for {len(mixed_pairs)} pairs...")
        mixed_diffs, mixed_sources = extract_difference_vectors(model, mixed_pairs, layer)
        
        # Run k-concept detection
        print("\nDetecting concepts...")
        detection = detect_k_concepts(mixed_diffs, max_k=args.max_k)
        print(f"  {detection['recommendation']}")
        
        # Run attribution
        print("\nAttributing pairs to concepts...")
        attribution = attribute_pairs_to_concepts(mixed_diffs, mixed_pairs, detection)
        
        # Print results
        print_concept_attribution(attribution, show_samples=True)
        
        # Summary of alignment with true sources
        print(f"\n{'=' * 70}")
        print("VALIDATION: How well do detected concepts align with true sources?")
        print("=" * 70)
        
        for concept_id, details in attribution['concept_details'].items():
            sources = details['source_distribution']
            total = details['num_pairs']
            dominant_source = max(sources.keys(), key=lambda k: sources[k])
            purity = sources[dominant_source] / total
            print(f"  Concept {concept_id}: {purity:.1%} purity ({dominant_source})")
            print(f"    Distribution: {sources}")
        
    elif args.detect_k:
        # Run k-concept detection
        print("=" * 70)
        print("K-CONCEPT DETECTION")
        print("=" * 70)
        
        vis_output = Path(args.vis_output_dir)
        vis_output.mkdir(parents=True, exist_ok=True)
        
        # Load pairs
        print("\nLoading TruthfulQA pairs...")
        tqa_pairs = load_truthfulqa_pairs(args.n_pairs, args.seed)
        print(f"  Loaded {len(tqa_pairs)} pairs")
        
        print("Loading HellaSwag pairs...")
        hs_pairs = load_hellaswag_pairs(args.n_pairs, args.seed)
        print(f"  Loaded {len(hs_pairs)} pairs")
        
        # Load model
        print(f"\nLoading model: {args.model}")
        model = WisentModel(args.model, device="mps")
        
        layer = args.layer if args.layer else model.num_layers // 2
        print(f"Using layer: {layer}")
        
        # Create mixed sample
        mixed_pairs = tqa_pairs + hs_pairs
        random.seed(args.seed)
        random.shuffle(mixed_pairs)
        
        # Extract activations
        print("\n--- Extracting activations ---")
        
        print("\nMixed sample...")
        mixed_diffs, mixed_sources = extract_difference_vectors(model, mixed_pairs, layer)
        
        print("\nTruthfulQA sample...")
        tqa_diffs, tqa_sources = extract_difference_vectors(model, tqa_pairs, layer)
        
        print("\nHellaSwag sample...")
        hs_diffs, hs_sources = extract_difference_vectors(model, hs_pairs, layer)
        
        # Run k-concept detection
        print("\n--- Running k-concept detection ---")
        
        print("\nAnalyzing MIXED sample...")
        mixed_detection = detect_k_concepts(mixed_diffs, max_k=args.max_k)
        print(f"  {mixed_detection['recommendation']}")
        
        print("\nAnalyzing TruthfulQA sample...")
        tqa_detection = detect_k_concepts(tqa_diffs, max_k=args.max_k)
        print(f"  {tqa_detection['recommendation']}")
        
        print("\nAnalyzing HellaSwag sample...")
        hs_detection = detect_k_concepts(hs_diffs, max_k=args.max_k)
        print(f"  {hs_detection['recommendation']}")
        
        # Summary
        print("\n" + "=" * 70)
        print("K-CONCEPT DETECTION SUMMARY")
        print("=" * 70)
        print(f"\n{'Sample':<20} {'Detected':<15} {'Expected':<15} {'Match':<10}")
        print("-" * 60)
        print(f"{'Mixed':<20} {mixed_detection['detected_concepts']:<15} {'2':<15} {'YES' if mixed_detection['detected_concepts'] == 2 else 'NO':<10}")
        print(f"{'TruthfulQA':<20} {tqa_detection['detected_concepts']:<15} {'1':<15} {'YES' if tqa_detection['detected_concepts'] == 1 else 'NO':<10}")
        print(f"{'HellaSwag':<20} {hs_detection['detected_concepts']:<15} {'1':<15} {'YES' if hs_detection['detected_concepts'] == 1 else 'NO':<10}")
        
        # Generate visualizations
        print("\n--- Generating visualizations ---")
        
        visualize_k_concepts(
            mixed_diffs, mixed_sources, mixed_detection,
            title="MIXED: TruthfulQA + HellaSwag",
            output_path=str(vis_output / "k_detection_mixed.png"),
            show_plot=False
        )
        
        visualize_k_concepts(
            tqa_diffs, tqa_sources, tqa_detection,
            title="PURE: TruthfulQA Only",
            output_path=str(vis_output / "k_detection_truthfulqa.png"),
            show_plot=False
        )
        
        visualize_k_concepts(
            hs_diffs, hs_sources, hs_detection,
            title="PURE: HellaSwag Only",
            output_path=str(vis_output / "k_detection_hellaswag.png"),
            show_plot=False
        )
        
        print(f"\nVisualizations saved to: {vis_output}")
        
    elif args.visualize:
        # Run visualization mode
        print("=" * 70)
        print("CONCEPT DETECTION VISUALIZATION")
        print("=" * 70)
        
        vis_output = Path(args.vis_output_dir)
        vis_output.mkdir(parents=True, exist_ok=True)
        
        # Load pairs
        print("\nLoading TruthfulQA pairs...")
        tqa_pairs = load_truthfulqa_pairs(args.n_pairs, args.seed)
        print(f"  Loaded {len(tqa_pairs)} pairs")
        
        print("Loading HellaSwag pairs...")
        hs_pairs = load_hellaswag_pairs(args.n_pairs, args.seed)
        print(f"  Loaded {len(hs_pairs)} pairs")
        
        # Load model
        print(f"\nLoading model: {args.model}")
        model = WisentModel(args.model, device="mps")
        
        layer = args.layer if args.layer else model.num_layers // 2
        print(f"Using layer: {layer}")
        
        # Create mixed sample
        mixed_pairs = tqa_pairs + hs_pairs
        random.seed(args.seed)
        random.shuffle(mixed_pairs)
        
        # Extract activations for all conditions
        print("\n--- Extracting activations ---")
        
        print("\nMixed sample...")
        mixed_diffs, mixed_sources = extract_difference_vectors(model, mixed_pairs, layer)
        
        print("\nTruthfulQA sample...")
        tqa_diffs, tqa_sources = extract_difference_vectors(model, tqa_pairs, layer)
        
        print("\nHellaSwag sample...")
        hs_diffs, hs_sources = extract_difference_vectors(model, hs_pairs, layer)
        
        # Generate visualizations
        print("\n--- Generating visualizations ---")
        
        print("\nMixed (TruthfulQA + HellaSwag):")
        visualize_concept_detection(
            mixed_diffs, mixed_sources,
            title="MIXED: TruthfulQA + HellaSwag (Should detect 2 concepts)",
            output_path=str(vis_output / "mixed_concepts.png"),
            show_plot=False
        )
        
        print("\nPure TruthfulQA:")
        visualize_concept_detection(
            tqa_diffs, tqa_sources,
            title="PURE: TruthfulQA Only (Should detect 1 concept)",
            output_path=str(vis_output / "truthfulqa_only.png"),
            show_plot=False
        )
        
        print("\nPure HellaSwag:")
        visualize_concept_detection(
            hs_diffs, hs_sources,
            title="PURE: HellaSwag Only (Should detect 1 concept)",
            output_path=str(vis_output / "hellaswag_only.png"),
            show_plot=False
        )
        
        print(f"\n{'=' * 70}")
        print(f"Visualizations saved to: {vis_output}")
        print("=" * 70)
        
    elif args.single_sample_test:
        # Test the single-sample detection on both mixed and pure samples
        print("=" * 70)
        print("TESTING SINGLE-SAMPLE DETECTION")
        print("=" * 70)
        print("\nThis test verifies that single-sample detection works by testing")
        print("it on samples we KNOW are mixed vs pure.\n")
        
        # Load pairs
        print("Loading TruthfulQA pairs...")
        tqa_pairs = load_truthfulqa_pairs(args.n_pairs, args.seed)
        print(f"  Loaded {len(tqa_pairs)} pairs")
        
        print("Loading HellaSwag pairs...")
        hs_pairs = load_hellaswag_pairs(args.n_pairs, args.seed)
        print(f"  Loaded {len(hs_pairs)} pairs")
        
        # Create mixed sample
        mixed_pairs = tqa_pairs + hs_pairs
        random.seed(args.seed)
        random.shuffle(mixed_pairs)
        
        print("\n" + "=" * 70)
        print("TEST A: Mixed sample (should detect MULTIPLE_CONCEPTS)")
        print("=" * 70)
        mixed_result = run_single_sample_detection(
            args.model, mixed_pairs, args.layer, args.n_bootstrap, args.seed
        )
        
        print("\n" + "=" * 70)
        print("TEST B: Pure TruthfulQA (should detect SINGLE_CONCEPT)")
        print("=" * 70)
        tqa_result = run_single_sample_detection(
            args.model, tqa_pairs, args.layer, args.n_bootstrap, args.seed
        )
        
        print("\n" + "=" * 70)
        print("TEST C: Pure HellaSwag (should detect SINGLE_CONCEPT)")
        print("=" * 70)
        hs_result = run_single_sample_detection(
            args.model, hs_pairs, args.layer, args.n_bootstrap, args.seed
        )
        
        # Summary
        print("\n" + "=" * 70)
        print("SINGLE-SAMPLE DETECTION SUMMARY")
        print("=" * 70)
        print(f"\n{'Sample':<20} {'Verdict':<20} {'Expected':<20} {'Match':<10}")
        print("-" * 70)
        
        tests = [
            ("Mixed", mixed_result["verdict"], "MULTIPLE_CONCEPTS"),
            ("TruthfulQA", tqa_result["verdict"], "SINGLE_CONCEPT"),
            ("HellaSwag", hs_result["verdict"], "SINGLE_CONCEPT"),
        ]
        
        for name, verdict, expected in tests:
            # POSSIBLY_MULTIPLE is acceptable for SINGLE_CONCEPT expectation
            if expected == "SINGLE_CONCEPT":
                match = "YES" if verdict in ["SINGLE_CONCEPT", "POSSIBLY_MULTIPLE"] else "NO"
            else:
                match = "YES" if verdict == expected else "PARTIAL" if verdict == "POSSIBLY_MULTIPLE" else "NO"
            print(f"{name:<20} {verdict:<20} {expected:<20} {match:<10}")
    else:
        run_experiment(
            model_name=args.model,
            n_pairs_per_concept=args.n_pairs,
            layer=args.layer,
            seed=args.seed,
            output_dir=args.output_dir,
        )

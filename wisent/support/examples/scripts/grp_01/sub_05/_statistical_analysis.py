"""Statistical analysis functions for mixed concept detection."""

import random
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score
from wisent.core.constants import (
    ZERO_THRESHOLD, DEFAULT_RANDOM_SEED, LINEARITY_N_INIT,
    LINEARITY_PCA_COMPONENTS, LINEARITY_N_BOOTSTRAP, STAT_ALPHA,
    CV_FOLDS, DIP_TEST_SIMULATIONS, DIRECTION_N_SPLITS, CV_DEFAULT_N_FOLDS,
    N_BOOTSTRAP_DEFAULT, DISPLAY_TOP_N_SMALL, DIP_TEST_N_SIMULATIONS,
    PERCENTILE_HIGH,
)


def compute_eigenvalue_analysis(diff_vectors: np.ndarray) -> Dict:
    """Analyze eigenvalue spectrum of difference vectors."""
    pca = PCA(n_components=min(LINEARITY_PCA_COMPONENTS, len(diff_vectors) - 1))
    pca.fit(diff_vectors)
    
    eigenvalues = pca.explained_variance_
    
    # Ratios relative to first eigenvalue
    ratios = eigenvalues / eigenvalues[0]
    
    return {
        "eigenvalue_ratio": ratios[1] if len(ratios) > 1 else 0,
        "top_5_ratios": ratios[:5].tolist(),
        "explained_variance_2d": sum(pca.explained_variance_ratio_[:2]),
        "eigenvalues": eigenvalues[:DISPLAY_TOP_N_SMALL].tolist(),
    }


def compute_clustering_analysis(diff_vectors: np.ndarray) -> Dict:
    """Analyze clustering quality for k=1, 2, 3."""
    results = {}
    
    # k=1: just compute inertia (no silhouette for k=1)
    km1 = KMeans(n_clusters=1, random_state=DEFAULT_RANDOM_SEED, n_init=LINEARITY_N_INIT)
    km1.fit(diff_vectors)
    results["inertia_k1"] = km1.inertia_
    results["silhouette_k1"] = 0.0  # undefined for k=1
    
    # k=2
    km2 = KMeans(n_clusters=2, random_state=DEFAULT_RANDOM_SEED, n_init=LINEARITY_N_INIT)
    labels2 = km2.fit_predict(diff_vectors)
    results["inertia_k2"] = km2.inertia_
    results["silhouette_k2"] = silhouette_score(diff_vectors, labels2)
    
    # k=3
    km3 = KMeans(n_clusters=3, random_state=DEFAULT_RANDOM_SEED, n_init=LINEARITY_N_INIT)
    labels3 = km3.fit_predict(diff_vectors)
    results["inertia_k3"] = km3.inertia_
    results["silhouette_k3"] = silhouette_score(diff_vectors, labels3)
    
    results["inertia_ratio"] = results["inertia_k2"] / results["inertia_k1"]
    
    return results


def compute_direction_consistency(
    diff_vectors: np.ndarray,
    n_splits: int = DIRECTION_N_SPLITS,
    seed: int = DEFAULT_RANDOM_SEED
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
        direction = direction / (np.linalg.norm(direction) + ZERO_THRESHOLD)
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


def compute_cv_variance(diff_vectors: np.ndarray, n_folds: int = CV_DEFAULT_N_FOLDS) -> Dict:
    """
    Compute cross-validation variance for linear probe.
    
    High variance indicates mixed concepts (some folds get lucky splits).
    """
    # Create labels: we'll use a proxy task of predicting positive vs negative
    # by using the sign of projection onto mean direction
    mean_dir = diff_vectors.mean(axis=0)
    mean_dir = mean_dir / (np.linalg.norm(mean_dir) + ZERO_THRESHOLD)
    
    projections = diff_vectors @ mean_dir
    labels = (projections > np.median(projections)).astype(int)
    
    # Cross-validate a linear classifier
    clf = LogisticRegression(random_state=DEFAULT_RANDOM_SEED, )
    scores = cross_val_score(clf, diff_vectors, labels, cv=n_folds)
    
    return {
        "mean": scores.mean(),
        "std": scores.std(),
        "scores": scores.tolist(),
        "variance_ratio": scores.std() / (scores.mean() + ZERO_THRESHOLD),
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
    data_norm = (data - data.min()) / (data.max() - data.min() + ZERO_THRESHOLD)
    
    # Dip = max difference between empirical CDF and closest unimodal CDF
    dip = np.max(np.abs(ecdf - data_norm))
    
    # Approximate p-value using Monte Carlo
    dip_null = []
    for _ in range(DIP_TEST_N_SIMULATIONS):
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
    
    gmm1 = GaussianMixture(n_components=1, random_state=DEFAULT_RANDOM_SEED)
    gmm1.fit(projections_2d)
    bic1 = gmm1.bic(projections_2d)
    
    gmm2 = GaussianMixture(n_components=2, random_state=DEFAULT_RANDOM_SEED)
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
    n_bootstrap: int = N_BOOTSTRAP_DEFAULT,
    seed: int = DEFAULT_RANDOM_SEED
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
        gmm1 = GaussianMixture(n_components=1, random_state=DEFAULT_RANDOM_SEED)
        gmm1.fit(proj)
        gmm2 = GaussianMixture(n_components=2, random_state=DEFAULT_RANDOM_SEED)
        gmm2.fit(proj)
        null_bic_diffs.append(gmm1.bic(proj) - gmm2.bic(proj))
        
        # Silhouette for k=2
        km = KMeans(n_clusters=2, random_state=DEFAULT_RANDOM_SEED, n_init=LINEARITY_N_INIT)
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
            "p95": np.percentile(null_bic_diffs, PERCENTILE_HIGH),
            "values": null_bic_diffs,
        },
        "silhouette": {
            "mean": np.mean(null_silhouettes),
            "std": np.std(null_silhouettes),
            "p95": np.percentile(null_silhouettes, PERCENTILE_HIGH),
            "values": null_silhouettes,
        },
        "eigenvalue_ratio": {
            "mean": np.mean(null_eigenvalue_ratios),
            "std": np.std(null_eigenvalue_ratios),
            "p95": np.percentile(null_eigenvalue_ratios, PERCENTILE_HIGH),
            "values": null_eigenvalue_ratios,
        },
    }

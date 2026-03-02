"""Detect concepts and run single-sample detection."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from wisent.core.utils.config_tools.constants import CONCEPT_BIC_STRONG_THRESHOLD, DEFAULT_RANDOM_SEED, N_BOOTSTRAP_DEFAULT, CONFIDENCE_HIGH_VOTES, CONFIDENCE_MEDIUM_VOTES, SEPARATOR_WIDTH_WIDE, SEPARATOR_WIDTH_MEDIUM
from wisent.core.primitives.models.wisent_model import WisentModel

from ._data_loading import extract_difference_vectors
from ._statistical_analysis import (
    compute_eigenvalue_analysis,
    compute_clustering_analysis,
    compute_direction_consistency,
    compute_cv_variance,
    compute_bimodality_analysis,
)
from ._detection_single import detect_multiple_concepts_single_sample


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
    dip_statistic: float  # Hartigans dip test
    dip_pvalue: float
    is_bimodal: bool
    
    # GMM comparison
    bic_1_component: float
    bic_2_components: float
    bic_difference: float  # 1-component - 2-component (positive = 2 is better)
    
    # Final verdict
    num_concepts_detected: int
    confidence: str  # high, medium, low
    evidence_summary: str


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
    if bimodal_results["bic_difference"] > CONCEPT_BIC_STRONG_THRESHOLD:
        evidence.append(f"BIC favors 2 components (diff={bimodal_results['bic_difference']:.1f})")
        num_concepts_votes.append(2)
    else:
        evidence.append(f"BIC favors 1 component (diff={bimodal_results['bic_difference']:.1f})")
        num_concepts_votes.append(1)
    
    # Final verdict
    num_concepts = 2 if sum(num_concepts_votes) / len(num_concepts_votes) > 1.5 else 1
    confidence = "high" if sum(v == num_concepts for v in num_concepts_votes) >= CONFIDENCE_HIGH_VOTES else \
                 "medium" if sum(v == num_concepts for v in num_concepts_votes) >= CONFIDENCE_MEDIUM_VOTES else "low"
    
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

def run_single_sample_detection(
    model_name: str,
    pairs: List[Dict],
    layer: int = None,
    n_bootstrap: int = N_BOOTSTRAP_DEFAULT,
    seed: int = DEFAULT_RANDOM_SEED,
):
    """
    Run detection on a SINGLE sample to determine if it contains multiple concepts.
    
    This is the answer to: "I have this data, how do I know if it's mixed?"
    """
    print("=" * SEPARATOR_WIDTH_WIDE)
    print("SINGLE SAMPLE CONCEPT DETECTION")
    print("=" * SEPARATOR_WIDTH_WIDE)
    
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
    
    print(f"\n{'=' * SEPARATOR_WIDTH_MEDIUM}")
    print(f"VERDICT: {result['verdict']}")
    print(f"CONFIDENCE: {result['confidence']}")
    print(f"EVIDENCE SCORE: {result['evidence_score']}/{result['max_evidence']}")
    print(f"{'=' * SEPARATOR_WIDTH_MEDIUM}")
    
    print(f"\nEvidence breakdown:")
    for detail in result["evidence_details"]:
        print(f"  - {detail}")
    
    print(f"\nRaw metrics:")
    for metric_name, value in result["metrics"].items():
        print(f"  {metric_name}: {value}")
    
    print(f"\n{result['interpretation']}")
    
    return result

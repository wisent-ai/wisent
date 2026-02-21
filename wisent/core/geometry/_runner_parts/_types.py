"""Auxiliary result type dataclasses for the geometry runner."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import torch
import numpy as np


@dataclass
class ZwiadResult:
    """Complete result from run_full_zwiad()."""
    benchmark: str
    model_name: str
    layer: int
    # Pair optimization
    optimal_n_pairs: int
    pair_search_history: List[Dict[str, Any]]
    saturation_status: str  # "OPTIMAL", "MAX_REACHED", "NO_SIGNAL"
    # Signal classification
    signal_exists: bool
    signal_type: str  # "LINEAR", "NONLINEAR", "MULTIMODAL", "NO_SIGNAL"
    # Method recommendation
    recommended_method: str  # "CAA", "TECZA", "TETNO", "GROM", "NO_METHOD"
    confidence: float
    reason: str
    # Core metrics
    linear_probe_accuracy: float
    best_nonlinear_accuracy: float
    knn_umap_accuracy: float
    knn_pacmap_accuracy: float
    # Geometry metrics (diff vectors)
    icd: float
    icd_top1_variance: float
    diff_mean_alignment: float
    caa_probe_alignment: float
    steerability_score: float
    effective_steering_dims: int
    # Multi-direction metrics
    multi_dir_gain: float
    spherical_silhouette_k2: float
    cluster_direction_angle: float
    # Quality metrics
    cohens_d: float
    bootstrap_std: float
    direction_stability: float
    # Steering vector (if signal exists)
    steering_direction: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding tensor)."""
        return {
            "benchmark": self.benchmark,
            "model_name": self.model_name,
            "layer": self.layer,
            "optimal_n_pairs": self.optimal_n_pairs,
            "pair_search_history": self.pair_search_history,
            "saturation_status": self.saturation_status,
            "signal_exists": self.signal_exists,
            "signal_type": self.signal_type,
            "recommended_method": self.recommended_method,
            "confidence": self.confidence,
            "reason": self.reason,
            "linear_probe_accuracy": self.linear_probe_accuracy,
            "best_nonlinear_accuracy": self.best_nonlinear_accuracy,
            "knn_umap_accuracy": self.knn_umap_accuracy,
            "knn_pacmap_accuracy": self.knn_pacmap_accuracy,
            "icd": self.icd,
            "icd_top1_variance": self.icd_top1_variance,
            "diff_mean_alignment": self.diff_mean_alignment,
            "caa_probe_alignment": self.caa_probe_alignment,
            "steerability_score": self.steerability_score,
            "effective_steering_dims": self.effective_steering_dims,
            "multi_dir_gain": self.multi_dir_gain,
            "spherical_silhouette_k2": self.spherical_silhouette_k2,
            "cluster_direction_angle": self.cluster_direction_angle,
            "cohens_d": self.cohens_d,
            "bootstrap_std": self.bootstrap_std,
            "direction_stability": self.direction_stability,
        }


@dataclass
class ZwiadLayerResult:
    """Result from layer search - extends ZwiadResult with layer analysis."""
    best_result: ZwiadResult
    optimal_layer: int
    optimal_layer_range: List[int]
    layer_search_history: List[Dict[str, Any]]
    num_layers: int
    layers_tested: List[int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = self.best_result.to_dict()
        d.update({
            "optimal_layer": self.optimal_layer,
            "optimal_layer_range": self.optimal_layer_range,
            "layer_search_history": self.layer_search_history,
            "num_layers": self.num_layers,
            "layers_tested": self.layers_tested,
        })
        return d


@dataclass
class ComponentAnalysisResult:
    """Result from analyzing different transformer components."""
    best_component: str
    best_component_accuracy: float
    residual_accuracy: float
    mlp_accuracy: float
    attn_accuracy: float
    head_accuracies: Dict[int, float]
    top_heads: List[Tuple[int, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_component": self.best_component,
            "best_component_accuracy": self.best_component_accuracy,
            "residual_accuracy": self.residual_accuracy,
            "mlp_accuracy": self.mlp_accuracy,
            "attn_accuracy": self.attn_accuracy,
            "head_accuracies": self.head_accuracies,
            "top_heads": self.top_heads,
        }


@dataclass
class MultiConceptAnalysis:
    """Result from analyzing if multiple concepts exist in a contrastive pair set."""
    num_concepts_detected: int
    is_multi_concept: bool
    confidence: float
    icd: float
    icd_suggests_multi: bool
    cluster_count: int
    cluster_silhouette: float
    clusters_suggest_multi: bool
    pca_variance_ratio: List[float]
    pca_effective_rank: float
    pca_suggests_multi: bool
    multi_dir_accuracy: Dict[int, float]
    multi_dir_gain: float
    directions_suggest_multi: bool
    concept_directions: Optional[List[torch.Tensor]] = None
    concept_sizes: Optional[List[int]] = None
    concept_accuracies: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_concepts_detected": self.num_concepts_detected,
            "is_multi_concept": self.is_multi_concept,
            "confidence": self.confidence,
            "icd": self.icd,
            "icd_suggests_multi": self.icd_suggests_multi,
            "cluster_count": self.cluster_count,
            "cluster_silhouette": self.cluster_silhouette,
            "clusters_suggest_multi": self.clusters_suggest_multi,
            "pca_variance_ratio": self.pca_variance_ratio,
            "pca_effective_rank": self.pca_effective_rank,
            "pca_suggests_multi": self.pca_suggests_multi,
            "multi_dir_accuracy": self.multi_dir_accuracy,
            "multi_dir_gain": self.multi_dir_gain,
            "directions_suggest_multi": self.directions_suggest_multi,
            "concept_sizes": self.concept_sizes,
            "concept_accuracies": self.concept_accuracies,
        }


@dataclass
class ConceptValidityResult:
    """Result from analyzing if a set of pairs represents a valid concept."""
    is_valid_concept: bool
    validity_score: float
    concept_level: str  # "instance", "category", "domain", "noise"
    coherence_score: float
    coherence_std: float
    is_coherent: bool
    stability_score: float
    stability_std: float
    is_stable: bool
    signal_strength: float
    signal_to_noise: float
    has_signal: bool
    icd: float
    specificity: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid_concept": self.is_valid_concept,
            "validity_score": self.validity_score,
            "concept_level": self.concept_level,
            "coherence_score": self.coherence_score,
            "coherence_std": self.coherence_std,
            "is_coherent": self.is_coherent,
            "stability_score": self.stability_score,
            "stability_std": self.stability_std,
            "is_stable": self.is_stable,
            "signal_strength": self.signal_strength,
            "signal_to_noise": self.signal_to_noise,
            "has_signal": self.has_signal,
            "icd": self.icd,
            "specificity": self.specificity,
        }


@dataclass
class ConceptDecomposition:
    """Result from decomposing a contrastive pair set into constituent concepts."""
    n_concepts: int
    decomposition_method: str  # "clustering", "nmf", "ica"
    concept_directions: List[torch.Tensor]
    concept_sizes: List[int]
    concept_coherences: List[float]
    concept_validities: List[bool]
    concept_relationships: List[Dict[str, Any]]
    pair_to_concept: List[int]
    pair_concept_scores: np.ndarray
    total_explained_variance: float
    reconstruction_error: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_concepts": self.n_concepts,
            "decomposition_method": self.decomposition_method,
            "concept_sizes": self.concept_sizes,
            "concept_coherences": self.concept_coherences,
            "concept_validities": self.concept_validities,
            "concept_relationships": self.concept_relationships,
            "pair_to_concept": self.pair_to_concept,
            "total_explained_variance": self.total_explained_variance,
            "reconstruction_error": self.reconstruction_error,
        }

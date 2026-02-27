"""Concept analysis metric functions and layer-wise analysis."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

from wisent.core.models.wisent_model import WisentModel
from wisent.core.activations.activations_collector import ActivationCollector
from wisent.core.activations import ExtractionStrategy
from wisent.core.contrastive_pairs.diagnostics.control_vectors import (
    detect_geometry_structure,
    GeometryAnalysisConfig,
)

from wisent.core.constants import (
    NORM_EPS, GEOMETRY_OPTIMIZATION_STEPS_SMALL,
    CV_FOLDS, KMEANS_N_INIT_SMALL, PAIR_GENERATORS_DEFAULT_N,
)
from wisent.examples.scripts._pair_generators_neutral import (
    ConceptMetrics,
    create_pairs_for_concept,
)


def compute_concept_direction(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> torch.Tensor:
    """Compute mean difference direction (CAA-style)."""
    pos_mean = pos_activations.mean(dim=0)
    neg_mean = neg_activations.mean(dim=0)
    direction = pos_mean - neg_mean
    return direction


def compute_cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    v1_norm = F.normalize(v1.unsqueeze(0), p=2, dim=1)
    v2_norm = F.normalize(v2.unsqueeze(0), p=2, dim=1)
    return float((v1_norm @ v2_norm.T).item())


def compute_linear_probe_accuracy(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_folds: int = CV_FOLDS,
) -> float:
    """Compute linear probe cross-validation accuracy."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        
        n_pos = len(pos_activations)
        n_neg = len(neg_activations)
        
        if n_pos < 3 or n_neg < 3:
            return 0.5
        
        X = torch.cat([pos_activations, neg_activations], dim=0).float().cpu().numpy()
        y = np.array([1] * n_pos + [0] * n_neg)
        
        n_folds = min(n_folds, min(n_pos, n_neg))
        if n_folds < 2:
            return 0.5
        
        clf = LogisticRegression( solver='lbfgs')
        scores = cross_val_score(clf, X, y, cv=n_folds, scoring='accuracy')
        return float(scores.mean())
    except Exception as e:
        print(f"  Warning: Linear probe failed: {e}")
        return 0.5


def compute_knn_accuracy(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    k: int = KMEANS_N_INIT_SMALL,
    n_folds: int = CV_FOLDS,
) -> float:
    """Compute k-NN cross-validation accuracy."""
    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score
        
        n_pos = len(pos_activations)
        n_neg = len(neg_activations)
        
        if n_pos < k + 1 or n_neg < k + 1:
            return 0.5
        
        X = torch.cat([pos_activations, neg_activations], dim=0).float().cpu().numpy()
        y = np.array([1] * n_pos + [0] * n_neg)
        
        n_folds = min(n_folds, min(n_pos, n_neg))
        if n_folds < 2:
            return 0.5
        
        clf = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(clf, X, y, cv=n_folds, scoring='accuracy')
        return float(scores.mean())
    except Exception as e:
        print(f"  Warning: k-NN failed: {e}")
        return 0.5


def compute_cohens_d(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> float:
    """Compute Cohen's d effect size along the mean difference direction."""
    direction = compute_concept_direction(pos_activations, neg_activations)
    direction_norm = F.normalize(direction.unsqueeze(0), p=2, dim=1).squeeze(0)
    
    pos_proj = (pos_activations @ direction_norm)
    neg_proj = (neg_activations @ direction_norm)
    
    pos_mean, pos_std = pos_proj.mean(), pos_proj.std()
    neg_mean, neg_std = neg_proj.mean(), neg_proj.std()
    
    pooled_std = ((pos_std**2 + neg_std**2) / 2).sqrt()
    cohens_d = abs(pos_mean - neg_mean) / (pooled_std + NORM_EPS)
    
    return float(cohens_d)


def analyze_concept(
    model: WisentModel,
    collector: ActivationCollector,
    concept_name: str,
    concept_data: Dict,
    layers_to_analyze: List[int],
    n_pairs: int = PAIR_GENERATORS_DEFAULT_N,
    strategy: ExtractionStrategy = ExtractionStrategy.CHAT_LAST,
) -> Tuple[Dict[int, ConceptMetrics], Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
    """Analyze a single concept across multiple layers."""
    
    print(f"\n{'='*60}")
    print(f"Analyzing concept: {concept_name}")
    print(f"Description: {concept_data['description']}")
    print(f"{'='*60}")
    
    pairs = create_pairs_for_concept(concept_data, n_pairs)
    print(f"Created {len(pairs)} contrastive pairs")
    
    # Collect activations for all pairs
    layer_names = [str(l) for l in layers_to_analyze]
    
    all_pos_activations = {l: [] for l in layers_to_analyze}
    all_neg_activations = {l: [] for l in layers_to_analyze}
    
    for i, pair in enumerate(pairs):
        print(f"  Processing pair {i+1}/{len(pairs)}...", end='\r')
        pair_with_acts = collector.collect(
            pair,
            strategy=strategy,
            layers=layer_names,
        )
        
        for layer in layers_to_analyze:
            layer_name = str(layer)
            pos_act = pair_with_acts.positive_response.layers_activations.to_dict().get(layer_name)
            neg_act = pair_with_acts.negative_response.layers_activations.to_dict().get(layer_name)
            
            if pos_act is not None and neg_act is not None:
                all_pos_activations[layer].append(pos_act)
                all_neg_activations[layer].append(neg_act)
    
    print(f"  Collected activations for {len(pairs)} pairs" + " " * 20)
    
    # Analyze each layer
    metrics_by_layer = {}
    activations_by_layer = {}
    
    for layer in layers_to_analyze:
        if not all_pos_activations[layer]:
            print(f"  Layer {layer}: No activations collected")
            continue
        
        pos_tensor = torch.stack(all_pos_activations[layer])
        neg_tensor = torch.stack(all_neg_activations[layer])
        
        activations_by_layer[layer] = (pos_tensor, neg_tensor)
        
        # Compute metrics
        linear_acc = compute_linear_probe_accuracy(pos_tensor, neg_tensor)
        knn_acc = compute_knn_accuracy(pos_tensor, neg_tensor, k=3)
        cohens_d = compute_cohens_d(pos_tensor, neg_tensor)
        
        direction = compute_concept_direction(pos_tensor, neg_tensor)
        direction_norm = float(direction.norm())
        
        # Signal detection (thresholds from Zwiad paper)
        has_signal = max(linear_acc, knn_acc) >= 0.6
        is_linear = has_signal and linear_acc >= knn_acc - 0.15
        
        # Geometry analysis
        try:
            config = GeometryAnalysisConfig(optimization_steps=GEOMETRY_OPTIMIZATION_STEPS_SMALL)
            geometry_result = detect_geometry_structure(pos_tensor, neg_tensor, config)
            best_structure = geometry_result.best_structure.value
            linear_score = geometry_result.all_scores.get("linear", type('', (), {'score': 0.0})()).score
            cone_score = geometry_result.all_scores.get("cone", type('', (), {'score': 0.0})()).score
        except Exception as e:
            print(f"  Layer {layer}: Geometry analysis failed: {e}")
            best_structure = "unknown"
            linear_score = 0.0
            cone_score = 0.0
        
        metrics = ConceptMetrics(
            concept=concept_name,
            layer=layer,
            linear_probe_accuracy=linear_acc,
            knn_accuracy=knn_acc,
            has_signal=has_signal,
            is_linear=is_linear,
            best_structure=best_structure,
            linear_score=linear_score,
            cone_score=cone_score,
            mean_direction_norm=direction_norm,
            cohens_d=cohens_d,
        )
        
        metrics_by_layer[layer] = metrics
        
        print(f"  Layer {layer:2d}: linear_acc={linear_acc:.3f}, knn={knn_acc:.3f}, "
              f"cohens_d={cohens_d:.2f}, structure={best_structure}")
    
    return metrics_by_layer, activations_by_layer


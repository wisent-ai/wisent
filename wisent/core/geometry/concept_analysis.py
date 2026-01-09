"""
Concept analysis for detecting and decomposing multiple concepts.

These functions analyze whether activations contain a single concept
or multiple interleaved concepts, and how to separate them.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


def detect_multiple_concepts(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    max_concepts: int = 5,
) -> Dict[str, Any]:
    """
    Detect if activations contain multiple concepts.
    
    Uses clustering on difference vectors to find distinct concepts.
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        n_pairs = min(len(pos_activations), len(neg_activations))
        if n_pairs < 10:
            return {"n_concepts": 1, "silhouette_scores": {}}
        
        diff_vectors = (pos_activations[:n_pairs] - neg_activations[:n_pairs]).float().cpu().numpy()
        
        # Normalize
        norms = np.linalg.norm(diff_vectors, axis=1, keepdims=True)
        valid_mask = norms.squeeze() > 1e-8
        diff_normalized = diff_vectors[valid_mask] / norms[valid_mask]
        
        if len(diff_normalized) < 10:
            return {"n_concepts": 1, "silhouette_scores": {}}
        
        silhouette_scores = {}
        for k in range(2, min(max_concepts + 1, len(diff_normalized) // 3)):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(diff_normalized)
            
            try:
                score = silhouette_score(diff_normalized, labels)
                silhouette_scores[k] = float(score)
            except:
                silhouette_scores[k] = 0.0
        
        if not silhouette_scores:
            return {"n_concepts": 1, "silhouette_scores": {}}
        
        # Best k is the one with highest silhouette
        best_k = max(silhouette_scores, key=silhouette_scores.get)
        best_score = silhouette_scores[best_k]
        
        # Only consider multiple concepts if silhouette > 0.2
        n_concepts = best_k if best_score > 0.2 else 1
        
        return {
            "n_concepts": n_concepts,
            "best_silhouette": best_score,
            "silhouette_scores": silhouette_scores,
            "is_multi_concept": n_concepts > 1,
        }
    except Exception:
        return {"n_concepts": 1, "silhouette_scores": {}}


def split_by_concepts(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_concepts: int = 2,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Split activations into separate concepts using clustering.
    
    Returns list of (pos, neg) pairs for each concept.
    """
    try:
        from sklearn.cluster import KMeans
        
        n_pairs = min(len(pos_activations), len(neg_activations))
        diff_vectors = (pos_activations[:n_pairs] - neg_activations[:n_pairs]).float().cpu().numpy()
        
        norms = np.linalg.norm(diff_vectors, axis=1, keepdims=True)
        valid_mask = norms.squeeze() > 1e-8
        diff_normalized = diff_vectors[valid_mask] / norms[valid_mask]
        
        km = KMeans(n_clusters=n_concepts, random_state=42, n_init=10)
        labels = km.fit_predict(diff_normalized)
        
        # Map back to original indices
        valid_indices = np.where(valid_mask)[0]
        
        result = []
        for c in range(n_concepts):
            concept_mask = labels == c
            original_indices = valid_indices[concept_mask]
            
            if len(original_indices) > 0:
                pos_c = pos_activations[original_indices]
                neg_c = neg_activations[original_indices]
                result.append((pos_c, neg_c))
        
        return result
    except Exception:
        return [(pos_activations, neg_activations)]


def analyze_concept_independence(
    concepts: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, Any]:
    """
    Analyze how independent different concepts are.
    
    Measures angle between concept directions.
    """
    try:
        if len(concepts) < 2:
            return {"n_concepts": len(concepts), "independence": 1.0}
        
        directions = []
        for pos, neg in concepts:
            n = min(len(pos), len(neg))
            if n < 2:
                continue
            diff_mean = (pos[:n] - neg[:n]).float().cpu().numpy().mean(axis=0)
            norm = np.linalg.norm(diff_mean)
            if norm > 1e-8:
                directions.append(diff_mean / norm)
        
        if len(directions) < 2:
            return {"n_concepts": len(concepts), "independence": 1.0}
        
        # Compute pairwise angles
        angles = []
        for i in range(len(directions)):
            for j in range(i + 1, len(directions)):
                cos_angle = np.dot(directions[i], directions[j])
                angle = np.degrees(np.arccos(np.clip(np.abs(cos_angle), 0, 1)))
                angles.append(angle)
        
        mean_angle = float(np.mean(angles))
        independence = mean_angle / 90.0  # Normalize: 90 degrees = fully independent
        
        return {
            "n_concepts": len(concepts),
            "mean_angle": mean_angle,
            "independence": float(np.clip(independence, 0, 1)),
            "angles": angles,
        }
    except Exception:
        return {"n_concepts": len(concepts), "independence": 0.5}


def compute_concept_coherence(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> float:
    """
    Compute how coherent the concept is (single direction vs. spread).
    
    High coherence = all pairs point same direction
    Low coherence = pairs point different directions
    """
    try:
        n_pairs = min(len(pos_activations), len(neg_activations))
        if n_pairs < 3:
            return 0.0
        
        diff_vectors = (pos_activations[:n_pairs] - neg_activations[:n_pairs]).float().cpu().numpy()
        
        norms = np.linalg.norm(diff_vectors, axis=1, keepdims=True)
        valid_mask = norms.squeeze() > 1e-8
        diff_normalized = diff_vectors[valid_mask] / norms[valid_mask]
        
        if len(diff_normalized) < 3:
            return 0.0
        
        # Coherence = how much variance is explained by first PC
        U, S, Vh = np.linalg.svd(diff_normalized, full_matrices=False)
        
        if len(S) == 0 or S.sum() == 0:
            return 0.0
        
        coherence = float((S[0] ** 2) / (S ** 2).sum())
        return coherence
    except Exception:
        return 0.0


def compute_concept_stability(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_bootstrap: int = 20,
) -> float:
    """
    Compute stability of concept direction across bootstrap samples.
    """
    try:
        from .direction_metrics import compute_direction_stability
        
        result = compute_direction_stability(
            pos_activations, neg_activations, n_bootstrap=n_bootstrap
        )
        return result.get("stability_score", 0.0)
    except Exception:
        return 0.0


def decompose_into_concepts(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_concepts: int = None,
) -> Dict[str, Any]:
    """
    Full concept decomposition: detect, split, and analyze.
    """
    detection = detect_multiple_concepts(pos_activations, neg_activations)
    
    if n_concepts is None:
        n_concepts = detection.get("n_concepts", 1)
    
    if n_concepts == 1:
        return {
            "n_concepts": 1,
            "concepts": [(pos_activations, neg_activations)],
            "independence": 1.0,
            "coherence": compute_concept_coherence(pos_activations, neg_activations),
        }
    
    concepts = split_by_concepts(pos_activations, neg_activations, n_concepts)
    independence = analyze_concept_independence(concepts)
    
    coherences = [compute_concept_coherence(p, n) for p, n in concepts]
    
    return {
        "n_concepts": len(concepts),
        "concepts": concepts,
        "independence": independence.get("independence", 0.5),
        "coherences": coherences,
        "mean_coherence": float(np.mean(coherences)) if coherences else 0.0,
    }


def find_mixed_pairs(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    threshold: float = 0.3,
) -> List[int]:
    """
    Find pairs that don't fit well into any cluster (mixed concepts).
    """
    try:
        from sklearn.cluster import KMeans
        
        n_pairs = min(len(pos_activations), len(neg_activations))
        diff_vectors = (pos_activations[:n_pairs] - neg_activations[:n_pairs]).float().cpu().numpy()
        
        norms = np.linalg.norm(diff_vectors, axis=1, keepdims=True)
        valid_mask = norms.squeeze() > 1e-8
        diff_normalized = diff_vectors[valid_mask] / norms[valid_mask]
        
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        km.fit(diff_normalized)
        
        distances = km.transform(diff_normalized).min(axis=1)
        median_dist = np.median(distances)
        
        mixed_mask = distances > median_dist * (1 + threshold)
        valid_indices = np.where(valid_mask)[0]
        
        return list(valid_indices[mixed_mask])
    except Exception:
        return []


def get_pure_concept_pairs(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    concept_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get pairs belonging to a specific concept cluster.
    """
    concepts = split_by_concepts(pos_activations, neg_activations, n_concepts=2)
    
    if concept_idx < len(concepts):
        return concepts[concept_idx]
    
    return pos_activations, neg_activations


def recommend_per_concept_steering(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> Dict[str, Any]:
    """
    Recommend whether to use single or per-concept steering.
    """
    decomposition = decompose_into_concepts(pos_activations, neg_activations)
    
    n_concepts = decomposition["n_concepts"]
    independence = decomposition["independence"]
    mean_coherence = decomposition.get("mean_coherence", 0.0)
    
    if n_concepts == 1:
        return {
            "recommendation": "single",
            "reason": "Single coherent concept detected",
            "n_concepts": 1,
        }
    elif independence > 0.7:
        return {
            "recommendation": "per_concept",
            "reason": f"{n_concepts} independent concepts detected",
            "n_concepts": n_concepts,
        }
    else:
        return {
            "recommendation": "multi_direction",
            "reason": f"{n_concepts} overlapping concepts (independence={independence:.2f})",
            "n_concepts": n_concepts,
        }

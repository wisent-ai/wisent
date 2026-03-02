"""Zwiad with automatic concept decomposition and naming."""
from typing import Dict, Any, Optional, List, Tuple, Set
from pathlib import Path
from dataclasses import asdict as _asdict
import json
import torch
import numpy as np
from wisent.core.utils.config_tools.constants import (
    DEFAULT_SCORE, DEFAULT_RANDOM_SEED, JSON_INDENT, VIZ_PCA_COMPONENTS,
    LINEARITY_MAX_PAIRS, MIN_CONCEPT_DIM,
)
from ..metrics.core.metrics_core import compute_geometry_metrics
from ..concepts import decompose_and_name_concepts_with_labels, find_optimal_layer_per_concept
from ..data.database_loaders import load_activations_from_database, load_pair_texts_from_database, load_available_layers_from_database

_GEO_KEYS = ["linear_accuracy", "nonlinear_accuracy", "gap", "diagnosis", "confidence",
    "p_value", "gap_ci_lower", "gap_ci_upper", "n_diagnostics_passed", "n_diagnostics_total",
    "t_statistic", "residual_silhouette", "residuals_cluster", "ramsey_improvement",
    "ramsey_significant", "diagnostics"]


def _save_checkpoint(results: Dict, output_path: Optional[str]) -> None:
    if not output_path:
        return
    def _ser(obj):
        if isinstance(obj, torch.Tensor): return obj.tolist()
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {str(k): _ser(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [_ser(v) for v in obj]
        if hasattr(obj, 'item'): return obj.item()
        return obj
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w') as f:
        json.dump(_ser(results), f, indent=JSON_INDENT)


def _load_checkpoint(output_path: Optional[str]) -> Dict:
    if not output_path:
        return {}
    p = Path(output_path)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    with open(p) as f:
        return json.load(f)

__all__ = [
    "run_zwiad_with_concept_naming", "extract_pair_texts_from_enriched_pairs",
    "load_activations_from_database", "load_pair_texts_from_database",
    "load_available_layers_from_database", "find_optimal_layer_per_concept",
]


def run_zwiad_with_concept_naming(
    activations_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    pair_texts: Optional[Dict[int, Dict[str, str]]] = None,
    generate_visualizations: bool = True,
    llm_model: str = "meta-llama/Llama-3.2-1B-Instruct",
    steps: str = "all", output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Zwiad with geometry metrics, 5-step protocol, and concept naming."""
    from .zwiad_protocol import (
        test_signal, test_geometry, test_decomposition, select_intervention,
        test_editability, SignalTestResult, GeometryTestResult,
        DecompositionTestResult, EditabilityTestResult,
    )
    import time as _time
    import sys as _sys
    def _log(msg): print(f"  [TRACE] {msg}", file=_sys.stderr, flush=True)
    existing = _load_checkpoint(output_path)
    _log(f"Checkpoint loaded: {list(existing.keys())}")
    if steps == "all":
        steps_to_run = {"signal", "geometry", "decomposition", "intervention", "editability"}
    else:
        steps_to_run = {s.strip().lower() for s in steps.split(",")}
    sorted_layers = sorted(activations_by_layer.keys())
    n_layers = len(sorted_layers)
    min_pairs = min(len(activations_by_layer[l][0]) for l in sorted_layers)
    pos_list = [activations_by_layer[l][0][:min_pairs] for l in sorted_layers]
    neg_list = [activations_by_layer[l][1][:min_pairs] for l in sorted_layers]
    pos_concat = torch.cat(pos_list, dim=1)
    neg_concat = torch.cat(neg_list, dim=1)
    n_pairs = len(pos_concat)
    from sklearn.decomposition import PCA as _PCA
    _pca_dims = min(n_pairs - 1, pos_concat.shape[1], VIZ_PCA_COMPONENTS)
    if _pca_dims < pos_concat.shape[1] and _pca_dims >= 2:
        _comb = torch.cat([pos_concat, neg_concat], dim=0).cpu().numpy()
        _comb_pca = _PCA(n_components=_pca_dims, random_state=DEFAULT_RANDOM_SEED).fit_transform(_comb)
        pos_pca = torch.tensor(_comb_pca[:n_pairs], dtype=pos_concat.dtype)
        neg_pca = torch.tensor(_comb_pca[n_pairs:], dtype=pos_concat.dtype)
    else:
        pos_pca, neg_pca = pos_concat, neg_concat
    if "metrics" in existing and existing["metrics"]:
        metrics = existing["metrics"]
    else:
        metrics = compute_geometry_metrics(pos_concat, neg_concat, generate_visualizations=generate_visualizations)
    results = {
        "n_pairs": n_pairs, "n_layers": n_layers, "layers_used": sorted_layers,
        "total_dims": pos_concat.shape[1], "steps_run": list(steps_to_run),
        "metrics": metrics, "recommended_method": metrics.get("recommended_method"),
        "recommendation_confidence": metrics.get("recommendation_confidence"),
    }
    _save_checkpoint(results, output_path)
    signal_result, geometry_result, decomposition_result, editability_result = None, None, None, None
    min_pairs_for_probes = MIN_CONCEPT_DIM
    # Step 1: Signal
    if "signal" in steps_to_run:
        if "signal_test" in existing:
            results["signal_test"] = existing["signal_test"]
            st = existing["signal_test"]
            if "max_z_score" in st:
                signal_result = SignalTestResult(
                    max_z_score=st["max_z_score"], min_p_value=st["min_p_value"],
                    passed=st["passed"], permutation_metrics=st.get("permutation_metrics", {}),
                    nonsense_metrics=st.get("nonsense_metrics"))
        elif n_pairs < min_pairs_for_probes:
            results["signal_test"] = {"status": "insufficient_data", "n_pairs": n_pairs}
        else:
            signal_result = test_signal(pos_pca, neg_pca, ["knn_accuracy", "knn_pca_accuracy", "mlp_probe_accuracy"])
            results["signal_test"] = {
                "max_z_score": signal_result.max_z_score, "min_p_value": signal_result.min_p_value,
                "passed": signal_result.passed, "permutation_metrics": signal_result.permutation_metrics,
                "nonsense_metrics": signal_result.nonsense_metrics}
        _save_checkpoint(results, output_path)
    # Step 2: Geometry
    if "geometry" in steps_to_run:
        if "geometry_test" in existing:
            results["geometry_test"] = existing["geometry_test"]
            gt = existing["geometry_test"]
            if "linear_accuracy" in gt:
                geometry_result = GeometryTestResult(**{k: gt.get(k, DEFAULT_SCORE) for k in _GEO_KEYS})
        elif n_pairs < min_pairs_for_probes:
            results["geometry_test"] = {"status": "insufficient_data", "n_pairs": n_pairs}
        else:
            _n_geo = min(n_pairs, LINEARITY_MAX_PAIRS)
            if n_pairs > LINEARITY_MAX_PAIRS:
                _idx = np.random.RandomState(DEFAULT_RANDOM_SEED).choice(n_pairs, LINEARITY_MAX_PAIRS, replace=False)
                _idx.sort()
                _pg, _ng = pos_pca[_idx], neg_pca[_idx]
            else:
                _pg, _ng = pos_pca, neg_pca
            geometry_result = test_geometry(_pg, _ng)
            results["geometry_test"] = {k: getattr(geometry_result, k) for k in _GEO_KEYS}
        _save_checkpoint(results, output_path)
    # Step 3: Decomposition + Concept Naming
    if "decomposition" in steps_to_run:
        if "decomposition_test" in existing:
            results["decomposition_test"] = existing["decomposition_test"]
            dt = existing["decomposition_test"]
            if "n_concepts" in dt and "status" not in dt:
                decomposition_result = DecompositionTestResult(
                    n_concepts=dt["n_concepts"], cluster_labels=dt.get("cluster_labels", [0]*n_pairs),
                    silhouette_score=dt.get("silhouette_score", DEFAULT_SCORE),
                    is_fragmented=dt.get("is_fragmented", False),
                    per_concept_sizes=dt.get("per_concept_sizes", {}))
            if "concept_decomposition" in existing:
                results["concept_decomposition"] = existing["concept_decomposition"]
        elif n_pairs < 3:
            results["decomposition_test"] = {"status": "insufficient_data", "n_pairs": n_pairs}
        else:
            decomposition_result = test_decomposition(pos_concat, neg_concat)
            results["decomposition_test"] = {
                "n_concepts": decomposition_result.n_concepts, "silhouette_score": decomposition_result.silhouette_score,
                "is_fragmented": decomposition_result.is_fragmented, "per_concept_sizes": decomposition_result.per_concept_sizes}
            decomp = decompose_and_name_concepts_with_labels(
                pos_concat, neg_concat, pair_texts, cluster_labels=decomposition_result.cluster_labels,
                n_concepts=decomposition_result.n_concepts,
                generate_visualizations=generate_visualizations, llm_model=llm_model)
            decomp["n_layers_used"] = n_layers
            labels_arr = np.array(decomposition_result.cluster_labels)
            pcl = find_optimal_layer_per_concept(activations_by_layer, labels_arr, decomposition_result.n_concepts)
            for concept in decomp.get("concepts", []):
                idx = concept["id"] - 1
                if idx in pcl:
                    concept["optimal_layer"] = pcl[idx]["best_layer"]
                    concept["optimal_layer_accuracy"] = pcl[idx]["best_accuracy"]
            decomp["per_concept_layers"] = pcl
            results["concept_decomposition"] = decomp
        _save_checkpoint(results, output_path)
    # Step 5: Editability
    if "editability" in steps_to_run:
        if "editability_test" in existing and existing["editability_test"]:
            results["editability_test"] = existing["editability_test"]
            et = existing["editability_test"]
            if "composite_editability" in et:
                editability_result = EditabilityTestResult(
                    composite_editability=et["composite_editability"],
                    steering_survival=et.get("steering_survival", DEFAULT_SCORE),
                    spectral_concentration=et.get("spectral_concentration", DEFAULT_SCORE),
                    spectral_sharpness=et.get("spectral_sharpness", DEFAULT_SCORE),
                    attention_entropy=et.get("attention_entropy", DEFAULT_SCORE),
                    jacobian_sensitivity=et.get("jacobian_sensitivity", DEFAULT_SCORE))
        elif n_pairs >= 2:
            editability_result = test_editability(pos_pca, neg_pca)
            results["editability_test"] = {
                "composite_editability": editability_result.composite_editability,
                "steering_survival": editability_result.steering_survival,
                "spectral_concentration": editability_result.spectral_concentration,
                "spectral_sharpness": editability_result.spectral_sharpness,
                "attention_entropy": editability_result.attention_entropy,
                "jacobian_sensitivity": editability_result.jacobian_sensitivity}
        _save_checkpoint(results, output_path)
    # Step 4: Intervention Selection
    if "intervention" in steps_to_run and signal_result and geometry_result and decomposition_result:
        intervention = select_intervention(signal_result, geometry_result, decomposition_result, metrics=metrics, editability=editability_result)
        results["intervention"] = {
            "recommended_method": intervention.recommended_method, "confidence": intervention.confidence,
            "reasoning": intervention.reasoning, "method_scores": intervention.method_scores}
        results["recommended_method"] = intervention.recommended_method
        results["recommendation_confidence"] = intervention.confidence
        _save_checkpoint(results, output_path)
    return results


def extract_pair_texts_from_enriched_pairs(enriched_pairs: List[Dict]) -> Dict[int, Dict[str, str]]:
    """Extract pair texts from enriched pairs format (wisent CLI output)."""
    pair_texts = {}
    for i, pair in enumerate(enriched_pairs):
        pair_texts[i] = {
            "prompt": pair.get("prompt", ""),
            "positive": pair.get("positive_response", {}).get("model_response", ""),
            "negative": pair.get("negative_response", {}).get("model_response", ""),
        }
    return pair_texts

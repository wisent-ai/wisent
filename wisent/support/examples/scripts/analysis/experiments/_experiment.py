"""Full experiment runner for mixed concept detection."""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans

from wisent.core.utils.config_tools.constants import (
    DEFAULT_RANDOM_SEED,
    SEPARATOR_WIDTH_WIDE, SEPARATOR_WIDTH_MEDIUM, SEPARATOR_WIDTH_WIDE_PLUS,
    JSON_INDENT,
)
from wisent.core.primitives.models.wisent_model import WisentModel

from ._data_loading import (
    load_truthfulqa_pairs,
    load_hellaswag_pairs,
    extract_difference_vectors,
)
from ._orchestration import detect_concepts


def run_experiment(
    model_name: str,
    n_pairs_per_concept: int,
    layer: int = None,
    seed: int = DEFAULT_RANDOM_SEED,
    output_dir: str = "/tmp/concept_detection",
    *, linearity_n_init: int, stability_n_clusters: int,
):
    """
    Run the full experiment: load pairs from both benchmarks,
    test MIXED (should detect multiple concepts),
    test PURE TruthfulQA (should detect single concept),
    and test PURE HellaSwag (should detect single concept).
    """
    print("=" * SEPARATOR_WIDTH_WIDE)
    print("MIXED CONCEPT DETECTION EXPERIMENT")
    print("=" * SEPARATOR_WIDTH_WIDE)

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
    print("\n" + "=" * SEPARATOR_WIDTH_WIDE)
    print("TEST 1: MIXED CONCEPTS (TruthfulQA + HellaSwag)")
    print("Expected: Should detect 2 distinct concepts")
    print("=" * SEPARATOR_WIDTH_WIDE)
    
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
    km = KMeans(n_clusters=stability_n_clusters, random_state=DEFAULT_RANDOM_SEED, n_init=linearity_n_init)
    cluster_labels = km.fit_predict(mixed_diffs)
    
    # Compute alignment with true sources
    from collections import Counter
    cluster_0_sources = [mixed_sources[i] for i in range(len(mixed_sources)) if cluster_labels[i] == 0]
    cluster_1_sources = [mixed_sources[i] for i in range(len(mixed_sources)) if cluster_labels[i] == 1]
    
    print(f"\n[VALIDATION - using hidden labels]")
    print(f"Cluster 0: {Counter(cluster_0_sources)}")
    print(f"Cluster 1: {Counter(cluster_1_sources)}")
    
    # ===== TEST 2: PURE TruthfulQA (should detect 1 concept) =====
    print("\n" + "=" * SEPARATOR_WIDTH_WIDE)
    print("TEST 2: PURE TruthfulQA")
    print("Expected: Should detect 1 concept")
    print("=" * SEPARATOR_WIDTH_WIDE)
    
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
    print("\n" + "=" * SEPARATOR_WIDTH_WIDE)
    print("TEST 3: PURE HellaSwag")
    print("Expected: Should detect 1 concept")
    print("=" * SEPARATOR_WIDTH_WIDE)
    
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
    print("\n" + "=" * SEPARATOR_WIDTH_WIDE)
    print("EXPERIMENT SUMMARY")
    print("=" * SEPARATOR_WIDTH_WIDE)
    
    print(f"\n{'Condition':<20} {'Detected':<10} {'Expected':<10} {'Match':<10}")
    print("-" * SEPARATOR_WIDTH_MEDIUM)
    
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
    print("-" * SEPARATOR_WIDTH_WIDE_PLUS)
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
        }, f, indent=JSON_INDENT)
    
    print(f"\nResults saved to: {output_file}")
    
    return results

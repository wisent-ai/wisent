#!/usr/bin/env python3
"""Full rigorous analysis of TruthfulQA data."""

import torch
from wisent.core.geometry.database_loaders import load_activations_from_database
from wisent.core.geometry.signal_null_tests import compute_signal_vs_null, compute_aggregate_signal
from wisent.core.geometry.is_linear import test_linearity
from wisent.core.geometry.effective_dim_null import compute_effective_dimensions_vs_null
from wisent.core.geometry.decomposition_metrics import find_optimal_clustering
from wisent.core.geometry.intervention_selection import rigorous_select_intervention


def main():
    # Load data
    pos, neg = load_activations_from_database(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        task_name="truthfulqa_custom",
        layer=8,
    )
    print(f"Loaded {len(pos)} pairs, dim={pos.shape[1]}")

    # Step 1: Signal test
    print("\n=== STEP 1: SIGNAL TEST ===")
    signal = compute_signal_vs_null(pos, neg, ["knn_accuracy", "mlp_probe_accuracy"])
    z, p, details = compute_aggregate_signal(signal)
    print(f"Aggregate z-score: {z:.2f}")
    print(f"p-value: {p:.4f}")
    print(f"Signal significant: {z > 2}")

    # Step 2: Linearity test
    print("\n=== STEP 2: LINEARITY TEST ===")
    lin = test_linearity(pos, neg)
    print(f"Linear accuracy: {lin.linear_accuracy:.3f}")
    print(f"Nonlinear accuracy: {lin.nonlinear_accuracy:.3f}")
    print(f"Gap: {lin.gap:.3f}")
    diagnosis = "LINEAR" if lin.is_linear else "NONLINEAR"
    print(f"Diagnosis: {diagnosis} (confidence={lin.confidence:.2f})")

    # Step 3: Effective dimension
    print("\n=== STEP 3: EFFECTIVE DIMENSION ===")
    eff = compute_effective_dimensions_vs_null(pos, neg)
    print(f"Diff effective rank: {eff['real']['effective_rank']:.1f}")
    print(f"Null effective rank: {eff['null']['effective_rank_null']:.1f}")
    eff_z = eff["z_scores"]["effective_rank_z"]
    print(f"Z-score: {eff_z:.2f}")
    print(f"Compression vs null: {eff['compression_vs_null']:.2f}x")

    # Step 4: Decomposition
    print("\n=== STEP 4: DECOMPOSITION ===")
    diff = pos - neg
    n_concepts, labels, sil = find_optimal_clustering(diff)
    print(f"Concepts found: {n_concepts}")
    print(f"Silhouette: {sil:.3f}")
    print(f"Multiple concepts supported: {n_concepts > 1}")

    # Final recommendation
    print("\n" + "=" * 50)
    print("FINAL VERDICT")
    print("=" * 50)
    geo_diag = "LINEAR" if lin.is_linear else "NONLINEAR"
    result = rigorous_select_intervention(
        signal_z=z,
        signal_p=p,
        geometry_diagnosis=geo_diag,
        geometry_confidence=lin.confidence,
        effective_dim_z=eff_z,
        n_concepts=n_concepts,
        silhouette=sil,
        geometry_type_z={},
    )
    print(f"Recommended method: {result.recommended_method}")
    print(f"Confidence: {result.confidence:.2f} [{result.confidence_lower:.2f}, {result.confidence_upper:.2f}]")
    print("Reasoning:")
    for r in result.reasoning:
        print(f"  - {r}")
    if result.warnings:
        print("Warnings:")
        for w in result.warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()

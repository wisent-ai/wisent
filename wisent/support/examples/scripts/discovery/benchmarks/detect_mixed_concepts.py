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

from wisent.core.utils.config_tools.constants import DEFAULT_RANDOM_SEED, N_BOOTSTRAP_DEFAULT, MAX_K_DEFAULT

from ._data_loading import (
    load_truthfulqa_pairs,
    load_hellaswag_pairs,
    get_activations,
    extract_difference_vectors,
)
from ._statistical_analysis import (
    compute_eigenvalue_analysis,
    compute_clustering_analysis,
    compute_direction_consistency,
    compute_cv_variance,
    hartigans_dip_test,
    compute_bimodality_analysis,
    compute_null_distribution,
)
from ._detection_single import detect_multiple_concepts_single_sample
from ._detection_k import detect_k_concepts
from ._layer_analysis import (
    get_activations_all_layers,
    extract_difference_vectors_all_layers,
    compute_projection,
    analyze_layer_separability,
)
from ._visualization import (
    visualize_multi_method,
    visualize_layer_analysis,
    attribute_pairs_to_concepts,
    print_concept_attribution,
)
from ._visualization_advanced import (
    visualize_k_concepts,
    visualize_concept_detection,
)
from ._orchestration import (
    ConceptDetectionResult,
    detect_concepts,
    run_single_sample_detection,
)
from ._experiment import run_experiment
from ._cli_modes import (
    run_analyze_layers_mode,
    run_attribute_mode,
    run_detect_k_mode,
)
from ._cli_modes_vis import (
    run_visualize_mode,
    run_single_sample_test_mode,
)


# Re-export all public functions for backward compatibility
__all__ = [
    "ConceptDetectionResult",
    "load_truthfulqa_pairs",
    "load_hellaswag_pairs",
    "get_activations",
    "extract_difference_vectors",
    "compute_eigenvalue_analysis",
    "compute_clustering_analysis",
    "compute_direction_consistency",
    "compute_cv_variance",
    "hartigans_dip_test",
    "compute_bimodality_analysis",
    "compute_null_distribution",
    "detect_multiple_concepts_single_sample",
    "detect_k_concepts",
    "get_activations_all_layers",
    "extract_difference_vectors_all_layers",
    "compute_projection",
    "analyze_layer_separability",
    "visualize_multi_method",
    "visualize_layer_analysis",
    "attribute_pairs_to_concepts",
    "print_concept_attribution",
    "visualize_k_concepts",
    "visualize_concept_detection",
    "detect_concepts",
    "run_experiment",
    "run_single_sample_detection",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect multiple concepts in mixed samples")
    parser.add_argument("--model", type=str, required=True,
                        help="Model to use")
    parser.add_argument("--n-pairs", type=int, default=N_BOOTSTRAP_DEFAULT,
                        help="Number of pairs per concept")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer to extract activations from (default: middle)")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--single-sample-test", action="store_true",
                        help="Run single-sample detection test (tests if we can detect mixed vs pure)")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT,
                        help="Number of bootstrap samples for null distribution")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations")
    parser.add_argument("--vis-output-dir", type=str, required=True,
                        help="Directory for visualization outputs")
    parser.add_argument("--detect-k", action="store_true",
                        help="Run k-concept detection (find how many concepts exist)")
    parser.add_argument("--max-k", type=int, default=MAX_K_DEFAULT,
                        help="Maximum k to try for concept detection")
    parser.add_argument("--attribute", action="store_true",
                        help="Run attribution to trace pairs back to detected concepts")
    parser.add_argument("--analyze-layers", action="store_true",
                        help="Analyze separability across all layers")
    parser.add_argument("--projection-method", type=str, required=True,
                        choices=["pca", "umap", "pacmap", "all"],
                        help="Projection method for visualization")
    
    args = parser.parse_args()

    if args.analyze_layers:
        run_analyze_layers_mode(args)
    elif args.attribute:
        run_attribute_mode(args)
    elif args.detect_k:
        run_detect_k_mode(args)
    elif args.visualize:
        run_visualize_mode(args)
    elif args.single_sample_test:
        run_single_sample_test_mode(args)
    else:
        run_experiment(
            model_name=args.model,
            n_pairs_per_concept=args.n_pairs,
            layer=args.layer,
            seed=args.seed,
            output_dir=args.output_dir,
        )

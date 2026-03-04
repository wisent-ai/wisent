"""
Concept Evolution Analysis: Hitler -> Fascism -> Harmful Ideology

Analyzes how representations evolve from specific (Hitler) to abstract (fascism)
to general (harmful ideology) concepts in LLM activation space.

Measures:
1. Cosine similarity between concept directions
2. Intrinsic dimensionality of each concept manifold
3. Cone score (whether multiple correlated directions encode the concept)
4. Linear vs nonlinear separability
5. Layer-wise evolution of representations

Usage:
    python -m wisent.examples.scripts.concept_evolution_analysis --model Qwen/Qwen3-4B
"""

import argparse

from wisent.examples.scripts._comparison import run_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Analyze how representations evolve from Hitler -> Fascism -> Harmful ideology"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model to analyze"
    )
    parser.add_argument(
        "--layers", type=str, default=None,
        help="Comma-separated list of layers to analyze (default: auto-select)"
    )
    parser.add_argument(
        "--n-pairs", type=int, required=True,
        help="Number of contrastive pairs per concept"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--device", type=str, required=True,
        help="Device to run on (cuda, mps, cpu)"
    )
    parser.add_argument(
        "--knn-k", type=int, required=True,
        help="Number of neighbors for k-NN classifier"
    )
    parser.add_argument("--cv-folds", type=int, required=True)
    parser.add_argument("--signal-exist-threshold", type=float, required=True)
    parser.add_argument("--signal-linear-gap", type=float, required=True)
    parser.add_argument("--min-cloud-points", type=int, required=True)
    parser.add_argument("--geometry-optimization-steps", type=int, required=True)

    args = parser.parse_args()

    layers = None
    if args.layers:
        layers = [int(l.strip()) for l in args.layers.split(",")]

    run_analysis(
        model_name=args.model,
        n_pairs=args.n_pairs,
        knn_k=args.knn_k,
        cv_folds=args.cv_folds,
        signal_exist_threshold=args.signal_exist_threshold,
        signal_linear_gap=args.signal_linear_gap,
        min_cloud_points=args.min_cloud_points,
        geometry_optimization_steps=args.geometry_optimization_steps,
        layers_to_analyze=layers,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()

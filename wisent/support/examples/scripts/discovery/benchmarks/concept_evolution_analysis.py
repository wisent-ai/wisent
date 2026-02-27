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

from wisent.core.constants import PAIR_GENERATORS_DEFAULT_N
from wisent.examples.scripts._comparison import run_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Analyze how representations evolve from Hitler -> Fascism -> Harmful ideology"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-4B",
        help="Model to analyze (default: Qwen/Qwen3-4B)"
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
        "--output-dir", type=str, default="/tmp/concept_evolution",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on (cuda, mps, cpu)"
    )

    args = parser.parse_args()

    layers = None
    if args.layers:
        layers = [int(l.strip()) for l in args.layers.split(",")]

    run_analysis(
        model_name=args.model,
        layers_to_analyze=layers,
        n_pairs=args.n_pairs,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()

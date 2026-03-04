"""Parser for cluster-benchmarks command."""

import argparse



def setup_cluster_benchmarks_parser(parser: argparse.ArgumentParser) -> None:
    """Set up arguments for the cluster-benchmarks command."""
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path (e.g., meta-llama/Llama-3.2-1B-Instruct)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--pairs-per-benchmark",
        type=int,
        required=True,
        help="Number of contrastive pairs per benchmark"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/mps/cpu). Auto-detected if not specified."
    )
    parser.add_argument(
        "--cluster-progress-interval", type=int, required=True,
        help="Log progress every N benchmarks during loading"
    )
    parser.add_argument(
        "--cluster-min-pairs", type=int, required=True,
        help="Minimum number of pairs required per benchmark"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        required=True,
        help="Fraction of data for training vs evaluation"
    )

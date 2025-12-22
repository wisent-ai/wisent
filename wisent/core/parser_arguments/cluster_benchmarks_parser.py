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
        default="./cluster_output",
        help="Output directory for results (default: ./cluster_output)"
    )
    parser.add_argument(
        "--pairs-per-benchmark",
        type=int,
        default=50,
        help="Number of contrastive pairs per benchmark (default: 50)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/mps/cpu). Auto-detected if not specified."
    )

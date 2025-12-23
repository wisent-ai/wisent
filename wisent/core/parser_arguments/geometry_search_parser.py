"""Parser for geometry-search command."""

import argparse


def setup_geometry_search_parser(parser: argparse.ArgumentParser) -> None:
    """Set up the geometry-search command parser."""
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path (e.g., meta-llama/Llama-3.2-1B-Instruct)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/ubuntu/output/geometry_results.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--pairs-per-benchmark",
        type=int,
        default=50,
        help="Number of pairs to sample per benchmark (default: 50)",
    )
    parser.add_argument(
        "--max-layer-combo-size",
        type=int,
        default=3,
        help="Maximum layers in combination (default: 3 = individual + pairs + triplets)",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Comma-separated list of strategies (default: all 7)",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=None,
        help="Comma-separated list of benchmarks, or path to .txt file (default: all)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for activation cache (default: /tmp/wisent_geometry_cache_<model>)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for model (auto/cuda/mps/cpu, default: auto)",
    )

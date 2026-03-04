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
        required=True,
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--pairs-per-benchmark",
        type=int,
        required=True,
        help="Number of pairs to sample per benchmark",
    )
    parser.add_argument(
        "--max-layer-combo-size",
        type=int,
        required=True,
        help="Maximum layers in combination (e.g., 3 = individual + pairs + triplets)",
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
        default=None,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Device for model (auto/cuda/mps/cpu)",
    )
    parser.add_argument(
        "--report-interval",
        type=int,
        required=True,
        help="How often to report progress during activation collection (in pairs)",
    )
    parser.add_argument(
        "--estimated-time-per-extraction",
        type=float,
        required=True,
        help="Estimated seconds per extraction operation",
    )
    parser.add_argument(
        "--geometry-minority-pct",
        type=int,
        required=True,
        help="Percentage threshold for geometry minority structure detection",
    )

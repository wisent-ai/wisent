"""Parser for the 'discover-steering' command - find optimal steering directions."""

from wisent.core.utils.config_tools.constants import COMPARISON_STEERING_LAYER, DEFAULT_STRENGTH


def setup_discover_steering_parser(parser):
    """Set up the discover-steering command parser."""
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model name (default: meta-llama/Llama-3.2-1B-Instruct)"
    )
    parser.add_argument(
        "--task", type=str, required=True,
        help="Task/benchmark name in database (e.g., truthfulqa_custom)"
    )
    parser.add_argument(
        "--layer", type=int, default=COMPARISON_STEERING_LAYER,
        help="Primary layer to test (default: 12)"
    )
    parser.add_argument(
        "--layer-range", type=str, default=None,
        help="Layer range to search, e.g., '8-16' (default: None, only test --layer)"
    )
    parser.add_argument(
        "--strength", type=float, default=DEFAULT_STRENGTH,
        help="Steering strength multiplier (default: 1.0)"
    )
    parser.add_argument(
        "--n-test-samples", type=int, required=True,
        help="Number of test samples for evaluation"
    )
    parser.add_argument(
        "--n-random-directions", type=int, required=True,
        help="Number of random directions to try"
    )
    parser.add_argument(
        "--database-url", type=str, default=None,
        help="Database URL (default: DATABASE_URL env var)"
    )
    parser.add_argument(
        "--output", type=str, default="./steering_discovery_results.json",
        help="Output JSON file path (default: ./steering_discovery_results.json)"
    )
    parser.add_argument(
        "--skip-layer-search", action="store_true",
        help="Skip the layer search (faster but less thorough)"
    )
    parser.add_argument(
        "--skip-direction-search", action="store_true",
        help="Skip the direction search (faster but less thorough)"
    )

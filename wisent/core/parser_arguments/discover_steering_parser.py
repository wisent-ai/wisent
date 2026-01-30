"""Parser for the 'discover-steering' command - find optimal steering directions."""


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
        "--layer", type=int, default=12,
        help="Primary layer to test (default: 12)"
    )
    parser.add_argument(
        "--layer-range", type=str, default=None,
        help="Layer range to search, e.g., '8-16' (default: None, only test --layer)"
    )
    parser.add_argument(
        "--strength", type=float, default=1.0,
        help="Steering strength multiplier (default: 1.0)"
    )
    parser.add_argument(
        "--n-test-samples", type=int, default=50,
        help="Number of test samples for evaluation (default: 50)"
    )
    parser.add_argument(
        "--n-random-directions", type=int, default=20,
        help="Number of random directions to try (default: 20)"
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

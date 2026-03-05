"""Parser for the 'discover-steering' command - find optimal steering directions."""

def setup_discover_steering_parser(parser):
    """Set up the discover-steering command parser."""
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model name"
    )
    parser.add_argument(
        "--task", type=str, required=True,
        help="Task/benchmark name in database (e.g., truthfulqa_custom)"
    )
    parser.add_argument(
        "--layer", type=int, required=True,
        help="Primary layer to test"
    )
    parser.add_argument(
        "--layer-range", type=str, default=None,
        help="Layer range to search, e.g., '8-16' (default: None, only test --layer)"
    )
    parser.add_argument(
        "--strength", type=float, required=True,
        help="Steering strength multiplier"
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
        "--output", type=str, required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--skip-layer-search", action="store_true",
        help="Skip the layer search (faster but less thorough)"
    )
    parser.add_argument(
        "--skip-direction-search", action="store_true",
        help="Skip the direction search (faster but less thorough)"
    )
    parser.add_argument(
        "--discover-task-limit", type=int, required=True,
        help="Maximum number of task pairs to load for discovery"
    )
    parser.add_argument(
        "--discover-train-limit", type=int, required=True,
        help="Maximum number of activation pairs to load for training"
    )

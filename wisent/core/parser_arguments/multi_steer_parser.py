"""Parser setup for the 'multi-steer' command."""


def setup_multi_steer_parser(parser):
    """Set up the multi-steer subcommand parser for dynamic vector combination."""
    # Vector inputs - can specify multiple vector-weight pairs
    parser.add_argument(
        "--vector",
        type=str,
        action="append",
        required=False,
        default=None,
        metavar="PATH:WEIGHT",
        help="Path to steering vector and its weight (format: path/to/vector.pt:0.5). Can be specified multiple times. If omitted, generates unsteered baseline.",
    )

    # Model configuration
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--layer", type=int, required=False, default=None, help="Layer index to apply combined steering (required when using vectors)")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (default: auto-detect)")

    # Steering method configuration
    parser.add_argument(
        "--method",
        type=str,
        default="CAA",
        choices=["CAA"],
        help="Steering method to use for combination (default: CAA)",
    )

    # Generation configuration
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to generate with combined steering")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum new tokens to generate (default: 100)")

    # Weight normalization
    parser.add_argument("--normalize-weights", action="store_true", help="Normalize weights to sum to 1.0")
    parser.add_argument(
        "--allow-unnormalized", action="store_true", help="Allow weights that don't sum to 1.0 (for stronger effects)"
    )
    parser.add_argument(
        "--target-norm", type=float, default=None, help="Scale the combined vector to have this norm (e.g., 10.0)"
    )

    # Output options
    parser.add_argument(
        "--save-combined", type=str, default=None, help="Save the combined steering vector to this path"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output showing weight calculations")

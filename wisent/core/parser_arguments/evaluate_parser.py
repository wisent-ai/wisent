"""Parser setup for the 'evaluate' command."""


def setup_evaluate_parser(parser):
    """Set up the evaluate subcommand parser for single-prompt evaluation."""

    # Required arguments
    parser.add_argument("--vector", type=str, required=True, help="Path to steering vector file (.pt)")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to evaluate")
    parser.add_argument(
        "--model", type=str, required=True, help="Model name or path (used for both generation and evaluation)"
    )
    parser.add_argument("--trait", type=str, required=True, help="Trait name (e.g., 'catholic', 'cynical')")

    # Optional model configuration
    parser.add_argument("--device", type=str, default=None, help="Device to run on (default: auto-detect)")

    # Optional steering parameters
    parser.add_argument(
        "--steering-strength", type=float, default=2.0, help="Steering strength to apply (default: 2.0)"
    )
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum new tokens to generate (default: 100)")
    parser.add_argument(
        "--trait-description",
        type=str,
        default=None,
        help="Optional description of the trait (default: use trait name)",
    )

    # Optional threshold parameters
    parser.add_argument(
        "--trait-threshold", type=float, default=None, help="Minimum trait quality threshold (-1 to 1 scale)"
    )
    parser.add_argument(
        "--answer-threshold", type=float, default=None, help="Minimum answer quality threshold (0 to 1 scale)"
    )

    # Output options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

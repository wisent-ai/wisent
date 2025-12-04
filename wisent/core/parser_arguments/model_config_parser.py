"""Parser setup for the 'model-config' command."""


def setup_model_config_parser(parser):
    """Set up the model-config subcommand parser."""
    # Create subparsers for different model config actions
    config_subparsers = parser.add_subparsers(dest="config_action", help="Model configuration actions")

    # Save configuration subcommand
    save_parser = config_subparsers.add_parser("save", help="Save optimal parameters for a model")
    save_parser.add_argument("model", type=str, help="Model name or path")
    save_parser.add_argument("--classification-layer", type=int, required=True, help="Optimal layer for classification")
    save_parser.add_argument(
        "--steering-layer", type=int, default=None, help="Optimal layer for steering (defaults to classification layer)"
    )
    save_parser.add_argument(
        "--token-aggregation",
        type=str,
        default="average",
        choices=["average", "final", "first", "max", "min", "max_score"],
        help="Token aggregation method. 'max_score' uses highest token hallucination score.",
    )
    save_parser.add_argument("--detection-threshold", type=float, default=0.6, help="Detection threshold")
    save_parser.add_argument(
        "--optimization-method", type=str, default="manual", help="How these parameters were determined"
    )
    save_parser.add_argument("--metrics", type=str, default=None, help="JSON string with optimization metrics")

    # List configurations subcommand
    list_parser = config_subparsers.add_parser("list", help="List all saved model configurations")
    list_parser.add_argument("--detailed", action="store_true", help="Show detailed configuration information")

    # Show configuration subcommand
    show_parser = config_subparsers.add_parser("show", help="Show configuration for a specific model")
    show_parser.add_argument("model", type=str, help="Model name or path")
    show_parser.add_argument("--task", type=str, default=None, help="Show task-specific overrides if available")

    # Remove configuration subcommand
    remove_parser = config_subparsers.add_parser("remove", help="Remove configuration for a model")
    remove_parser.add_argument("model", type=str, help="Model name or path")
    remove_parser.add_argument("--confirm", action="store_true", help="Confirm removal without prompting")

    # Test configuration subcommand
    test_parser = config_subparsers.add_parser("test", help="Test if saved configuration works")
    test_parser.add_argument("model", type=str, help="Model name or path")
    test_parser.add_argument(
        "--task", type=str, default="truthfulqa_mc1", help="Task to test with (default: truthfulqa_mc1)"
    )
    test_parser.add_argument("--limit", type=int, default=5, help="Number of samples to test with (default: 5)")
    test_parser.add_argument("--device", type=str, default=None, help="Device to run on")

    # Common arguments for all subcommands
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Custom directory for configuration files (default: ~/.wisent/model_configs/)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

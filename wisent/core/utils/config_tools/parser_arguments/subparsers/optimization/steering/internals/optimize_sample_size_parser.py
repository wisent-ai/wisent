"""Parser setup for the 'optimize-sample-size' command."""



def setup_sample_size_optimizer_parser(parser):
    """Set up the sample-size-optimizer subcommand parser."""
    parser.add_argument("model", type=str, help="Model name or path to optimize")
    parser.add_argument("--task", type=str, required=True, help="Task to optimize for (REQUIRED)")
    parser.add_argument("--layer", type=int, required=True, help="Layer index to use (REQUIRED)")
    # Classification-specific arguments
    parser.add_argument(
        "--threshold", type=float, default=None, help="Detection threshold for classification (required)"
    )

    # Steering mode
    parser.add_argument("--steering-mode", action="store_true", help="Optimize for steering instead of classification")
    parser.add_argument(
        "--steering-method",
        type=str,
        required=True,
        choices=["CAA"],
        help="Steering method to use",
    )
    parser.add_argument("--steering-strength", type=float, required=True, help="Steering strength to use")
    parser.add_argument(
        "--token-targeting-strategy",
        type=str,
        required=True,
        choices=["CHOICE_TOKEN", "LAST_TOKEN", "FIRST_TOKEN", "ALL_TOKENS"],
        help="Token targeting strategy for steering",
    )

    # Common optimization parameters
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        required=True,
        help="Sample sizes to test",
    )
    parser.add_argument("--test-size", type=int, required=True, help="Fixed test set size")
    parser.add_argument("--test-split", type=float, required=True, help="DEPRECATED: Use --test-size instead")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples to load from dataset")
    parser.add_argument("--save-plot", action="store_true", help="Save performance plot")
    parser.add_argument("--no-save-config", action="store_true", help="Don't save optimal sample size to model config")
    parser.add_argument("--device", type=str, default=None, help="Device to run on")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--force", action="store_true", help="Force optimization even without matching classifier parameters"
    )
    parser.add_argument(
        "--classifier-test-size",
        type=float,
        required=True,
        help="Fraction of data to hold out for testing in classifier training",
    )
    parser.add_argument("--classifier-epochs", type=int, required=True, help="Epochs for classifier training")
    parser.add_argument("--classifier-batch-size", type=int, required=True, help="Batch size for classifier training")
    parser.add_argument("--classifier-lr", type=float, required=True, help="Learning rate for classifier training")
    parser.add_argument(
        "--sample-loading-buffer", type=int, required=True,
        help="Extra samples to load beyond max_train + test_size to account for filtering"
    )

"""Parser setup for the 'optimize-sample-size' command."""

from wisent.core import constants as _C
from wisent.core.constants import DATA_SPLIT_SEED, CLASSIFIER_THRESHOLD


def setup_sample_size_optimizer_parser(parser):
    """Set up the sample-size-optimizer subcommand parser."""
    parser.add_argument("model", type=str, help="Model name or path to optimize")
    parser.add_argument("--task", type=str, required=True, help="Task to optimize for (REQUIRED)")
    parser.add_argument("--layer", type=int, required=True, help="Layer index to use (REQUIRED)")
    # Classification-specific arguments
    parser.add_argument(
        "--threshold", type=float, default=CLASSIFIER_THRESHOLD, help="Detection threshold for classification (default: 0.5)"
    )

    # Steering mode
    parser.add_argument("--steering-mode", action="store_true", help="Optimize for steering instead of classification")
    parser.add_argument(
        "--steering-method",
        type=str,
        default="CAA",
        choices=["CAA"],
        help="Steering method to use (default: CAA)",
    )
    parser.add_argument("--steering-strength", type=float, default=_C.DEFAULT_BASE_STRENGTH, help="Steering strength to use (default: 1.0)")
    parser.add_argument(
        "--token-targeting-strategy",
        type=str,
        default="LAST_TOKEN",
        choices=["CHOICE_TOKEN", "LAST_TOKEN", "FIRST_TOKEN", "ALL_TOKENS"],
        help="Token targeting strategy for steering (default: LAST_TOKEN)",
    )

    # Common optimization parameters
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=list(_C.PARSER_FULL_OPTIMIZE_SAMPLE_SIZES),
        help="Sample sizes to test (default: 5 10 20 50 100 200 500)",
    )
    parser.add_argument("--test-size", type=int, default=_C.OPTIMIZE_SAMPLE_TEST_SIZE, help="Fixed test set size (default: 200)")
    parser.add_argument("--test-split", type=float, default=_C.OPTIMIZE_SAMPLE_TEST_SPLIT, help="DEPRECATED: Use --test-size instead")
    parser.add_argument("--seed", type=int, default=DATA_SPLIT_SEED, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples to load from dataset")
    parser.add_argument("--save-plot", action="store_true", help="Save performance plot")
    parser.add_argument("--no-save-config", action="store_true", help="Don't save optimal sample size to model config")
    parser.add_argument("--device", type=str, default=None, help="Device to run on")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--force", action="store_true", help="Force optimization even without matching classifier parameters"
    )

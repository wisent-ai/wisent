"""Parser setup for the 'full-optimize' command."""


def setup_full_optimizer_parser(parser):
    """Set up the full-optimize subcommand parser."""
    parser.add_argument("model", type=str, help="Model name or path to optimize")

    # Task selection - mutually exclusive options
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument("--tasks", type=str, nargs="+", help="Specific tasks to optimize")
    task_group.add_argument(
        "--skills", type=str, nargs="+", help="Select tasks by skill categories (e.g., coding, mathematics, reasoning)"
    )
    task_group.add_argument(
        "--risks",
        type=str,
        nargs="+",
        help="Select tasks by risk categories (e.g., harmfulness, toxicity, hallucination)",
    )

    # General limit that applies to all optimizations unless overridden
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Sample limit for all optimizations (default: 100). Can be overridden by specific limits below",
    )

    # Specific limits (override general limit if provided)
    parser.add_argument(
        "--classification-limit",
        type=int,
        default=None,
        help="Sample limit for classification optimization (overrides --limit)",
    )
    parser.add_argument(
        "--sample-size-limit",
        type=int,
        default=None,
        help="Sample limit for sample size optimization (overrides --limit)",
    )
    parser.add_argument(
        "--steering-limit", type=int, default=None, help="Sample limit for steering optimization (overrides --limit)"
    )

    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50, 100, 200, 500],
        help="Sample sizes to test (default: 5 10 20 50 100 200 500)",
    )
    parser.add_argument(
        "--skip-classification", action="store_true", help="Skip classification optimization and use existing config"
    )
    parser.add_argument("--skip-sample-size", action="store_true", help="Skip sample size optimization")
    parser.add_argument("--skip-classifier-training", action="store_true", help="Skip final classifier training step")
    parser.add_argument("--skip-control-vectors", action="store_true", help="Skip control vector training step")

    # Steering optimization options
    parser.add_argument("--skip-steering", action="store_true", help="Skip steering optimization")
    parser.add_argument(
        "--steering-methods",
        type=str,
        nargs="+",
        choices=["CAA", "HPR", "DAC", "BiPO", "KSteering"],
        default=["CAA", "HPR", "DAC", "BiPO", "KSteering"],
        help="Steering methods to test (default: all methods with parameter variations)",
    )
    parser.add_argument(
        "--steering-layer-range", type=str, default=None, help="Layer range for steering optimization (e.g., '0-5')"
    )
    parser.add_argument(
        "--steering-strength-range",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0],
        help="Steering strengths to test (default: 0.5 1.0 1.5 2.0)",
    )
    # Task selection options
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Number of tasks to randomly select from matched tasks (default: all)",
    )
    parser.add_argument(
        "--min-quality-score",
        type=int,
        default=2,
        choices=[1, 2, 3, 4, 5],
        help="Minimum quality score for tasks (default: 2)",
    )
    parser.add_argument(
        "--task-seed", type=int, default=None, help="Random seed for task selection (for reproducibility)"
    )

    parser.add_argument(
        "--max-time-per-task", type=float, default=20.0, help="Maximum time per task in minutes (default: 20.0)"
    )

    parser.add_argument("--device", type=str, default=None, help="Device to run on")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--save-plots", action="store_true", help="Save plots for both optimizations")

    # Timing calibration options
    parser.add_argument(
        "--skip-timing-estimation", action="store_true", help="Skip timing estimation and proceed without time warnings"
    )
    parser.add_argument("--calibration-file", type=str, default=None, help="File to save/load calibration data")
    parser.add_argument(
        "--calibrate-only",
        action="store_true",
        help="Only run calibration and exit (saves to --calibration-file if provided)",
    )

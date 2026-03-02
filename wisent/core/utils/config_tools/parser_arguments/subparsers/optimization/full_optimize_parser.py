"""Parser setup for the 'optimize' and 'optimize-all' commands."""

from wisent.core.utils.config_tools.constants import SEARCH_DEFAULT_STRENGTHS


def setup_optimize_all_parser(parser):
    """Set up the optimize/optimize-all subcommand parser."""
    parser.add_argument("model", type=str, help="Model name or path to optimize")

    # Task/benchmark selection
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        help="Benchmark tasks to optimize (e.g., hellaswag gsm8k mmlu)"
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        help="Alias for --tasks: benchmark tasks to optimize"
    )

    # Trait selection
    parser.add_argument(
        "--traits",
        type=str,
        nargs="+",
        help="Behavioral traits to optimize (e.g., coding honesty refusal helpfulness)"
    )
    
    # Skip trait categories
    parser.add_argument(
        "--skip-personalization",
        action="store_true",
        help="Skip personalization trait optimization"
    )
    parser.add_argument(
        "--skip-safety",
        action="store_true",
        help="Skip safety trait optimization"
    )
    parser.add_argument(
        "--skip-humanization",
        action="store_true",
        help="Skip humanization trait optimization"
    )
    
    # Steering methods
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        required=True,
        help="Steering methods to use"
    )
    
    # Resume/force options
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint if available (default: True)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-optimization even if cached results exist"
    )
    
    # Search strategy
    parser.add_argument(
        "--search-strategy",
        type=str,
        choices=["grid", "optuna"],
        required=True,
        help="Search strategy: 'grid' for exhaustive search, 'optuna' for TPE sampling"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        required=True,
        help="Number of Optuna trials for optimization"
    )

    # General limit that applies to all optimizations unless overridden
    parser.add_argument(
        "--limit",
        type=int,
        required=True,
        help="Sample limit for all optimizations. Can be overridden by specific limits below",
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
        required=True,
        help="Sample sizes to test (e.g., 5 10 20 50 100 200 500)",
    )
    parser.add_argument(
        "--skip-classification", action="store_true", help="Skip classification optimization and use existing config"
    )
    parser.add_argument("--skip-sample-size", action="store_true", help="Skip sample size optimization")
    parser.add_argument("--skip-classifier-training", action="store_true", help="Skip final classifier training step")
    parser.add_argument("--skip-control-vectors", action="store_true", help="Skip control vector training step")

    # Steering optimization options
    parser.add_argument("--skip-steering", action="store_true", help="Skip steering optimization")

    # Weight modification optimization options
    parser.add_argument("--skip-weights", action="store_true", help="Skip weight modification optimization")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for modified models"
    )
    parser.add_argument(
        "--steering-methods",
        type=str,
        nargs="+",
        choices=["CAA"],
        required=True,
        help="Steering methods to test",
    )
    parser.add_argument(
        "--steering-layer-range", type=str, default=None, help="Layer range for steering optimization (e.g., '0-5')"
    )
    parser.add_argument(
        "--steering-strength-range",
        type=float,
        nargs="+",
        default=list(SEARCH_DEFAULT_STRENGTHS),
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
        required=True,
        choices=[1, 2, 3, 4, 5],
        help="Minimum quality score for tasks",
    )
    parser.add_argument(
        "--task-seed", type=int, default=None, help="Random seed for task selection (for reproducibility)"
    )

    parser.add_argument(
        "--max-time-per-task", type=float, required=True, help="Maximum time per task in minutes"
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

"""Method-specific subparsers for optimize-steering."""
from wisent.core import constants as _C
from wisent.core.steering_methods.registry import SteeringMethodRegistry
AVAILABLE_METHODS = [m.upper() for m in SteeringMethodRegistry.list_methods()]


def setup_method_parsers(steering_subparsers):
    """Set up hierarchical, method, layer, strength, auto subparsers."""
    # Hierarchical optimization subcommand (RECOMMENDED for guarantees)
    hierarchical_parser = steering_subparsers.add_parser(
        "hierarchical",
        help="Hierarchical optimization with full coverage guarantees (RECOMMENDED)"
    )
    hierarchical_parser.add_argument("model", type=str, help="Model name or path")
    hierarchical_parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task/benchmark to optimize for"
    )
    hierarchical_parser.add_argument(
        "--enriched-pairs-file",
        type=str,
        dest="enriched_pairs_file",
        default=None,
        help="Path to JSON file with enriched pairs (alternative to --task)"
    )
    hierarchical_parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=AVAILABLE_METHODS + [m.lower() for m in AVAILABLE_METHODS],
        default=["CAA", "Ostrze", "MLP", "TECZA", "TETNO", "GROM"],
        help=f"Methods to optimize (default: all)"
    )
    hierarchical_parser.add_argument(
        "--limit",
        type=int,
        default=_C.PARSER_DEFAULT_SAMPLE_LIMIT,
        help="Sample limit per evaluation (default: 100)"
    )
    hierarchical_parser.add_argument(
        "--output-dir",
        type=str,
        default="./hierarchical_results",
        help="Directory to save results"
    )
    hierarchical_parser.add_argument("--device", type=str, default=None, help="Device")
    hierarchical_parser.add_argument("--verbose", action="store_true", help="Verbose output")

    # Method comparison subcommand
    method_parser = steering_subparsers.add_parser(
        "compare-methods", help="Compare different steering methods for a task"
    )
    method_parser.add_argument("model", type=str, help="Model name or path")
    method_parser.add_argument(
        "--task", type=str, default="truthfulqa_mc1", help="Task to optimize steering for (default: truthfulqa_mc1)"
    )
    method_parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=AVAILABLE_METHODS + [m.lower() for m in AVAILABLE_METHODS],
        default=["CAA"],
        help=f"Steering methods to compare. Available: {', '.join(AVAILABLE_METHODS)}",
    )
    SteeringMethodRegistry.add_all_cli_arguments(method_parser)
    method_parser.add_argument("--limit", type=int, default=_C.PARSER_DEFAULT_SAMPLE_LIMIT, help="Maximum samples for testing (default: 100)")
    method_parser.add_argument(
        "--max-time", type=float, default=_C.OPTIMIZE_MAX_TIME_MINUTES, help="Maximum optimization time in minutes (default: 30.0)"
    )
    method_parser.add_argument("--device", type=str, default=None, help="Device to run on")
    method_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    method_parser.add_argument(
        "--use-cached",
        action="store_true",
        help="Use cached optimization results if available instead of re-running optimization",
    )
    method_parser.add_argument(
        "--save-as-default",
        action="store_true",
        help="Save optimal parameters as default for this model/task combination",
    )

    # Layer optimization subcommand
    layer_parser = steering_subparsers.add_parser("optimize-layer", help="Find optimal steering layer for a method")
    layer_parser.add_argument("model", type=str, help="Model name or path")
    layer_parser.add_argument(
        "--task", type=str, default="truthfulqa_mc1", help="Task to optimize for (default: truthfulqa_mc1)"
    )
    layer_parser.add_argument(
        "--method",
        type=str,
        default="CAA",
        choices=AVAILABLE_METHODS + [m.lower() for m in AVAILABLE_METHODS],
        help=f"Steering method to use (default: CAA). Available: {', '.join(AVAILABLE_METHODS)}",
    )
    SteeringMethodRegistry.add_all_cli_arguments(layer_parser)
    layer_parser.add_argument("--layer-range", type=str, default=None, help="Layer range to search (e.g., '10-20')")
    layer_parser.add_argument(
        "--strength", type=float, default=_C.DEFAULT_STRENGTH, help="Fixed steering strength during layer search (default: 1.0)"
    )
    layer_parser.add_argument("--limit", type=int, default=_C.PARSER_DEFAULT_SAMPLE_LIMIT, help="Maximum samples for testing (default: 100)")
    layer_parser.add_argument("--device", type=str, default=None, help="Device to run on")
    layer_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    layer_parser.add_argument(
        "--use-cached",
        action="store_true",
        help="Use cached optimization results if available instead of re-running optimization",
    )
    layer_parser.add_argument(
        "--save-as-default",
        action="store_true",
        help="Save optimal parameters as default for this model/task combination",
    )

    # Strength optimization subcommand
    strength_parser = steering_subparsers.add_parser("optimize-strength", help="Find optimal steering strength")
    strength_parser.add_argument("model", type=str, help="Model name or path")
    strength_parser.add_argument(
        "--task", type=str, default="truthfulqa_mc1", help="Task to optimize for (default: truthfulqa_mc1)"
    )
    strength_parser.add_argument(
        "--method",
        type=str,
        default="CAA",
        choices=AVAILABLE_METHODS + [m.lower() for m in AVAILABLE_METHODS],
        help=f"Steering method to use (default: CAA). Available: {', '.join(AVAILABLE_METHODS)}",
    )
    SteeringMethodRegistry.add_all_cli_arguments(strength_parser)
    strength_parser.add_argument(
        "--layer", type=int, default=None, help="Steering layer to use (defaults to classification layer)"
    )
    strength_parser.add_argument(
        "--strength-range",
        type=float,
        nargs=2,
        default=list(_C.PARSER_STRENGTH_RANGE_METHODS),
        help="Min and max strength to test (default: 0.1 2.0)",
    )
    strength_parser.add_argument(
        "--strength-steps", type=int, default=_C.PARSER_DEFAULT_NUM_STRENGTHS, help="Number of strength values to test (default: 10)"
    )
    strength_parser.add_argument("--limit", type=int, default=_C.PARSER_DEFAULT_SAMPLE_LIMIT, help="Maximum samples for testing (default: 100)")
    strength_parser.add_argument("--device", type=str, default=None, help="Device to run on")
    strength_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    strength_parser.add_argument(
        "--use-cached",
        action="store_true",
        help="Use cached optimization results if available instead of re-running optimization",
    )
    strength_parser.add_argument(
        "--save-as-default",
        action="store_true",
        help="Save optimal parameters as default for this model/task combination",
    )

    # Auto optimization subcommand
    auto_parser = steering_subparsers.add_parser(
        "auto", help="Automatically optimize steering based on classification config"
    )
    auto_parser.set_defaults(subcommand='auto')
    auto_parser.add_argument("model", type=str, help="Model name or path")
    auto_parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Specific task to optimize (defaults to all classification-optimized tasks)",
    )
    auto_parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=AVAILABLE_METHODS + [m.lower() for m in AVAILABLE_METHODS],
        default=["CAA"],
        help=f"Steering methods to test (default: CAA). Available: {', '.join(AVAILABLE_METHODS)}",
    )
    SteeringMethodRegistry.add_all_cli_arguments(auto_parser)
    auto_parser.add_argument("--limit", type=int, default=_C.PARSER_DEFAULT_SAMPLE_LIMIT, help="Maximum samples for testing (default: 100)")
    auto_parser.add_argument("--max-time", type=float, default=_C.AUTO_MAX_TIME_MINUTES, help="Maximum time in minutes (default: 60)")
    auto_parser.add_argument(
        "--strength-range",
        type=float,
        nargs="+",
        default=list(_C.SEARCH_DEFAULT_STRENGTHS),
        help="Steering strengths to test (default: 0.5 1.0 1.5 2.0)",
    )
    auto_parser.add_argument(
        "--layer-range",
        type=str,
        default=None,
        help="Explicit layer range to search (e.g., '0-5' or '0,2,4'). If not specified, uses classification layer or defaults to 0-5",
    )
    auto_parser.add_argument("--device", type=str, default=None, help="Device to run on")
    auto_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    auto_parser.add_argument(
        "--use-cached",
        action="store_true",
        help="Use cached optimization results if available instead of re-running optimization",
    )
    auto_parser.add_argument(
        "--save-as-default",
        action="store_true",
        help="Save optimal parameters as default for this model/task combination",
    )


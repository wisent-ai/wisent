"""Method-specific subparsers for optimize-steering."""
from wisent.core.control.steering_methods.registry import SteeringMethodRegistry
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
        required=True,
        help=f"Methods to optimize. Available: {', '.join(AVAILABLE_METHODS)}"
    )
    hierarchical_parser.add_argument(
        "--limit",
        type=int,
        required=True,
        help="Sample limit per evaluation"
    )
    hierarchical_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
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
        "--task", type=str, required=True, help="Task to optimize steering for"
    )
    method_parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=AVAILABLE_METHODS + [m.lower() for m in AVAILABLE_METHODS],
        required=True,
        help=f"Steering methods to compare. Available: {', '.join(AVAILABLE_METHODS)}",
    )
    SteeringMethodRegistry.add_all_cli_arguments(method_parser)
    method_parser.add_argument("--limit", type=int, required=True, help="Maximum samples for testing")
    method_parser.add_argument(
        "--max-time", type=float, default=None, help="Maximum optimization time in minutes"
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
        "--task", type=str, required=True, help="Task to optimize for"
    )
    layer_parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=AVAILABLE_METHODS + [m.lower() for m in AVAILABLE_METHODS],
        help=f"Steering method to use. Available: {', '.join(AVAILABLE_METHODS)}",
    )
    SteeringMethodRegistry.add_all_cli_arguments(layer_parser)
    layer_parser.add_argument("--layer-range", type=str, default=None, help="Layer range to search (e.g., '10-20')")
    layer_parser.add_argument(
        "--strength", type=float, required=True, help="Fixed steering strength during layer search"
    )
    layer_parser.add_argument("--limit", type=int, required=True, help="Maximum samples for testing")
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
        "--task", type=str, required=True, help="Task to optimize for"
    )
    strength_parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=AVAILABLE_METHODS + [m.lower() for m in AVAILABLE_METHODS],
        help=f"Steering method to use. Available: {', '.join(AVAILABLE_METHODS)}",
    )
    SteeringMethodRegistry.add_all_cli_arguments(strength_parser)
    strength_parser.add_argument(
        "--layer", type=int, default=None, help="Steering layer to use (defaults to classification layer)"
    )
    strength_parser.add_argument(
        "--strength-range",
        type=float,
        nargs=2,
        required=True,
        help="Min and max strength to test",
    )
    strength_parser.add_argument(
        "--strength-steps", type=int, required=True, help="Number of strength values to test"
    )
    strength_parser.add_argument("--limit", type=int, required=True, help="Maximum samples for testing")
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
        required=True,
        help=f"Steering methods to test. Available: {', '.join(AVAILABLE_METHODS)}",
    )
    SteeringMethodRegistry.add_all_cli_arguments(auto_parser)
    auto_parser.add_argument("--limit", type=int, required=True, help="Maximum samples for testing")
    auto_parser.add_argument("--max-time", type=float, default=None, help="Maximum time in minutes")
    auto_parser.add_argument(
        "--strength-range",
        type=float,
        nargs="+",
        default=None,
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
    auto_parser.add_argument(
        "--train-ratio",
        type=float,
        required=True,
        help="Fraction of docs to use for training vs evaluation",
    )


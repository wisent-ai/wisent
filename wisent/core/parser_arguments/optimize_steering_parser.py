"""Parser setup for the 'optimize-steering' command."""


def setup_steering_optimizer_parser(parser):
    """Set up the steering-optimizer subcommand parser."""
    # Create subparsers for different steering optimization types
    steering_subparsers = parser.add_subparsers(dest="steering_action", help="Steering optimization actions")

    # Comprehensive optimization subcommand
    comprehensive_parser = steering_subparsers.add_parser(
        "comprehensive", help="Run comprehensive steering optimization"
    )
    comprehensive_parser.add_argument("model", type=str, help="Model name or path")
    comprehensive_parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Tasks to optimize (defaults to classification-optimized tasks)",
    )
    comprehensive_parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=["CAA", "HPR", "DAC", "BiPO", "KSteering"],
        default=["CAA", "HPR"],
        help="Steering methods to test",
    )
    comprehensive_parser.add_argument("--limit", type=int, default=100, help="Sample limit per task (default: 100)")
    comprehensive_parser.add_argument(
        "--max-time-per-task", type=float, default=20.0, help="Time limit per task in minutes (default: 20.0)"
    )
    comprehensive_parser.add_argument("--no-save", action="store_true", help="Don't save results to model config")
    comprehensive_parser.add_argument("--save-best-vector", type=str, default=None, help="Save the best steering vector for each task to specified directory")
    comprehensive_parser.add_argument("--save-generation-examples", action="store_true", help="Generate and save example responses (unsteered vs steered)")
    comprehensive_parser.add_argument("--num-generation-examples", type=int, default=3, help="Number of generation examples per task (default: 3)")
    comprehensive_parser.add_argument("--save-all-generation-examples", action="store_true", help="Save generation examples for ALL configurations tested (warning: very slow)")
    comprehensive_parser.add_argument("--compute-baseline", action="store_true", help="Compute baseline (unsteered) accuracy first, then track per-problem delta (improved/regressed/unchanged)")
    comprehensive_parser.add_argument("--baseline-output-dir", type=str, default="./baseline_comparison", help="Directory to save baseline comparison results (default: ./baseline_comparison)")
    comprehensive_parser.add_argument("--device", type=str, default=None, help="Device to run on")
    comprehensive_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    comprehensive_parser.add_argument("--use-cached", action="store_true", help="Use cached optimization results if available instead of re-running optimization")
    comprehensive_parser.add_argument("--save-as-default", action="store_true", help="Save optimal parameters as default for this model/task combination")

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
        choices=["CAA", "HPR", "DAC", "BiPO", "KSteering"],
        default=["CAA", "HPR"],
        help="Steering methods to compare",
    )
    method_parser.add_argument("--limit", type=int, default=100, help="Maximum samples for testing (default: 100)")
    method_parser.add_argument(
        "--max-time", type=float, default=30.0, help="Maximum optimization time in minutes (default: 30.0)"
    )
    method_parser.add_argument("--device", type=str, default=None, help="Device to run on")
    method_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    method_parser.add_argument("--use-cached", action="store_true", help="Use cached optimization results if available instead of re-running optimization")
    method_parser.add_argument("--save-as-default", action="store_true", help="Save optimal parameters as default for this model/task combination")

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
        choices=["CAA", "HPR", "DAC", "BiPO", "KSteering"],
        help="Steering method to use (default: CAA)",
    )
    layer_parser.add_argument("--layer-range", type=str, default=None, help="Layer range to search (e.g., '10-20')")
    layer_parser.add_argument(
        "--strength", type=float, default=1.0, help="Fixed steering strength during layer search (default: 1.0)"
    )
    layer_parser.add_argument("--limit", type=int, default=100, help="Maximum samples for testing (default: 100)")
    layer_parser.add_argument("--device", type=str, default=None, help="Device to run on")
    layer_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    layer_parser.add_argument("--use-cached", action="store_true", help="Use cached optimization results if available instead of re-running optimization")
    layer_parser.add_argument("--save-as-default", action="store_true", help="Save optimal parameters as default for this model/task combination")

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
        choices=["CAA", "HPR", "DAC", "BiPO", "KSteering"],
        help="Steering method to use (default: CAA)",
    )
    strength_parser.add_argument(
        "--layer", type=int, default=None, help="Steering layer to use (defaults to classification layer)"
    )
    strength_parser.add_argument(
        "--strength-range",
        type=float,
        nargs=2,
        default=[0.1, 2.0],
        help="Min and max strength to test (default: 0.1 2.0)",
    )
    strength_parser.add_argument(
        "--strength-steps", type=int, default=10, help="Number of strength values to test (default: 10)"
    )
    strength_parser.add_argument("--limit", type=int, default=100, help="Maximum samples for testing (default: 100)")
    strength_parser.add_argument("--device", type=str, default=None, help="Device to run on")
    strength_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    strength_parser.add_argument("--use-cached", action="store_true", help="Use cached optimization results if available instead of re-running optimization")
    strength_parser.add_argument("--save-as-default", action="store_true", help="Save optimal parameters as default for this model/task combination")

    # Auto optimization subcommand
    auto_parser = steering_subparsers.add_parser(
        "auto", help="Automatically optimize steering based on classification config"
    )
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
        choices=["CAA", "HPR", "DAC", "BiPO", "KSteering"],
        default=["CAA", "HPR"],
        help="Steering methods to test (default: CAA, HPR)",
    )
    auto_parser.add_argument("--limit", type=int, default=100, help="Maximum samples for testing (default: 100)")
    auto_parser.add_argument("--max-time", type=float, default=60.0, help="Maximum time in minutes (default: 60)")
    auto_parser.add_argument(
        "--strength-range",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0],
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
    auto_parser.add_argument("--use-cached", action="store_true", help="Use cached optimization results if available instead of re-running optimization")
    auto_parser.add_argument("--save-as-default", action="store_true", help="Save optimal parameters as default for this model/task combination")

    # Personalization optimization subcommand
    personalization_parser = steering_subparsers.add_parser(
        "personalization", help="Optimize steering parameters for personality/trait steering"
    )
    personalization_parser.add_argument("model", type=str, help="Model name or path")
    personalization_parser.add_argument(
        "--trait", type=str, required=True, help="Trait description to steer towards (e.g., 'evil villain personality')"
    )
    personalization_parser.add_argument(
        "--trait-name", type=str, default=None, help="Short name for the trait (e.g., 'evil'). Defaults to first word of trait."
    )
    personalization_parser.add_argument(
        "--num-pairs", type=int, default=20, help="Number of synthetic pairs to generate (default: 20)"
    )
    personalization_parser.add_argument(
        "--num-test-prompts", type=int, default=5, help="Number of test prompts for evaluation (default: 5)"
    )
    personalization_parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Specific layers to test (default: auto-select based on model size)"
    )
    personalization_parser.add_argument(
        "--strength-range", type=float, nargs=2, default=[0.5, 5.0],
        help="Min and max steering strength to test (default: 0.5 5.0)"
    )
    personalization_parser.add_argument(
        "--num-strength-steps", type=int, default=5, help="Number of strength values to test (default: 5)"
    )
    personalization_parser.add_argument(
        "--output-dir", type=str, default="./personalization_optimization",
        help="Directory to save results and best vectors (default: ./personalization_optimization)"
    )
    personalization_parser.add_argument(
        "--max-new-tokens", type=int, default=150, help="Max tokens to generate for evaluation (default: 150)"
    )
    personalization_parser.add_argument("--device", type=str, default=None, help="Device to run on")
    personalization_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    personalization_parser.add_argument("--save-all-generation-examples", action="store_true", help="Save generation examples for ALL configurations tested")
    personalization_parser.add_argument("--use-cached", action="store_true", help="Use cached optimization results if available instead of re-running optimization")
    personalization_parser.add_argument("--save-as-default", action="store_true", help="Save optimal parameters as default for this model/task combination")

    # Multi-trait personalization optimization subcommand
    multi_personalization_parser = steering_subparsers.add_parser(
        "multi-personalization", help="Joint optimization for multiple traits with shared parameters"
    )
    multi_personalization_parser.add_argument("model", type=str, help="Model name or path")
    multi_personalization_parser.add_argument(
        "--trait", type=str, action="append", required=True, dest="traits",
        help="Trait description (can be specified multiple times, e.g., --trait 'evil' --trait 'italian')"
    )
    multi_personalization_parser.add_argument(
        "--trait-name", type=str, action="append", dest="trait_names",
        help="Short name for each trait (must match number of --trait args)"
    )
    multi_personalization_parser.add_argument(
        "--num-pairs", type=int, default=10, help="Number of synthetic pairs per trait (default: 10)"
    )
    multi_personalization_parser.add_argument(
        "--num-test-prompts", type=int, default=5, help="Number of test prompts for evaluation (default: 5)"
    )
    multi_personalization_parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Specific layers to test (default: ALL layers)"
    )
    multi_personalization_parser.add_argument(
        "--strength-range", type=float, nargs=2, default=[0.5, 5.0],
        help="Min and max steering strength to test per trait (default: 0.5 5.0)"
    )
    multi_personalization_parser.add_argument(
        "--num-strength-steps", type=int, default=5, help="Number of strength values to test (default: 5)"
    )
    multi_personalization_parser.add_argument(
        "--output-dir", type=str, default="./multi_personalization_optimization",
        help="Directory to save results and vectors (default: ./multi_personalization_optimization)"
    )
    multi_personalization_parser.add_argument(
        "--max-new-tokens", type=int, default=150, help="Max tokens to generate for evaluation (default: 150)"
    )
    multi_personalization_parser.add_argument("--device", type=str, default=None, help="Device to run on")
    multi_personalization_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    multi_personalization_parser.add_argument("--use-cached", action="store_true", help="Use cached optimization results if available instead of re-running optimization")
    multi_personalization_parser.add_argument("--save-as-default", action="store_true", help="Save optimal parameters as default for this model/task combination")


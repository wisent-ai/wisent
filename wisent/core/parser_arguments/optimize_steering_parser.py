"""Parser setup for the 'optimize-steering' command."""

from wisent.core.steering_methods.registry import SteeringMethodRegistry

# Get available steering methods from registry
AVAILABLE_METHODS = [m.upper() for m in SteeringMethodRegistry.list_methods()]


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
        choices=AVAILABLE_METHODS,
        default=["CAA"],
        help=f"Steering methods to test. Available: {', '.join(AVAILABLE_METHODS)}",
    )
    # Add method-specific arguments from registry
    SteeringMethodRegistry.add_all_cli_arguments(comprehensive_parser)
    comprehensive_parser.add_argument("--limit", type=int, default=100, help="Sample limit per task (default: 100)")
    comprehensive_parser.add_argument(
        "--max-time-per-task", type=float, default=20.0, help="Time limit per task in minutes (default: 20.0)"
    )
    comprehensive_parser.add_argument("--no-save", action="store_true", help="Don't save results to model config")
    comprehensive_parser.add_argument(
        "--save-best-vector",
        type=str,
        default=None,
        help="Save the best steering vector for each task to specified directory",
    )
    comprehensive_parser.add_argument(
        "--save-generation-examples",
        action="store_true",
        help="Generate and save example responses (unsteered vs steered)",
    )
    comprehensive_parser.add_argument(
        "--num-generation-examples", type=int, default=3, help="Number of generation examples per task (default: 3)"
    )
    comprehensive_parser.add_argument(
        "--save-all-generation-examples",
        action="store_true",
        help="Save generation examples for ALL configurations tested (warning: very slow)",
    )
    comprehensive_parser.add_argument(
        "--compute-baseline",
        action="store_true",
        help="Compute baseline (unsteered) accuracy first, then track per-problem delta (improved/regressed/unchanged)",
    )
    comprehensive_parser.add_argument(
        "--baseline-output-dir",
        type=str,
        default="./baseline_comparison",
        help="Directory to save baseline comparison results (default: ./baseline_comparison)",
    )
    comprehensive_parser.add_argument(
        "--output-dir",
        type=str,
        default="./optimization_results",
        help="Directory to save optimization results (default: ./optimization_results)",
    )
    comprehensive_parser.add_argument(
        "--show-comparisons",
        type=int,
        default=0,
        metavar="N",
        help="Show N before/after response comparisons with biggest score changes in console. Default: 0 (disabled)",
    )
    comprehensive_parser.add_argument(
        "--save-comparisons",
        type=str,
        default=None,
        metavar="PATH",
        help="Save all comparisons to JSON file (use with --show-comparisons to also display in console)",
    )
    comprehensive_parser.add_argument("--device", type=str, default=None, help="Device to run on")
    comprehensive_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    comprehensive_parser.add_argument(
        "--use-cached",
        action="store_true",
        help="Use cached optimization results if available instead of re-running optimization",
    )
    comprehensive_parser.add_argument(
        "--save-as-default",
        action="store_true",
        help="Save optimal parameters as default for this model/task combination",
    )
    comprehensive_parser.add_argument(
        "--custom-evaluator",
        type=str,
        default=None,
        help=(
            "Custom evaluator specification. Can be: "
            "(1) Python module path e.g. 'my_evaluators.gptzero', "
            "(2) File path with function e.g. './my_eval.py:score_fn', "
            "(3) Built-in example e.g. 'wisent.core.evaluators.custom.examples.gptzero'. "
            "The module must define 'create_evaluator', 'evaluator', or 'evaluate' function."
        )
    )
    comprehensive_parser.add_argument(
        "--custom-evaluator-kwargs",
        type=str,
        default=None,
        help="JSON string of kwargs for custom evaluator, e.g. '{\"api_key\": \"xxx\", \"optimize_for\": \"human_prob\"}'"
    )
    
    # Early rejection options
    comprehensive_parser.add_argument(
        "--disable-early-rejection",
        action="store_true",
        default=False,
        help="Disable early rejection of low-quality vectors during optimization (slower but explores more)"
    )
    comprehensive_parser.add_argument(
        "--early-rejection-snr-threshold",
        type=float,
        default=5.0,
        help="Minimum SNR for early rejection during optimization (default: 5.0)"
    )
    comprehensive_parser.add_argument(
        "--early-rejection-cv-threshold",
        type=float,
        default=0.1,
        help="Minimum cross-validation score for early rejection during optimization (default: 0.1)"
    )
    
    # ==========================================================================
    # SEARCH STRATEGY CONFIGURATION
    # ==========================================================================
    
    comprehensive_parser.add_argument(
        "--search-strategy",
        type=str,
        choices=["grid", "optuna"],
        default="grid",
        help="Search strategy: 'grid' for exhaustive search, 'optuna' for TPE sampling (default: grid)"
    )
    comprehensive_parser.add_argument(
        "--n-trials",
        type=int,
        default=300,
        help="Number of Optuna trials when using --search-strategy optuna (default: 300)"
    )
    comprehensive_parser.add_argument(
        "--n-startup-trials",
        type=int,
        default=10,
        help="Number of random trials before TPE kicks in (default: 10)"
    )
    
    # ==========================================================================
    # SEARCH SPACE CONFIGURATION
    # ==========================================================================
    
    # Quick search mode
    comprehensive_parser.add_argument(
        "--quick-search",
        action="store_true",
        help="Use reduced search space for faster testing (fewer configurations)"
    )
    
    # Base search space overrides
    comprehensive_parser.add_argument(
        "--search-layers", "--layers",
        type=str,
        default=None,
        dest="search_layers",
        help="Comma-separated layer indices to search (e.g., '8,10,12,14')"
    )
    comprehensive_parser.add_argument(
        "--search-strengths", "--strengths",
        type=str,
        default=None,
        dest="search_strengths",
        help="Comma-separated strength values to search (e.g., '0.5,1.0,1.5,2.0')"
    )
    comprehensive_parser.add_argument(
        "--search-strategies", "--strategies",
        type=str,
        default=None,
        dest="search_strategies",
        help="Comma-separated steering strategies to search (e.g., 'constant,initial_only,diminishing,increasing,gaussian')"
    )
    comprehensive_parser.add_argument(
        "--search-token-aggregations", "--token-aggregations",
        type=str,
        default=None,
        dest="search_token_aggregations",
        help="Comma-separated token aggregation strategies (e.g., 'last_token,mean_pooling,first_token,max_pooling,continuation_token,choice_token')"
    )
    comprehensive_parser.add_argument(
        "--search-prompt-constructions", "--prompt-constructions",
        type=str,
        default=None,
        dest="search_prompt_constructions",
        help="Comma-separated prompt construction strategies (e.g., 'chat_template,direct_completion,instruction_following,multiple_choice,role_playing')"
    )
    
    # PRISM-specific search space
    comprehensive_parser.add_argument(
        "--search-num-directions",
        type=int,
        nargs="+",
        default=None,
        help="[PRISM/TITAN] Number of directions to search (e.g., 1 2 3 5)"
    )
    comprehensive_parser.add_argument(
        "--search-direction-weighting",
        type=str,
        nargs="+",
        default=None,
        choices=["primary_only", "equal", "learned", "decay"],
        help="[PRISM] Direction weighting strategies to search"
    )
    comprehensive_parser.add_argument(
        "--search-retain-weight",
        type=float,
        nargs="+",
        default=None,
        help="[PRISM/TITAN] Retain weight values to search (e.g., 0.0 0.1 0.3)"
    )
    
    # PULSE-specific search space
    comprehensive_parser.add_argument(
        "--search-sensor-layer",
        type=str,
        nargs="+",
        default=None,
        choices=["middle", "late", "last_quarter"],
        help="[PULSE/TITAN] Sensor layer configurations to search"
    )
    comprehensive_parser.add_argument(
        "--search-steering-layers",
        type=str,
        nargs="+",
        default=None,
        choices=["single_best", "range_3", "range_5", "all_late"],
        help="[PULSE/TITAN] Steering layer configurations to search"
    )
    comprehensive_parser.add_argument(
        "--search-threshold",
        type=float,
        nargs="+",
        default=None,
        help="[PULSE] Condition threshold values to search (e.g., 0.3 0.5 0.7)"
    )
    comprehensive_parser.add_argument(
        "--search-gate-temp",
        type=float,
        nargs="+",
        default=None,
        help="[PULSE] Gate temperature values to search (e.g., 0.1 0.5 1.0)"
    )
    comprehensive_parser.add_argument(
        "--search-max-alpha",
        type=float,
        nargs="+",
        default=None,
        help="[PULSE/TITAN] Max alpha values to search (e.g., 1.5 2.0 3.0)"
    )
    
    # TITAN-specific search space
    comprehensive_parser.add_argument(
        "--search-gate-hidden",
        type=int,
        nargs="+",
        default=None,
        help="[TITAN] Gate hidden dimension values to search (e.g., 32 64 128)"
    )
    comprehensive_parser.add_argument(
        "--search-intensity-hidden",
        type=int,
        nargs="+",
        default=None,
        help="[TITAN] Intensity hidden dimension values to search (e.g., 16 32 64)"
    )
    comprehensive_parser.add_argument(
        "--search-behavior-weight",
        type=float,
        nargs="+",
        default=None,
        help="[TITAN] Behavior weight values to search (e.g., 0.5 1.0)"
    )
    comprehensive_parser.add_argument(
        "--search-sparse-weight",
        type=float,
        nargs="+",
        default=None,
        help="[TITAN] Sparse weight values to search (e.g., 0.0 0.05 0.1)"
    )

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
        choices=AVAILABLE_METHODS,
        default=["CAA"],
        help=f"Steering methods to compare. Available: {', '.join(AVAILABLE_METHODS)}",
    )
    SteeringMethodRegistry.add_all_cli_arguments(method_parser)
    method_parser.add_argument("--limit", type=int, default=100, help="Maximum samples for testing (default: 100)")
    method_parser.add_argument(
        "--max-time", type=float, default=30.0, help="Maximum optimization time in minutes (default: 30.0)"
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
        choices=AVAILABLE_METHODS,
        help=f"Steering method to use (default: CAA). Available: {', '.join(AVAILABLE_METHODS)}",
    )
    SteeringMethodRegistry.add_all_cli_arguments(layer_parser)
    layer_parser.add_argument("--layer-range", type=str, default=None, help="Layer range to search (e.g., '10-20')")
    layer_parser.add_argument(
        "--strength", type=float, default=1.0, help="Fixed steering strength during layer search (default: 1.0)"
    )
    layer_parser.add_argument("--limit", type=int, default=100, help="Maximum samples for testing (default: 100)")
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
        choices=AVAILABLE_METHODS,
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
        default=[0.1, 2.0],
        help="Min and max strength to test (default: 0.1 2.0)",
    )
    strength_parser.add_argument(
        "--strength-steps", type=int, default=10, help="Number of strength values to test (default: 10)"
    )
    strength_parser.add_argument("--limit", type=int, default=100, help="Maximum samples for testing (default: 100)")
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
        choices=AVAILABLE_METHODS,
        default=["CAA"],
        help=f"Steering methods to test (default: CAA). Available: {', '.join(AVAILABLE_METHODS)}",
    )
    SteeringMethodRegistry.add_all_cli_arguments(auto_parser)
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

    # Personalization optimization subcommand
    personalization_parser = steering_subparsers.add_parser(
        "personalization", help="Optimize steering parameters for personality/trait steering"
    )
    personalization_parser.add_argument(
        "--task",
        type=str,
        default="personalization",
        help="Task type (default: personalization). For consistency with other commands.",
    )
    personalization_parser.add_argument(
        "--model", type=str, required=True, help="Model name or path"
    )
    personalization_parser.add_argument(
        "--trait", type=str, required=True, help="Trait description to steer towards (e.g., 'evil villain personality')"
    )
    personalization_parser.add_argument(
        "--trait-name",
        type=str,
        default=None,
        help="Short name for the trait (e.g., 'evil'). Defaults to first word of trait.",
    )
    personalization_parser.add_argument(
        "--num-pairs", type=int, default=20, help="Number of synthetic pairs to generate (default: 20)"
    )
    personalization_parser.add_argument(
        "--num-test-prompts", type=int, default=5, help="Number of test prompts for evaluation (default: 5)"
    )
    personalization_parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layers to test (default: auto-select based on model size)",
    )
    personalization_parser.add_argument(
        "--strength-range",
        type=float,
        nargs=2,
        default=[0.5, 5.0],
        help="Min and max steering strength to test (default: 0.5 5.0)",
    )
    personalization_parser.add_argument(
        "--num-strength-steps", type=int, default=5, help="Number of strength values to test (default: 5)"
    )
    personalization_parser.add_argument(
        "--output-dir",
        type=str,
        default="./personalization_optimization",
        help="Directory to save results and best vectors (default: ./personalization_optimization)",
    )
    personalization_parser.add_argument(
        "--max-new-tokens", type=int, default=150, help="Max tokens to generate for evaluation (default: 150)"
    )
    personalization_parser.add_argument("--device", type=str, default=None, help="Device to run on")
    personalization_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    personalization_parser.add_argument(
        "--save-all-generation-examples",
        action="store_true",
        help="Save generation examples for ALL configurations tested",
    )
    personalization_parser.add_argument(
        "--use-cached",
        action="store_true",
        help="Use cached optimization results if available instead of re-running optimization",
    )
    personalization_parser.add_argument(
        "--save-as-default",
        action="store_true",
        help="Save optimal parameters as default for this model/task combination",
    )
    personalization_parser.add_argument(
        "--custom-evaluator",
        type=str,
        default=None,
        help=(
            "Custom evaluator for trait optimization. Can be: "
            "(1) Python module path e.g. 'my_evaluators.gptzero', "
            "(2) File path with function e.g. './my_eval.py:score_fn', "
            "(3) Built-in example e.g. 'wisent.core.evaluators.custom.examples.gptzero'. "
            "Overrides the default personalization evaluator."
        )
    )
    personalization_parser.add_argument(
        "--custom-evaluator-kwargs",
        type=str,
        default=None,
        help="JSON string of kwargs for custom evaluator, e.g. '{\"api_key\": \"xxx\"}'"
    )
    personalization_parser.add_argument(
        "--search-strategy",
        type=str,
        choices=["grid", "optuna"],
        default="grid",
        help="Search strategy: 'grid' for exhaustive search, 'optuna' for TPE sampling (default: grid)"
    )
    personalization_parser.add_argument(
        "--n-trials",
        type=int,
        default=300,
        help="Number of Optuna trials when using --search-strategy optuna (default: 300)"
    )
    personalization_parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Sample limit for optimization (default: 50)"
    )

    # Multi-trait personalization optimization subcommand
    multi_personalization_parser = steering_subparsers.add_parser(
        "multi-personalization", help="Joint optimization for multiple traits with shared parameters"
    )
    multi_personalization_parser.add_argument("model", type=str, help="Model name or path")
    multi_personalization_parser.add_argument(
        "--trait",
        type=str,
        action="append",
        required=True,
        dest="traits",
        help="Trait description (can be specified multiple times, e.g., --trait 'evil' --trait 'italian')",
    )
    multi_personalization_parser.add_argument(
        "--trait-name",
        type=str,
        action="append",
        dest="trait_names",
        help="Short name for each trait (must match number of --trait args)",
    )
    multi_personalization_parser.add_argument(
        "--num-pairs", type=int, default=10, help="Number of synthetic pairs per trait (default: 10)"
    )
    multi_personalization_parser.add_argument(
        "--num-test-prompts", type=int, default=5, help="Number of test prompts for evaluation (default: 5)"
    )
    multi_personalization_parser.add_argument(
        "--layers", type=int, nargs="+", default=None, help="Specific layers to test (default: ALL layers)"
    )
    multi_personalization_parser.add_argument(
        "--strength-range",
        type=float,
        nargs=2,
        default=[0.5, 5.0],
        help="Min and max steering strength to test per trait (default: 0.5 5.0)",
    )
    multi_personalization_parser.add_argument(
        "--num-strength-steps", type=int, default=5, help="Number of strength values to test (default: 5)"
    )
    multi_personalization_parser.add_argument(
        "--output-dir",
        type=str,
        default="./multi_personalization_optimization",
        help="Directory to save results and vectors (default: ./multi_personalization_optimization)",
    )
    multi_personalization_parser.add_argument(
        "--max-new-tokens", type=int, default=150, help="Max tokens to generate for evaluation (default: 150)"
    )
    multi_personalization_parser.add_argument("--device", type=str, default=None, help="Device to run on")
    multi_personalization_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    multi_personalization_parser.add_argument(
        "--use-cached",
        action="store_true",
        help="Use cached optimization results if available instead of re-running optimization",
    )
    multi_personalization_parser.add_argument(
        "--save-as-default",
        action="store_true",
        help="Save optimal parameters as default for this model/task combination",
    )

    # ==========================================================================
    # UNIVERSAL METHOD OPTIMIZER (NEW)
    # ==========================================================================
    # This optimizer uses the universal train(pair_set) interface that ALL
    # steering methods implement, ensuring it works with any method including
    # future ones.
    
    universal_parser = steering_subparsers.add_parser(
        "universal",
        help="Universal optimizer that works with ANY steering method (recommended)"
    )
    universal_parser.add_argument("model", type=str, help="Model name or path")
    universal_parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task/benchmark to optimize for (e.g., truthfulqa_generation, arc_easy)"
    )
    universal_parser.add_argument(
        "--method",
        type=str,
        default="CAA",
        choices=AVAILABLE_METHODS + [m.lower() for m in AVAILABLE_METHODS],
        help=f"Steering method to optimize. Available: {', '.join(AVAILABLE_METHODS)} (default: CAA)"
    )
    universal_parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum samples to use (default: 100)"
    )
    universal_parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced search space for faster testing"
    )
    universal_parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Maximum number of configurations to test (default: all)"
    )
    universal_parser.add_argument(
        "--output-dir",
        type=str,
        default="./optimization_results",
        help="Directory to save results (default: ./optimization_results)"
    )
    universal_parser.add_argument(
        "--save-best-vector",
        action="store_true",
        help="Save the best steering vector to output directory"
    )
    universal_parser.add_argument("--device", type=str, default=None, help="Device to run on")
    universal_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    # Search space customization
    universal_parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to search (e.g., '8,10,12,14')"
    )
    universal_parser.add_argument(
        "--strengths",
        type=str,
        default=None,
        help="Comma-separated strength values to search (e.g., '0.5,1.0,1.5,2.0')"
    )
    universal_parser.add_argument(
        "--token-aggregations",
        type=str,
        nargs="+",
        default=None,
        choices=["last_token", "mean_pooling", "first_token", "max_pooling", "continuation_token"],
        help="Token aggregation strategies to search"
    )
    universal_parser.add_argument(
        "--prompt-strategies",
        type=str,
        nargs="+",
        default=None,
        choices=["chat_template", "direct_completion", "multiple_choice", "role_playing", "instruction_following"],
        help="Prompt construction strategies to search"
    )
    
    # Method-specific parameter overrides (JSON format)
    universal_parser.add_argument(
        "--method-params",
        type=str,
        default=None,
        help='JSON dict of method-specific parameter ranges, e.g., \'{"num_directions": [2, 3, 5]}\''
    )

"""Comprehensive subparser for optimize-steering."""
from wisent.core.control.steering_methods.registry import SteeringMethodRegistry
from wisent.core.control.steering_methods.definitions._definitions_tetno_grom import TETNO_STEERING_LAYER_CONFIGS
from wisent.core.utils.cli.commands.steering.core.configuration.settings.steering_search_space_classes import DirectionWeighting
AVAILABLE_METHODS = [m.upper() for m in SteeringMethodRegistry.list_methods()]

def setup_comprehensive_parser(steering_subparsers):
    """Set up the comprehensive optimization subparser."""
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
        "--enriched-pairs-file",
        type=str,
        dest="enriched_pairs_file",
        default=None,
        help="Path to JSON file with enriched pairs (activations). Alternative to --tasks for custom data.",
    )
    comprehensive_parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=AVAILABLE_METHODS + [m.lower() for m in AVAILABLE_METHODS],
        required=True,
        help=f"Steering methods to test. Available: {', '.join(AVAILABLE_METHODS)}",
    )
    # Add method-specific arguments from registry
    SteeringMethodRegistry.add_all_cli_arguments(comprehensive_parser)
    comprehensive_parser.add_argument("--limit", type=int, required=True, help="Sample limit per task")
    comprehensive_parser.add_argument(
        "--max-time-per-task", type=float, required=True, help="Time limit per task in minutes"
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
        "--num-generation-examples", type=int, required=True, help="Number of generation examples per task"
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
        required=True,
        help="Directory to save baseline comparison results",
    )
    comprehensive_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save optimization results",
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
            "(3) Built-in example e.g. 'wisent.core.reading.evaluators.custom.examples.gptzero'. "
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
        required=True,
        help="Minimum SNR for early rejection during optimization"
    )
    comprehensive_parser.add_argument(
        "--early-rejection-cv-threshold",
        type=float,
        required=True,
        help="Minimum cross-validation score for early rejection during optimization"
    )
    
    # SEARCH STRATEGY CONFIGURATION
    
    comprehensive_parser.add_argument(
        "--search-strategy",
        type=str,
        choices=["grid", "optuna"],
        required=True,
        help="Search strategy: 'grid' for exhaustive search, 'optuna' for TPE sampling"
    )
    comprehensive_parser.add_argument(
        "--n-trials",
        type=int,
        required=True,
        help="Number of Optuna trials when using --search-strategy optuna"
    )
    comprehensive_parser.add_argument(
        "--n-startup-trials",
        type=int,
        required=True,
        help="Number of random trials before TPE kicks in"
    )
    
    # SEARCH SPACE CONFIGURATION

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
    
    # TECZA-specific search space
    comprehensive_parser.add_argument(
        "--search-num-directions",
        type=int,
        nargs="+",
        default=None,
        help="[TECZA/GROM] Number of directions to search (e.g., 1 2 3 5)"
    )
    comprehensive_parser.add_argument(
        "--search-direction-weighting",
        type=str,
        nargs="+",
        default=None,
        choices=[d.value for d in DirectionWeighting],
        help="[TECZA] Direction weighting strategies to search"
    )
    comprehensive_parser.add_argument(
        "--search-retain-weight",
        type=float,
        nargs="+",
        default=None,
        help="[TECZA/GROM] Retain weight values to search (e.g., 0.0 0.1 0.3)"
    )
    
    # TETNO-specific search space
    comprehensive_parser.add_argument(
        "--search-sensor-layer",
        type=str,
        nargs="+",
        default=None,
        choices=["middle"],
        help="[TETNO/GROM] Sensor layer configurations to search"
    )
    comprehensive_parser.add_argument(
        "--search-steering-layers",
        type=str,
        nargs="+",
        default=None,
        choices=list(TETNO_STEERING_LAYER_CONFIGS),
        help="[TETNO/GROM] Steering layer configurations to search"
    )
    comprehensive_parser.add_argument(
        "--search-threshold",
        type=float,
        nargs="+",
        default=None,
        help="[TETNO] Condition threshold values to search (e.g., 0.3 0.5 0.7)"
    )
    comprehensive_parser.add_argument(
        "--search-gate-temp",
        type=float,
        nargs="+",
        default=None,
        help="[TETNO] Gate temperature values to search (e.g., 0.1 0.5 1.0)"
    )
    comprehensive_parser.add_argument(
        "--search-max-alpha",
        type=float,
        nargs="+",
        default=None,
        help="[TETNO/GROM] Max alpha values to search (e.g., 1.5 2.0 3.0)"
    )
    
    # GROM-specific search space
    comprehensive_parser.add_argument(
        "--search-gate-hidden",
        type=int,
        nargs="+",
        default=None,
        help="[GROM] Gate hidden dimension values to search (e.g., 32 64 128)"
    )
    comprehensive_parser.add_argument(
        "--search-intensity-hidden",
        type=int,
        nargs="+",
        default=None,
        help="[GROM] Intensity hidden dimension values to search (e.g., 16 32 64)"
    )
    comprehensive_parser.add_argument(
        "--search-behavior-weight",
        type=float,
        nargs="+",
        default=None,
        help="[GROM] Behavior weight values to search (e.g., 0.5 1.0)"
    )
    comprehensive_parser.add_argument(
        "--search-sparse-weight",
        type=float,
        nargs="+",
        default=None,
        help="[GROM] Sparse weight values to search (e.g., 0.0 0.05 0.1)"
    )

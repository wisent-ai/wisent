"""Welfare and universal subparsers for optimize-steering."""
from wisent.core.steering_methods.registry import SteeringMethodRegistry
AVAILABLE_METHODS = [m.upper() for m in SteeringMethodRegistry.list_methods()]


def setup_welfare_universal_parsers(steering_subparsers):
    """Set up welfare and universal subparsers."""
    # ==========================================================================
    # WELFARE OPTIMIZATION (AI Subjective States - ANIMA Framework)
    # ==========================================================================
    # Based on ANIMA: Activation-based Neural Investigation of Model Affect
    # Optimizes steering for AI welfare states: comfort/distress, satisfaction/dissatisfaction,
    # engagement/aversion, curiosity/boredom, affiliation/isolation, agency/helplessness

    welfare_parser = steering_subparsers.add_parser(
        "welfare", help="Optimize steering parameters for AI welfare states (ANIMA framework)"
    )
    welfare_parser.add_argument(
        "--task",
        type=str,
        default="welfare",
        help="Task type (default: welfare). For consistency with other commands.",
    )
    welfare_parser.add_argument(
        "--model", type=str, required=True, help="Model name or path"
    )
    welfare_parser.add_argument(
        "--trait",
        type=str,
        required=True,
        choices=[
            "comfort_distress",
            "satisfaction_dissatisfaction",
            "engagement_aversion",
            "curiosity_boredom",
            "affiliation_isolation",
            "agency_helplessness",
        ],
        help="Welfare state to steer (e.g., 'comfort_distress', 'agency_helplessness')"
    )
    welfare_parser.add_argument(
        "--direction",
        type=str,
        default="positive",
        choices=["positive", "negative"],
        help="Direction to steer: 'positive' (comfort, satisfaction, etc.) or 'negative' (distress, etc.)"
    )
    welfare_parser.add_argument(
        "--num-pairs", type=int, default=50, help="Number of pairs to use for training (default: 50)"
    )
    welfare_parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layers to test (default: auto-select based on model size)",
    )
    welfare_parser.add_argument(
        "--strength-range",
        type=float,
        nargs=2,
        default=[0.5, 3.0],
        help="Min and max steering strength to test (default: 0.5 3.0)",
    )
    welfare_parser.add_argument(
        "--num-strength-steps", type=int, default=5, help="Number of strength values to test (default: 5)"
    )
    welfare_parser.add_argument(
        "--output-dir",
        type=str,
        default="./welfare_optimization",
        help="Directory to save results and best vectors (default: ./welfare_optimization)",
    )
    welfare_parser.add_argument(
        "--max-new-tokens", type=int, default=150, help="Max tokens to generate for evaluation (default: 150)"
    )
    welfare_parser.add_argument("--device", type=str, default=None, help="Device to run on")
    welfare_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    welfare_parser.add_argument(
        "--use-cached",
        action="store_true",
        help="Use cached optimization results if available instead of re-running optimization",
    )
    welfare_parser.add_argument(
        "--save-as-default",
        action="store_true",
        help="Save optimal parameters as default for this model/task combination",
    )
    welfare_parser.add_argument(
        "--search-strategy",
        type=str,
        choices=["grid", "optuna"],
        default="grid",
        help="Search strategy: 'grid' for exhaustive search, 'optuna' for TPE sampling (default: grid)"
    )
    welfare_parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials when using --search-strategy optuna (default: 100)"
    )
    welfare_parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Sample limit for optimization (default: 50)"
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
        default=None,
        help="Task/benchmark to optimize for (e.g., truthfulqa_generation, arc_easy)"
    )
    universal_parser.add_argument(
        "--enriched-pairs-file",
        type=str,
        dest="enriched_pairs_file",
        default=None,
        help="Path to JSON file with enriched pairs (activations). Alternative to --task for custom data.",
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

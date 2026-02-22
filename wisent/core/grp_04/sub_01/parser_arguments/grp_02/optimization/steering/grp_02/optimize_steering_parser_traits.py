"""Personalization subparsers for optimize-steering."""


def setup_personalization_parsers(steering_subparsers):
    """Set up personalization and multi-personalization subparsers."""
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

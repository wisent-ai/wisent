"""Parser setup for the 'agent' command."""


def setup_agent_parser(parser):
    """Set up the agent subcommand parser."""
    parser.add_argument("prompt", type=str, help="Prompt to send to the autonomous agent")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model to use")
    parser.add_argument("--layer", type=int, help="Layer to use (overrides parameter file)")
    parser.add_argument(
        "--agent-strategy",
        type=str,
        default="synthetic_pairs_classifier_steering",
        choices=["synthetic_pairs_classifier_steering"],
        help="Agent strategy to use (default: synthetic_pairs_classifier_steering)"
    )
    parser.add_argument(
        "--quality-threshold", type=float, default=0.3, help="Quality threshold for classifiers (default: 0.3)"
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=10.0,
        help="Time budget in minutes for creating classifiers (default: 10.0)",
    )
    parser.add_argument("--max-attempts", type=int, default=3, help="Maximum improvement attempts (default: 3)")
    parser.add_argument(
        "--max-classifiers", type=int, default=None, help="Maximum classifiers to use (default: no limit)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # Synthetic pair generation arguments
    parser.add_argument(
        "--num-pairs", type=int, default=None, help="Number of pairs to generate (default: based on time budget)"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Similarity threshold for deduplication (0-1, higher = more strict)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=4, help="Number of parallel workers for generation (default: 4)"
    )

    # Activation collection arguments
    parser.add_argument(
        "--token-aggregation",
        type=str,
        default="average",
        choices=["average", "final", "first", "max", "min", "max_score"],
        help="How to aggregate token activations. 'max_score' uses highest token hallucination score (default: average)",
    )
    parser.add_argument(
        "--prompt-strategy",
        type=str,
        default="chat_template",
        choices=["chat_template", "direct_completion", "instruction_following", "multiple_choice", "role_playing"],
        help="Prompt construction strategy (default: chat_template)",
    )
    parser.add_argument(
        "--normalize-layers", action="store_true", help="Normalize layer activations"
    )
    parser.add_argument(
        "--return-full-sequence", action="store_true", help="Return full activation sequence instead of aggregated"
    )

    # Classifier training arguments
    parser.add_argument(
        "--classifier-epochs", type=int, default=50, help="Number of epochs for classifier training (default: 50)"
    )
    parser.add_argument(
        "--classifier-lr", type=float, default=1e-3, help="Learning rate for classifier training (default: 1e-3)"
    )
    parser.add_argument(
        "--classifier-batch-size",
        type=int,
        default=None,
        help="Batch size for classifier training (default: adaptive based on data size)",
    )
    parser.add_argument(
        "--classifier-type",
        type=str,
        default="logistic",
        choices=["logistic", "mlp"],
        help="Type of classifier to use (default: logistic)",
    )

    # Steering method arguments - uses centralized registry
    from wisent.core.steering_methods import SteeringMethodRegistry
    SteeringMethodRegistry.add_all_cli_arguments(parser)
    parser.add_argument(
        "--steering-strength", type=float, default=1.0, help="Strength of steering vector application (default: 1.0)"
    )
    parser.add_argument("--steering-mode", action="store_true", help="Enable steering mode")

    # Normalization parameters
    parser.add_argument("--normalize-mode", action="store_true", help="Enable normalization of steering vectors")
    parser.add_argument(
        "--normalization-method",
        type=str,
        default="none",
        choices=["none", "l2_unit", "l2_norm", "max_norm"],
        help="Normalization method for steering vectors (default: none)",
    )
    parser.add_argument("--target-norm", type=float, default=None, help="Target norm for steering vectors")

    # Quality Control System parameters
    parser.add_argument(
        "--enable-quality-control",
        action="store_true",
        default=True,
        help="Enable new quality control system (default: True)",
    )
    parser.add_argument(
        "--max-quality-attempts",
        type=int,
        default=5,
        help="Maximum attempts to achieve acceptable quality (default: 5)",
    )
    parser.add_argument(
        "--show-parameter-reasoning", action="store_true", help="Display model's reasoning for parameter choices"
    )

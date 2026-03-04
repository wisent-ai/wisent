"""Parser setup for the 'agent' command."""


def setup_agent_parser(parser):
    """Set up the agent subcommand parser."""
    parser.add_argument("prompt", type=str, help="Prompt to send to the autonomous agent")
    parser.add_argument("--model", type=str, required=True, help="Model to use")
    parser.add_argument("--layer", type=int, help="Layer to use (overrides parameter file)")
    parser.add_argument(
        "--agent-strategy",
        type=str,
        required=True,
        choices=["synthetic_pairs_classifier_steering"],
        help="Agent strategy to use"
    )
    parser.add_argument(
        "--quality-threshold", type=float, default=None, help="Quality threshold for classifiers"
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=None,
        help="Time budget in minutes for creating classifiers",
    )
    parser.add_argument("--max-attempts", type=int, default=None, help="Maximum improvement attempts")
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
        help="Similarity threshold for deduplication (higher = more strict)",
    )
    parser.add_argument(
        "--max-workers", type=int, required=True, help="Number of parallel workers for generation"
    )
    parser.add_argument(
        "--agent-synth-min-pairs", type=int, required=True,
        help="Minimum number of synthetic pairs to generate"
    )
    parser.add_argument(
        "--agent-synth-time-multiplier", type=int, required=True,
        help="Multiplier for time budget to determine pair count"
    )

    # Data infrastructure parameters
    parser.add_argument(
        "--generate-pairs-min-tokens", type=int, required=True,
        help="Minimum tokens for pair generation"
    )
    parser.add_argument(
        "--simhash-default-threshold-bits", type=int, required=True,
        help="SimHash threshold bits for deduplication"
    )
    parser.add_argument(
        "--tokens-per-pair-estimate", type=int, required=True,
        help="Estimated tokens per contrastive pair"
    )
    parser.add_argument(
        "--tokens-base-offset", type=int, required=True,
        help="Base token offset for generation"
    )
    parser.add_argument(
        "--trait-label-max-length", type=int, required=True,
        help="Maximum length for trait labels"
    )
    parser.add_argument(
        "--dedup-word-ngram", type=int, required=True,
        help="Word n-gram size for deduplication"
    )
    parser.add_argument(
        "--dedup-char-ngram", type=int, required=True,
        help="Character n-gram size for deduplication"
    )
    parser.add_argument(
        "--simhash-num-bands", type=int, required=True,
        help="Number of bands for SimHash LSH"
    )
    parser.add_argument(
        "--fast-diversity-seed", type=int, required=True,
        help="Random seed for diversity computation"
    )
    parser.add_argument(
        "--diversity-max-sample-size", type=int, required=True,
        help="Maximum sample size for diversity metrics"
    )
    parser.add_argument(
        "--retry-multiplier", type=int, required=True,
        help="Multiplier for max generation retry attempts"
    )

    # Activation collection arguments
    parser.add_argument(
        "--normalize-layers", action="store_true", help="Normalize layer activations"
    )
    parser.add_argument(
        "--return-full-sequence", action="store_true", help="Return full activation sequence instead of aggregated"
    )

    # Classifier training arguments
    parser.add_argument(
        "--classifier-epochs", type=int, default=None, help="Number of epochs for classifier training"
    )
    parser.add_argument(
        "--classifier-lr", type=float, default=None, help="Learning rate for classifier training"
    )
    parser.add_argument(
        "--classifier-batch-size",
        type=int,
        required=True,
        help="Batch size for classifier training",
    )
    parser.add_argument(
        "--classifier-test-size",
        type=float,
        required=True,
        help="Fraction of data to hold out for testing",
    )
    parser.add_argument(
        "--classifier-type",
        type=str,
        required=True,
        choices=["logistic", "mlp"],
        help="Type of classifier to use",
    )

    # Steering method arguments - uses centralized registry
    from wisent.core.control.steering_methods import SteeringMethodRegistry
    SteeringMethodRegistry.add_all_cli_arguments(parser)
    parser.add_argument(
        "--steering-strength", type=float, required=True, help="Strength of steering vector application"
    )
    parser.add_argument("--steering-mode", action="store_true", help="Enable steering mode")

    # Normalization parameters
    parser.add_argument("--normalize-mode", action="store_true", help="Enable normalization of steering vectors")
    parser.add_argument(
        "--normalization-method",
        type=str,
        required=True,
        choices=["none", "l2_unit", "l2_norm", "max_norm"],
        help="Normalization method for steering vectors",
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
        default=None,
        help="Maximum attempts to achieve acceptable quality",
    )
    parser.add_argument(
        "--show-parameter-reasoning", action="store_true", help="Display model's reasoning for parameter choices"
    )

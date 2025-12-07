"""Parser setup for the 'generate-vector' command."""


def setup_generate_vector_parser(parser):
    """Set up the generate-vector subcommand parser."""
    # Source of contrastive pairs - mutually exclusive for single property
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument(
        "--from-pairs",
        type=str,
        metavar="FILE",
        help="Path to JSON file containing contrastive pairs (single property)",
    )
    source_group.add_argument(
        "--from-description",
        type=str,
        metavar="TRAIT",
        help="Natural language description of the trait (single property)",
    )

    # Multi-property support
    parser.add_argument("--multi-property", action="store_true", help="Enable multi-property steering")
    parser.add_argument(
        "--property-files",
        type=str,
        nargs="+",
        metavar="NAME:FILE:LAYER",
        help="Property definitions from files (format: property_name:pairs_file:layer)",
    )
    parser.add_argument(
        "--property-descriptions",
        type=str,
        nargs="+",
        metavar="NAME:DESC:LAYER",
        help="Property definitions from descriptions (format: property_name:description:layer)",
    )

    # Model configuration
    parser.add_argument("--model", type=str, default="distilgpt2", help="Model name or path (default: distilgpt2)")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (default: auto-detect)")

    # Steering method configuration - uses centralized registry
    from wisent.core.steering_methods import SteeringMethodRegistry
    methods = SteeringMethodRegistry.list_methods()
    parser.add_argument(
        "--method",
        type=str,
        default=SteeringMethodRegistry.get_default_method().upper(),
        choices=[m.upper() for m in methods] + [m.lower() for m in methods],
        help=f"Steering method to use (default: {SteeringMethodRegistry.get_default_method().upper()})",
    )
    parser.add_argument("--layer", type=int, default=0, help="Layer index to apply steering (default: 0)")
    
    # Add method-specific arguments from registry
    for definition in SteeringMethodRegistry.list_definitions():
        definition.add_cli_arguments(parser)

    # Output configuration
    parser.add_argument("--output", type=str, required=True, help="Output path for the generated steering vector")

    # Pair generation options (only used with --from-description)
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=30,
        help="Number of pairs to generate when using --from-description (default: 30)",
    )
    parser.add_argument(
        "--save-pairs", type=str, default=None, help="Save generated pairs to this file when using --from-description"
    )

    # Activation extraction configuration
    parser.add_argument(
        "--prompt-construction",
        type=str,
        default="multiple_choice",
        choices=["multiple_choice", "role_playing", "direct_completion", "instruction_following"],
        help="Strategy for constructing prompts from question-answer pairs (default: multiple_choice)",
    )
    parser.add_argument(
        "--token-targeting",
        type=str,
        default="choice_token",
        choices=["choice_token", "continuation_token", "last_token", "first_token", "mean_pooling", "max_pooling"],
        help="Strategy for targeting tokens in activation extraction (default: choice_token)",
    )

    # General options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

"""Parser setup for the 'synthetic' command."""
from wisent.core.utils.config_tools.constants import NONSENSE_MAX_WORD_LENGTH


def setup_synthetic_parser(parser):
    """Set up the synthetic subcommand parser."""
    # Either generate new pairs or load existing ones
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--trait", type=str, help="Natural language description of the desired trait or behavior (generates new pairs)"
    )
    group.add_argument("--pairs-file", type=str, help="Path to existing JSON file with contrastive pairs")

    # Generation parameters (only used if --trait is specified)
    parser.add_argument(
        "--num-pairs",
        type=int,
        required=True,
        help="Number of contrastive pairs to generate (only used with --trait)",
    )
    parser.add_argument(
        "--save-pairs",
        type=str,
        default=None,
        help="Save generated pairs to this file (optional, only used with --trait)",
    )

    # Model and device
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--device", type=str, default=None, help="Device to run on")

    # Training/evaluation parameters
    parser.add_argument("--layer", type=str, required=True, help="Layer(s) to extract activations from")
    parser.add_argument(
        "--steering-method",
        type=str,
        required=True,
        choices=["CAA"],
        help="Steering method to use",
    )
    parser.add_argument("--steering-strength", type=float, required=True, help="Strength of steering vector application")
    parser.add_argument(
        "--test-questions", type=int, required=True, help="Number of test questions to generate for evaluation"
    )

    # Output
    parser.add_argument("--output", type=str, required=True, help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--timing", action="store_true", help="Show timing information")
    parser.add_argument("--similarity-threshold", type=float, required=True, help="Similarity threshold for deduplication")
    parser.add_argument("--intermediate-dir", type=str, default=None, help="Directory for intermediate files")
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate files after completion")
    parser.add_argument("--layers", type=str, required=True, help="Layers to extract activations from")

    # Steering method configuration - uses centralized registry
    from wisent.core.control.steering_methods import SteeringMethodRegistry
    methods = SteeringMethodRegistry.list_methods()
    parser.add_argument(
        "--method",
        type=str,
        default=SteeringMethodRegistry.get_default_method().upper(),
        choices=[m.upper() for m in methods] + [m.lower() for m in methods],
        help=f"Steering vector creation method (default: {SteeringMethodRegistry.get_default_method().upper()})",
    )
    parser.add_argument("--normalize", action="store_true", help="Normalize steering vectors")

    # Nonsense generation options (for creating nonsense pairs)
    parser.add_argument(
        "--nonsense",
        action="store_true",
        help="Generate nonsense contrastive pairs (negative responses are gibberish/nonsense)",
    )
    parser.add_argument(
        "--nonsense-mode",
        type=str,
        choices=["random_chars", "repetitive", "word_salad", "mixed"],
        required=True,
        help="Type of nonsense to generate: 'random_chars' (ahsdhashdahsdha), 'repetitive' (the the the), 'word_salad' (real words, no meaning), 'mixed' (combination)",
    )

    # Nonsense detection options (for detecting nonsense in responses)
    parser.add_argument(
        "--enable-nonsense-detection",
        action="store_true",
        help="Enable nonsense detection to stop lobotomized responses",
    )
    parser.add_argument(
        "--max-word-length",
        type=int,
        default=NONSENSE_MAX_WORD_LENGTH,
        help="Maximum reasonable word length for nonsense detection (default: 20)",
    )
    parser.add_argument(
        "--repetition-threshold",
        type=float,
        required=True,
        help="Threshold for repetitive content detection",
    )
    parser.add_argument(
        "--gibberish-threshold",
        type=float,
        required=True,
        help="Threshold for gibberish word detection",
    )
    parser.add_argument(
        "--disable-dictionary-check",
        action="store_true",
        help="Disable dictionary-based word validation (faster but less accurate)",
    )
    parser.add_argument(
        "--nonsense-action",
        type=str,
        required=True,
        choices=["regenerate", "stop", "flag"],
        help="Action when nonsense is detected: regenerate, stop generation, or flag for review",
    )

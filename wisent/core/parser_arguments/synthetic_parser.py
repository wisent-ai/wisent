"""Parser setup for the 'synthetic' command."""


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
        default=30,
        help="Number of contrastive pairs to generate (default: 30, only used with --trait)",
    )
    parser.add_argument(
        "--save-pairs",
        type=str,
        default=None,
        help="Save generated pairs to this file (optional, only used with --trait)",
    )

    # Model and device
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or path")
    parser.add_argument("--device", type=str, default=None, help="Device to run on")

    # Training/evaluation parameters
    parser.add_argument("--layer", type=str, default="15", help="Layer(s) to extract activations from")
    parser.add_argument(
        "--steering-method",
        type=str,
        default="CAA",
        choices=["CAA", "HPR", "DAC", "BiPO", "KSteering"],
        help="Steering method to use",
    )
    parser.add_argument("--steering-strength", type=float, default=1.0, help="Strength of steering vector application")
    parser.add_argument(
        "--test-questions", type=int, default=5, help="Number of test questions to generate for evaluation"
    )

    # Output
    parser.add_argument("--output", type=str, default="./results", help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # K-Steering specific parameters
    parser.add_argument(
        "--ksteering-target-labels", type=str, default="0", help="Comma-separated target label indices for K-steering"
    )
    parser.add_argument(
        "--ksteering-avoid-labels", type=str, default="", help="Comma-separated avoid label indices for K-steering"
    )
    parser.add_argument("--ksteering-alpha", type=float, default=50.0, help="Alpha parameter for K-steering")

    # Nonsense detection options
    parser.add_argument(
        "--enable-nonsense-detection",
        action="store_true",
        help="Enable nonsense detection to stop lobotomized responses",
    )
    parser.add_argument(
        "--max-word-length",
        type=int,
        default=20,
        help="Maximum reasonable word length for nonsense detection (default: 20)",
    )
    parser.add_argument(
        "--repetition-threshold",
        type=float,
        default=0.7,
        help="Threshold for repetitive content detection (0-1, default: 0.7)",
    )
    parser.add_argument(
        "--gibberish-threshold",
        type=float,
        default=0.3,
        help="Threshold for gibberish word detection (0-1, default: 0.3)",
    )
    parser.add_argument(
        "--disable-dictionary-check",
        action="store_true",
        help="Disable dictionary-based word validation (faster but less accurate)",
    )
    parser.add_argument(
        "--nonsense-action",
        type=str,
        default="regenerate",
        choices=["regenerate", "stop", "flag"],
        help="Action when nonsense is detected: regenerate, stop generation, or flag for review",
    )

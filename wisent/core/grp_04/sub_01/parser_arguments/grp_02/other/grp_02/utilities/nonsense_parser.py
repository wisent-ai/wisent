"""Parser setup for the 'test-nonsense' command."""

from wisent.core.constants import NONSENSE_REPETITION_THRESHOLD, NONSENSE_GIBBERISH_THRESHOLD, NONSENSE_MAX_WORD_LENGTH


def setup_test_nonsense_parser(parser):
    """Set up the test-nonsense subcommand parser."""
    parser.add_argument(
        "text", type=str, nargs="?", help="Text to analyze (if not provided, will use interactive mode)"
    )
    parser.add_argument("--max-word-length", type=int, default=NONSENSE_MAX_WORD_LENGTH, help="Maximum reasonable word length (default: 20)")
    parser.add_argument(
        "--repetition-threshold",
        type=float,
        default=NONSENSE_REPETITION_THRESHOLD,
        help="Threshold for repetitive content detection (0-1, default: 0.7)",
    )
    parser.add_argument(
        "--gibberish-threshold",
        type=float,
        default=NONSENSE_GIBBERISH_THRESHOLD,
        help="Threshold for gibberish word detection (0-1, default: 0.3)",
    )
    parser.add_argument(
        "--disable-dictionary-check", action="store_true", help="Disable dictionary-based word validation"
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed analysis")
    parser.add_argument("--examples", action="store_true", help="Test with built-in example texts")

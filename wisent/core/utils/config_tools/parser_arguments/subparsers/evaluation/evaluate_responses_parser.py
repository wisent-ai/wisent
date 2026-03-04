"""Parser setup for the 'evaluate-responses' command."""


def setup_evaluate_responses_parser(parser):
    """Set up the evaluate-responses command parser."""
    parser.add_argument("--input", type=str, required=True, help="Input JSON file with generated responses")
    parser.add_argument("--baseline", type=str, help="Baseline responses JSON file (for personalization comparison)")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file for evaluation results")
    parser.add_argument("--task", type=str, help="Task name (optional, overrides task from input JSON)")
    parser.add_argument("--trait", type=str, help="Personality trait to evaluate (optional, for personalization tasks)")
    parser.add_argument("--trait-description", type=str, help="Description of the personality trait")
    parser.add_argument("--subprocess-timeout", type=int, required=True,
                        help="Timeout in seconds for subprocess execution (e.g. Docker code evaluation)")
    parser.add_argument("--personalization-good-threshold", type=int, required=True,
                        help="Threshold score for good personalization")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument(
        "--fast-diversity-seed", type=int, required=True,
        help="Seed for fast diversity computation"
    )
    parser.add_argument(
        "--diversity-max-sample-size", type=int, required=True,
        help="Maximum sample size for diversity computation"
    )
    parser.add_argument(
        "--min-sentence-length", type=int, required=True,
        help="Minimum sentence length for coherence evaluation"
    )
    parser.add_argument(
        "--nonsense-min-tokens", type=int, required=True,
        help="Minimum token count for nonsense word detection"
    )
    parser.add_argument(
        "--quality-min-response-length", type=int, required=True,
        help="Minimum response length for quality scoring"
    )
    parser.add_argument(
        "--quality-repetition-ratio-threshold", type=float, required=True,
        help="Threshold for repetitive token ratio penalty"
    )
    parser.add_argument(
        "--quality-bigram-repeat-threshold", type=int, required=True,
        help="Threshold for repeated bigram count penalty"
    )
    parser.add_argument(
        "--quality-bigram-repeat-penalty", type=float, required=True,
        help="Penalty multiplier for repeated bigrams"
    )
    parser.add_argument(
        "--quality-special-char-ratio-threshold", type=float, required=True,
        help="Threshold for special character ratio penalty"
    )
    parser.add_argument(
        "--quality-special-char-penalty", type=float, required=True,
        help="Penalty multiplier for excessive special characters"
    )
    parser.add_argument(
        "--quality-char-repeat-count", type=int, required=True,
        help="Minimum consecutive character repeats to trigger penalty"
    )
    parser.add_argument(
        "--quality-char-repeat-penalty", type=float, required=True,
        help="Penalty multiplier for character repetition"
    )

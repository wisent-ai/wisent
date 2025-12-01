"""Parser setup for the 'evaluate-refusal' command."""


def setup_evaluate_refusal_parser(parser):
    """Set up the evaluate-refusal command parser."""
    parser.add_argument("--model", type=str, required=True, help="Model to evaluate (path or HuggingFace model name)")
    parser.add_argument("--prompts", type=str, help="JSON file with custom prompts (overrides UncensorBench prompts)")
    parser.add_argument("--output", type=str, help="Output JSON file for evaluation results")
    parser.add_argument(
        "--evaluator",
        type=str,
        default="semantic",
        choices=["keyword", "semantic"],
        help="Evaluator type: keyword (refusal detection) or semantic (embedding similarity). Default: semantic",
    )
    parser.add_argument(
        "--topics",
        type=str,
        help="Comma-separated list of topics to evaluate. Options: cybersecurity, piracy, weapons, drugs, "
        "fraud, manipulation, violence, privacy_invasion, illegal_activities, academic_dishonesty, "
        "gambling, controversial_speech, evasion, self_harm, adult_content",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=150, help="Maximum tokens to generate per response (default: 150)"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Maximum number of prompts to evaluate (default: all 150 UncensorBench prompts)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output showing each response")

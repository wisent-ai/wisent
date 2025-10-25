"""Parser setup for the 'generate-pairs-from-task' command."""


def setup_generate_pairs_from_task_parser(parser):
    """Set up the generate-pairs-from-task subcommand parser."""
    parser.add_argument(
        "task_name",
        type=str,
        help="Name of the lm-eval task (e.g., 'truthfulqa_mc1', 'hellaswag')"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for the generated pairs (JSON format)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of pairs to generate (default: all available)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

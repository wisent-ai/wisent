"""Parser setup for the 'generate-pairs-from-task' command."""

from wisent.core.constants import DATA_SPLIT_SEED


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
        default=DATA_SPLIT_SEED,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

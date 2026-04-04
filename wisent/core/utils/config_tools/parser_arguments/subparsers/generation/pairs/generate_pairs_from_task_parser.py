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
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of contrastive pairs to generate (default: all)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.5,
        dest="train_ratio",
        help="Train/test split ratio (default: 0.5)"
    )
    parser.add_argument(
        "--allow-subtasks",
        action="store_true",
        dest="allow_subtasks",
        help="Accept subtasks whose parent is a working benchmark"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

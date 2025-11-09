"""Parser for diagnose-pairs command."""


def setup_diagnose_pairs_parser(parser):
    """Set up the diagnose-pairs command parser."""
    parser.add_argument(
        'pairs_file',
        type=str,
        help='Path to JSON file containing contrastive pairs'
    )

    parser.add_argument(
        '--show-sample',
        action='store_true',
        help='Show a sample pair from the file'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser.set_defaults(command='diagnose-pairs')
    return parser

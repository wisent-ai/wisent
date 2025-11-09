"""Parser for diagnose-vectors command."""


def setup_diagnose_vectors_parser(parser):
    """Set up the diagnose-vectors command parser."""
    parser.add_argument(
        'vectors_file',
        type=str,
        help='Path to JSON file containing steering vectors'
    )

    parser.add_argument(
        '--show-sample',
        action='store_true',
        help='Show sample vector values from the first layer'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser.set_defaults(command='diagnose-vectors')
    return parser

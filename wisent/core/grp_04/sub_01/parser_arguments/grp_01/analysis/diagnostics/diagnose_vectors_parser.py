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

    parser.add_argument(
        '--check-cone',
        action='store_true',
        help='Check for cone structure (requires --activations-file)'
    )

    parser.add_argument(
        '--activations-file',
        type=str,
        default=None,
        help='Path to file containing positive/negative activations for cone analysis (.pt or .json)'
    )

    parser.add_argument(
        '--cone-threshold',
        type=float,
        default=0.7,
        help='Threshold for cone detection score (default: 0.7)'
    )

    parser.add_argument(
        '--cone-directions',
        type=int,
        default=5,
        help='Number of cone directions to search for (default: 5)'
    )

    parser.add_argument(
        '--detect-geometry',
        action='store_true',
        help='Detect geometric structure of activations (requires --activations-file)'
    )

    parser.add_argument(
        '--max-clusters',
        type=int,
        default=5,
        help='Maximum clusters to try for cluster detection (default: 5)'
    )

    parser.add_argument(
        '--manifold-neighbors',
        type=int,
        default=10,
        help='Number of neighbors for manifold analysis (default: 10)'
    )

    parser.set_defaults(command='diagnose-vectors')
    return parser

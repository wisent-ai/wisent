"""Parser for check-linearity command."""


def setup_check_linearity_parser(parser):
    """Set up the check-linearity command parser."""
    parser.add_argument(
        'pairs_file',
        type=str,
        help='Path to JSON file containing contrastive pairs'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='meta-llama/Llama-3.2-1B-Instruct',
        help='Model to use for activation collection'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run model on (cuda, mps, cpu)'
    )
    
    parser.add_argument(
        '--max-pairs',
        type=int,
        default=50,
        help='Maximum number of pairs to use for analysis'
    )
    
    parser.add_argument(
        '--linear-threshold',
        type=float,
        default=0.7,
        help='Linear score threshold to declare LINEAR (default: 0.7)'
    )
    
    parser.add_argument(
        '--weak-threshold',
        type=float,
        default=0.5,
        help='Linear score threshold to declare WEAKLY_LINEAR (default: 0.5)'
    )
    
    parser.add_argument(
        '--min-cohens-d',
        type=float,
        default=1.0,
        help='Minimum Cohen\'s d for meaningful separation (default: 1.0)'
    )
    
    parser.add_argument(
        '--layers',
        type=str,
        default=None,
        help='Comma-separated layer indices to test (default: auto-select)'
    )
    
    parser.add_argument(
        '--optimization-steps',
        type=int,
        default=50,
        help='Optimization steps for geometry detection (default: 50)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path for results JSON'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed results for all configurations'
    )
    
    parser.set_defaults(command='check-linearity')
    return parser

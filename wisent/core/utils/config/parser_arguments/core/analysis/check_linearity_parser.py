"""Parser for check-linearity command."""

from wisent.core.activations import ExtractionStrategy
from wisent.core.constants import DIAG_OPTIMIZATION_STEPS


def setup_check_linearity_parser(parser):
    """Set up the check-linearity command parser."""
    parser.add_argument(
        'pairs_file',
        type=str,
        help='Path to JSON file containing contrastive pairs'
    )
    
    parser.add_argument(
        '--extraction-strategy',
        type=str,
        default=None,
        choices=ExtractionStrategy.list_all(),
        help=f'Extraction strategy to use. If not specified, tests multiple strategies. Options: {", ".join(ExtractionStrategy.list_all())}'
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
        default='auto',
        help='Device to run model on (auto, cuda, mps, cpu)'
    )
    
    parser.add_argument(
        '--max-pairs',
        type=int,
        required=True,
        help='Maximum number of pairs to use for analysis'
    )
    
    parser.add_argument(
        '--linear-threshold',
        type=float,
        required=True,
        help='Linear score threshold to declare LINEAR'
    )
    
    parser.add_argument(
        '--weak-threshold',
        type=float,
        required=True,
        help='Linear score threshold to declare WEAKLY_LINEAR'
    )
    
    parser.add_argument(
        '--min-cohens-d',
        type=float,
        required=True,
        help='Minimum Cohen\'s d for meaningful separation'
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
        default=DIAG_OPTIMIZATION_STEPS,
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

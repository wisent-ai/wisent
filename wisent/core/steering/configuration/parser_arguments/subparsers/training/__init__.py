"""Training-related parser arguments."""

from .synthetic_parser import setup_synthetic_parser
from .train_unified_goodness_parser import setup_train_unified_goodness_parser
from .get_activations_parser import setup_get_activations_parser

__all__ = [
    'setup_synthetic_parser',
    'setup_train_unified_goodness_parser',
    'setup_get_activations_parser',
]

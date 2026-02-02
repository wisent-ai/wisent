"""Generation-related parser arguments."""

from .pairs import (
    setup_generate_pairs_parser,
    setup_generate_pairs_from_task_parser,
    setup_generate_responses_parser,
)
from .vectors import (
    setup_generate_vector_parser,
    setup_generate_vector_from_task_parser,
    setup_generate_vector_from_synthetic_parser,
)

__all__ = [
    'setup_generate_pairs_parser',
    'setup_generate_pairs_from_task_parser',
    'setup_generate_responses_parser',
    'setup_generate_vector_parser',
    'setup_generate_vector_from_task_parser',
    'setup_generate_vector_from_synthetic_parser',
]

"""Vector generation parser arguments."""

from .generate_vector_parser import setup_generate_vector_parser
from .generate_vector_from_task_parser import setup_generate_vector_from_task_parser
from .generate_vector_from_synthetic_parser import setup_generate_vector_from_synthetic_parser

__all__ = [
    'setup_generate_vector_parser',
    'setup_generate_vector_from_task_parser',
    'setup_generate_vector_from_synthetic_parser',
]

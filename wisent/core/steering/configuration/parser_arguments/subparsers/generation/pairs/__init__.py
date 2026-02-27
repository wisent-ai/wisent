"""Pair generation parser arguments."""

from .generate_pairs_parser import setup_generate_pairs_parser
from .generate_pairs_from_task_parser import setup_generate_pairs_from_task_parser
from .generate_responses_parser import setup_generate_responses_parser

__all__ = [
    'setup_generate_pairs_parser',
    'setup_generate_pairs_from_task_parser',
    'setup_generate_responses_parser',
]

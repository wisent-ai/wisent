"""Diagnostic analysis parser arguments."""

from .diagnose_pairs_parser import setup_diagnose_pairs_parser
from .diagnose_vectors_parser import setup_diagnose_vectors_parser

__all__ = [
    'setup_diagnose_pairs_parser',
    'setup_diagnose_vectors_parser',
]

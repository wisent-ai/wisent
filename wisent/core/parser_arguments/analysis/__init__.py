"""Analysis-related parser arguments."""

from .check_linearity_parser import setup_check_linearity_parser
from .cluster_benchmarks_parser import setup_cluster_benchmarks_parser
from .geometry_search_parser import setup_geometry_search_parser
from .repscan_parser import setup_repscan_parser
from .diagnostics import (
    setup_diagnose_pairs_parser,
    setup_diagnose_vectors_parser,
)

__all__ = [
    'setup_check_linearity_parser',
    'setup_cluster_benchmarks_parser',
    'setup_geometry_search_parser',
    'setup_repscan_parser',
    'setup_diagnose_pairs_parser',
    'setup_diagnose_vectors_parser',
]

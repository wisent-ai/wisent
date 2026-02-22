import os as _os
_base = _os.path.dirname(__file__)
for _root, _dirs, _files in _os.walk(_base):
    if _root != _base:
        __path__.append(_root)

"""
Steering optimization utilities.

This module provides metrics and utilities for steering optimization.
The main optimization logic is in wisent.core.cli.optimize_steering.
"""

from .metrics import (
    calculate_comprehensive_metrics,
    evaluate_benchmark_performance,
    evaluate_probe_performance,
    generate_performance_summary,
)

__all__ = [
    "calculate_comprehensive_metrics",
    "evaluate_benchmark_performance",
    "evaluate_probe_performance",
    "generate_performance_summary",
]

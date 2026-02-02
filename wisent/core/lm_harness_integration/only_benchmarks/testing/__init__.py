"""Benchmark testing functions and CLI."""

from __future__ import annotations

from .tests import (
    test_single_benchmark_direct,
    test_benchmark_creation,
    extract_contrastive_pairs_from_output,
    test_readme_updates,
    test_benchmark_matching,
)
from .cli import main, cli_entry


__all__ = [
    "test_single_benchmark_direct",
    "test_benchmark_creation",
    "extract_contrastive_pairs_from_output",
    "test_readme_updates",
    "test_benchmark_matching",
    "main",
    "cli_entry",
]

"""Test all benchmarks to verify extractor and evaluator work."""

import sys
import signal
from contextlib import contextmanager
from wisent.support.examples.scripts.discovery.validation.test_one_benchmark import test_benchmark


class TimeoutError(Exception):
    """Raised when a test times out."""
    pass


@contextmanager
def timeout(seconds):
    """Context manager (timeout functionality removed)."""
    yield

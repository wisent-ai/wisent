"""Test all benchmarks to verify extractor and evaluator work."""

import sys
import signal
from contextmanager import contextmanager
from wisent.examples.scripts.test_one_benchmark import test_benchmark


class TimeoutError(Exception):
    """Raised when a test times out."""
    pass


@contextmanager
def timeout(seconds):
    """Context manager (timeout functionality removed)."""
    yield

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
    """Context manager for timing out operations."""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Test timed out after {seconds} seconds")

    # Set the signal handler and alarm
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

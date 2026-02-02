"""Utility parser arguments."""

from .monitor_parser import setup_monitor_parser
from .nonsense_parser import setup_test_nonsense_parser
from .tasks_parser import setup_tasks_parser

__all__ = [
    'setup_monitor_parser',
    'setup_test_nonsense_parser',
    'setup_tasks_parser',
]

"""CLI execution logic for Wisent commands."""

from .tasks import execute_tasks
from .generate_pairs_from_task import execute_generate_pairs_from_task
from .generate_pairs import execute_generate_pairs
from .get_activations import execute_get_activations

__all__ = ['execute_tasks', 'execute_generate_pairs_from_task', 'execute_generate_pairs', 'execute_get_activations']

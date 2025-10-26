"""CLI execution logic for Wisent commands."""

from .tasks import execute_tasks
from .generate_pairs_from_task import execute_generate_pairs_from_task
from .generate_pairs import execute_generate_pairs
from .get_activations import execute_get_activations
from .create_steering_vector import execute_create_steering_vector
from .generate_vector_from_task import execute_generate_vector_from_task
from .generate_vector_from_synthetic import execute_generate_vector_from_synthetic
from .optimize_classification import execute_optimize_classification
from .optimize_steering import execute_optimize_steering

__all__ = ['execute_tasks', 'execute_generate_pairs_from_task', 'execute_generate_pairs', 'execute_get_activations', 'execute_create_steering_vector', 'execute_generate_vector_from_task', 'execute_generate_vector_from_synthetic', 'execute_optimize_classification', 'execute_optimize_steering']

"""CLI execution logic for Wisent commands."""

from .tasks import execute_tasks
from .generate_pairs_from_task import execute_generate_pairs_from_task
from .generate_pairs import execute_generate_pairs
from .diagnose_pairs import execute_diagnose_pairs
from .get_activations import execute_get_activations
from .diagnose_vectors import execute_diagnose_vectors
from .create_steering_vector import execute_create_steering_vector
from .generate_vector_from_task import execute_generate_vector_from_task
from .generate_vector_from_synthetic import execute_generate_vector_from_synthetic
from .optimize_classification import execute_optimize_classification
from .optimize_steering import execute_optimize_steering
from .optimize_sample_size import execute_optimize_sample_size
from .generate_responses import execute_generate_responses
from .evaluate_responses import execute_evaluate_responses
from .multi_steer import execute_multi_steer
from .agent import execute_agent
from .modify_weights import execute_modify_weights
from .evaluate_refusal import execute_evaluate_refusal
from .inference_config_cli import execute_inference_config
from .optimization_cache import execute_optimization_cache
from .optimize_weights import execute_optimize_weights
from .optimize import execute_optimize
from .geometry_search import execute_geometry_search

__all__ = ['execute_tasks', 'execute_generate_pairs_from_task', 'execute_generate_pairs', 'execute_diagnose_pairs', 'execute_get_activations', 'execute_diagnose_vectors', 'execute_create_steering_vector', 'execute_generate_vector_from_task', 'execute_generate_vector_from_synthetic', 'execute_optimize_classification', 'execute_optimize_steering', 'execute_optimize_sample_size', 'execute_generate_responses', 'execute_evaluate_responses', 'execute_multi_steer', 'execute_agent', 'execute_modify_weights', 'execute_evaluate_refusal', 'execute_inference_config', 'execute_optimization_cache', 'execute_optimize_weights', 'execute_optimize', 'execute_geometry_search']

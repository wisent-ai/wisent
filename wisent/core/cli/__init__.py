"""CLI execution logic for Wisent commands."""

# Analysis config
from .analysis.config.tasks import execute_tasks
from .analysis.config.inference_config_cli import execute_inference_config

# Analysis diagnosis
from .analysis.diagnosis.diagnose_pairs import execute_diagnose_pairs
from .analysis.diagnosis.diagnose_vectors import execute_diagnose_vectors
from .analysis.diagnosis.cluster_benchmarks import execute_cluster_benchmarks

# Analysis evaluation
from .analysis.evaluation.evaluate_responses import execute_evaluate_responses
from .analysis.evaluation.evaluate_refusal import execute_evaluate_refusal
from .analysis.evaluation.check_linearity import execute_check_linearity

# Analysis geometry
from .analysis.geometry.get_activations import execute_get_activations
from .analysis.geometry.geometry_search import execute_geometry_search
from .analysis.geometry.repscan import execute_repscan

# Analysis training
from .analysis.training.modify_weights import execute_modify_weights
from .analysis.training.train_unified_goodness import execute_train_unified_goodness

# Generation pairs
from .generation.pairs.generate_pairs_from_task import execute_generate_pairs_from_task
from .generation.pairs.generate_pairs import execute_generate_pairs

# Generation vectors
from .generation.vectors.generate_vector_from_task import execute_generate_vector_from_task
from .generation.vectors.generate_vector_from_synthetic import execute_generate_vector_from_synthetic
from .generation.vectors.generate_responses import execute_generate_responses

# Optimization core
from .optimization.core.optimize_steering import execute_optimize_steering
from .optimization.core.optimization_cache import execute_optimization_cache
from .optimization.core.optimize import execute_optimize

# Optimization specific
from .optimization.specific.optimize_classification import execute_optimize_classification
from .optimization.specific.optimize_sample_size import execute_optimize_sample_size
from .optimization.specific.optimize_weights import execute_optimize_weights

# Steering core
from .steering.core.create_steering_object import execute_create_steering_object
from .steering.core.multi_steer import execute_multi_steer
from .steering.core.verify_steering import execute_verify_steering
from .steering.core.discover_steering import execute_discover_steering

# Steering viz
from .steering.viz.steering_viz import execute_steering_viz
from .steering.viz.per_concept_steering_viz import execute_per_concept_steering_viz

# Agent
from .agent.main import execute_agent

__all__ = [
    'execute_tasks',
    'execute_generate_pairs_from_task',
    'execute_generate_pairs',
    'execute_diagnose_pairs',
    'execute_get_activations',
    'execute_diagnose_vectors',
    'execute_create_steering_object',
    'execute_generate_vector_from_task',
    'execute_generate_vector_from_synthetic',
    'execute_optimize_classification',
    'execute_optimize_steering',
    'execute_optimize_sample_size',
    'execute_generate_responses',
    'execute_evaluate_responses',
    'execute_multi_steer',
    'execute_agent',
    'execute_modify_weights',
    'execute_evaluate_refusal',
    'execute_inference_config',
    'execute_optimization_cache',
    'execute_optimize_weights',
    'execute_optimize',
    'execute_geometry_search',
    'execute_verify_steering',
    'execute_repscan',
    'execute_train_unified_goodness',
    'execute_check_linearity',
    'execute_cluster_benchmarks',
    'execute_steering_viz',
    'execute_per_concept_steering_viz',
    'execute_discover_steering',
]

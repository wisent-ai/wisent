import os as _os
_base = _os.path.dirname(__file__)
for _root, _dirs, _files in _os.walk(_base):
    _dirs[:] = sorted(d for d in _dirs if not d.startswith((".", "_")))
    if _root != _base:
        __path__.append(_root)

"""CLI execution logic for Wisent commands.

All imports are lazy to avoid circular import chains on Python 3.10
when this package is loaded as a side effect of namespace resolution.
"""

_IMPORT_MAP = {
    'execute_tasks': '.task_configs',
    'execute_inference_config': '.inference_config_cli',
    'execute_diagnose_pairs': '.diagnose_pairs',
    'execute_diagnose_vectors': '.diagnose_vectors',
    'execute_cluster_benchmarks': '.cluster_benchmarks',
    'execute_evaluate_responses': '.evaluate_responses',
    'execute_evaluate_refusal': '.evaluate_refusal',
    'execute_check_linearity': '.check_linearity',
    'execute_get_activations': '.get_activations',
    'execute_geometry_search': '.geometry_search',
    'execute_zwiad': '.zwiad',
    'execute_modify_weights': '.entry',
    'execute_train_unified_goodness': '.train_unified_goodness',
    'execute_generate_pairs_from_task': '.generate_pairs_from_task',
    'execute_generate_pairs': '.generate_pairs',
    'execute_generate_vector_from_task': '.generate_vector_from_task',
    'execute_generate_vector_from_synthetic': '.generate_vector_from_synthetic',
    'execute_generate_responses': '.generate_responses',
    'execute_optimize_steering': '.optimize_steering',
    'execute_optimization_cache': '.optimization_cache',
    'execute_optimize': '.optimize',
    'execute_optimize_classification': '.optimize_classification',
    'execute_optimize_sample_size': '.optimize_sample_size',
    'execute_optimize_weights': '.optimize_weights',
    'execute_tune_recommendation': '.tune_recommendation',
    'execute_create_steering_object': '.create_steering_object',
    'execute_multi_steer': '.multi_steer',
    'execute_verify_steering': '.verify_steering',
    'execute_discover_steering': '.discover_steering',
    'execute_steering_viz': '.steering_viz',
    'execute_per_concept_steering_viz': '.per_concept_steering_viz',
    'execute_agent': '.agent_entry',
    'execute_migrate_activations': '.migrate_activations',
}


def __getattr__(name):
    if name == 'execute_compare_steering':
        from wisent.core.reading.comparison import execute_compare_steering
        return execute_compare_steering
    if name in _IMPORT_MAP:
        import importlib
        mod = importlib.import_module(_IMPORT_MAP[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_IMPORT_MAP.keys()) + ['execute_compare_steering']

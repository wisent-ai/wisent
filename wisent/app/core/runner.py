"""Execute Wisent CLI commands from the Gradio interface.

Builds CLI argument lists, parses them through the standard argparse
setup, looks up handlers via the command dispatch map, and captures
stdout/stderr output.
"""

import io
import contextlib
import traceback


_COMMAND_MAP = {
    'tasks': 'execute_tasks',
    'generate-pairs': 'execute_generate_pairs',
    'diagnose-pairs': 'execute_diagnose_pairs',
    'generate-pairs-from-task': 'execute_generate_pairs_from_task',
    'get-activations': 'execute_get_activations',
    'diagnose-vectors': 'execute_diagnose_vectors',
    'create-steering-vector': 'execute_create_steering_object',
    'generate-vector-from-task': 'execute_generate_vector_from_task',
    'generate-vector-from-synthetic': 'execute_generate_vector_from_synthetic',
    'synthetic': 'execute_generate_vector_from_synthetic',
    'optimize-classification': 'execute_optimize_classification',
    'optimize-steering': 'execute_optimize_steering',
    'optimize-sample-size': 'execute_optimize_sample_size',
    'generate-responses': 'execute_generate_responses',
    'evaluate-responses': 'execute_evaluate_responses',
    'multi-steer': 'execute_multi_steer',
    'agent': 'execute_agent',
    'modify-weights': 'execute_modify_weights',
    'evaluate-refusal': 'execute_evaluate_refusal',
    'inference-config': 'execute_inference_config',
    'optimization-cache': 'execute_optimization_cache',
    'optimize-weights': 'execute_optimize_weights',
    'optimize-all': 'execute_optimize',
    'optimize': 'execute_optimize',
    'train-unified-goodness': 'execute_train_unified_goodness',
    'check-linearity': 'execute_check_linearity',
    'cluster-benchmarks': 'execute_cluster_benchmarks',
    'geometry-search': 'execute_geometry_search',
    'verify-steering': 'execute_verify_steering',
    'zwiad': 'execute_zwiad',
    'discover-steering': 'execute_discover_steering',
    'migrate-activations': 'execute_migrate_activations',
    'compare-steering': 'execute_compare_steering',
    'steering-viz': 'execute_steering_viz',
    'generate-vector': 'execute_generate_vector_from_synthetic',
}


def run_command(command_name, arg_list):
    """Run a Wisent CLI command and capture its output.

    Args:
        command_name: The CLI command name (e.g. 'generate-pairs').
        arg_list: List of CLI argument strings.

    Returns:
        Captured stdout + stderr as a single string.
    """
    import wisent.core.utils.cli as _cli
    from wisent.core.utils.config_tools.parser_arguments import setup_parser

    full_args = [command_name] + arg_list

    try:
        parser = setup_parser()
        args = parser.parse_args(full_args)
    except SystemExit:
        return f"Argument parsing failed for: wisent {' '.join(full_args)}"

    func_name = _resolve_func_name(command_name, args)
    if func_name is None:
        return f"Unknown command: {command_name}"

    try:
        handler = getattr(_cli, func_name)
    except AttributeError:
        return f"Handler not found: {func_name}"

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buf), \
                contextlib.redirect_stderr(stderr_buf):
            result = handler(args)
    except Exception:
        captured = stdout_buf.getvalue() + stderr_buf.getvalue()
        return captured + "\n" + traceback.format_exc()

    output = stdout_buf.getvalue()
    errors = stderr_buf.getvalue()
    combined = output
    if errors:
        combined += "\n--- stderr ---\n" + errors
    if result is not None:
        combined += "\n--- result ---\n" + str(result)
    if not combined.strip():
        combined = "Command completed successfully (no output)."
    return combined


def _resolve_func_name(command_name, args):
    """Resolve handler function name, handling special cases."""
    if command_name == 'steering-viz':
        if getattr(args, 'per_concept', False):
            return 'execute_per_concept_steering_viz'
        return 'execute_steering_viz'
    return _COMMAND_MAP.get(command_name)

"""
Main entry point for the Wisent CLI.

This module connects the argparse parser (wisent/core/parser_arguments/) to execution logic
and provides the main() function that serves as the CLI entry point.
"""

import sys
from pathlib import Path
from wisent.core.parser_arguments import setup_parser
from wisent.core.branding import print_banner
from wisent.core.cli import execute_tasks, execute_generate_pairs_from_task, execute_generate_pairs, execute_diagnose_pairs, execute_get_activations, execute_diagnose_vectors, execute_create_steering_vector, execute_generate_vector_from_task, execute_generate_vector_from_synthetic, execute_optimize_classification, execute_optimize_steering, execute_optimize_sample_size, execute_generate_responses, execute_evaluate_responses, execute_multi_steer, execute_agent, execute_modify_weights, execute_evaluate_refusal, execute_inference_config, execute_optimization_cache, execute_optimize_weights, execute_optimize
from wisent.core.cli.train_unified_goodness import execute_train_unified_goodness
from wisent.core.cli.check_linearity import execute_check_linearity
from wisent.core.cli.cluster_benchmarks import execute_cluster_benchmarks
from wisent.core.cli.geometry_search import execute_geometry_search


def _should_show_banner() -> bool:
    """Check if this is the first use and banner should be shown."""
    wisent_dir = Path.home() / ".wisent"
    banner_flag = wisent_dir / ".banner_shown"

    if banner_flag.exists():
        return False

    # First use - create the flag file
    wisent_dir.mkdir(parents=True, exist_ok=True)
    banner_flag.touch()
    return True


def main():
    """Main entry point for the Wisent CLI."""
    # Show banner only on first use
    if _should_show_banner():
        print_banner("Wisent CLI", width=64, use_color=True)

    # Parse arguments
    parser = setup_parser()
    args = parser.parse_args()

    # If no command specified, show help
    if not hasattr(args, 'command') or args.command is None:
        parser.print_help()
        sys.exit(0)

    # Execute based on command
    if args.command == 'tasks':
        execute_tasks(args)
    elif args.command == 'generate-pairs':
        execute_generate_pairs(args)
    elif args.command == 'diagnose-pairs':
        execute_diagnose_pairs(args)
    elif args.command == 'generate-pairs-from-task':
        execute_generate_pairs_from_task(args)
    elif args.command == 'get-activations':
        execute_get_activations(args)
    elif args.command == 'diagnose-vectors':
        execute_diagnose_vectors(args)
    elif args.command == 'create-steering-vector':
        execute_create_steering_vector(args)
    elif args.command == 'generate-vector-from-task':
        execute_generate_vector_from_task(args)
    elif args.command == 'generate-vector-from-synthetic' or args.command == 'synthetic':
        execute_generate_vector_from_synthetic(args)
    elif args.command == 'optimize-classification':
        execute_optimize_classification(args)
    elif args.command == 'optimize-steering':
        execute_optimize_steering(args)
    elif args.command == 'optimize-sample-size':
        execute_optimize_sample_size(args)
    elif args.command == 'generate-responses':
        execute_generate_responses(args)
    elif args.command == 'evaluate-responses':
        execute_evaluate_responses(args)
    elif args.command == 'multi-steer':
        execute_multi_steer(args)
    elif args.command == 'agent':
        execute_agent(args)
    elif args.command == 'modify-weights':
        execute_modify_weights(args)
    elif args.command == 'evaluate-refusal':
        execute_evaluate_refusal(args)
    elif args.command == 'inference-config':
        execute_inference_config(args)
    elif args.command == 'optimization-cache':
        execute_optimization_cache(args)
    elif args.command == 'optimize-weights':
        execute_optimize_weights(args)
    elif args.command == 'optimize-all' or args.command == 'optimize':
        execute_optimize(args)
    elif args.command == 'train-unified-goodness':
        execute_train_unified_goodness(args)
    elif args.command == 'check-linearity':
        execute_check_linearity(args)
    elif args.command == 'cluster-benchmarks':
        execute_cluster_benchmarks(args)
    elif args.command == 'geometry-search':
        execute_geometry_search(args)
    else:
        print(f"\n✗ Command '{args.command}' is not yet implemented")
        sys.exit(1)


if __name__ == '__main__':
    main()

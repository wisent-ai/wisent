"""
Main entry point for the Wisent CLI.

This module connects the argparse parser (wisent/core/parser_arguments/) to execution logic
and provides the main() function that serves as the CLI entry point.
"""

import sys
from wisent.core.parser_arguments import setup_parser
from wisent.core.branding import print_banner
from wisent.core.cli import execute_tasks, execute_generate_pairs_from_task, execute_generate_pairs, execute_diagnose_pairs, execute_get_activations, execute_diagnose_vectors, execute_create_steering_vector, execute_generate_vector_from_task, execute_generate_vector_from_synthetic, execute_optimize_classification, execute_optimize_steering, execute_optimize_sample_size, execute_generate_responses, execute_evaluate_responses, execute_multi_steer, execute_agent


def main():
    """Main entry point for the Wisent CLI."""
    # Show banner
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
    else:
        print(f"\n✗ Command '{args.command}' is not yet implemented")
        sys.exit(1)


if __name__ == '__main__':
    main()

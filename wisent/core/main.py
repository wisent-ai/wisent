"""
Main entry point for the Wisent CLI.

This module connects the argparse parser (wisent/core/parser_arguments/) to command handlers
and provides the main() function that serves as the CLI entry point.
"""

import sys
from wisent.core.parser_arguments import setup_parser
from wisent.core.branding import print_banner


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

    # Route to appropriate command handler
    try:
        if args.command == 'tasks':
            from wisent.core.commands.tasks import handle_tasks_command
            handle_tasks_command(args)
        elif args.command == 'generate-pairs':
            from wisent.core.commands.generate_pairs import handle_generate_pairs_command
            handle_generate_pairs_command(args)
        elif args.command == 'synthetic':
            from wisent.core.commands.synthetic import handle_synthetic_command
            handle_synthetic_command(args)
        elif args.command == 'test-nonsense':
            from wisent.core.commands.test_nonsense import handle_test_nonsense_command
            handle_test_nonsense_command(args)
        elif args.command == 'monitor':
            from wisent.core.commands.monitor import handle_monitor_command
            handle_monitor_command(args)
        elif args.command == 'agent':
            from wisent.core.commands.agent import handle_agent_command
            handle_agent_command(args)
        elif args.command == 'model-config':
            from wisent.core.commands.model_config import handle_model_config_command
            handle_model_config_command(args)
        elif args.command == 'configure-model':
            from wisent.core.commands.configure_model import handle_configure_model_command
            handle_configure_model_command(args)
        elif args.command == 'optimize-classification':
            from wisent.core.commands.optimize_classification import handle_optimize_classification_command
            handle_optimize_classification_command(args)
        elif args.command == 'optimize-steering':
            from wisent.core.commands.optimize_steering import handle_optimize_steering_command
            handle_optimize_steering_command(args)
        elif args.command == 'optimize-sample-size':
            from wisent.core.commands.optimize_sample_size import handle_optimize_sample_size_command
            handle_optimize_sample_size_command(args)
        elif args.command == 'full-optimize':
            from wisent.core.commands.full_optimize import handle_full_optimize_command
            handle_full_optimize_command(args)
        elif args.command == 'generate-vector':
            from wisent.core.commands.generate_vector import handle_generate_vector_command
            handle_generate_vector_command(args)
        elif args.command == 'multi-steer':
            from wisent.core.commands.multi_steer import handle_multi_steer_command
            handle_multi_steer_command(args)
        elif args.command == 'evaluate':
            from wisent.core.commands.evaluate import handle_evaluate_command
            handle_evaluate_command(args)
        else:
            print(f"Error: Unknown command '{args.command}'")
            parser.print_help()
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose if hasattr(args, 'verbose') else False:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

"""
Parser setup for the 'tasks' command.

This command runs evaluation tasks on language models.
"""
from wisent.core.utils.config_tools.parser_arguments.other.utilities.tasks_parser_basic import setup_basic_task_args
from wisent.core.utils.config_tools.parser_arguments.other.utilities.tasks_parser_steering import setup_steering_task_args


def setup_tasks_parser(parser):
    """Set up the tasks subcommand parser."""
    setup_basic_task_args(parser)
    setup_steering_task_args(parser)

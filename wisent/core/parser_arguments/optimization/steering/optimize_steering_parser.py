"""Parser setup for the 'optimize-steering' command."""

from wisent.core.steering_methods.registry import SteeringMethodRegistry

# Get available steering methods from registry
AVAILABLE_METHODS = [m.upper() for m in SteeringMethodRegistry.list_methods()]



from wisent.core.parser_arguments.optimization.steering.optimize_steering_parser_comprehensive import (
    setup_comprehensive_parser,
)
from wisent.core.parser_arguments.optimization.steering.optimize_steering_parser_methods import (
    setup_method_parsers,
)
from wisent.core.parser_arguments.optimization.steering.optimize_steering_parser_traits import (
    setup_personalization_parsers,
)
from wisent.core.parser_arguments.optimization.steering.optimize_steering_parser_welfare import (
    setup_welfare_universal_parsers,
)


def setup_steering_optimizer_parser(parser):
    """Set up the optimize-steering command parser."""
    steering_subparsers = parser.add_subparsers(
        dest="steering_action", help="Steering optimization actions"
    )
    setup_comprehensive_parser(steering_subparsers)
    setup_method_parsers(steering_subparsers)
    setup_personalization_parsers(steering_subparsers)
    setup_welfare_universal_parsers(steering_subparsers)

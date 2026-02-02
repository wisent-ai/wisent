"""Other/miscellaneous parser arguments."""

from .agent_parser import setup_agent_parser
from .modify_weights_parser import setup_modify_weights_parser
from .steering import (
    setup_multi_steer_parser,
    setup_steering_viz_parser,
    setup_discover_steering_parser,
)
from .utilities import (
    setup_monitor_parser,
    setup_test_nonsense_parser,
    setup_tasks_parser,
)

__all__ = [
    'setup_agent_parser',
    'setup_modify_weights_parser',
    'setup_multi_steer_parser',
    'setup_steering_viz_parser',
    'setup_discover_steering_parser',
    'setup_monitor_parser',
    'setup_test_nonsense_parser',
    'setup_tasks_parser',
]

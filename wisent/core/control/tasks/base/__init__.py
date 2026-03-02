"""Base task infrastructure for wisent."""

from .task_interface import (
    TaskInterface,
    TaskRegistry,
    register_task,
    get_task,
    list_tasks,
    get_task_info,
    list_task_info,
)
from .task_selector import (
    TaskSelector,
    get_tasks_for_skills_and_risks,
    expand_task_if_skill_or_risk,
)
from .diversity_processors import (
    OpenerPenaltyProcessor,
    TriePenaltyProcessor,
    PhraseLedger,
    build_diversity_processors,
)
from .file_task import (
    FileTask,
    create_file_task,
    register_file_task,
    load_tasks_from_directory,
)

__all__ = [
    'TaskInterface',
    'TaskRegistry',
    'register_task',
    'get_task',
    'list_tasks',
    'get_task_info',
    'list_task_info',
    'TaskSelector',
    'get_tasks_for_skills_and_risks',
    'expand_task_if_skill_or_risk',
    'OpenerPenaltyProcessor',
    'TriePenaltyProcessor',
    'PhraseLedger',
    'build_diversity_processors',
    'FileTask',
    'create_file_task',
    'register_file_task',
    'load_tasks_from_directory',
]

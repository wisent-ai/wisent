"""Task manager package for lm-evaluation-harness integration."""

from .core import (
    load_available_tasks,
    load_docs,
    get_available_tasks,
    is_valid_task,
    resolve_task_name,
    TaskManager,
    _task_manager,
)

from .group_handling import (
    find_working_task_from_group,
    handle_configurable_group_task,
)

from .extraction import (
    extract_individual_tasks_from_yaml,
    try_find_related_working_task,
    try_extract_working_tasks_from_group,
)

from .yaml_support import (
    save_custom_task_yaml,
    create_task_yaml_from_user_content,
    load_with_env_config,
    create_flan_held_in_files,
    load_task_with_config_dir,
)

__all__ = [
    # Core functions
    'load_available_tasks',
    'load_docs',
    'get_available_tasks',
    'is_valid_task',
    'resolve_task_name',
    'TaskManager',
    '_task_manager',
    # Group handling
    'find_working_task_from_group',
    'handle_configurable_group_task',
    # Extraction
    'extract_individual_tasks_from_yaml',
    'try_find_related_working_task',
    'try_extract_working_tasks_from_group',
    # YAML support
    'save_custom_task_yaml',
    'create_task_yaml_from_user_content',
    'load_with_env_config',
    'create_flan_held_in_files',
    'load_task_with_config_dir',
]

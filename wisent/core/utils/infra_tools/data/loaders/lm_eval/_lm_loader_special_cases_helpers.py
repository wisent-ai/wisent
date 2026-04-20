"""Additional special case handlers for lm-eval tasks (evalita, mmlusr, registry)."""

from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING

import yaml
from lm_eval.tasks import get_task_dict
from lm_eval.tasks import TaskManager as LMTaskManager
from wisent.core.utils.infra_tools.data.core.atoms import DataLoaderError

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask

__all__ = [
    "handle_evalita_llm",
    "handle_mmlusr",
    "SPECIAL_CASE_HANDLERS",
    "get_special_case_handler",
]

log = logging.getLogger(__name__)


def handle_evalita_llm(task_manager: LMTaskManager) -> dict[str, ConfigurableTask]:
    """Handle evalita_LLM special case - map to evalita-mp subtasks.

    The task 'evalita_LLM' doesn't exist in lm-eval but is an alias
    for a set of evalita-mp subtasks.

    Args:
        task_manager: Initialized LMTaskManager instance.

    Returns:
        Dictionary mapping task names to ConfigurableTask instances.

    Raises:
        DataLoaderError: If no subtasks could be loaded.
    """
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.api.group import ConfigurableGroup

    # These are the group tasks from the evalita_llm.yaml group file
    group_tasks = [
        'evalita-mp_te',
        'evalita-mp_sa',
        'evalita-mp_wic',
        'evalita-mp_hs',
        'evalita-mp_at',
        'evalita-mp_faq',
        'evalita-mp_sum_fp',
        'evalita-mp_ls',
        'evalita-mp_ner_group',
        'evalita-mp_re',
    ]
    task_dict = {}
    for group_task in group_tasks:
        try:
            result = get_task_dict([group_task], task_manager=task_manager)
            # result might contain ConfigurableGroup keys with dict values
            # or string keys with ConfigurableTask values
            for key, value in result.items():
                # If key is a ConfigurableGroup and value is a dict, extract the dict items
                if not isinstance(key, str) and isinstance(value, dict):
                    # Extract all tasks from the nested dict
                    for task_name, task_obj in value.items():
                        if isinstance(task_name, str) and isinstance(task_obj, ConfigurableTask):
                            task_dict[task_name] = task_obj
                    continue

                # If key is a string
                if isinstance(key, str):
                    if isinstance(value, dict):
                        # Nested group - add all subtasks
                        task_dict.update(value)
                    elif isinstance(value, ConfigurableTask):
                        # Direct task
                        task_dict[key] = value
        except Exception as e:
            # If a subtask fails to load, skip it but continue with others
            log.warning(f"Failed to load group task '{group_task}': {e}")
            continue
    if not task_dict:
        raise DataLoaderError("No subtasks could be loaded for evalita_LLM")
    return task_dict


def handle_mmlusr(task_manager: LMTaskManager) -> dict[str, ConfigurableTask]:
    """Handle mmlusr special case - fix dataset_name to use correct config.

    The YAML configs set dataset_name to things like 'answer_only_world_religions',
    but the actual dataset only has three configs: 'answer_only', 'question_only',
    'question_and_answer'. We need to fix the dataset_name to use the correct config.

    Args:
        task_manager: Initialized LMTaskManager instance.

    Returns:
        Dictionary mapping task names to ConfigurableTask instances.

    Raises:
        DataLoaderError: If no subtasks could be loaded.
    """
    from lm_eval.api.task import ConfigurableTask

    # Create custom YAML constructor for !function tags
    def function_constructor(loader, node):
        """Handle !function tags by returning a placeholder string."""
        return f"!function {loader.construct_scalar(node)}"

    yaml.SafeLoader.add_constructor('!function', function_constructor)

    # Find the root mmlusr yaml directory
    yaml_dir = None
    for task_name, task_info in task_manager.task_index.items():
        if 'mmlusr' in task_name:
            # Go up to the mmlusr root directory (the yaml_path might be in a subdirectory)
            # lm-eval >= 0.4.9 returns an Entry namedtuple; older returned dict.
            path = getattr(task_info, 'yaml_path', None) if not isinstance(task_info, dict) else task_info.get('yaml_path')
            while path and os.path.basename(path) != 'mmlusr':
                path = os.path.dirname(path)
            if path and os.path.basename(path) == 'mmlusr':
                yaml_dir = path
                break

    if not yaml_dir:
        raise DataLoaderError("Could not find mmlusr yaml directory")

    # Determine which variant directory to search based on task_manager task names
    # Look for a task that matches one of the three variants to determine which to load
    variant_dir = None
    for task_name in task_manager.task_index.keys():
        if 'mmlusr_answer_only' in task_name:
            variant_dir = os.path.join(yaml_dir, 'answer_only')
            break
        elif 'mmlusr_question_only' in task_name:
            variant_dir = os.path.join(yaml_dir, 'question_only')
            break
        elif 'mmlusr_question_and_answer' in task_name:
            variant_dir = os.path.join(yaml_dir, 'question_and_answer')
            break

    # If no specific variant found, default to answer_only
    if not variant_dir:
        variant_dir = os.path.join(yaml_dir, 'answer_only')

    # Get all mmlusr task files from the specific variant directory only
    task_dict = {}
    for root, dirs, files in os.walk(variant_dir):
        for file in files:
            if file.endswith('.yaml') and not file.startswith('_'):
                try:
                    yaml_path = os.path.join(root, file)
                    with open(yaml_path) as f:
                        cfg = yaml.safe_load(f)

                    # Skip if not a task config
                    if 'task' not in cfg:
                        continue

                    task_name = cfg['task']

                    # Resolve includes if present
                    if 'include' in cfg:
                        include_file = cfg.pop('include')
                        include_path = os.path.join(os.path.dirname(yaml_path), include_file)
                        with open(include_path) as inc_f:
                            base_cfg = yaml.safe_load(inc_f)
                            # Remove process_docs if it's a string placeholder
                            if 'process_docs' in base_cfg and isinstance(base_cfg['process_docs'], str):
                                del base_cfg['process_docs']
                            base_cfg.update(cfg)
                            cfg = base_cfg

                    # Remove process_docs if it's a string placeholder
                    if 'process_docs' in cfg and isinstance(cfg['process_docs'], str):
                        del cfg['process_docs']

                    # Fix doc_to_text to use actual dataset fields (column_0, column_1, etc.)
                    # instead of processed fields (question, choices, etc.)
                    if 'doc_to_text' in cfg:
                        cfg['doc_to_text'] = "{{column_0.strip()}}\nA. {{column_1}}\nB. {{column_2}}\nC. {{column_3}}\nD. {{column_4}}\nAnswer:"

                    # Fix doc_to_target to use column_5 instead of answer
                    if 'doc_to_target' in cfg:
                        cfg['doc_to_target'] = 'column_5'

                    # Fix dataset_name to use correct config
                    if 'dataset_name' in cfg:
                        dataset_name = cfg['dataset_name']
                        # Map task-specific dataset_name to actual config
                        if 'answer_only' in dataset_name:
                            cfg['dataset_name'] = 'answer_only'
                        elif 'question_only' in dataset_name:
                            cfg['dataset_name'] = 'question_only'
                        elif 'question_and_answer' in dataset_name:
                            cfg['dataset_name'] = 'question_and_answer'

                    # Create task instance
                    task = ConfigurableTask(config=cfg)
                    task_dict[task_name] = task
                    log.info(f"Loaded mmlusr subtask '{task_name}'")

                except Exception as e:
                    log.warning(f"Failed to load mmlusr subtask from '{file}': {e}")
                    import traceback
                    log.debug(traceback.format_exc())
                    continue

    if not task_dict:
        raise DataLoaderError("No subtasks could be loaded for mmlusr")

    return task_dict


def _build_registry(handler_modules: dict) -> dict:
    """Build the SPECIAL_CASE_HANDLERS registry.

    Args:
        handler_modules: Dict with handler functions to include.

    Returns:
        Registry mapping task names to handler functions.
    """
    registry = dict(handler_modules)
    registry['evalita_llm'] = handle_evalita_llm
    registry['evalita_LLM'] = handle_evalita_llm
    registry['meddialog_qsumm'] = handler_modules.get('meddialog', None)
    registry['meddialog_qsumm_perplexity'] = handler_modules.get('meddialog', None)
    registry['meddialog_raw_dialogues'] = handler_modules.get('meddialog', None)
    registry['meddialog_raw_perplexity'] = handler_modules.get('meddialog', None)
    return {k: v for k, v in registry.items() if v is not None}


def get_special_case_handler(task_name: str, registry: dict):
    """Get the special case handler for a task if one exists.

    Args:
        task_name: Name of the task to check.
        registry: The SPECIAL_CASE_HANDLERS registry dict.

    Returns:
        Handler function if one exists, None otherwise.
    """
    # Check exact match first
    if task_name in registry:
        return registry[task_name]

    # Check if task matches mmlusr pattern
    if task_name.startswith('mmlusr'):
        return handle_mmlusr

    return None

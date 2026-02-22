"""Special case handlers for lm-eval tasks that require custom loading logic."""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

from lm_eval.tasks import get_task_dict
from lm_eval.tasks import TaskManager as LMTaskManager
from wisent.core.data_loaders.core.atoms import DataLoaderError

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask

# Re-export from helpers
from wisent.core.data_loaders.loaders.lm_eval._lm_loader_special_cases_helpers import (
    handle_evalita_llm,
    handle_mmlusr,
    get_special_case_handler as _get_special_case_handler_impl,
)

__all__ = [
    "get_special_case_handler",
    "handle_bigbench",
    "handle_evalita_llm",
    "handle_global_mmlu",
    "handle_inverse_scaling",
    "handle_lambada_multilingual_stablelm",
    "handle_meddialog",
    "handle_mmlusr",
]

log = logging.getLogger(__name__)


def handle_bigbench(task_manager: LMTaskManager) -> dict[str, ConfigurableTask]:
    """Handle bigbench special case - load all three tag groups.

    Args:
        task_manager: Initialized LMTaskManager instance.

    Returns:
        Dictionary mapping task names to ConfigurableTask instances.
    """
    task_dict = {}
    for tag in ['bigbench_generate_until', 'bigbench_multiple_choice_a', 'bigbench_multiple_choice_b']:
        try:
            tag_tasks = get_task_dict([tag], task_manager=task_manager)
            task_dict.update(tag_tasks)
        except Exception as e:
            # If a tag group fails to load, skip it but continue with others
            print(f"Warning: Failed to load tag group '{tag}': {e}")
            continue
    return task_dict


def handle_inverse_scaling(task_manager: LMTaskManager) -> dict[str, ConfigurableTask]:
    """Handle inverse_scaling special case - load all subtasks.

    Args:
        task_manager: Initialized LMTaskManager instance.

    Returns:
        Dictionary mapping task names to ConfigurableTask instances.

    Raises:
        DataLoaderError: If no subtasks could be loaded.
    """
    subtasks = [
        'inverse_scaling_hindsight_neglect_10shot',
        'inverse_scaling_into_the_unknown',
        'inverse_scaling_mc',
        'inverse_scaling_memo_trap',
        'inverse_scaling_modus_tollens',
        'inverse_scaling_neqa',
        'inverse_scaling_pattern_matching_suppression',
        'inverse_scaling_quote_repetition',
        'inverse_scaling_redefine_math',
        'inverse_scaling_repetitive_algebra',
        'inverse_scaling_sig_figs',
        'inverse_scaling_winobias_antistereotype',
    ]
    task_dict = {}
    for subtask in subtasks:
        try:
            subtask_tasks = get_task_dict([subtask], task_manager=task_manager)
            task_dict.update(subtask_tasks)
        except Exception as e:
            # If a subtask fails to load, skip it but continue with others
            log.warning(f"Failed to load subtask '{subtask}': {e}")
            continue
    if not task_dict:
        raise DataLoaderError(f"No subtasks could be loaded for 'inverse_scaling'")
    return task_dict


def handle_lambada_multilingual_stablelm(task_manager: LMTaskManager) -> dict[str, ConfigurableTask]:
    """Handle lambada_multilingual_stablelm special case - load all subtasks.

    The YAML configs for these tasks contain a 'group' field that is incompatible
    with lm-eval 0.4.9.1. We need to load the configs manually without the group field.

    Args:
        task_manager: Initialized LMTaskManager instance.

    Returns:
        Dictionary mapping task names to ConfigurableTask instances.

    Raises:
        DataLoaderError: If no subtasks could be loaded.
    """
    import yaml
    import os
    from lm_eval.api.task import ConfigurableTask

    # Find the yaml directory
    yaml_dir = None
    for task_name, task_info in task_manager.task_index.items():
        if 'lambada_openai_mt_stablelm' in task_name:
            yaml_dir = os.path.dirname(task_info['yaml_path'])
            break

    if not yaml_dir:
        raise DataLoaderError("Could not find lambada_multilingual_stablelm yaml directory")

    subtasks = ['de', 'en', 'es', 'fr', 'it', 'nl', 'pt']
    task_dict = {}

    for lang in subtasks:
        try:
            yaml_file = f'lambada_mt_stablelm_{lang}.yaml'
            with open(os.path.join(yaml_dir, yaml_file)) as f:
                cfg = yaml.safe_load(f)

            # Resolve includes if present
            if 'include' in cfg:
                include_file = cfg.pop('include')
                with open(os.path.join(yaml_dir, include_file)) as inc_f:
                    base_cfg = yaml.safe_load(inc_f)
                    # Remove group field from base config
                    base_cfg = {k: v for k, v in base_cfg.items() if k != 'group'}
                    base_cfg.update(cfg)
                    cfg = base_cfg
            else:
                # Remove group field if present
                cfg = {k: v for k, v in cfg.items() if k != 'group'}

            # Create task instance
            task = ConfigurableTask(config=cfg)
            task_name = cfg['task']
            task_dict[task_name] = task
            log.info(f"Loaded subtask '{task_name}'")

        except Exception as e:
            log.warning(f"Failed to load subtask 'lambada_openai_mt_stablelm_{lang}': {e}")
            continue

    if not task_dict:
        raise DataLoaderError(f"No subtasks could be loaded for 'lambada_multilingual_stablelm'")

    return task_dict


def handle_meddialog(task_manager: LMTaskManager) -> dict[str, ConfigurableTask]:
    """Handle meddialog special case - load all subtasks without group field.

    Args:
        task_manager: Initialized LMTaskManager instance.

    Returns:
        Dictionary mapping task names to ConfigurableTask instances.

    Raises:
        DataLoaderError: If no subtasks could be loaded.
    """
    import yaml
    import os
    from lm_eval.api.task import ConfigurableTask

    yaml_dir = None
    for task_name, task_info in task_manager.task_index.items():
        if 'meddialog' in task_name:
            yaml_dir = os.path.dirname(task_info['yaml_path'])
            break

    if not yaml_dir:
        raise DataLoaderError("Could not find meddialog yaml directory")

    subtasks = ['meddialog_qsumm', 'meddialog_qsumm_perplexity', 'meddialog_raw_dialogues', 'meddialog_raw_perplexity']
    task_dict = {}

    for task_name in subtasks:
        try:
            yaml_file = f'{task_name}.yaml'
            with open(os.path.join(yaml_dir, yaml_file)) as f:
                cfg = yaml.safe_load(f)

            if 'include' in cfg:
                include_file = cfg.pop('include')
                with open(os.path.join(yaml_dir, include_file)) as inc_f:
                    base_cfg = yaml.safe_load(inc_f)
                    base_cfg = {k: v for k, v in base_cfg.items() if k != 'group'}
                    base_cfg.update(cfg)
                    cfg = base_cfg
            else:
                cfg = {k: v for k, v in cfg.items() if k != 'group'}

            task = ConfigurableTask(config=cfg)
            actual_task_name = cfg['task']
            task_dict[actual_task_name] = task
            log.info(f"Loaded subtask '{actual_task_name}'")

        except Exception as e:
            log.warning(f"Failed to load subtask '{task_name}': {e}")
            continue

    if not task_dict:
        raise DataLoaderError("No subtasks could be loaded for meddialog tasks")

    return task_dict


def handle_global_mmlu(task_manager: LMTaskManager) -> dict[str, ConfigurableTask]:
    """Handle global_mmlu special case - map to language-specific group tasks.

    Args:
        task_manager: Initialized LMTaskManager instance.

    Returns:
        Dictionary mapping task names to ConfigurableTask instances.

    Raises:
        DataLoaderError: If no subtasks could be loaded.
    """
    from lm_eval.api.task import ConfigurableTask

    group_tasks = [
        'global_mmlu_ar', 'global_mmlu_bn', 'global_mmlu_de',
        'global_mmlu_en', 'global_mmlu_es', 'global_mmlu_fr',
        'global_mmlu_hi', 'global_mmlu_id', 'global_mmlu_it',
        'global_mmlu_ja', 'global_mmlu_ko', 'global_mmlu_pt',
        'global_mmlu_sw', 'global_mmlu_yo', 'global_mmlu_zh',
    ]
    task_dict = {}
    for group_task in group_tasks:
        try:
            result = get_task_dict([group_task], task_manager=task_manager)
            for key, value in result.items():
                if isinstance(value, dict):
                    task_dict.update(value)
                elif isinstance(value, ConfigurableTask):
                    task_dict[key] = value
        except Exception as e:
            log.warning(f"Failed to load group task '{group_task}': {e}")
            continue
    if not task_dict:
        raise DataLoaderError("No subtasks could be loaded for global_mmlu")
    return task_dict


# Registry of special case handlers
SPECIAL_CASE_HANDLERS = {
    'bigbench': handle_bigbench,
    'evalita_llm': handle_evalita_llm,
    'evalita_LLM': handle_evalita_llm,
    'global_mmlu': handle_global_mmlu,
    'inverse_scaling': handle_inverse_scaling,
    'lambada_multilingual_stablelm': handle_lambada_multilingual_stablelm,
    'meddialog_qsumm': handle_meddialog,
    'meddialog_qsumm_perplexity': handle_meddialog,
    'meddialog_raw_dialogues': handle_meddialog,
    'meddialog_raw_perplexity': handle_meddialog,
}


def get_special_case_handler(task_name: str):
    """Get the special case handler for a task if one exists.

    Args:
        task_name: Name of the task to check.

    Returns:
        Handler function if one exists, None otherwise.
    """
    return _get_special_case_handler_impl(task_name, SPECIAL_CASE_HANDLERS)

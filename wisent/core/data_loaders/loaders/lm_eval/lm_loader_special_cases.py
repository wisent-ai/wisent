"""Special case handlers for lm-eval tasks that require custom loading logic."""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

from lm_eval.tasks import get_task_dict
from lm_eval.tasks import TaskManager as LMTaskManager
from wisent.core.data_loaders.core.atoms import DataLoaderError

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask

__all__ = ["get_special_case_handler", "handle_bigbench", "handle_evalita_llm", "handle_global_mmlu", "handle_inverse_scaling", "handle_lambada_multilingual_stablelm", "handle_meddialog", "handle_mmlusr"]

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

    The task 'global_mmlu' doesn't exist in lm-eval but is an alias
    for all language-specific global_mmlu tasks.

    Args:
        task_manager: Initialized LMTaskManager instance.

    Returns:
        Dictionary mapping task names to ConfigurableTask instances.

    Raises:
        DataLoaderError: If no subtasks could be loaded.
    """
    from lm_eval.api.task import ConfigurableTask

    # These are the language-specific group tasks
    group_tasks = [
        'global_mmlu_ar',
        'global_mmlu_bn',
        'global_mmlu_de',
        'global_mmlu_en',
        'global_mmlu_es',
        'global_mmlu_fr',
        'global_mmlu_hi',
        'global_mmlu_id',
        'global_mmlu_it',
        'global_mmlu_ja',
        'global_mmlu_ko',
        'global_mmlu_pt',
        'global_mmlu_sw',
        'global_mmlu_yo',
        'global_mmlu_zh',
    ]
    task_dict = {}
    for group_task in group_tasks:
        try:
            result = get_task_dict([group_task], task_manager=task_manager)
            # result might contain ConfigurableGroup or actual tasks
            # Flatten any nested structures
            for key, value in result.items():
                if isinstance(value, dict):
                    # Nested group - add all subtasks
                    task_dict.update(value)
                elif isinstance(value, ConfigurableTask):
                    # Direct task
                    task_dict[key] = value
                # Skip ConfigurableGroup objects - we want the actual tasks
        except Exception as e:
            # If a subtask fails to load, skip it but continue with others
            log.warning(f"Failed to load group task '{group_task}': {e}")
            continue
    if not task_dict:
        raise DataLoaderError("No subtasks could be loaded for global_mmlu")
    return task_dict


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
    import yaml
    import os
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
            path = task_info['yaml_path']
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


# Registry of special case handlers
SPECIAL_CASE_HANDLERS = {
    'bigbench': handle_bigbench,
    'evalita_llm': handle_evalita_llm,  # Lowercase version (normalized by loader)
    'evalita_LLM': handle_evalita_llm,  # Original capitalization (just in case)
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
    # Check exact match first
    if task_name in SPECIAL_CASE_HANDLERS:
        return SPECIAL_CASE_HANDLERS[task_name]

    # Check if task matches mmlusr pattern
    if task_name.startswith('mmlusr'):
        return handle_mmlusr

    return None

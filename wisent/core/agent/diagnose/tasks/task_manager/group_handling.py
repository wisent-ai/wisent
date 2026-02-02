"""ConfigurableGroup task handling functions."""

import os
import glob
import random
from typing import Optional, Tuple, Any


def find_working_task_from_group(group_dict, max_depth=3, current_depth=0):
    """Recursively search through nested ConfigurableGroup structures to find a working individual task."""
    if current_depth >= max_depth:
        print(f"   {'  ' * current_depth}⚠️  Max depth reached")
        return None, None
    try:
        if hasattr(group_dict, 'items') and callable(group_dict.items):
            items = list(group_dict.items())
        elif isinstance(group_dict, dict):
            items = list(group_dict.items())
        else:
            return None, None
        random.shuffle(items)
        for key, value in items[:5]:
            print(f"   {'  ' * current_depth}🔍 Checking: {key}")
            if hasattr(value, 'items') and callable(value.items):
                result_task, result_name = find_working_task_from_group(value, max_depth, current_depth + 1)
                if result_task is not None:
                    return result_task, result_name
            elif hasattr(value, 'validation_docs') or hasattr(value, 'test_docs') or hasattr(value, 'training_docs'):
                try:
                    if hasattr(value, 'validation_docs') and value.has_validation_docs():
                        docs = list(value.validation_docs())
                        if docs:
                            print(f"   {'  ' * current_depth}✅ Found working task: {key}")
                            return value, key
                    elif hasattr(value, 'test_docs') and value.has_test_docs():
                        docs = list(value.test_docs())
                        if docs:
                            print(f"   {'  ' * current_depth}✅ Found working task: {key}")
                            return value, key
                    elif hasattr(value, 'training_docs') and value.has_training_docs():
                        docs = list(value.training_docs())
                        if docs:
                            print(f"   {'  ' * current_depth}✅ Found working task: {key}")
                            return value, key
                except Exception as e:
                    print(f"   {'  ' * current_depth}❌ Task {key} failed: {str(e)[:50]}")
            if hasattr(value, 'items') and 'ConfigurableGroup' in str(type(key)):
                result_task, result_name = find_working_task_from_group(value, max_depth, current_depth + 1)
                if result_task is not None:
                    return result_task, result_name
        return None, None
    except Exception as e:
        print(f"Error exploring group: {e}")
        return None, None


def handle_configurable_group_task(task_name: str):
    """Consolidated function to handle ConfigurableGroup tasks for both CLI and processing scripts."""
    from .extraction import try_extract_working_tasks_from_group, try_find_related_working_task
    from .yaml_support import create_flan_held_in_files, load_task_with_config_dir, extract_individual_tasks_from_yaml
    try:
        from lm_eval.tasks import get_task_dict
    except ImportError as e:
        raise ImportError("lm-evaluation-harness is required. Install with: pip install lm-eval") from e
    print(f"🔍 Loading task: {task_name}")
    try:
        from lm_eval.tasks import TaskManager as LMTaskManager
        task_manager = LMTaskManager()
        task_manager.initialize_tasks()
        task_dict = get_task_dict([task_name], task_manager=task_manager)
        if task_name in task_dict:
            task = task_dict[task_name]
            print(f"   ✅ Found {task_name} in registry")
            return task, task_name
    except Exception as e:
        print(f"   ⚠️  Registry loading failed: {e}")
        try:
            from lm_eval.tasks import TaskManager as LMTaskManager
            task_manager = LMTaskManager()
            task_manager.initialize_tasks()
            all_tasks = getattr(task_manager, 'all_tasks', set())
            all_groups = getattr(task_manager, 'all_groups', set())
            print(f"   📊 Registry check: {len(all_tasks)} tasks, {len(all_groups)} groups available")
            if task_name in all_tasks or task_name in all_groups:
                if task_name in all_groups:
                    print(f"   💡 Found {task_name} as a ConfigurableGroup - extracting individual tasks...")
                    result = try_extract_working_tasks_from_group(task_name, task_manager)
                    if result:
                        return result
                    return None
                print(f"   💡 Found {task_name} as individual task - trying alternatives...")
                return try_find_related_working_task(task_name)
            print(f"   🔄 Task {task_name} not found in registry, trying alternatives...")
            return try_find_related_working_task(task_name)
        except Exception as registry_error:
            print(f"   ⚠️  Registry check failed: {registry_error}")
            return try_find_related_working_task(task_name)
    print(f"   🔍 Searching for custom YAML configuration for {task_name}")
    if task_name == "flan_held_in":
        yaml_file_path = create_flan_held_in_files()
        if yaml_file_path:
            config_dir = os.path.dirname(yaml_file_path)
            print(f"   🔍 Loading flan_held_in from: {config_dir}")
            try:
                task_dict = load_task_with_config_dir(task_name, config_dir)
                if task_name in task_dict:
                    print(f"   ✅ Successfully loaded {task_name}")
                    return task_dict[task_name], task_name
                print(f"   🔍 Extracting individual tasks from group...")
                individual_tasks = extract_individual_tasks_from_yaml(yaml_file_path, task_name)
                if individual_tasks:
                    for extracted_task_name in individual_tasks:
                        try:
                            individual_dict = load_task_with_config_dir(extracted_task_name, config_dir)
                            if extracted_task_name in individual_dict:
                                print(f"   ✅ Successfully loaded individual task: {extracted_task_name}")
                                return individual_dict[extracted_task_name], extracted_task_name
                        except Exception:
                            continue
            except Exception as e:
                print(f"   ❌ Failed to load flan_held_in: {e}")
    yaml_candidates = []
    for search_dir in ["wisent/parameters/tasks", ".", "tasks", "configs"]:
        if os.path.exists(search_dir):
            yaml_candidates.extend(glob.glob(os.path.join(search_dir, f"{task_name}.yaml")))
            yaml_candidates.extend(glob.glob(os.path.join(search_dir, f"{task_name}.yml")))
    for yaml_file in yaml_candidates:
        if os.path.exists(yaml_file):
            print(f"   🔍 Found YAML file: {yaml_file}")
            try:
                task_dict = load_task_with_config_dir(task_name, os.path.dirname(yaml_file))
                if task_name in task_dict:
                    print(f"   ✅ Successfully loaded {task_name}")
                    return task_dict[task_name], task_name
            except Exception as e:
                print(f"   ❌ Failed to load from {yaml_file}: {str(e)[:100]}")
    print(f"   🔄 Falling back to ConfigurableGroup handling for {task_name}")
    try:
        from lm_eval.tasks import TaskManager as LMTaskManager
        task_manager = LMTaskManager()
        task_manager.initialize_tasks()
        all_tasks = set(getattr(task_manager, 'all_tasks', []))
        all_groups = set(getattr(task_manager, 'all_groups', []))
        if task_name in all_tasks or task_name in all_groups:
            if task_name in all_groups:
                result = try_extract_working_tasks_from_group(task_name, task_manager)
                if result:
                    return result
                return None
            return try_find_related_working_task(task_name)
        return try_find_related_working_task(task_name)
    except Exception:
        return try_find_related_working_task(task_name)
    try:
        task_dict = get_task_dict([task_name])
        if task_name not in task_dict:
            return try_find_related_working_task(task_name)
        task = task_dict[task_name]
        if hasattr(task, '__dict__') and isinstance(getattr(task, '__dict__', {}), dict):
            task_dict_items = getattr(task, '__dict__', {})
            if any(isinstance(v, dict) for v in task_dict_items.values()):
                working_task = find_working_task_from_group(task_dict_items)
                if working_task:
                    return working_task
        try:
            if hasattr(task, 'validation_docs'):
                docs = list(task.validation_docs())
                if docs:
                    return task, task_name
            elif hasattr(task, 'test_docs'):
                docs = list(task.test_docs())
                if docs:
                    return task, task_name
            elif hasattr(task, 'training_docs'):
                docs = list(task.training_docs())
                if docs:
                    return task, task_name
        except Exception:
            return try_find_related_working_task(task_name)
        return try_find_related_working_task(task_name)
    except Exception:
        return try_find_related_working_task(task_name)

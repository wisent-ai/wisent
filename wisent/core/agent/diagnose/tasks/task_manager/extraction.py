"""Task extraction functions for finding working tasks from groups and registries."""

import os
import random
from typing import Optional, Tuple, List, Any

from wisent.core.errors import TaskLoadError


def extract_individual_tasks_from_yaml(yaml_file: str, group_name: str, _visited_files=None) -> List[str]:
    """Extract individual task names from a YAML configuration file."""
    try:
        import yaml
        if _visited_files is None:
            _visited_files = set()
        yaml_path_normalized = os.path.abspath(yaml_file)
        if yaml_path_normalized in _visited_files:
            print(f"   Cycle detected: {yaml_file} - skipping")
            return []
        _visited_files.add(yaml_path_normalized)
        with open(yaml_file, 'r') as f:
            yaml_content = yaml.safe_load(f)
        individual_tasks = []

        def extract_tasks_recursive(obj, depth=0):
            if depth > 5:
                return
            if isinstance(obj, dict):
                if 'task' in obj:
                    task_value = obj['task']
                    if isinstance(task_value, str):
                        individual_tasks.append(task_value)
                    elif isinstance(task_value, list):
                        for item in task_value:
                            extract_tasks_recursive(item, depth + 1)
                    elif isinstance(task_value, dict):
                        extract_tasks_recursive(task_value, depth + 1)
                for key, value in obj.items():
                    if key != 'task':
                        extract_tasks_recursive(value, depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    extract_tasks_recursive(item, depth + 1)
            elif isinstance(obj, str):
                individual_tasks.append(obj)

        extract_tasks_recursive(yaml_content)
        potential_tasks = list(set([task for task in individual_tasks if task and isinstance(task, str)]))
        print(f"   Found potential tasks/groups: {potential_tasks[:5]}...")
        resolved_tasks = []
        yaml_dir = os.path.dirname(yaml_file)
        max_tasks_to_process = 5
        for i, task_name in enumerate(potential_tasks[:max_tasks_to_process]):
            if any(suffix in task_name for suffix in ['_zeroshot_', '_fewshot_', '_cot_', '_prompt-', '_task_']):
                resolved_tasks.append(task_name)
                continue
            if len(_visited_files) < 3:
                potential_group_file = os.path.join(yaml_dir, f"{task_name}.yaml")
                if os.path.exists(potential_group_file):
                    print(f"   Found nested group file: {os.path.basename(potential_group_file)}")
                    nested_tasks = extract_individual_tasks_from_yaml(potential_group_file, task_name, _visited_files.copy())
                    resolved_tasks.extend(nested_tasks[:3])
                    continue
                for subdir in ['zeroshot', 'fewshot', 'cot']:
                    subdir_path = os.path.join(yaml_dir, task_name, subdir)
                    if os.path.isdir(subdir_path):
                        subdir_yaml = os.path.join(subdir_path, f"_{task_name}_{subdir}.yaml")
                        if os.path.exists(subdir_yaml):
                            print(f"   Found nested group in subdir: {subdir}")
                            nested_tasks = extract_individual_tasks_from_yaml(subdir_yaml, f"{task_name}_{subdir}", _visited_files.copy())
                            resolved_tasks.extend(nested_tasks[:3])
                            break
                else:
                    resolved_tasks.append(task_name)
            else:
                resolved_tasks.append(task_name)
        final_tasks = list(set(resolved_tasks))[:10]
        print(f"   Extracted individual tasks from YAML: {final_tasks}")
        return final_tasks
    except Exception as e:
        print(f"   Error extracting tasks from YAML {yaml_file}: {e}")
        return []


def try_find_related_working_task(task_name: str):
    """Aggressively find related tasks that work when the main task has issues."""
    try:
        from lm_eval.tasks import get_task_dict
        from lm_eval.tasks import TaskManager as LMTaskManager
        task_manager = LMTaskManager()
        task_manager.initialize_tasks()
        all_tasks = getattr(task_manager, 'all_tasks', set())
        all_groups = getattr(task_manager, 'all_groups', set())
        if isinstance(all_tasks, list):
            all_tasks = set(all_tasks)
        if isinstance(all_groups, list):
            all_groups = set(all_groups)
        all_available_tasks = all_tasks | all_groups
        print(f"   TaskManager has {len(all_tasks)} tasks, {len(all_groups)} groups")
        print(f"   AGGRESSIVE SEARCH for working alternatives to '{task_name}' ({len(all_available_tasks)} tasks available)...")
        if '_group' in task_name:
            base_name = task_name.replace('_group', '')
            print(f"   Trying base name: {base_name}")
            try:
                from .group_handling import handle_configurable_group_task
                return handle_configurable_group_task(base_name)
            except:
                pass
        parts = task_name.split('_')
        if len(parts) > 1:
            for i in range(len(parts) - 1, 0, -1):
                parent_name = '_'.join(parts[:i])
                print(f"   Trying parent: {parent_name}")
                try:
                    from .group_handling import handle_configurable_group_task
                    return handle_configurable_group_task(parent_name)
                except:
                    continue
        prefix = parts[0] if parts else task_name
        print(f"   Searching for ANY task starting with '{prefix}_'...")
        matching_tasks = [t for t in all_available_tasks if t.startswith(prefix + '_') and t != task_name]
        for candidate in matching_tasks[:10]:
            print(f"   Trying candidate: {candidate}")
            try:
                from .group_handling import handle_configurable_group_task
                result = handle_configurable_group_task(candidate)
                print(f"   SUCCESS! Found working alternative: {candidate}")
                return result
            except:
                continue
        if prefix in all_available_tasks:
            print(f"   Trying exact prefix: {prefix}")
            try:
                from .group_handling import handle_configurable_group_task
                return handle_configurable_group_task(prefix)
            except:
                pass
        keywords = [part for part in parts if len(part) > 2]
        for keyword in keywords:
            print(f"   Searching for tasks containing '{keyword}'...")
            keyword_tasks = [t for t in all_available_tasks if keyword in t and t != task_name]
            for candidate in keyword_tasks[:5]:
                print(f"   Trying keyword match: {candidate}")
                try:
                    from .group_handling import handle_configurable_group_task
                    result = handle_configurable_group_task(candidate)
                    print(f"   SUCCESS! Found working keyword match: {candidate}")
                    return result
                except:
                    continue
        print(f"   FAILED TO FIND CORRECT TASK: {task_name} - NO RANDOM FALLBACKS ALLOWED!")
        return None
    except Exception as e:
        print(f"   Search failed: {e}")
        return None


def try_extract_working_tasks_from_group(group_name: str, task_manager):
    """Try to extract and load individual working tasks from a problematic group."""
    try:
        from lm_eval.tasks import get_task_dict
        import yaml
        print(f"   Extracting working tasks from group: {group_name}")
        if hasattr(task_manager, 'task_index') and group_name in task_manager.task_index:
            group_info = task_manager.task_index[group_name]
            yaml_path = group_info.get('yaml_path')
            if yaml_path and os.path.exists(yaml_path):
                print(f"   Found group YAML: {yaml_path}")
                try:
                    with open(yaml_path, 'r') as f:
                        yaml_content = yaml.safe_load(f)
                    initial_tasks = []
                    if isinstance(yaml_content, dict):
                        if 'task' in yaml_content:
                            if isinstance(yaml_content['task'], list):
                                initial_tasks.extend(yaml_content['task'])
                            elif isinstance(yaml_content['task'], str):
                                initial_tasks.append(yaml_content['task'])
                        for key, value in yaml_content.items():
                            if isinstance(value, list) and key not in ['metric_list', 'generation_kwargs', 'metadata']:
                                for item in value:
                                    if isinstance(item, str) and ('_' in item or item.isalpha()):
                                        if item not in initial_tasks:
                                            initial_tasks.append(item)
                    if initial_tasks:
                        print(f"   Found {len(initial_tasks)} initial tasks from main YAML: {initial_tasks[:5]}...")
                        for task_name in initial_tasks[:15]:
                            try:
                                print(f"   Trying initial task: {task_name}")
                                result = get_task_dict([task_name], task_manager=task_manager)
                                if task_name in result:
                                    task = result[task_name]
                                    print(f"   SUCCESS: Found working initial task {task_name}")
                                    return task, task_name
                            except Exception as e:
                                print(f"      Initial task {task_name} failed: {str(e)[:50]}")
                                continue
                except Exception as yaml_parse_error:
                    print(f"   Main YAML parsing failed: {str(yaml_parse_error)[:100]}")
                try:
                    individual_tasks = extract_individual_tasks_from_yaml(yaml_path, group_name)
                    if individual_tasks:
                        print(f"   Found {len(individual_tasks)} individual tasks in group")
                        base_tasks_to_try = []
                        for task in individual_tasks:
                            if '_prompt-' in task:
                                base_task = task.split('_prompt-')[0]
                                if base_task not in base_tasks_to_try:
                                    base_tasks_to_try.append(base_task)
                        for base_task in base_tasks_to_try:
                            try:
                                print(f"   Trying base task: {base_task}")
                                result = get_task_dict([base_task], task_manager=task_manager)
                                if base_task in result:
                                    task = result[base_task]
                                    print(f"   SUCCESS: Found working base task {base_task}")
                                    return task, base_task
                            except Exception as e:
                                print(f"      Base task {base_task} failed: {str(e)[:50]}")
                                continue
                        valid_tasks = [t for t in individual_tasks if not any(x in t for x in ['{{', '}}', '_common_yaml', 'sentence:'])]
                        for individual_task in valid_tasks[:5]:
                            try:
                                print(f"   Trying individual task: {individual_task}")
                                result = get_task_dict([individual_task], task_manager=task_manager)
                                if individual_task in result:
                                    task = result[individual_task]
                                    print(f"   SUCCESS: Found working individual task {individual_task}")
                                    return task, individual_task
                            except Exception as e:
                                print(f"      Individual task {individual_task} failed: {str(e)[:50]}")
                                continue
                except Exception as yaml_error:
                    print(f"   YAML extraction failed: {str(yaml_error)[:100]}")
        print(f"   FINAL CATCH-ALL: Searching registry for tasks matching group pattern...")
        all_tasks = getattr(task_manager, 'all_tasks', set())
        if isinstance(all_tasks, list):
            all_tasks = set(all_tasks)
        candidates = []
        if group_name in all_tasks:
            candidates.append(group_name)
        group_prefix_tasks = [t for t in all_tasks if t.startswith(group_name + '_')]
        candidates.extend(group_prefix_tasks[:10])
        group_parts = [part for part in group_name.split('_') if len(part) > 2]
        for part in group_parts:
            matching_tasks = [t for t in all_tasks if part in t and t not in candidates]
            matching_tasks.sort(key=lambda x: (part in x.split('_'), len(x)), reverse=True)
            candidates.extend(matching_tasks[:3])
        seen = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate not in seen:
                unique_candidates.append(candidate)
                seen.add(candidate)
        print(f"   Found {len(unique_candidates)} candidate tasks to try...")
        for candidate in unique_candidates[:20]:
            try:
                print(f"   Trying candidate: {candidate}")
                result = get_task_dict([candidate], task_manager=task_manager)
                if candidate in result:
                    task = result[candidate]
                    print(f"   SUCCESS: Found working candidate {candidate}")
                    return task, candidate
            except Exception as e:
                print(f"      Candidate {candidate} failed: {str(e)[:50]}")
                continue
        print(f"   FAILED: Group {group_name} has no working tasks")
        return None
    except Exception as e:
        print(f"   Group extraction failed: {e}")
        return None

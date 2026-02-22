"""Core task discovery and document loading functions."""

import json
import os
import re
import random
from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher

from wisent.core.errors import TaskLoadError, TaskNotFoundError, NoDocsAvailableError


def load_available_tasks() -> List[str]:
    """Load available tasks from local tasks.json file or lm-eval registry."""
    try:
        tasks_json_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "parameters", "tasks", "tasks.json")
        if not os.path.exists(tasks_json_path):
            tasks_json_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tasks.json")
        if os.path.exists(tasks_json_path):
            with open(tasks_json_path, 'r') as f:
                tasks_data = json.load(f)
                if 'task_list' in tasks_data and tasks_data['task_list']:
                    print(f"Loaded {len(tasks_data['task_list'])} tasks from local tasks.json")
                    return tasks_data['task_list']
                elif 'tasks' in tasks_data:
                    task_names = list(tasks_data['tasks'].keys())
                    print(f"Loaded {len(task_names)} tasks from local tasks.json")
                    return task_names
    except Exception as e:
        print(f"Warning: Could not load from local tasks.json: {e}")
    try:
        from lm_eval.api.registry import ALL_TASKS
        return list(ALL_TASKS)
    except ImportError:
        try:
            import subprocess
            result = subprocess.run(['lm_eval', '--tasks', 'list'], capture_output=True, text=True, timeout=30)
            task_names = []
            for line in result.stdout.split('\n'):
                if '|' in line and not line.startswith('|---') and 'Group' not in line and 'Config Location' not in line:
                    parts = line.split('|')
                    if len(parts) >= 2:
                        task_name = parts[1].strip()
                        if task_name and not task_name.startswith('-') and task_name != 'Group':
                            task_names.append(task_name)
            return task_names
        except Exception:
            try:
                import lm_eval.tasks
                from lm_eval.tasks import get_task_dict
                try:
                    import lm_eval.tasks.openbookqa
                    from lm_eval.api.registry import TASK_REGISTRY
                    return list(TASK_REGISTRY.keys())
                except:
                    pass
                import pkgutil
                import lm_eval.tasks as tasks_pkg
                task_names = []
                for importer, modname, ispkg in pkgutil.iter_modules(tasks_pkg.__path__):
                    if not ispkg and not modname.startswith('_'):
                        task_names.append(modname)
                return task_names
            except Exception as e:
                raise TaskLoadError(task_name="lm-eval task discovery", cause=e)


def load_docs(task, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load documents from the most appropriate split (validation -> test -> train -> fewshot)."""
    docs = []
    if task.has_validation_docs():
        docs = list(task.validation_docs())
    elif task.has_test_docs():
        docs = list(task.test_docs())
    elif task.has_training_docs():
        docs = list(task.training_docs())
    elif hasattr(task, 'has_fewshot_docs') and task.has_fewshot_docs():
        docs = list(task.fewshot_docs())
    else:
        if hasattr(task, 'dataset') and hasattr(task, 'fewshot_split'):
            try:
                from datasets import load_dataset
                dataset = load_dataset(
                    task.dataset_path if hasattr(task, 'dataset_path') else task.dataset_name,
                    task.dataset_config_name if hasattr(task, 'dataset_config_name') else None,
                    split=task.fewshot_split
                )
                docs = [dict(item) for item in dataset]
            except Exception as e:
                raise NoDocsAvailableError(task_name=task.NAME)
        else:
            raise NoDocsAvailableError(task_name=task.NAME)
    if limit is not None and limit > 0:
        docs = docs[:limit]
    return docs


def get_available_tasks() -> List[str]:
    """Get list of all available task names."""
    return load_available_tasks()


def is_valid_task(task_name: str) -> bool:
    """Check if a task name is valid."""
    return task_name in load_available_tasks()


def resolve_task_name(task_name: str) -> str:
    """Resolve a task name to its canonical form."""
    return task_name


class TaskManager:
    """Manages lm-eval task discovery, validation, and loading."""

    def __init__(self):
        self._available_tasks = None
        self._task_name_mappings = {}

    @property
    def available_tasks(self) -> List[str]:
        """Get list of available tasks, loading if necessary."""
        if self._available_tasks is None:
            self._available_tasks = load_available_tasks()
        return self._available_tasks

    def get_available_tasks(self) -> List[str]:
        """Get list of all available tasks."""
        return self.available_tasks

    def is_valid_task(self, task_name: str) -> bool:
        """Check if a task name is valid."""
        try:
            resolved_name = self.resolve_task_name(task_name)
            return resolved_name in self.available_tasks
        except ValueError:
            return False

    def resolve_task_name(self, task_name: str) -> str:
        """Resolve a task name to its canonical form with fuzzy matching."""
        if task_name in self.available_tasks:
            return task_name
        if task_name in self._task_name_mappings:
            return self._task_name_mappings[task_name]
        best_match, best_similarity = None, 0.0
        for available_task in self.available_tasks:
            similarity = self._calculate_task_name_similarity(task_name, available_task)
            if similarity > best_similarity and similarity >= 0.6:
                best_similarity, best_match = similarity, available_task
        if best_match:
            self._task_name_mappings[task_name] = best_match
            return best_match
        suggestions = [t for t in self.available_tasks if any(w.lower() in t.lower() for w in task_name.split('_'))][:5]
        raise TaskNotFoundError(task_name=task_name, available_tasks=suggestions if suggestions else None)

    def _calculate_task_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two task names."""
        base_similarity = SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
        words1 = set(re.split(r'[_\-\s]+', name1.lower()))
        words2 = set(re.split(r'[_\-\s]+', name2.lower()))
        if words1 and words2:
            word_overlap = len(words1.intersection(words2)) / max(len(words1), len(words2))
            return (base_similarity + word_overlap) / 2
        return base_similarity

    def load_task(self, task_name: str, limit: Optional[int] = None):
        """Load a task from lm-evaluation-harness with dynamic task name resolution."""
        from .group_handling import handle_configurable_group_task
        actual_task_name = self.resolve_task_name(task_name)
        try:
            task, _ = handle_configurable_group_task(actual_task_name)
            task._limit = limit
            return task
        except Exception as e:
            if not self.is_valid_task(actual_task_name):
                raise TaskNotFoundError(task_name=task_name)
            raise TaskLoadError(task_name=task_name, cause=e)

    def split_task_data(self, task_data, split_ratio: float = 0.8, random_seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split task data into training and testing sets."""
        limit = getattr(task_data, '_limit', None)
        docs = load_docs(task_data, limit)
        random.seed(random_seed)
        shuffled_docs = docs.copy()
        random.shuffle(shuffled_docs)
        split_point = int(len(shuffled_docs) * split_ratio)
        return shuffled_docs[:split_point], shuffled_docs[split_point:]

    def prepare_prompts_from_docs(self, task, docs: List[Dict[str, Any]]) -> List[str]:
        """Prepare prompts from task documents."""
        prompts = []
        for doc in docs:
            try:
                if hasattr(task, 'doc_to_text'):
                    prompt = task.doc_to_text(doc)
                elif hasattr(task, 'doc_format'):
                    prompt = task.doc_format(doc)
                elif 'input' in doc:
                    prompt = doc['input']
                elif 'question' in doc:
                    prompt = doc['question']
                elif 'prompt' in doc:
                    prompt = doc['prompt']
                else:
                    text_fields = ['text', 'passage', 'context', 'story']
                    prompt = None
                    for field in text_fields:
                        if field in doc and isinstance(doc[field], str):
                            prompt = doc[field]
                            break
                    if prompt is None:
                        prompt = str(doc)
                prompts.append(prompt)
            except Exception as e:
                print(f"Warning: Could not create prompt from document: {e}")
                continue
        return prompts

    def get_reference_answers(self, task, docs: List[Dict[str, Any]]) -> List[str]:
        """Extract reference answers from task documents."""
        answers = []
        for doc in docs:
            try:
                if hasattr(task, 'doc_to_target'):
                    answer = task.doc_to_target(doc)
                elif hasattr(task, 'get_answer'):
                    answer = task.get_answer(doc)
                elif 'answer' in doc:
                    answer = doc['answer']
                elif 'target' in doc:
                    answer = doc['target']
                elif 'label' in doc:
                    answer = doc['label']
                elif 'output' in doc:
                    answer = doc['output']
                else:
                    answer_fields = ['correct_answer', 'gold', 'truth', 'solution']
                    answer = None
                    for field in answer_fields:
                        if field in doc:
                            answer = doc[field]
                            break
                    if answer is None:
                        answer = "UNKNOWN"
                answers.append(str(answer))
            except Exception as e:
                print(f"Warning: Could not extract answer from document: {e}")
                answers.append("UNKNOWN")
        return answers

    def register_custom_task_yaml(self, task_name: str, yaml_content: str) -> bool:
        """Register a custom YAML task configuration that can be loaded later."""
        try:
            from .yaml_support import create_task_yaml_from_user_content
            yaml_file_path = create_task_yaml_from_user_content(task_name, yaml_content)
            if yaml_file_path:
                print(f"Registered custom task configuration for '{task_name}'")
                print(f"   Saved to: {yaml_file_path}")
                return True
            return False
        except Exception as e:
            print(f"Failed to register custom task '{task_name}': {e}")
            return False


# Global instance for convenience
_task_manager = TaskManager()

"""Group task handling for populate_tasks."""

import random
from typing import Dict, Any, List, Optional

from wisent.core.utils import get_all_docs_from_task


def find_working_task_from_group(group_dict: Dict, depth: int = 0, max_depth: int = 3) -> Any:
    """Recursively search through a ConfigurableGroup to find a task with usable documents."""
    if depth > max_depth:
        print(f"   Warning: Maximum recursion depth ({max_depth}) reached, stopping search")
        return None

    items = list(group_dict.items())
    random.shuffle(items)

    for item_name, item in items[:3]:
        indent = "   " + "  " * depth
        print(f"{indent}Checking '{item_name}'...")

        if hasattr(item, 'items') and callable(item.items):
            print(f"{indent}'{item_name}' is a nested group, going deeper...")
            nested_task = find_working_task_from_group(item, depth + 1, max_depth)
            if nested_task:
                print(f"{indent}Found working task in '{item_name}'")
                return nested_task
            else:
                print(f"{indent}No working tasks in '{item_name}', continuing...")
                continue

        try:
            all_docs, split_counts = get_all_docs_from_task(item)
            if all_docs:
                print(f"{indent}Found working task '{item_name}' with {len(all_docs)} documents (splits: {split_counts})")
                return item
            else:
                print(f"{indent}'{item_name}' has no documents")
        except Exception as e:
            print(f"{indent}'{item_name}' failed: {e}")
            continue

    return None


def expand_group_task(task_name: str, get_task_dict) -> List[str]:
    """Expand a group task name into its individual subtasks."""
    try:
        task_dict = get_task_dict([task_name])
        if task_name not in task_dict:
            return [task_name]

        task = task_dict[task_name]

        if hasattr(task, 'items') and callable(task.items):
            subtasks = []
            for subtask_name, subtask in task.items():
                if hasattr(subtask, 'items') and callable(subtask.items):
                    nested = expand_group_task(subtask_name, get_task_dict)
                    subtasks.extend(nested)
                else:
                    subtasks.append(subtask_name)
            return subtasks
        else:
            return [task_name]
    except Exception as e:
        print(f"   Warning: Could not expand '{task_name}': {e}")
        return [task_name]


def get_samples_from_group_task(group_name: str, subtasks: List[str], num_samples: int = 5) -> Dict[str, Any]:
    """Get samples from a group task by sampling from its subtasks."""
    from lm_eval import evaluator

    print(f"Getting samples from group task '{group_name}' with {len(subtasks)} subtasks...")

    all_samples = []
    samples_per_task = max(1, num_samples // len(subtasks))

    for i, subtask in enumerate(subtasks[:num_samples]):
        try:
            print(f"   Getting samples from subtask {i+1}/{min(len(subtasks), num_samples)}: '{subtask}'...")

            task_dict = evaluator.get_task_dict([subtask])
            if subtask not in task_dict:
                print(f"   Subtask '{subtask}' not found, skipping...")
                continue

            task = task_dict[subtask]
            all_docs, split_counts = get_all_docs_from_task(task)

            if not all_docs:
                print(f"   No documents found for subtask '{subtask}', skipping...")
                continue

            sample_docs = all_docs[:samples_per_task] if len(all_docs) >= samples_per_task else all_docs

            for doc in sample_docs:
                sample = {"sample_id": len(all_samples) + 1, "subtask": subtask}
                sample["question"] = _extract_question(task, doc)
                sample["correct_answer"] = _extract_answer(task, doc)
                sample["choices"], sample["format"], sample["correct_choice_index"] = _extract_choices(doc)
                all_samples.append(sample)

                if len(all_samples) >= num_samples:
                    break

        except Exception as e:
            print(f"   Error processing subtask '{subtask}': {e}")
            continue

        if len(all_samples) >= num_samples:
            break

    if not all_samples:
        return {"task_name": group_name, "error": f"No samples could be retrieved from any subtasks of {group_name}", "samples": []}

    print(f"Successfully retrieved {len(all_samples)} samples from {group_name} group task")

    subtask_names = list(set([sample["subtask"] for sample in all_samples]))
    return {
        "task_name": group_name,
        "description": f"Group task '{group_name}' containing subtasks: {', '.join(subtask_names)}",
        "samples": all_samples[:num_samples],
        "num_subtasks": len(subtasks),
        "sampled_subtasks": subtask_names
    }


def process_group_task(group_name: str, sub_tasks: List[str], get_task_dict) -> Dict[str, Any]:
    """Process a group task and extract information from its subtasks."""
    from .sample_extraction import get_evaluation_method, get_category

    print(f"\nProcessing group task: {group_name} with {len(sub_tasks)} subtasks")

    result = {"name": group_name, "is_group": True, "subtasks": sub_tasks, "num_subtasks": len(sub_tasks)}

    try:
        task_dict = get_task_dict([sub_tasks[0]] if sub_tasks else [group_name])
        first_subtask_name = sub_tasks[0] if sub_tasks else group_name

        if first_subtask_name in task_dict:
            task = task_dict[first_subtask_name]
            result["evaluation_method"] = get_evaluation_method(task)
            result["category"] = get_category(task)
            result["description"] = f"Group task containing: {', '.join(sub_tasks[:5])}" + ("..." if len(sub_tasks) > 5 else "")
        else:
            result["evaluation_method"] = "unknown"
            result["category"] = "unknown"
            result["description"] = f"Group task with {len(sub_tasks)} subtasks"
    except Exception as e:
        print(f"   Error processing group: {e}")
        result["error"] = str(e)

    return result


def _extract_question(task, doc) -> str:
    """Extract question from document."""
    try:
        if hasattr(task, 'doc_to_text'):
            return str(task.doc_to_text(doc))
        for key in ['question', 'text', 'prompt']:
            if key in doc:
                return str(doc[key])
        return "Question format not recognized"
    except Exception as e:
        return f"Error extracting question: {e}"


def _extract_answer(task, doc) -> str:
    """Extract answer from document."""
    try:
        if hasattr(task, 'doc_to_target'):
            return str(task.doc_to_target(doc))
        for key in ['answer', 'target']:
            if key in doc:
                return str(doc[key])
        return "Answer format not recognized"
    except Exception as e:
        return f"Error extracting answer: {e}"


def _extract_choices(doc) -> tuple:
    """Extract choices from document."""
    try:
        if 'choices' in doc and isinstance(doc['choices'], list):
            choices = [str(choice) for choice in doc['choices']]
            gold = doc.get('gold', doc.get('label', None))
            if isinstance(gold, list) and len(gold) > 0:
                idx = gold[0]
            elif isinstance(gold, int):
                idx = gold
            else:
                idx = None
            return choices, "multiple_choice", idx
        return [], "open_ended", None
    except Exception:
        return [], "unknown", None

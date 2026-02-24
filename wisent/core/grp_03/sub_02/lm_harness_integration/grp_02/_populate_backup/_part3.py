#!/usr/bin/env python3
"""
Part 3 of populate_tasks_backup: Core task discovery and sampling utilities.
Split from populate_tasks_backup.py to meet 300-line limit.
"""

import json
import os
import sys
import random
from typing import Dict, Any, List, Optional

from wisent.core.constants import MAX_RECURSION_DEPTH, EVAL_HARNESS_NUM_SAMPLES_SMALL


def find_working_task_from_group(group_dict: Dict, depth: int = 0,
                                  max_depth: int = MAX_RECURSION_DEPTH) -> Any:
    """
    Recursively search through a ConfigurableGroup to find a task with
    usable documents.

    Args:
        group_dict: Dictionary containing tasks or nested groups
        depth: Current recursion depth
        max_depth: Maximum recursion depth allowed

    Returns:
        A task object with usable documents, or None if none found
    """
    if depth > max_depth:
        print(f"   Maximum recursion depth ({max_depth}) reached, stopping search")
        return None
    items = list(group_dict.items())
    random.shuffle(items)
    for item_name, item in items[:3]:
        indent = "   " + "  " * depth
        print(f"{indent} Checking '{item_name}'...")
        if hasattr(item, 'items') and callable(item.items):
            print(f"{indent} '{item_name}' is a nested group, going deeper...")
            nested_task = find_working_task_from_group(item, depth + 1, max_depth)
            if nested_task:
                print(f"{indent} Found working task in '{item_name}'")
                return nested_task
            else:
                print(f"{indent} No working tasks in '{item_name}', continuing...")
                continue
        try:
            has_docs = False
            test_docs = []
            if hasattr(item, 'validation_docs') and item.has_validation_docs():
                test_docs = list(item.validation_docs())
                if test_docs:
                    has_docs = True
            elif hasattr(item, 'test_docs') and item.has_test_docs():
                test_docs = list(item.test_docs())
                if test_docs:
                    has_docs = True
            elif hasattr(item, 'training_docs') and item.has_training_docs():
                test_docs = list(item.training_docs())
                if test_docs:
                    has_docs = True
            if has_docs:
                print(f"{indent} Found working task '{item_name}' with "
                      f"{len(test_docs)} documents")
                return item
            else:
                print(f"{indent} '{item_name}' has no documents")
        except Exception as e:
            print(f"{indent} '{item_name}' failed: {e}")
            continue
    return None


def get_samples_from_group_task(group_name: str, subtasks: List[str],
                                 num_samples: int = EVAL_HARNESS_NUM_SAMPLES_SMALL) -> Dict[str, Any]:
    """
    Get samples from a group task by sampling from its subtasks.

    Args:
        group_name: Name of the group task (e.g., "glue")
        subtasks: List of subtask names (e.g., ["cola", "sst2", "mrpc"])
        num_samples: Total number of samples to retrieve across all subtasks

    Returns:
        Dictionary containing samples from multiple subtasks
    """
    print(f"Getting samples from group task '{group_name}' with "
          f"{len(subtasks)} subtasks...")
    all_samples = []
    samples_per_task = max(1, num_samples // len(subtasks))
    from lm_eval import evaluator
    for i, subtask in enumerate(subtasks[:num_samples]):
        try:
            print(f"   Getting samples from subtask {i+1}/"
                  f"{min(len(subtasks), num_samples)}: '{subtask}'...")
            task_dict = evaluator.get_task_dict([subtask])
            if subtask not in task_dict:
                print(f"   Subtask '{subtask}' not found, skipping...")
                continue
            task = task_dict[subtask]
            docs = _get_task_docs(task)
            if not docs:
                print(f"   No documents found for subtask '{subtask}', skipping...")
                continue
            sample_docs = docs[:samples_per_task] if len(docs) >= samples_per_task else docs
            for j, doc in enumerate(sample_docs):
                sample = {"sample_id": len(all_samples) + 1, "subtask": subtask}
                sample["question"] = _extract_question(task, doc)
                sample["correct_answer"] = _extract_answer(task, doc)
                sample["choices"], sample["format"] = _extract_choices(doc)
                if sample["choices"] and isinstance(sample.get("choices"), list):
                    gold = doc.get('gold', doc.get('label', None))
                    if isinstance(gold, list) and len(gold) > 0:
                        sample["correct_choice_index"] = gold[0]
                    elif isinstance(gold, int):
                        sample["correct_choice_index"] = gold
                    else:
                        sample["correct_choice_index"] = None
                all_samples.append(sample)
                if len(all_samples) >= num_samples:
                    break
        except Exception as e:
            print(f"   Error processing subtask '{subtask}': {e}")
            continue
        if len(all_samples) >= num_samples:
            break
    if not all_samples:
        return {
            "task_name": group_name,
            "error": f"No samples could be retrieved from any subtasks of {group_name}",
            "samples": []
        }
    print(f"Successfully retrieved {len(all_samples)} samples from "
          f"{group_name} group task")
    subtask_names = list(set([sample["subtask"] for sample in all_samples]))
    description = (f"Group task '{group_name}' containing subtasks: "
                   f"{', '.join(subtask_names)}")
    return {
        "task_name": group_name,
        "description": description,
        "samples": all_samples[:num_samples],
        "num_subtasks": len(subtasks),
        "sampled_subtasks": subtask_names
    }


def _get_task_docs(task) -> list:
    """Extract documents from a task, trying validation, test, then training."""
    docs = []
    if hasattr(task, 'validation_docs') and task.has_validation_docs():
        docs = list(task.validation_docs())
    elif hasattr(task, 'test_docs') and task.has_test_docs():
        docs = list(task.test_docs())
    elif hasattr(task, 'training_docs') and task.has_training_docs():
        docs = list(task.training_docs())
    return docs


def _extract_question(task, doc) -> str:
    """Extract question text from a task document."""
    try:
        if hasattr(task, 'doc_to_text'):
            return str(task.doc_to_text(doc))
        for key in ('question', 'text', 'prompt'):
            if key in doc:
                return str(doc[key])
        return "Question format not recognized"
    except Exception as e:
        return f"Error extracting question: {e}"


def _extract_answer(task, doc) -> str:
    """Extract answer text from a task document."""
    try:
        if hasattr(task, 'doc_to_target'):
            return str(task.doc_to_target(doc))
        for key in ('answer', 'target'):
            if key in doc:
                return str(doc[key])
        return "Answer format not recognized"
    except Exception as e:
        return f"Error extracting answer: {e}"


def _extract_choices(doc) -> tuple:
    """Extract choices and format from a document. Returns (choices, format)."""
    try:
        if 'choices' in doc and isinstance(doc['choices'], list):
            return [str(c) for c in doc['choices']], "multiple_choice"
        return [], "open_ended"
    except Exception:
        return [], "unknown"


def load_lm_eval():
    """Load lm_eval library and handle import errors."""
    try:
        from lm_eval.tasks import get_task_dict
        return get_task_dict, None
    except ImportError as e:
        print(f"Error: lm-evaluation-harness is required. "
              f"Install with: pip install lm-eval")
        print(f"Import error: {e}")
        sys.exit(1)


def expand_group_task(task_name: str, get_task_dict) -> List[str]:
    """Try to expand a group task to get its individual sub-tasks."""
    try:
        from lm_eval.api.registry import get_group
        group_tasks = get_group(task_name)
        if group_tasks:
            return list(group_tasks)
    except Exception:
        pass
    try:
        from lm_eval.api.registry import ALL_TASKS
        sub_tasks = [t for t in ALL_TASKS if t.startswith(task_name + "_")]
        if sub_tasks:
            return sub_tasks
    except Exception:
        pass
    try:
        tasks_file = os.path.join(os.path.dirname(__file__), '..', '..', 'tasks.json')
        if os.path.exists(tasks_file):
            with open(tasks_file, 'r') as f:
                data = json.load(f)
                all_task_names = list(data.get('tasks', {}).keys())
                sub_tasks = [t for t in all_task_names
                             if t.startswith(task_name + "_")]
                if sub_tasks:
                    return sub_tasks
    except Exception as e:
        print(f"    Error reading tasks.json for group expansion: {e}")
    return []


def get_evaluation_method(task) -> str:
    """Get evaluation method from a task."""
    eval_method = "Unknown evaluation method"
    if hasattr(task, 'process_results'):
        try:
            if hasattr(task, 'OUTPUT_TYPE'):
                output_type = task.OUTPUT_TYPE
                if output_type == "multiple_choice":
                    eval_method = ("Multiple choice accuracy "
                                   "(argmax of log-likelihoods vs gold labels)")
                elif output_type == "generate_until":
                    eval_method = ("Text generation "
                                   "(exact match, F1, BLEU, ROUGE depending on task)")
                elif output_type == "loglikelihood":
                    eval_method = "Log-likelihood evaluation (perplexity, accuracy)"
                elif output_type == "loglikelihood_rolling":
                    eval_method = "Rolling log-likelihood (word/byte perplexity)"
                else:
                    eval_method = f"Unknown output type: {output_type}"
            else:
                eval_method = "Has process_results but no OUTPUT_TYPE found"
        except Exception as e:
            eval_method = f"Has process_results but couldn't inspect: {e}"
    else:
        eval_method = "No process_results method found"
    return eval_method


def get_category(task) -> str:
    """Get category from a task."""
    category = "unknown"
    if hasattr(task, 'OUTPUT_TYPE'):
        output_type = task.OUTPUT_TYPE
        if output_type == "multiple_choice":
            category = "multiple_choice"
        elif output_type == "generate_until":
            category = "open_ended_generation"
        elif output_type == "loglikelihood":
            category = "log_likelihood"
        elif output_type == "loglikelihood_rolling":
            category = "rolling_log_likelihood"
        else:
            category = f"other_{output_type}"
    else:
        category = "no_output_type"
    return category

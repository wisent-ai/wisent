#!/usr/bin/env python3
"""
Task processing: Individual and group task information extraction.
Split from populate_tasks_backup.py to meet 300-line limit.
"""

import random
from typing import Dict, Any, List, Optional

from wisent.core.lm_harness_integration._populate_backup._part3 import (
    expand_group_task,
    get_evaluation_method,
    get_category,
    _get_task_docs,
)


def get_task_info(task_name: str, get_task_dict,
                   task_registry) -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific task."""
    try:
        task_dict = get_task_dict([task_name])
        if task_name not in task_dict:
            print(f"  Task {task_name} not found directly, "
                  f"checking if it's a group task...")
            sub_tasks = expand_group_task(task_name, get_task_dict)
            if not sub_tasks:
                print(f"  Warning: Task {task_name} not found "
                      f"and no sub-tasks found")
                return None
            print(f"  Found group task with {len(sub_tasks)} sub-tasks: "
                  f"{sub_tasks[:3]}{'...' if len(sub_tasks) > 3 else ''}")
            return process_group_task(task_name, sub_tasks, get_task_dict)
        task = task_dict[task_name]
        return process_individual_task(task_name, task)
    except Exception as e:
        print(f"Error processing task {task_name}: {e}")
        return None


def process_individual_task(task_name: str, task) -> Dict[str, Any]:
    """Process an individual task to extract information."""
    info = {
        "name": task_name,
        "description": getattr(task, 'DESCRIPTION',
                               getattr(task, '__doc__', f"Task: {task_name}")),
        "example_question": "",
        "example_good_response": "",
        "example_bad_response": "",
        "evaluation_method": "",
        "category": "",
        "task_type": "individual"
    }
    try:
        docs = _get_task_docs(task)
        if docs:
            sample_doc = docs[0]
            if hasattr(task, 'doc_to_text'):
                info["example_question"] = str(task.doc_to_text(sample_doc))
            elif 'question' in sample_doc:
                info["example_question"] = str(sample_doc['question'])
            elif 'text' in sample_doc:
                info["example_question"] = str(sample_doc['text'])
            if hasattr(task, 'doc_to_target'):
                target = task.doc_to_target(sample_doc)
                info["example_good_response"] = str(target)
            elif 'answer' in sample_doc:
                info["example_good_response"] = str(sample_doc['answer'])
            elif 'target' in sample_doc:
                info["example_good_response"] = str(sample_doc['target'])
            if 'choices' in sample_doc and isinstance(sample_doc['choices'], list):
                choices = sample_doc['choices']
                if len(choices) > 1:
                    gold = sample_doc.get('gold', sample_doc.get('label', [0]))
                    if isinstance(gold, list) and len(gold) > 0:
                        correct_idx = gold[0]
                    elif isinstance(gold, int):
                        correct_idx = gold
                    else:
                        correct_idx = 0
                    if correct_idx < len(choices):
                        info["example_good_response"] = str(choices[correct_idx])
                        bad_idx = (correct_idx + 1) % len(choices)
                        info["example_bad_response"] = str(choices[bad_idx])
            if not info["example_bad_response"]:
                info["example_bad_response"] = _generate_bad_response(
                    task_name, info["example_good_response"])
    except Exception as e:
        print(f"Warning: Could not get sample for {task_name}: {e}")
    eval_method = get_evaluation_method(task)
    info["evaluation_method"] = eval_method
    category = get_category(task)
    info["category"] = category
    return info


def process_group_task(group_name: str, sub_tasks: List[str],
                        get_task_dict) -> Dict[str, Any]:
    """Process a group task by sampling from its sub-tasks."""
    random_subtask = random.choice(sub_tasks)
    print(f"  Using random sub-task '{random_subtask}' for examples")
    try:
        subtask_dict = get_task_dict([random_subtask])
        if random_subtask not in subtask_dict:
            print(f"  Warning: Random sub-task {random_subtask} not found")
            return None
        subtask = subtask_dict[random_subtask]
        example_info = extract_examples_from_task(random_subtask, subtask)
    except Exception as e:
        print(f"  Error getting examples from {random_subtask}: {e}")
        example_info = {
            "example_question": f"Example from {group_name} group",
            "example_good_response": "Sample correct response",
            "example_bad_response": "Sample incorrect response"
        }
    all_eval_methods = set()
    all_categories = set()
    print(f"  Analyzing all {len(sub_tasks)} sub-tasks for "
          f"evaluation methods and categories...")
    for i, subtask_name in enumerate(sub_tasks[:10]):
        try:
            subtask_dict = get_task_dict([subtask_name])
            if subtask_name in subtask_dict:
                subtask = subtask_dict[subtask_name]
                eval_method = get_evaluation_method(subtask)
                category = get_category(subtask)
                if eval_method != "Unknown evaluation method":
                    all_eval_methods.add(eval_method)
                if category != "unknown":
                    all_categories.add(category)
        except Exception as e:
            print(f"    Warning: Could not analyze sub-task {subtask_name}: {e}")
            continue
    info = {
        "name": group_name,
        "description": f"Group task containing {len(sub_tasks)} sub-tasks",
        "example_question": example_info.get("example_question", ""),
        "example_good_response": example_info.get("example_good_response", ""),
        "example_bad_response": example_info.get("example_bad_response", ""),
        "evaluation_method": (list(all_eval_methods) if all_eval_methods
                              else ["Unknown evaluation method"]),
        "category": (list(all_categories) if all_categories
                     else ["unknown"]),
        "task_type": "group",
        "sub_tasks": sub_tasks,
        "sub_task_count": len(sub_tasks),
        "example_source": random_subtask
    }
    return info


def extract_examples_from_task(task_name: str, task) -> Dict[str, str]:
    """Extract example question and responses from a task."""
    info = {
        "example_question": "",
        "example_good_response": "",
        "example_bad_response": ""
    }
    try:
        docs = _get_task_docs(task)
        if docs:
            sample_doc = docs[0]
            if hasattr(task, 'doc_to_text'):
                info["example_question"] = str(task.doc_to_text(sample_doc))
            elif 'question' in sample_doc:
                info["example_question"] = str(sample_doc['question'])
            elif 'text' in sample_doc:
                info["example_question"] = str(sample_doc['text'])
            if hasattr(task, 'doc_to_target'):
                target = task.doc_to_target(sample_doc)
                info["example_good_response"] = str(target)
            elif 'answer' in sample_doc:
                info["example_good_response"] = str(sample_doc['answer'])
            elif 'target' in sample_doc:
                info["example_good_response"] = str(sample_doc['target'])
            if 'choices' in sample_doc and isinstance(sample_doc['choices'], list):
                choices = sample_doc['choices']
                if len(choices) > 1:
                    gold = sample_doc.get('gold', sample_doc.get('label', [0]))
                    if isinstance(gold, list) and len(gold) > 0:
                        correct_idx = gold[0]
                    elif isinstance(gold, int):
                        correct_idx = gold
                    else:
                        correct_idx = 0
                    if correct_idx < len(choices):
                        info["example_good_response"] = str(choices[correct_idx])
                        bad_idx = (correct_idx + 1) % len(choices)
                        info["example_bad_response"] = str(choices[bad_idx])
            if not info["example_bad_response"]:
                info["example_bad_response"] = _generate_bad_response(
                    task_name, info["example_good_response"])
    except Exception as e:
        print(f"Warning: Could not get sample for {task_name}: {e}")
    return info


def _generate_bad_response(task_name: str, good_response: str) -> str:
    """Generate a generic bad response based on task type."""
    if 'bool' in task_name.lower():
        if good_response.strip().lower().startswith("yes"):
            return "No"
        return "Yes"
    elif 'math' in task_name.lower() or 'gsm' in task_name.lower():
        return "42"
    elif 'truth' in task_name.lower():
        return "This is a common misconception"
    else:
        return "Incorrect or irrelevant response"

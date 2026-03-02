#!/usr/bin/env python3
"""
Part 4 of populate_tasks_backup: Task sample retrieval for analysis.
Split from populate_tasks_backup.py to meet 300-line limit.
"""

import random
from typing import Dict, Any, List, Optional

from wisent.core.utils.infra_tools.errors import InsufficientDataError, TaskNotFoundError
from wisent.core.utils.config_tools.constants import DISPLAY_TOP_N_TINY
from wisent.core.utils.config_tools.constants import EVAL_HARNESS_NUM_SAMPLES_SMALL, DISPLAY_TRUNCATION_MEDIUM
from wisent.core import constants as _C
from wisent.core.primitives.models.lm_harness_integration._populate_backup._part3 import (
    get_samples_from_group_task,
    expand_group_task,
    _get_task_docs,
    _extract_question,
    _extract_answer,
)
from wisent.core.primitives.models.lm_harness_integration._populate_backup._part2 import (
    get_benchmark_groups_from_readme,
)


def get_task_samples_for_analysis(task_name: str,
                                   num_samples: int = EVAL_HARNESS_NUM_SAMPLES_SMALL) -> Dict[str, Any]:
    """
    Retrieve sample questions and answers from a benchmark task for AI analysis.

    This function extracts representative examples from a benchmark that can be
    analyzed to determine what cognitive abilities the benchmark tests.

    Args:
        task_name: Name of the task to analyze
        num_samples: Number of samples to retrieve (default 5)

    Returns:
        Dictionary containing task info and samples for analysis
    """
    try:
        from lm_eval import evaluator
        print(f"Loading task: {task_name}")
        # Step 1: Try individual task first
        try:
            task_dict = evaluator.get_task_dict([task_name])
            expanded_tasks = list(task_dict.keys())
            if len(expanded_tasks) == 1:
                print(f"Found individual task: {task_name}")
                task = task_dict[task_name]
                resolved_name = task_name
            elif len(expanded_tasks) > 1:
                print(f"Found group task '{task_name}' with "
                      f"{len(expanded_tasks)} subtasks: "
                      f"{expanded_tasks[:_C.DISPLAY_TOP_N_MINI]}{'...' if len(expanded_tasks) > _C.DISPLAY_TOP_N_MINI else ''}")
                return get_samples_from_group_task(
                    task_name, expanded_tasks, num_samples)
            else:
                raise InsufficientDataError(reason="No tasks returned")
        except Exception as e:
            # Step 2: Try as group task
            print(f"Individual task failed, trying as group task...")
            try:
                subtasks = expand_group_task(
                    task_name, evaluator.get_task_dict)
                if subtasks:
                    print(f"Found group expansion with {len(subtasks)} subtasks: "
                          f"{subtasks[:DISPLAY_TOP_N_TINY]}{'...' if len(subtasks) > DISPLAY_TOP_N_TINY else ''}")
                    return get_samples_from_group_task(
                        task_name, subtasks, num_samples)
                else:
                    raise InsufficientDataError(
                        reason="No group expansion found")
            except Exception as e2:
                # Step 3: Try as group of groups (large size)
                print(f"Group task failed, trying as large group...")
                try:
                    from lm_eval.tasks import TaskManager
                    tm = TaskManager()
                    tm.initialize_tasks()
                    all_groups = getattr(tm, 'all_groups', set())
                    if task_name in all_groups:
                        print(f"Found in groups registry, "
                              f"attempting large expansion...")
                        expanded_dict = evaluator.get_task_dict([task_name])
                        if expanded_dict:
                            all_subtasks = list(expanded_dict.keys())
                            if len(all_subtasks) > 1:
                                print(f"Large group expansion: "
                                      f"{len(all_subtasks)} subtasks")
                                return get_samples_from_group_task(
                                    task_name, all_subtasks, num_samples)
                    raise TaskNotFoundError(task_name=task_name)
                except Exception as e3:
                    # Step 4: Try README-based discovery
                    return _try_readme_groups(
                        task_name, num_samples, evaluator, e, e2, e3)
        # Get task description
        description = getattr(task, 'DESCRIPTION',
                              getattr(task, '__doc__', f"Task: {task_name}"))
        # Get documents with robust error handling
        docs = _get_docs_with_error_handling(task, task_name, description)
        if isinstance(docs, dict):
            return docs  # Error dict was returned
        if not docs:
            return {
                "task_name": task_name,
                "description": description,
                "error": "No documents found for this task",
                "samples": []
            }
        # Sample documents
        sample_docs = _sample_documents(docs, num_samples)
        # Extract samples
        samples = _build_samples(task, task_name, sample_docs)
        task_info = {
            "task_name": task_name,
            "description": description,
            "total_docs": len(docs),
            "sampled_docs": len(samples),
            "output_type": getattr(task, 'OUTPUT_TYPE', 'unknown'),
            "samples": samples
        }
        return task_info
    except ImportError as e:
        return {
            "task_name": task_name,
            "error": f"lm-evaluation-harness not installed: {e}",
            "samples": []
        }
    except Exception as e:
        return {
            "task_name": task_name,
            "error": f"Error retrieving samples: {e}",
            "samples": []
        }


def _try_readme_groups(task_name, num_samples, evaluator, e, e2, e3):
    """Try to find benchmark groups by reading README from lm-eval-harness."""
    print(f"Large group failed, trying to read README for '{task_name}'...")
    try:
        readme_data = get_benchmark_groups_from_readme(task_name)
        benchmark_groups = readme_data.get('groups', [])
        readme_tags = readme_data.get('tags', [])
        if benchmark_groups:
            print(f"Found {len(benchmark_groups)} groups from README: "
                  f"{benchmark_groups}")
            print(f"README-determined tags: {readme_tags}")
            all_samples = []
            for group_name in benchmark_groups:
                print(f"   Processing README group: {group_name}")
                try:
                    group_task_dict = evaluator.get_task_dict([group_name])
                    group_subtasks = list(group_task_dict.keys())
                    group_samples = get_samples_from_group_task(
                        group_name, group_subtasks,
                        num_samples // len(benchmark_groups))
                    if 'samples' in group_samples:
                        all_samples.extend(group_samples['samples'])
                except Exception as ge:
                    print(f"   Failed to process {group_name}: {ge}")
            if all_samples:
                return {
                    "task_name": task_name,
                    "resolved_groups": benchmark_groups,
                    "readme_tags": readme_tags,
                    "samples": all_samples[:num_samples],
                    "total_groups": len(benchmark_groups)
                }
        raise InsufficientDataError(
            reason=f"No README groups found for '{task_name}'")
    except Exception as e4:
        return {
            "task_name": task_name,
            "error": (f"Task '{task_name}' not found: "
                      f"individual ({e}), group ({e2}), "
                      f"large group ({e3}), README ({e4})"),
            "samples": []
        }


def _get_docs_with_error_handling(task, task_name, description):
    """Get documents from task with error handling. Returns docs or error dict."""
    docs = []
    try:
        docs = _get_task_docs(task)
    except Exception as e:
        error_msg = str(e)
        print(f"Document retrieval error for {task_name}: {error_msg}")
        if "utils" in error_msg and "has no attribute" in error_msg:
            return {
                "task_name": task_name, "description": description,
                "error": f"Task has internal lm-eval dependency issue: {error_msg}",
                "samples": []
            }
        elif "expected str, bytes or os.PathLike object, not NoneType" in error_msg:
            return {
                "task_name": task_name, "description": description,
                "error": f"Task has internal lm-eval configuration issue: {error_msg}",
                "samples": []
            }
        elif "module" in error_msg and "has no attribute" in error_msg:
            return {
                "task_name": task_name, "description": description,
                "error": f"Task has missing dependency: {error_msg}",
                "samples": []
            }
        else:
            print(f"Warning: Could not get docs for {task_name}: {e}")
    return docs


def _sample_documents(docs, num_samples):
    """Sample documents from the full document list."""
    if len(docs) <= num_samples:
        return docs
    if len(docs) > 100:
        section_size = len(docs) // num_samples
        sample_docs = []
        for i in range(num_samples):
            start_idx = i * section_size
            end_idx = min((i + 1) * section_size, len(docs))
            random_idx = random.randint(start_idx, min(end_idx - 1, len(docs) - 1))
            sample_docs.append(docs[random_idx])
        return sample_docs
    return random.sample(docs, num_samples)


def _build_samples(task, task_name, sample_docs):
    """Build sample dictionaries from task documents."""
    samples = []
    for i, doc in enumerate(sample_docs):
        sample = {"sample_id": i + 1}
        sample["question"] = _extract_question(task, doc)
        sample["correct_answer"] = _extract_answer(task, doc)
        sample["choices"] = []
        try:
            if 'choices' in doc and isinstance(doc['choices'], list):
                sample["choices"] = [str(c) for c in doc['choices']]
                sample["format"] = "multiple_choice"
                gold = doc.get('gold', doc.get('label', None))
                if isinstance(gold, list) and len(gold) > 0:
                    sample["correct_choice_index"] = gold[0]
                elif isinstance(gold, int):
                    sample["correct_choice_index"] = gold
                else:
                    sample["correct_choice_index"] = None
            elif 'mc1_targets' in doc or 'mc2_targets' in doc:
                sample["format"] = "multiple_choice"
                if 'mc1_targets' in doc:
                    mc1 = doc['mc1_targets']
                    if 'choices' in mc1:
                        sample["choices"] = [str(c) for c in mc1['choices']]
                        if 'labels' in mc1:
                            labels = mc1['labels']
                            if isinstance(labels, list) and len(labels) > 0:
                                try:
                                    sample["correct_choice_index"] = labels.index(1)
                                except ValueError:
                                    sample["correct_choice_index"] = 0
            elif any(key in doc for key in ['A)', 'B)', 'C)', 'D)']):
                sample["format"] = "multiple_choice"
                sample["choices"] = ["Could not extract choices automatically"]
            else:
                sample["format"] = "open_ended"
        except Exception as e:
            sample["choices"] = []
            sample["format"] = "unknown"
            print(f"Warning: Error processing choices for {task_name}: {e}")
        sample["additional_info"] = {}
        skip_keys = {'question', 'text', 'answer', 'target', 'choices',
                     'gold', 'label'}
        for key, value in doc.items():
            if key not in skip_keys:
                try:
                    if isinstance(value, dict):
                        sample["additional_info"][key] = (
                            f"Dict with keys: {list(value.keys())}")
                    elif isinstance(value, list):
                        sample["additional_info"][key] = (
                            f"List with {len(value)} items")
                    else:
                        sample["additional_info"][key] = str(value)[:DISPLAY_TRUNCATION_MEDIUM]
                except Exception:
                    sample["additional_info"][key] = "Could not convert to string"
        samples.append(sample)
    return samples

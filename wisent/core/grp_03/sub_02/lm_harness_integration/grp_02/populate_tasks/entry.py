"""Main entry point for populate_tasks."""

import json
import os
import sys
import subprocess
import random
from typing import Dict, Any, List, Optional

from wisent.core.utils import get_all_docs_from_task
from wisent.core.errors import InsufficientDataError, TaskNotFoundError

from .sample_extraction import get_evaluation_method, get_category, extract_examples_from_task
from .group_handling import expand_group_task, get_samples_from_group_task


def load_lm_eval():
    """Load lm_eval library and handle import errors."""
    try:
        from lm_eval.tasks import get_task_dict
        return get_task_dict, None
    except ImportError as e:
        print(f"Error: lm-evaluation-harness is required. Install with: pip install lm-eval")
        print(f"Import error: {e}")
        sys.exit(1)


def get_task_info(task_name: str, get_task_dict, task_registry) -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific task."""
    try:
        task_dict = get_task_dict([task_name])
        if task_name not in task_dict:
            print(f"  Task {task_name} not found directly, checking if it's a group task...")
            sub_tasks = expand_group_task(task_name, get_task_dict)
            if not sub_tasks:
                print(f"  Warning: Task {task_name} not found and no sub-tasks found")
                return None
            print(f"  Found group task with {len(sub_tasks)} sub-tasks: {sub_tasks[:3]}{'...' if len(sub_tasks) > 3 else ''}")
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
        "description": getattr(task, 'DESCRIPTION', getattr(task, '__doc__', f"Task: {task_name}")),
        "example_question": "", "example_good_response": "", "example_bad_response": "",
        "evaluation_method": "", "category": "", "task_type": "individual"
    }
    try:
        docs, split_counts = get_all_docs_from_task(task)
        if not docs:
            docs = []
        if docs:
            sample_doc = docs[0]
            if hasattr(task, 'doc_to_text'):
                info["example_question"] = str(task.doc_to_text(sample_doc))
            elif 'question' in sample_doc:
                info["example_question"] = str(sample_doc['question'])
            elif 'text' in sample_doc:
                info["example_question"] = str(sample_doc['text'])
            if hasattr(task, 'doc_to_target'):
                info["example_good_response"] = str(task.doc_to_target(sample_doc))
            elif 'answer' in sample_doc:
                info["example_good_response"] = str(sample_doc['answer'])
            elif 'target' in sample_doc:
                info["example_good_response"] = str(sample_doc['target'])
            if 'choices' in sample_doc and isinstance(sample_doc['choices'], list):
                choices = sample_doc['choices']
                if len(choices) > 1:
                    gold = sample_doc.get('gold', sample_doc.get('label', [0]))
                    correct_idx = gold[0] if isinstance(gold, list) and len(gold) > 0 else (gold if isinstance(gold, int) else 0)
                    if correct_idx < len(choices):
                        info["example_good_response"] = str(choices[correct_idx])
                        bad_idx = (correct_idx + 1) % len(choices)
                        info["example_bad_response"] = str(choices[bad_idx])
            if not info["example_bad_response"]:
                if 'bool' in task_name.lower():
                    info["example_bad_response"] = "No" if info["example_good_response"].strip().lower().startswith("yes") else "Yes"
                elif 'math' in task_name.lower() or 'gsm' in task_name.lower():
                    info["example_bad_response"] = "42"
                elif 'truth' in task_name.lower():
                    info["example_bad_response"] = "This is a common misconception"
                else:
                    info["example_bad_response"] = "Incorrect or irrelevant response"
    except Exception as e:
        print(f"Warning: Could not get sample for {task_name}: {e}")
    info["evaluation_method"] = get_evaluation_method(task)
    info["category"] = get_category(task)
    return info


def process_group_task(group_name: str, sub_tasks: List[str], get_task_dict) -> Dict[str, Any]:
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
        example_info = {"example_question": f"Example from {group_name} group", "example_good_response": "Sample correct response", "example_bad_response": "Sample incorrect response"}
    all_eval_methods, all_categories = set(), set()
    print(f"  Analyzing all {len(sub_tasks)} sub-tasks for evaluation methods and categories...")
    for subtask_name in sub_tasks[:10]:
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
    return {
        "name": group_name, "description": f"Group task containing {len(sub_tasks)} sub-tasks",
        "example_question": example_info.get("example_question", ""),
        "example_good_response": example_info.get("example_good_response", ""),
        "example_bad_response": example_info.get("example_bad_response", ""),
        "evaluation_method": list(all_eval_methods) if all_eval_methods else ["Unknown evaluation method"],
        "category": list(all_categories) if all_categories else ["unknown"],
        "task_type": "group", "sub_tasks": sub_tasks, "sub_task_count": len(sub_tasks), "example_source": random_subtask
    }


def test_sample_retrieval(task_name: str = "truthfulqa_mc1"):
    """Test function to demonstrate the get_task_samples_for_analysis function."""
    from .sample_extraction import get_task_samples_for_analysis
    print(f"\n=== Testing Sample Retrieval for '{task_name}' ===")
    result = get_task_samples_for_analysis(task_name, num_samples=3)
    if "error" in result:
        print(f"Error: {result['error']}")
        return False
    print(f"Successfully retrieved samples from '{result['task_name']}'")
    description = result.get('description') or "No description available"
    print(f"Description: {description[:200]}...")
    print(f"Total documents: {result['total_docs']}")
    print(f"Sampled documents: {result['sampled_docs']}")
    print(f"Output type: {result['output_type']}")
    print(f"\n--- Sample Questions ---")
    for sample in result['samples']:
        print(f"\nSample {sample['sample_id']}:")
        print(f"Question: {sample.get('question', 'No question')[:300]}...")
        print(f"Correct Answer: {sample.get('correct_answer', 'No answer')}")
        print(f"Format: {sample.get('format', 'unknown')}")
    return True


def test_specific_task():
    """Test specific problematic tasks for error handling."""
    from .sample_extraction import get_task_samples_for_analysis
    print("\nTesting specific problematic tasks...")
    test_tasks = ['evalita-mp_ner_fic_group', 'flan_held_in']
    for task_name in test_tasks:
        print(f"\nTesting task: {task_name}")
        try:
            result = get_task_samples_for_analysis(task_name, num_samples=3)
            print(f"Result keys: {list(result.keys())}")
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Success: Retrieved {len(result.get('samples', []))} samples")
        except Exception as e:
            print(f"Exception raised: {e}")


def main():
    """Main function to populate tasks.json."""
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        task_name = sys.argv[2] if len(sys.argv) > 2 else "truthfulqa_mc1"
        print(f"Running in TEST MODE")
        success = test_sample_retrieval(task_name)
        print(f"\n{'Test completed successfully!' if success else 'Test failed!'}")
        return
    print("Loading lm_eval library...")
    get_task_dict, task_registry = load_lm_eval()
    print("Getting all available tasks from lm_eval...")
    try:
        print("Using subprocess to get task list...")
        result = subprocess.run(['lm_eval', '--tasks', 'list'], capture_output=True, text=True, timeout=60)
        task_names = []
        for line in result.stdout.split('\n'):
            if '|' in line and not line.startswith('|---') and 'Group' not in line and 'Config Location' not in line:
                parts = line.split('|')
                if len(parts) >= 2:
                    task_name = parts[1].strip()
                    if task_name and not task_name.startswith('-') and task_name != 'Group':
                        task_names.append(task_name)
        available_tasks = task_names
        print(f"Found {len(available_tasks)} total tasks via subprocess")
    except Exception as e:
        print(f"Error getting tasks from lm_eval: {e}")
        available_tasks = ["truthfulqa_mc1", "hellaswag"]
    tasks_file = "wisent/core/tasks.json"
    print(f"\nPhase 1: Saving {len(available_tasks)} task names to {tasks_file}")
    initial_data = {"tasks": {task_name: {} for task_name in available_tasks}, "task_list": available_tasks}
    with open(tasks_file, 'w') as f:
        json.dump(initial_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(available_tasks)} task names to {tasks_file}")
    print(f"\nPhase 2: Populating detailed information for 2 test tasks...")
    tasks_to_populate = ["truthfulqa_mc1", "hellaswag"]
    with open(tasks_file, 'r') as f:
        current_data = json.load(f)
    processed = 0
    for i, task_name in enumerate(tasks_to_populate):
        print(f"Processing {i+1}/{len(tasks_to_populate)}: {task_name}")
        task_info = get_task_info(task_name, get_task_dict, task_registry)
        if task_info:
            current_data["tasks"][task_name] = task_info
            processed += 1
            with open(tasks_file, 'w') as f:
                json.dump(current_data, f, indent=2, ensure_ascii=False)
            print(f"  Updated {task_name}")
        else:
            print(f"  Failed to process {task_name}")
    print(f"\nCompleted processing. Successfully populated {processed}/{len(tasks_to_populate)} tasks")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_specific_task()
    else:
        main()
